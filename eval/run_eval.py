"""評価指標をまとめて計算し、results/ に表とグラフを出力する。

  python run_eval.py --outputs data/outputs.jsonl
  python run_eval.py --outputs data/outputs.jsonl --no-bertscore   # 形式遵守率だけ素早く確認

評価対象は後処理後の最終出力（output）。生出力（raw_outputs）の人的評価はここでは扱わず、
human_eval_sheet.py aggregate --target raw で別に集計する（summary・stats には混ぜない）。

出力:
  results/format_detail.csv     1件ごとの形式チェック結果
  results/bertscore_detail.csv  1件ごとの BERTScore（トークン数・切り捨ての有無つき）
  results/bertscore_chunk_detail.csv  分割ごとの BERTScore（分割情報がある場合）
  results/human_final_*.csv     人的評価（final）。human/final/ に記入済みシートがある場合
  results/summary.csv           データセット × モデルごとの主要指標
  results/stats.csv             base と lora の対応のある検定
  results/summary.png           スライド用のグラフ
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

import metric_format
from common import ROOT, load_config, load_records, paired_test
from human_eval_sheet import HumanEvalError, run_aggregate

FORMAT_COL = "後処理後の形式遵守率(%)"

SERIES_COLORS = ["#2a78d6", "#eb6834"]  # base=青, target=オレンジ


# BERTScore の列（平均は切り捨てのない件だけ。n_ 列は使った件数・切り捨てで除いた件数）
BS_COLS = {
    "vs_ref_F": "BERTScore_F1(正解比)",
    "vs_ref_n_used": "n_正解比_使用",
    "vs_ref_n_truncated": "n_正解比_切り捨て除外",
    "vs_src_F": "BERTScore_F1(入力比・全文)",
    "vs_src_n_used": "n_入力比全文_使用",
    "vs_src_n_truncated": "n_入力比全文_切り捨て除外",
    "chunk_F": "BERTScore_F1(入力比・分割単位)",
    "chunk_n_used": "n_分割単位_使用",
    "chunk_n_truncated": "n_分割単位_切り捨て除外",
    "chunk_n_no_info": "n_分割単位_分割情報なし",
}
TRUNC_OF = {"BERTScore_F1(正解比)": "n_正解比_切り捨て除外",
            "BERTScore_F1(入力比・全文)": "n_入力比全文_切り捨て除外",
            "BERTScore_F1(入力比・分割単位)": "n_分割単位_切り捨て除外"}
REVIEW_COL = "日本語要確認率(%)"  # グラフでは単独の棒にせず、日本語合格率の棒に（+要確認）として添える


def build_summary(fmt_sum, bs_sum, human) -> pd.DataFrame:
    s = fmt_sum[["set", "model", "n", "all_pass", "ja_pass", "ja_review", "chunks_all_ok", "n_chunk_known"]] \
        .rename(columns={"all_pass": FORMAT_COL, "ja_pass": "日本語合格率(%)", "ja_review": REVIEW_COL,
                         "chunks_all_ok": "全分割合格率(%)", "n_chunk_known": "n_分割情報あり"})
    if bs_sum is not None:
        s = s.merge(bs_sum[["set", "model", *BS_COLS]].rename(columns=BS_COLS), on=["set", "model"], how="left")
    if human is not None:
        h = human["summary"].pivot_table(index=["set", "model"], columns="criterion", values="mean")
        h.columns = [f"人的評価(final):{c}" for c in h.columns]
        s = s.merge(h.reset_index(), on=["set", "model"], how="left")
    return s


def build_stats(fmt_detail, bs_detail, human, base, target) -> pd.DataFrame:
    rows = []
    metrics = [(fmt_detail.assign(v=fmt_detail["all_pass"].astype(float)), "v", "後処理後の形式遵守(0/1)"),
               (fmt_detail.assign(v=(fmt_detail["ja_status"] == "合格").astype(float)), "v", "日本語合格(0/1)")]
    if fmt_detail["chunks_all_ok"].notna().any():
        metrics.append((fmt_detail, "chunks_all_ok", "全分割合格(0/1)"))
    if bs_detail is not None:  # 切り捨てのある件は NaN（_valid 列）なので、検定の組から外れる
        metrics += [(bs_detail, "vs_ref_F_valid", "BERTScore_F1(正解比)"),
                    (bs_detail, "vs_src_F_valid", "BERTScore_F1(入力比・全文)"),
                    (bs_detail, "chunk_F_valid", "BERTScore_F1(入力比・分割単位)")]
    for df, col, name in metrics:
        for s, g in pd.concat([df, df.assign(set="ALL")]).groupby("set"):
            rows.append({"set": s, "metric": name, **paired_test(g, col, base, target)})
    out = pd.DataFrame(rows)
    if human is not None:
        out = pd.concat([out, human["tests"]], ignore_index=True)
    return out.round(4)


def plot_summary(summary: pd.DataFrame, base: str, target: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    jp = [f for f in ("Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "Noto Sans CJK JP") if f in installed]
    plt.rcParams["font.family"] = jp[:1] or ["sans-serif"]

    metrics = [c for c in summary.columns
               if c not in ("set", "model", "n", REVIEW_COL) and not c.startswith("n_")
               and (summary[c].notna().any() or c == "全分割合格率(%)")]  # 3種類の判定は常に並べる
    sets = [s for s in summary["set"].unique() if s != "ALL"] or ["ALL"]

    ncols = min(3, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), squeeze=False)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    width = 0.36
    for ax, m in zip(axes.flat, metrics):
        for k, (model, color) in enumerate(zip((base, target), SERIES_COLORS)):
            vals = [summary.loc[(summary.set == s) & (summary.model == model), m].mean() for s in sets]
            xs = [i + (k - 0.5) * width for i in range(len(sets))]
            bars = ax.bar(xs, [0 if math.isnan(v) else v for v in vals], width * 0.92,
                          color=color, label=model, edgecolor="white", linewidth=1)
            sel = [(summary.set == s) & (summary.model == model) for s in sets]
            reviews = [summary.loc[q, REVIEW_COL].mean() if m == "日本語合格率(%)" else 0 for q in sel]
            excluded = [summary.loc[q, TRUNC_OF[m]].sum() if m in TRUNC_OF else 0 for q in sel]
            for b, v, rv, ex in zip(bars, vals, reviews, excluded):
                if math.isnan(v):
                    if ex > 0:  # 全件が切り捨てで除かれた
                        ax.annotate(f"除外{int(ex)}", (b.get_x() + b.get_width() / 2, 0), ha="center",
                                    va="bottom", fontsize=7, color="#6b6a63", xytext=(0, 2),
                                    textcoords="offset points")
                    continue
                label = f"{v:.1f}" if "%" in m else f"{v:.2f}"
                if rv > 0:
                    label += f"\n(+要確認{rv:.1f})"
                if ex > 0:
                    label += f"\n(除外{int(ex)})"
                ax.annotate(label, (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8, color="#3d3d3a",
                            xytext=(0, 2), textcoords="offset points")
        ymax = 110 if "%" in m else 1.05 if m.startswith("BERTScore") else 5.5
        for i, s in enumerate(sets):
            rows_s = summary.loc[summary.set == s]
            if rows_s[m].isna().all():
                n_ex = rows_s[TRUNC_OF[m]].sum() if m in TRUNC_OF else 0
                if n_ex == 0:
                    ax.text(i, ymax * 0.05, "正解なし" if "正解比" in m else "対象外",
                            ha="center", fontsize=8, color="#6b6a63")
        if m in TRUNC_OF and summary[TRUNC_OF[m]].sum() > 0:
            ax.text(0.5, -0.22, "（除外n＝512トークン超で集計から除いた件数）", transform=ax.transAxes,
                    ha="center", fontsize=7, color="#6b6a63")
        ax.set_title(m, fontsize=10, color="#1f1f1e")
        ax.set_xticks(range(len(sets)), [f"データ{s}" for s in sets])
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#e4e3dc", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8, colors="#6b6a63")
        ax.tick_params(axis="x", labelsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.08 / nrows))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="対話変換システムの評価")
    ap.add_argument("--outputs", default=str(ROOT / "data" / "outputs.jsonl"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--human-dir", default=str(ROOT / "human" / "final"),
                    help="final（後処理後の出力）の人的評価シートのフォルダ")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--no-bertscore", action="store_true", help="BERTScore を計算しない")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base, target = cfg["compare"]["base"], cfg["compare"]["target"]
    records = load_records(args.outputs)
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} 件を読み込みました（{args.outputs}）")

    print("[指標3] 後処理後の形式遵守率・日本語判定・全分割合格 ...")
    fmt_detail = metric_format.evaluate(records, cfg)
    fmt_sum = metric_format.summarize(fmt_detail)
    fmt_detail.to_csv(out / "format_detail.csv", index=False, encoding="utf-8-sig")
    fmt_sum.to_csv(out / "format_summary.csv", index=False, encoding="utf-8-sig")

    bs_detail = bs_sum = None
    if not args.no_bertscore:
        import metric_bertscore
        print("[指標1] BERTScore ...")
        bs_detail, bs_chunks = metric_bertscore.evaluate(records, cfg)
        bs_sum = metric_bertscore.summarize(bs_detail)
        bs_detail.to_csv(out / "bertscore_detail.csv", index=False, encoding="utf-8-sig")
        if len(bs_chunks):
            bs_chunks.to_csv(out / "bertscore_chunk_detail.csv", index=False, encoding="utf-8-sig")
        n_tr = {k: int((bs_detail[f"{k}_truncated"] == True).sum()) for k in metric_bertscore.KINDS}  # noqa: E712
        n_tr["chunk"] = int((bs_detail["chunk_n_truncated"] > 0).sum())
        if any(n_tr.values()):
            print(f"  [注意] 512トークン超のため集計から除いた件数: 正解比 {n_tr['vs_ref']} / "
                  f"入力比・全文 {n_tr['vs_src']} / 入力比・分割単位 {n_tr['chunk']}（detail には値を残しています）")

    try:
        human = run_aggregate(Path(args.human_dir), cfg, target="final")
    except HumanEvalError as e:
        print(f"[停止] 人的評価: {e}", file=sys.stderr)
        sys.exit(1)
    if human is None:
        print("[指標2] 人的評価(final): シートがないためスキップ")
    else:
        d = human["detail"]
        print(f"[指標2] 人的評価(final): 評価者 {d['rater'].nunique()} 人分を集計"
              f"（数値 {(d.status == '数値').sum()} / 評価不能 {(d.status == '評価不能').sum()}"
              f" / 未記入 {(d.status == '未記入').sum()}）")
        for name in ("detail", "summary", "alpha", "reasons"):
            human[name].to_csv(out / f"human_final_{name}.csv", index=False, encoding="utf-8-sig")

    summary = build_summary(fmt_sum, bs_sum, human).round(3)
    stats = build_stats(fmt_detail, bs_detail, human, base, target)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(out / "stats.csv", index=False, encoding="utf-8-sig")
    plot_summary(summary, base, target, out / "summary.png")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n== summary ==\n" + summary.to_string(index=False))
        print("\n== stats（base vs " + target + "）==\n" + stats.to_string(index=False))
    print(f"\n結果を {out} に保存しました")


if __name__ == "__main__":
    main()
