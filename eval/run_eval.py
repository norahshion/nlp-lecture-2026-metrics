"""評価指標をまとめて計算し、results/ に表とグラフを出力する。

  python run_eval.py --outputs data/outputs.jsonl
  python run_eval.py --outputs data/outputs.jsonl --no-bertscore   # 形式遵守率だけ素早く確認

出力:
  results/format_detail.csv     1件ごとの形式チェック結果
  results/bertscore_detail.csv  1件ごとの BERTScore
  results/human_*.csv           人的評価（human/ に記入済みシートがある場合）
  results/summary.csv           データセット × モデルごとの主要指標
  results/stats.csv             base と lora の対応のある検定
  results/summary.png           スライド用のグラフ
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

import metric_format
from common import ROOT, load_config, load_records, paired_test
from human_eval_sheet import run_aggregate

SERIES_COLORS = ["#2a78d6", "#eb6834"]  # base=青, target=オレンジ


def build_summary(fmt_sum, bs_sum, human) -> pd.DataFrame:
    s = fmt_sum[["set", "model", "n", "all_pass"]].rename(columns={"all_pass": "形式遵守率(%)"})
    if bs_sum is not None:
        s = s.merge(bs_sum[["set", "model", "vs_ref_F", "vs_src_F"]]
                    .rename(columns={"vs_ref_F": "BERTScore_F1(正解比)", "vs_src_F": "BERTScore_F1(入力比)"}),
                    on=["set", "model"], how="left")
    if human is not None:
        h = human["summary"].pivot_table(index=["set", "model"], columns="criterion", values="mean")
        h.columns = [f"人的評価:{c}" for c in h.columns]
        s = s.merge(h.reset_index(), on=["set", "model"], how="left")
    return s


def build_stats(fmt_detail, bs_detail, human, base, target) -> pd.DataFrame:
    rows = []
    metrics = [(fmt_detail.assign(v=fmt_detail["all_pass"].astype(float)), "v", "形式遵守(0/1)")]
    if bs_detail is not None:
        metrics += [(bs_detail, "vs_ref_F", "BERTScore_F1(正解比)"),
                    (bs_detail, "vs_src_F", "BERTScore_F1(入力比)")]
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

    metrics = [c for c in summary.columns if c not in ("set", "model", "n")]
    metrics = [m for m in metrics if summary[m].notna().any()]
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
            for b, v in zip(bars, vals):
                if not math.isnan(v):
                    ax.annotate(f"{v:.1f}" if "%" in m else f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                                ha="center", va="bottom", fontsize=8, color="#3d3d3a",
                                xytext=(0, 2), textcoords="offset points")
        ymax = 110 if "%" in m else 1.05 if m.startswith("BERTScore") else 5.5
        for i, s in enumerate(sets):
            if summary.loc[summary.set == s, m].isna().all():
                ax.text(i, ymax * 0.05, "正解なし", ha="center", fontsize=8, color="#6b6a63")
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
    ap.add_argument("--human-dir", default=str(ROOT / "human"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--no-bertscore", action="store_true", help="BERTScore を計算しない")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base, target = cfg["compare"]["base"], cfg["compare"]["target"]
    records = load_records(args.outputs)
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} 件を読み込みました（{args.outputs}）")

    print("[指標3] 形式遵守率モデル読み込み(初回のみ時間がかかります)...")
    fmt_detail = metric_format.evaluate(records, cfg)
    fmt_sum = metric_format.summarize(fmt_detail)
    fmt_detail.to_csv(out / "format_detail.csv", index=False, encoding="utf-8-sig")
    fmt_sum.to_csv(out / "format_summary.csv", index=False, encoding="utf-8-sig")

    bs_detail = bs_sum = None
    if not args.no_bertscore:
        import metric_bertscore
        print("[指標1] BERTScore ...")
        bs_detail = metric_bertscore.evaluate(records, cfg)
        bs_sum = metric_bertscore.summarize(bs_detail)
        bs_detail.to_csv(out / "bertscore_detail.csv", index=False, encoding="utf-8-sig")

    human = run_aggregate(Path(args.human_dir), cfg)
    if human is None:
        print("[指標2] 人的評価: 記入済みシートがないためスキップ")
    else:
        print(f"[指標2] 人的評価: 評価者 {human['detail']['rater'].nunique()} 人分を集計")
        for name in ("detail", "summary", "alpha"):
            human[name].to_csv(out / f"human_{name}.csv", index=False, encoding="utf-8-sig")

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
