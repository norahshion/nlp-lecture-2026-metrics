"""指標2：人的評価（評価シートの作成と集計）

使い方:
  # 1) 評価シートを作る（どちらのモデルの出力かは伏せ、順番もシャッフル）
  python human_eval_sheet.py make --outputs data/outputs.jsonl --raters member1 member2 member3 //shhe_member1.xlsxのように生成，メンバー数だけ入力

  # 2) 各評価者が human/sheet_<名前>.xlsx の「評価」シートに 1〜5 を記入

  # 3) 集計（human/ 内の sheet_*.xlsx をすべて読む）
  python human_eval_sheet.py aggregate
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from common import ROOT, load_config, load_records, paired_test

SHEET_NAME = "評価"
KEY_FILE = "key.csv"


# ---------------------------------------------------------------- シート作成
def make_sheets(records: list[dict], cfg: dict, out_dir: Path, raters: list[str],
                *, dummy_scores: bool = False) -> list[Path]:
    hcfg = cfg["human_eval"]
    criteria, scale = hcfg["criteria"], int(hcfg["scale"])
    models = {cfg["compare"]["base"], cfg["compare"]["target"]}
    items = [r for r in records if r["model"] in models]
    random.Random(hcfg["seed"]).shuffle(items)

    out_dir.mkdir(parents=True, exist_ok=True)
    key = pd.DataFrame([{"item_id": f"H{i:03d}", "id": r["id"], "set": r["set"], "model": r["model"]}
                        for i, r in enumerate(items, 1)])
    key.to_csv(out_dir / KEY_FILE, index=False, encoding="utf-8-sig")

    paths = []
    score_rng = random.Random(hcfg["seed"])
    for name in raters or ["template"]:
        wb = Workbook()
        ws = wb.active
        ws.title = SHEET_NAME
        header = ["item_id", "入力文", "生成された対話", *criteria, "コメント"]
        ws.append(header)
        for (_, k), r in zip(key.iterrows(), items):
            scores = ([score_rng.randint(1, scale) for _ in criteria]
                      if dummy_scores else [None] * len(criteria))
            ws.append([k["item_id"], r["source"], r["output"], *scores,
                       "動作確認用の仮点数（評価結果ではありません）" if dummy_scores else None])

        head_fill = PatternFill("solid", fgColor="DDE7F5")
        for c in ws[1]:
            c.font, c.fill = Font(bold=True), head_fill
        for col, width in zip("ABC", (9, 45, 60)):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "D2"

        dv = DataValidation(type="whole", operator="between", formula1="1", formula2=str(scale),
                            showErrorMessage=True, error=f"1〜{scale} の整数で入力してください")
        ws.add_data_validation(dv)
        for j in range(len(criteria)):
            col = chr(ord("D") + j)
            ws.column_dimensions[col].width = 12
            dv.add(f"{col}2:{col}{len(items) + 1}")
        ws.column_dimensions[chr(ord("D") + len(criteria))].width = 30

        info = wb.create_sheet("説明")
        info.append(["評価者", name])
        if dummy_scores:
            info.append(["動作確認専用", "点数は乱数で生成した仮値です。モデルの性能評価には使えません。"])
        info.append(["採点", f"各項目を 1（悪い）〜 {scale}（良い）の整数で記入してください"])
        info.append(["注意", "どのモデルの出力かは伏せてあります。入力文と見比べて評価してください"])
        info.append([])
        for c in criteria:
            info.append([c, hcfg.get("descriptions", {}).get(c, "")])
        info.column_dimensions["A"].width = 14
        info.column_dimensions["B"].width = 90

        path = out_dir / f"sheet_{name}.xlsx"
        wb.save(path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------- 集計
def krippendorff_alpha_ordinal(units: list[list[float]]) -> float:
    """順序尺度の Krippendorff's α。units は「1項目に対する評価者たちの点数」のリスト。"""
    units = [[v for v in u if not pd.isna(v)] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    values = sorted({v for u in units for v in u})
    if len(values) < 2:
        return float("nan")  # 全員が同じ点しか付けていないと計算できない
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)

    o = np.zeros((k, k))  # 一致行列
    for u in units:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[u[a]], idx[u[b]]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()

    delta = np.zeros((k, k))
    for c in range(k):
        for d in range(k):
            lo, hi = min(c, d), max(c, d)
            delta[c, d] = (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[d]) / 2) ** 2

    d_o = (o * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum()
    return float(1 - (n - 1) * d_o / d_e) if d_e > 0 else float("nan")


def load_filled(sheet_paths: list[Path], key: pd.DataFrame, criteria: list[str]) -> pd.DataFrame:
    """記入済みシートを縦長の表（評価者・項目・観点・点数）にする。"""
    frames = []
    for p in sheet_paths:
        rater = p.stem.removeprefix("sheet_")
        df = pd.read_excel(p, sheet_name=SHEET_NAME)
        missing = [c for c in criteria if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name} に評価列がありません: {missing}")
        long = df.melt(id_vars="item_id", value_vars=criteria, var_name="criterion", value_name="score")
        long["score"] = pd.to_numeric(long["score"], errors="coerce")
        n_blank = long["score"].isna().sum()
        if n_blank:
            print(f"  [注意] {p.name}: 未記入・数値以外が {n_blank} 個あります（集計から除外）")
        long["rater"] = rater
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["rater", "item_id", "id", "set", "model", "criterion", "score"])
    return pd.concat(frames).merge(key, on="item_id", how="left")


def aggregate(detail: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    base, target = cfg["compare"]["base"], cfg["compare"]["target"]
    d = detail.dropna(subset=["score"])
    both = pd.concat([d, d.assign(set="ALL")])

    summary = (both.groupby(["set", "model", "criterion"])["score"]
               .agg(["mean", "std", "count"]).round(3).reset_index())

    alpha_rows = []
    for crit, g in d.groupby("criterion"):
        units = g.groupby("item_id")["score"].apply(list).tolist()
        alpha_rows.append({"criterion": crit, "n_raters": g["rater"].nunique(),
                           "alpha_ordinal": round(krippendorff_alpha_ordinal(units), 3)})
    alpha = pd.DataFrame(alpha_rows)

    # 評価者平均をとってから base と lora を同じ入力どうしで比較
    per_item = d.groupby(["set", "id", "model", "criterion"])["score"].mean().reset_index()
    test_rows = []
    for (s, crit), g in pd.concat([per_item, per_item.assign(set="ALL")]).groupby(["set", "criterion"]):
        test_rows.append({"set": s, "metric": f"人的評価:{crit}", **paired_test(g, "score", base, target)})
    return {"summary": summary, "alpha": alpha, "tests": pd.DataFrame(test_rows)}


def run_aggregate(human_dir: Path, cfg: dict) -> dict[str, pd.DataFrame] | None:
    key_path = human_dir / KEY_FILE
    sheets = sorted(p for p in human_dir.glob("sheet_*.xlsx") if p.stem != "sheet_template")
    if not key_path.exists() or not sheets:
        return None
    key = pd.read_csv(key_path, encoding="utf-8-sig")
    detail = load_filled(sheets, key, cfg["human_eval"]["criteria"])
    res = aggregate(detail, cfg)
    res["detail"] = detail
    return res


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="人的評価シートの作成と集計")
    ap.add_argument("--config", default=None)
    ap.add_argument("--human-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="評価シートを作成")
    mk.add_argument("--outputs", default=str(ROOT / "data" / "outputs.jsonl"))
    mk.add_argument("--raters", nargs="*", default=[], help="評価者名（人数分のシートを作る）")
    mk.add_argument("--dummy-scores", action="store_true",
                    help="見本データ専用。仮点数を埋め、既定では human_sample/ に保存")
    sub.add_parser("aggregate", help="記入済みシートを集計")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dummy_scores = args.cmd == "make" and args.dummy_scores
    human_dir = Path(args.human_dir) if args.human_dir else ROOT / ("human_sample" if dummy_scores else "human")
    if dummy_scores:
        if Path(args.outputs).resolve() != (ROOT / "data" / "sample_outputs.jsonl").resolve():
            ap.error("--dummy-scores は data/sample_outputs.jsonl にのみ使用できます")
        if human_dir.resolve() == (ROOT / "human").resolve():
            ap.error("仮点数は本番用 human/ に保存できません。別フォルダを指定してください")
        print("[動作確認専用] 仮点数を生成します。モデルの性能評価には使えません。")
    if args.cmd == "make":
        paths = make_sheets(load_records(args.outputs), cfg, human_dir, args.raters,
                            dummy_scores=dummy_scores)
        print("作成しました:", *[str(p) for p in paths], sep="\n  ")
        print(f"対応表（評価が終わるまで評価者には見せない）: {human_dir / KEY_FILE}")
    else:
        res = run_aggregate(human_dir, cfg)
        if res is None:
            print(f"{human_dir} に key.csv または sheet_*.xlsx がありません")
            return
        for name in ("summary", "alpha", "tests"):
            print(f"\n== {name} ==\n{res[name].to_string(index=False)}")


if __name__ == "__main__":
    main()
