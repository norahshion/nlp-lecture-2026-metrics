"""指標2：人的評価（評価シートの作成と集計）

評価対象（--target）は2種類あり、シート・対応表・集計先を分ける。
  final : 後処理後の最終出力（outputs.jsonl の output）… システム全体の評価（既定）
  raw   : モデルが生成した文章そのもの（raw_outputs）… 生成文の品質評価
          分割ごとに【分割1/3】のような境界を付けて順番どおりに並べる。本文は一切変えない。

使い方（eval/ で実行。評価者名には member1 などの ID を使う）:
  # 1) 評価シートを作る（モデル名は伏せ、順番もシャッフル）
  python human_eval_sheet.py make --raters member1 member2 member3              # final → human/final/
  python human_eval_sheet.py make --target raw --raters member1 member2 member3 # raw   → human/raw/

  # 2) 各評価者が sheet_<ID>.xlsx の「評価」シートに 1〜5 または「評価不能」を記入

  # 3) 集計（そのフォルダの sheet_*.xlsx をすべて読む）
  python human_eval_sheet.py aggregate                 # → results/human_final/
  python human_eval_sheet.py aggregate --target raw    # → results/human_raw/

既にシートや対応表があるフォルダには作成しない（上書きは --overwrite を付けたときだけ）。
集計では、評価対象・評価項目・基準の版・対応表がシートと一致しなければ停止する。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.exceptions import IllegalCharacterError
from openpyxl.worksheet.datavalidation import DataValidation

from common import ROOT, load_config, load_records, paired_test

SHEET_NAME = "評価"
META_SHEET = "_meta"
KEY_FILE = "key.csv"
META_FILE = "meta.json"
REASON_COL = "評価不能の理由"
COMMENT_COL = "コメント"
NOTE_COL = "表示上の注意"
EXCEL_CELL_LIMIT = 32767
TARGETS = ("final", "raw")
TARGET_LABEL = {
    "final": "後処理後の最終出力（システム全体の評価）",
    "raw": "モデルが生成した文章（後処理前の生出力）",
}
CONTENT_HEADER = {"final": "評価対象：後処理後の対話", "raw": "評価対象：生成された文章（分割ごと）"}


class HumanEvalError(Exception):
    pass


# ---------------------------------------------------------------- 表示用テキスト
def chunk_marker(i: int, n: int, raw: str) -> str:
    return f"【分割{i}/{n}】" + ("（空の生出力）" if not raw.strip() else "")


def raw_cell_text(raws: list[str]) -> str:
    """生出力を分割境界つきで順番どおりに並べる。本文は加工しない。"""
    n = len(raws)
    return "\n\n".join(f"{chunk_marker(i, n, x)}\n{x}" for i, x in enumerate(raws, 1))


def build_items(records: list[dict], cfg: dict, target: str) -> list[dict]:
    models = {cfg["compare"]["base"], cfg["compare"]["target"]}
    recs = [r for r in records if r["model"] in models]
    items = []
    if target == "final":
        for r in recs:
            out = r["output"]
            items.append({**r, "content": out, "note": "" if out.strip() else "出力が空です"})
        return items

    missing = [f"{r['id']}({r['model']})" for r in recs
               if not isinstance(r.get("raw_outputs"), list) or len(r["raw_outputs"]) == 0]
    if missing:
        raise HumanEvalError("raw_outputs がないため raw のシートを作れません（final への代替はしません）: "
                             + ", ".join(missing))
    bad = [f"{r['id']}({r['model']})" for r in recs if not all(isinstance(x, str) for x in r["raw_outputs"])]
    if bad:
        raise HumanEvalError("raw_outputs に文字列以外が含まれています: " + ", ".join(bad))
    for r in recs:
        raws = r["raw_outputs"]
        empty = [str(i) for i, x in enumerate(raws, 1) if not x.strip()]
        if len(empty) == len(raws):
            note = "全分割の生出力が空です"
        elif empty:
            note = "空の分割: " + ", ".join(empty)
        else:
            note = ""
        items.append({**r, "content": raw_cell_text(raws), "note": note})
    return items


# ---------------------------------------------------------------- シート作成
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta(cfg: dict, target: str, key_hash: str, n_items: int, dummy: bool) -> dict:
    h = cfg["human_eval"]
    return {"target": target, "criteria": list(h["criteria"]), "criteria_version": h.get("criteria_version"),
            "scale": int(h["scale"]), "key_sha256": key_hash, "n_items": n_items, "dummy_scores": dummy}


def make_sheets(records: list[dict], cfg: dict, out_dir: Path, raters: list[str], *,
                target: str = "final", dummy_scores: bool = False, overwrite: bool = False) -> list[Path]:
    if target not in TARGETS:
        raise HumanEvalError(f"--target は {TARGETS} のどれかです")
    hcfg = cfg["human_eval"]
    criteria, scale, unable = list(hcfg["criteria"]), int(hcfg["scale"]), hcfg.get("unable_label", "評価不能")
    names = raters or ["template"]

    existing = [p for p in [out_dir / KEY_FILE, out_dir / META_FILE, *out_dir.glob("sheet_*.xlsx")] if p.exists()]
    if existing and not overwrite:
        raise HumanEvalError(f"{out_dir} には既に評価シート等があります（{', '.join(p.name for p in existing)}）。"
                             "別のフォルダを指定するか、作り直す場合だけ --overwrite を付けてください")

    items = build_items(records, cfg, target)
    random.Random(f"{hcfg['seed']}-{target}").shuffle(items)
    keyed = [{**it, "item_id": f"{'H' if target == 'final' else 'R'}{i:03d}"} for i, it in enumerate(items, 1)]

    # Excel のセル上限を超えるものは、切り捨てずに停止する
    over = [f"{it['item_id']}={it['id']}({it['model']}) の{col} {len(it[key])}文字"
            for it in keyed for key, col in (("source", "入力文"), ("content", "評価対象"))
            if len(it[key]) > EXCEL_CELL_LIMIT]
    if over:
        raise HumanEvalError(f"Excel の1セルの上限（{EXCEL_CELL_LIMIT}文字）を超えるため作成できません: " + "; ".join(over))

    out_dir.mkdir(parents=True, exist_ok=True)
    key = pd.DataFrame([{"item_id": it["item_id"], "id": it["id"], "set": it["set"], "model": it["model"],
                         "target": target} for it in keyed])
    key.to_csv(out_dir / KEY_FILE, index=False, encoding="utf-8-sig")
    meta = _meta(cfg, target, _sha256(out_dir / KEY_FILE), len(keyed), dummy_scores)
    (out_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = []
    score_rng = random.Random(hcfg["seed"])
    for name in names:
        wb = Workbook()
        ws = wb.active
        ws.title = SHEET_NAME
        header = ["item_id", "入力文", CONTENT_HEADER[target], NOTE_COL, *criteria, REASON_COL, COMMENT_COL]
        ws.append(header)
        for it in keyed:
            scores = ([score_rng.randint(1, scale) for _ in criteria] if dummy_scores else [None] * len(criteria))
            row = [it["item_id"], it["source"], it["content"], it["note"], *scores, None,
                   "動作確認用の仮点数（評価結果ではありません）" if dummy_scores else None]
            try:
                ws.append(row)
            except IllegalCharacterError as e:
                raise HumanEvalError(f"{it['item_id']}={it['id']}({it['model']}) に Excel で扱えない制御文字が"
                                     "含まれるため作成できません（本文は変更しません）") from e

        head_fill = PatternFill("solid", fgColor="DDE7F5")
        for c in ws[1]:
            c.font, c.fill = Font(bold=True), head_fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for col, width in zip("ABCD", (8, 45, 70, 16)):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "E2"

        first, last = 5, 4 + len(criteria)
        n_rows = len(keyed) + 1
        dv = DataValidation(type="list", formula1='"' + ",".join([*map(str, range(1, scale + 1)), unable]) + '"',
                            allow_blank=True, showErrorMessage=True,
                            error=f"1〜{scale} の整数か「{unable}」を入力してください")
        ws.add_data_validation(dv)
        for j in range(first, last + 1):
            col = get_column_letter(j)
            ws.column_dimensions[col].width = 13
            dv.add(f"{col}2:{col}{n_rows}")
        reason_col = get_column_letter(last + 1)
        dv_r = DataValidation(type="list", formula1='"' + ",".join(hcfg.get("unable_reasons", [])) + '"',
                              allow_blank=True)
        ws.add_data_validation(dv_r)
        dv_r.add(f"{reason_col}2:{reason_col}{n_rows}")
        ws.column_dimensions[reason_col].width = 18
        ws.column_dimensions[get_column_letter(last + 2)].width = 30

        _write_info_sheet(wb.create_sheet("説明"), name, target, hcfg, scale, unable, dummy_scores)

        ms = wb.create_sheet(META_SHEET)
        ms["A1"] = json.dumps(meta, ensure_ascii=False)
        ms.sheet_state = "hidden"

        path = out_dir / f"sheet_{name}.xlsx"
        wb.save(path)
        _verify_saved(path, keyed)
        paths.append(path)
    return paths


def _write_info_sheet(info, name, target, hcfg, scale, unable, dummy):
    bold = Font(bold=True)
    info.append(["評価者", name])
    if dummy:
        info.append(["動作確認専用", "点数は乱数で生成した仮値です。モデルの性能評価には使えません。"])
    info.append(["評価対象", TARGET_LABEL[target]])
    if target == "raw":
        info.append(["表示", "【分割1/3】などは分割の境界を示す表示で、評価対象ではありません。"
                             "生成された文章は、翻訳・修正・抽出をせず、そのまま順番どおりに表示しています。"])
    else:
        info.append(["表示", "後処理後の出力をそのまま表示しています。出力が空の場合、評価対象の欄は空欄です"
                             "（「表示上の注意」に「出力が空です」と表示）。"])
    info.append(["採点", f"各項目に 1（悪い）〜 {scale}（良い）の整数を記入してください。"
                         f"下の目安は 1・3・5 点のもので、2・4 はその中間です。"
                         f"判断できない項目は「{unable}」と記入し、「{REASON_COL}」を選んでください。"])
    info.append(["注意", "どのモデルの出力かは伏せてあります。入力文と見比べて評価してください。"])
    info.append([])
    info.append(["特別な場合の扱い"])
    info.cell(info.max_row, 1).font = bold
    for rule in hcfg.get("rules", []):
        info.append(["", rule])
    info.append([])
    for c in hcfg["criteria"]:
        info.append([c, hcfg.get("descriptions", {}).get(c, "")])
        info.cell(info.max_row, 1).font = bold
        for pt in (5, 3, 1):
            text = hcfg.get("anchors", {}).get(c, {}).get(pt, "")
            info.append([f"  {pt}点", text])
        info.append(["  2・4点", "それぞれ上下の目安の中間"])
    info.column_dimensions["A"].width = 26
    info.column_dimensions["B"].width = 100
    for row in info.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def _verify_saved(path: Path, keyed: list[dict]):
    """保存したシートを読み直し、入力文・評価対象が1文字も変わっていないことを確かめる。"""
    ws = load_workbook(path, read_only=True)[SHEET_NAME]
    for it, row in zip(keyed, ws.iter_rows(min_row=2, max_col=3, values_only=True)):
        got_src, got_content = row[1] or "", row[2] or ""
        if row[0] != it["item_id"] or got_src != it["source"] or got_content != it["content"]:
            raise HumanEvalError(f"{path.name} の {it['item_id']} が書き込み前と一致しません")


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


def parse_cell(v, scale: int, unable: str) -> tuple[str, float]:
    """セルの値を（状態, 点数）にする。状態は 数値 / 評価不能 / 未記入 / 不正値。"""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
        return "未記入", np.nan
    if isinstance(v, str):
        s = v.strip()
        if s == unable:
            return "評価不能", np.nan
        try:
            v = float(s)
        except ValueError:
            return "不正値", np.nan
    if isinstance(v, (int, float)) and float(v).is_integer() and 1 <= v <= scale:
        return "数値", float(v)
    return "不正値", np.nan


def _read_sheet_meta(path: Path) -> dict | None:
    wb = load_workbook(path, read_only=True)
    if META_SHEET not in wb.sheetnames:
        return None
    try:
        return json.loads(wb[META_SHEET]["A1"].value)
    except (TypeError, ValueError):
        return None


def load_filled(human_dir: Path, cfg: dict, target: str) -> pd.DataFrame | None:
    hcfg = cfg["human_eval"]
    criteria, scale, unable = list(hcfg["criteria"]), int(hcfg["scale"]), hcfg.get("unable_label", "評価不能")
    sheets = sorted(p for p in human_dir.glob("sheet_*.xlsx") if p.stem != "sheet_template")
    meta_path, key_path = human_dir / META_FILE, human_dir / KEY_FILE
    if not sheets:
        return None
    if not meta_path.exists() or not key_path.exists():
        raise HumanEvalError(f"{human_dir} に {META_FILE} または {KEY_FILE} がありません。"
                             "旧形式（3項目）のシートの可能性があります。新しい基準のシートと混ぜて集計できません")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("target") != target:
        raise HumanEvalError(f"{human_dir} は評価対象 '{meta.get('target')}' のシートです（'{target}' として集計しようとしました）")
    if meta.get("criteria") != criteria or meta.get("criteria_version") != hcfg.get("criteria_version"):
        raise HumanEvalError(f"{human_dir} のシートは評価項目・基準の版が現在の設定と違います"
                             f"（シート: {meta.get('criteria')} v{meta.get('criteria_version')}）")
    if _sha256(key_path) != meta.get("key_sha256"):
        raise HumanEvalError(f"{KEY_FILE} がシート作成時から変わっています")
    key = pd.read_csv(key_path, encoding="utf-8-sig")

    frames, invalid = [], []
    for p in sheets:
        sm = _read_sheet_meta(p)
        if sm is None:
            raise HumanEvalError(f"{p.name} は旧形式のシートです（評価条件の情報がありません）。混ぜて集計できません")
        for k in ("target", "criteria", "criteria_version", "key_sha256"):
            if sm.get(k) != meta.get(k):
                raise HumanEvalError(f"{p.name} は他のシートと条件が違います（{k}）。別の回・別の評価対象のシートが混ざっていないか確認してください")
        df = pd.read_excel(p, sheet_name=SHEET_NAME, dtype=object)
        missing = [c for c in ["item_id", *criteria, REASON_COL] if c not in df.columns]
        if missing:
            raise HumanEvalError(f"{p.name} に必要な列がありません: {missing}")
        if set(df["item_id"]) != set(key["item_id"]):
            raise HumanEvalError(f"{p.name} の item_id が {KEY_FILE} と一致しません")
        rater = p.stem.removeprefix("sheet_")
        for _, row in df.iterrows():
            reason = row.get(REASON_COL)
            reason = "" if reason is None or (isinstance(reason, float) and np.isnan(reason)) else str(reason).strip()
            for c in criteria:
                status, score = parse_cell(row[c], scale, unable)
                if status == "不正値":
                    invalid.append(f"{p.name} {row['item_id']} {c}={row[c]!r}")
                frames.append({"rater": rater, "item_id": row["item_id"], "criterion": c, "status": status,
                               "score": score,
                               "unable_reason": (reason or "（理由未記入）") if status == "評価不能" else ""})
    if invalid:
        raise HumanEvalError("1〜{0} の整数・「{1}」・空欄以外の値があります: ".format(scale, unable) + "; ".join(invalid[:20]))
    detail = pd.DataFrame(frames).merge(key, on="item_id", how="left")
    n_no_reason = (detail["unable_reason"] == "（理由未記入）").sum()
    if n_no_reason:
        print(f"  [注意] 理由が選ばれていない「{unable}」が {n_no_reason} 個あります")
    return detail


def aggregate(detail: pd.DataFrame, cfg: dict, target: str) -> dict[str, pd.DataFrame]:
    base, target_model = cfg["compare"]["base"], cfg["compare"]["target"]
    both = pd.concat([detail, detail.assign(set="ALL")])
    g = both.groupby(["set", "model", "criterion"])
    summary = g["score"].agg(["mean", "std"]).round(3)
    summary["n_valid"] = g["status"].apply(lambda s: (s == "数値").sum())
    summary["n_unable"] = g["status"].apply(lambda s: (s == "評価不能").sum())
    summary["n_blank"] = g["status"].apply(lambda s: (s == "未記入").sum())
    summary["n_total"] = g.size()
    summary = summary.reset_index()
    summary.insert(0, "target", target)

    valid = detail[detail["status"] == "数値"]
    alpha_rows = []
    for crit in cfg["human_eval"]["criteria"]:
        gc = valid[valid["criterion"] == crit]
        units = gc.groupby("item_id")["score"].apply(list).tolist()
        alpha_rows.append({"target": target, "criterion": crit, "n_raters": gc["rater"].nunique(),
                           "n_items_with_2plus": sum(1 for u in units if len(u) >= 2),
                           "n_valid_ratings": len(gc),
                           "alpha_ordinal": round(krippendorff_alpha_ordinal(units), 3)})
    alpha = pd.DataFrame(alpha_rows)

    # 評価者平均（数値のみ）をとってから、同じ入力どうしで base と lora を比べる。n は有効な組の数
    per_item = valid.groupby(["set", "id", "model", "criterion"])["score"].mean().reset_index()
    test_rows = []
    for (s, crit), gg in pd.concat([per_item, per_item.assign(set="ALL")]).groupby(["set", "criterion"]):
        test_rows.append({"set": s, "metric": f"人的評価({target}):{crit}",
                          **paired_test(gg, "score", base, target_model)})
    tests = pd.DataFrame(test_rows)

    unable = detail[detail["status"] == "評価不能"]
    reasons = (unable.groupby(["model", "criterion", "unable_reason"]).size().rename("count").reset_index()
               if len(unable) else pd.DataFrame(columns=["model", "criterion", "unable_reason", "count"]))
    return {"summary": summary, "alpha": alpha, "tests": tests, "reasons": reasons}


def run_aggregate(human_dir: Path, cfg: dict, target: str = "final") -> dict[str, pd.DataFrame] | None:
    """シートがなければ None。条件の食い違いがあれば HumanEvalError。"""
    detail = load_filled(human_dir, cfg, target)
    if detail is None:
        return None
    res = aggregate(detail, cfg, target)
    res["detail"] = detail
    return res


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="人的評価シートの作成と集計")
    ap.add_argument("--config", default=None)
    ap.add_argument("--human-dir", default=None, help="既定: human/<target>/（--dummy-scores では human_sample/<target>/）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="評価シートを作成")
    mk.add_argument("--target", choices=TARGETS, default="final", help="final=後処理後の出力（既定）, raw=生出力")
    mk.add_argument("--outputs", default=str(ROOT / "data" / "outputs.jsonl"))
    mk.add_argument("--raters", nargs="*", default=[], help="評価者 ID（member1 など。人数分のシートを作る）")
    mk.add_argument("--dummy-scores", action="store_true",
                    help="見本データ専用。仮点数を埋め、既定では human_sample/<target>/ に保存")
    mk.add_argument("--overwrite", action="store_true", help="既存のシート・対応表があっても作り直す")
    ag = sub.add_parser("aggregate", help="記入済みシートを集計")
    ag.add_argument("--target", choices=TARGETS, default="final")
    ag.add_argument("--results", default=None, help="既定: results/human_<target>/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dummy_scores = args.cmd == "make" and args.dummy_scores
    human_dir = (Path(args.human_dir) if args.human_dir
                 else ROOT / ("human_sample" if dummy_scores else "human") / args.target)
    if dummy_scores:
        if Path(args.outputs).resolve() != (ROOT / "data" / "sample_outputs.jsonl").resolve():
            ap.error("--dummy-scores は data/sample_outputs.jsonl にのみ使用できます")
        if human_dir.resolve().is_relative_to((ROOT / "human").resolve()):
            ap.error("仮点数は本番用 human/ に保存できません。別フォルダを指定してください")
        print("[動作確認専用] 仮点数を生成します。モデルの性能評価には使えません。")
    try:
        if args.cmd == "make":
            paths = make_sheets(load_records(args.outputs), cfg, human_dir, args.raters, target=args.target,
                                dummy_scores=dummy_scores, overwrite=args.overwrite)
            print(f"評価対象: {args.target}（{TARGET_LABEL[args.target]}）")
            print("作成しました:", *[str(p) for p in paths], sep="\n  ")
            print(f"対応表（評価が終わるまで評価者には見せない）: {human_dir / KEY_FILE}")
        else:
            res = run_aggregate(human_dir, cfg, args.target)
            if res is None:
                print(f"{human_dir} に sheet_*.xlsx がありません")
                return
            out = Path(args.results) if args.results else ROOT / "results" / f"human_{args.target}"
            out.mkdir(parents=True, exist_ok=True)
            for name in ("detail", "summary", "alpha", "tests", "reasons"):
                res[name].to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                for name in ("summary", "alpha", "tests", "reasons"):
                    print(f"\n== {name} ==\n{res[name].to_string(index=False)}")
            print(f"\n結果を {out} に保存しました（評価対象: {args.target}）")
    except HumanEvalError as e:
        print(f"[停止] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
