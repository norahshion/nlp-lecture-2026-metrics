"""生成側の出力（pred_*.jsonl）を、評価プログラムの入力形式（outputs.jsonl）に変換する。

  python convert_pred.py --base ../pred_base.jsonl --target ../pred_lora.jsonl
  python convert_pred.py --base ../pred_base.jsonl --target ../pred_lora.jsonl --overwrite  # 既存ファイルを上書き

  # 分割ごとの入力・出力も付ける（分割単位の BERTScore に必要）
  python convert_pred.py --base ../pred_base.jsonl --target ../pred_lora.jsonl \
      --pipeline ../member_program/dialogue_pipeline.py

  # データA（test.jsonl の生成結果）。id がなければ index から test_000 のような id を作る
  python convert_pred.py --base ../pred_test_base.jsonl --target ../pred_test_lora.jsonl \
      --set A --id-from-index --out data/outputs_A.jsonl

項目の対応:
  input       → source   （入力した文章）
  prediction  → output   （後処理後の最終出力。空でも削除しない）
  source      → publisher（発信元の名前）
  format_ok   → gen_format_ok（生成側の形式チェック結果）
  issues      → gen_issues   （分割単位ごとの生成側チェック結果）
  raw_outputs → raw_outputs  （分割単位ごとの生出力。失敗理由の判定に使う）
  adapter     → 引き継がない（ローカルパスを含むため）。代わりに uses_adapter（真偽値）を付ける

--pipeline を指定したときだけ、生成側の共有コード（dialogue_pipeline.py）で次を作って付ける:
  source_chunks : 分割ごとの入力（preprocess(input) の結果）
  final_chunks  : 分割ごとの後処理後の出力（raw_outputs を1つずつ parse_dialogue したもの）
  分割数が raw_outputs と一致し、final_chunks を結合すると prediction が再現できることを確かめ、
  合わなければ停止する（生成に使ったコードと版が違う可能性があるため）。

元のファイルは読み込むだけで変更しない。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from common import ROOT, load_config

PRED_REQUIRED = ["input", "prediction"]
SET_BY_PREFIX = {"news_": "B", "jdd_": "A"}


class ConvertError(Exception):
    pass


def read_pred(path: Path) -> list[dict]:
    if not path.exists():
        raise ConvertError(f"ファイルがありません: {path}")
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ConvertError(f"{path.name}:{ln} 行目が JSON として読めません: {e}") from e
            missing = [k for k in PRED_REQUIRED if k not in r]
            if missing:
                raise ConvertError(f"{path.name}:{ln} 行目に必須項目がありません: {missing}")
            raw, issues = r.get("raw_outputs"), r.get("issues")
            if raw is not None and not (isinstance(raw, list) and all(isinstance(x, str) for x in raw)):
                raise ConvertError(f"{path.name}:{ln} 行目の raw_outputs は文字列のリストである必要があります")
            if issues is not None and not (isinstance(issues, list) and all(
                    isinstance(i, list) and all(isinstance(k, str) for k in i) for i in issues)):
                raise ConvertError(f"{path.name}:{ln} 行目の issues は「文字列のリスト」のリストである必要があります"
                                   "（未記録の分割に null は使えません。合格は [] で表します）")
            if raw is not None and issues is not None and len(raw) != len(issues):
                raise ConvertError(f"{path.name}:{ln} 行目の raw_outputs と issues の数が違います"
                                   f"（{len(raw)} と {len(issues)}）")
            rows.append(r)
    return rows


def assign_ids(rows: list[dict], path: Path, from_index: bool, prefix: str) -> None:
    """id を確定させる。id がない行があれば --id-from-index のときだけ index から作る。"""
    has_id = [isinstance(r.get("id"), str) and r["id"].strip() != "" for r in rows]
    if from_index:
        if any(has_id):
            raise ConvertError(f"{path.name} には id のある行があります。--id-from-index は id が1つもない場合だけ使えます")
        idx = [r.get("index") for r in rows]
        if not all(isinstance(i, int) for i in idx):
            raise ConvertError(f"{path.name} に整数の index がない行があるため、id を作れません")
        for r in rows:
            r["id"] = f"{prefix}{r['index']:03d}"
    elif not all(has_id):
        n = has_id.count(False)
        raise ConvertError(f"{path.name} に id のない行が {n} 件あります。"
                           "index から id を作る場合は --id-from-index を付けてください")
    ids = [r["id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ConvertError(f"{path.name} の id が重複しています: {dup}")


def load_pipeline(path: Path):
    """生成側の共有コードを、指定されたファイルだけ読み込む（フォルダを import パスに足さない）。"""
    if not path.is_file():
        raise ConvertError(f"--pipeline のファイルがありません: {path}")
    name = "_shared_dialogue_pipeline"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass は読み込み中のモジュールが登録されている必要がある
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    for name in ("preprocess", "parse_dialogue", "merge_chunks", "render_dialogue"):
        if not hasattr(mod, name):
            raise ConvertError(f"{path.name} に {name} がありません。生成に使った dialogue_pipeline.py を指定してください")
    return mod


def add_chunks(r: dict, mod, label: str) -> None:
    """分割ごとの入力と後処理後の出力を付け、prediction が再現できるか確かめる。"""
    raws = r.get("raw_outputs")
    if not isinstance(raws, list):
        raise ConvertError(f"{label}: raw_outputs がないため分割情報を作れません")
    sources = mod.preprocess(r["input"])
    if len(sources) != len(raws):
        raise ConvertError(f"{label}: 共有コードで分けた分割数（{len(sources)}）が raw_outputs の数（{len(raws)}）と違います。"
                           "生成時と別の版のコード、または別の --max-chunk-chars で生成された可能性があります")
    utts = [mod.parse_dialogue(x)[0] for x in raws]
    finals = [mod.render_dialogue(u) for u in utts]
    merged, _ = mod.merge_chunks(utts)
    if mod.render_dialogue(merged) != (r.get("prediction") or ""):
        raise ConvertError(f"{label}: 共有コードの後処理で prediction を再現できません。生成時と別の版のコードの可能性があります")
    r["source_chunks"] = sources
    r["final_chunks"] = finals


def validate_pair(base: list[dict], target: list[dict], names: tuple[str, str]) -> list[str]:
    """両モデルで id と入力文が一致しているか確認する。問題があれば例外、注意点は警告として返す。"""
    b = {r["id"]: r for r in base}
    t = {r["id"]: r for r in target}
    only_b, only_t = sorted(b.keys() - t.keys()), sorted(t.keys() - b.keys())
    if only_b or only_t:
        raise ConvertError(f"両モデルの id が一致しません（{names[0]} のみ: {only_b} / {names[1]} のみ: {only_t}）")
    diff = [i for i in b if b[i]["input"] != t[i]["input"]]
    if diff:
        raise ConvertError(f"同じ id なのに入力文が違います: {sorted(diff)}")
    diff_ref = [i for i in b if (b[i].get("reference") or None) != (t[i].get("reference") or None)]
    if diff_ref:
        raise ConvertError(f"同じ id なのに正解（reference）が違います: {sorted(diff_ref)}")

    warnings = []
    for i in sorted(b):
        nb, nt = len(b[i].get("raw_outputs") or []), len(t[i].get("raw_outputs") or [])
        if nb != nt:
            warnings.append(f"{i}: 分割数が違います（{names[0]}={nb}, {names[1]}={nt}）")
    if any(r.get("adapter") for r in base):
        warnings.append(f"{names[0]} 側のファイルにアダプタが指定されています。ファイルの取り違えがないか確認してください")
    if any(not r.get("adapter") for r in target) and any("adapter" in r for r in target):
        warnings.append(f"{names[1]} 側のファイルにアダプタのない行があります。ファイルの取り違えがないか確認してください")
    return warnings


def decide_set(rid: str, forced: str | None) -> str:
    if forced:
        return forced
    for prefix, s in SET_BY_PREFIX.items():
        if rid.startswith(prefix):
            return s
    raise ConvertError(f"id '{rid}' からデータセット（A/B）を判断できません。--set で指定してください")


def convert_row(r: dict, model: str, set_name: str) -> dict:
    raw = r.get("raw_outputs")
    issues = r.get("issues")
    out = {
        "id": r["id"],
        "set": set_name,
        "model": model,
        "source": r["input"],
        "reference": r.get("reference"),
        "output": r.get("prediction") or "",
        "publisher": r.get("source"),
        "base_model": r.get("base_model"),
        "uses_adapter": bool(r.get("adapter")),
        "gen_format_ok": r.get("format_ok"),
        "gen_issues": issues,
        "raw_outputs": raw,
    }
    if "source_chunks" in r:
        out["source_chunks"] = r["source_chunks"]
        out["final_chunks"] = r["final_chunks"]
    return out


def write_jsonl_safely(rows: list[dict], out: Path, overwrite: bool, inputs: list[Path]):
    if any(out.resolve() == p.resolve() for p in inputs):
        raise ConvertError("変換先が元ファイルと同じです。別のパスを指定してください")
    if out.exists() and not overwrite:
        raise ConvertError(f"{out} は既にあります。上書きする場合は --overwrite を付けてください")
    out.parent.mkdir(parents=True, exist_ok=True)
    # 一時ファイルは mkstemp で「存在しない名前」を排他的に新規作成する（既存ファイルは決して開かない）
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main(argv=None):
    cfg = load_config()
    base_name, target_name = cfg["compare"]["base"], cfg["compare"]["target"]
    ap = argparse.ArgumentParser(description="pred_*.jsonl を評価用の outputs.jsonl に変換")
    ap.add_argument("--base", required=True, help=f"{base_name}（未学習モデル）の生成結果")
    ap.add_argument("--target", required=True, help=f"{target_name}（追加学習済みモデル）の生成結果")
    ap.add_argument("--out", default=str(ROOT / "data" / "outputs.jsonl"))
    ap.add_argument("--set", choices=["A", "B"], default=None,
                    help="データセットを固定する（省略時は id の接頭辞 news_→B, jdd_→A で判断）")
    ap.add_argument("--overwrite", action="store_true", help="変換先が既にあれば上書きする")
    ap.add_argument("--id-from-index", action="store_true",
                    help="id が1つもないファイルで、index から id を作る（例: test_000）")
    ap.add_argument("--id-prefix", default="test_", help="--id-from-index で付ける接頭辞（既定: test_）")
    ap.add_argument("--pipeline", default=None,
                    help="生成に使った dialogue_pipeline.py。指定すると分割ごとの入力・出力を付ける")
    args = ap.parse_args(argv)

    base_path, target_path, out = Path(args.base), Path(args.target), Path(args.out)
    try:
        base, target = read_pred(base_path), read_pred(target_path)
        for rows_, p in ((base, base_path), (target, target_path)):
            assign_ids(rows_, p, args.id_from_index, args.id_prefix)
        warnings = validate_pair(base, target, (base_name, target_name))
        if args.pipeline:
            mod = load_pipeline(Path(args.pipeline))
            for rows_, name in ((base, base_name), (target, target_name)):
                for r in rows_:
                    add_chunks(r, mod, f"{r['id']}({name})")
        rows = [convert_row(r, base_name, decide_set(r["id"], args.set)) for r in base]
        rows += [convert_row(r, target_name, decide_set(r["id"], args.set)) for r in target]
        write_jsonl_safely(rows, out, args.overwrite, [base_path, target_path])
    except ConvertError as e:
        print(f"[停止] {e}", file=sys.stderr)
        sys.exit(1)

    for w in warnings:
        print(f"[注意] {w}")
    n_empty = {m: sum(1 for r in rows if r["model"] == m and not r["output"]) for m in (base_name, target_name)}
    print(f"{len(base)} 入力 × 2 モデル = {len(rows)} 件を {out} に書き出しました")
    print(f"  出力が空の件数（削除せず残しています）: {n_empty}")
    if args.pipeline:
        sha = hashlib.sha256(Path(args.pipeline).read_bytes()).hexdigest()[:12]
        print(f"  分割情報を付けました（共有コード {Path(args.pipeline).name} sha256:{sha}…、prediction の再現を確認済み）")


if __name__ == "__main__":
    main()
