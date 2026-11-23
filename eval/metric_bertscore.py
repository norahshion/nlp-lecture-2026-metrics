"""指標1：BERTScore（意味的類似度）

  vs_ref : 生成文 と 正解対話（reference）を比較 … データA（逆生成データ）のみ
  vs_src : 生成文 と 入力文（source）を比較     … 全データ。全文どうしで比べる
  chunk  : 分割ごとに「入力の分割 i」と「後処理後の出力の分割 i」を比較 … 分割情報がある場合のみ
           （convert_pred.py --pipeline で付けた source_chunks / final_chunks を使う）

どれも話者ラベル「A：」「B：」を外した発話内容だけで比較する。

■ 長さの上限について
BERT は最大512トークン（特殊トークンを含む）までしか読めず、超えた部分は切り捨てられる。
1件ごとに比べる2つの文章のトークン数を数え、どちらかが上限を超えたものは「切り捨てあり」とする。
切り捨てありの値は detail には残すが、集計（summary）と検定には使わない（全文の評価として扱わないため）。
集計には、使った件数と切り捨てで除いた件数を必ず添える。

■ 分割単位の値（chunk_F）
1件の値は、分割ごとの F1 の単純平均。後処理後の出力が空の分割は 0 点として平均に含める（削除しない）。
分割のどれか1つでも上限を超えたら、その件の chunk_F は集計から除く。
"""
from __future__ import annotations

import math

import pandas as pd

from parse_dialogue import to_plain

KINDS = ("vs_ref", "vs_src")


def _make_scorer(cfg: dict):
    import torch
    from bert_score import BERTScorer

    bcfg = cfg["bertscore"]
    device = bcfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return BERTScorer(model_type=bcfg["model_type"], num_layers=int(bcfg["num_layers"]),
                      batch_size=int(bcfg["batch_size"]), device=device)


def _token_counter(scorer, cfg: dict):
    # 長さを数えるだけなので「512を超えています」という transformers の警告は出さない（超過はこちらで記録する）
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = getattr(scorer, "_tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["bertscore"]["model_type"])
    limit = getattr(tok, "model_max_length", 512)
    if not isinstance(limit, int) or limit > 100_000:  # 上限が設定されていないトークナイザー
        limit = 512
    return (lambda s: len(tok(s, add_special_tokens=True, truncation=False)["input_ids"])), limit


def evaluate(records: list[dict], cfg: dict, scorer=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1件ごとの表（detail）と、分割ごとの表（chunk_detail）を返す。"""
    scorer = scorer or _make_scorer(cfg)
    count, limit = _token_counter(scorer, cfg)

    # ---- 比べるペアを集める: (キー, 生成文, 比較対象)
    pairs = []
    rows, chunk_rows = [], []
    for i, r in enumerate(records):
        cand = to_plain(r["output"])
        row = {"id": r["id"], "set": r["set"], "model": r["model"]}
        for k in KINDS:
            for m in ("P", "R", "F"):
                row[f"{k}_{m}"] = math.nan
            row[f"{k}_cand_tokens"] = math.nan
            row[f"{k}_ref_tokens"] = math.nan
            row[f"{k}_truncated"] = None
        refs = {"vs_src": r["source"].strip(), "vs_ref": to_plain(r["reference"]) if r["reference"] else None}
        for k, ref in refs.items():
            if ref is None:
                continue
            row[f"{k}_cand_tokens"], row[f"{k}_ref_tokens"] = count(cand), count(ref)
            row[f"{k}_truncated"] = bool(max(row[f"{k}_cand_tokens"], row[f"{k}_ref_tokens"]) > limit)
            pairs.append((("full", i, k), cand, ref))

        srcs, fins = r.get("source_chunks"), r.get("final_chunks")
        row["chunk_n"] = len(srcs) if isinstance(srcs, list) else math.nan
        if isinstance(srcs, list) and isinstance(fins, list) and len(srcs) == len(fins) and srcs:
            for j, (s, f) in enumerate(zip(srcs, fins)):
                c = to_plain(f) if f else ""
                ct, st = count(c) if c else 0, count(s)
                chunk_rows.append({"id": r["id"], "set": r["set"], "model": r["model"], "chunk": j + 1,
                                   "src_tokens": st, "cand_tokens": ct, "empty": not c,
                                   "truncated": bool(max(ct, st) > limit), "F": math.nan})
                pairs.append((("chunk", len(chunk_rows) - 1, None), c, s))
        rows.append(row)

    # ---- まとめて計算（空の生成文は計算できないので 0 点）
    valid = [p for p in pairs if p[1] and p[2]]
    scores = {}
    if valid:
        P, R, F = scorer.score([p[1] for p in valid], [p[2] for p in valid])
        for p, a, b, c in zip(valid, P.tolist(), R.tolist(), F.tolist()):
            scores[p[0]] = (a, b, c)
    for key, cand, ref in pairs:
        a, b, c = scores.get(key, (0.0, 0.0, 0.0))
        kind, idx, k = key
        if kind == "full":
            rows[idx][f"{k}_P"], rows[idx][f"{k}_R"], rows[idx][f"{k}_F"] = a, b, c
        else:
            chunk_rows[idx]["F"] = c

    detail = pd.DataFrame(rows)
    chunk_detail = pd.DataFrame(chunk_rows, columns=["id", "set", "model", "chunk", "src_tokens", "cand_tokens",
                                                     "empty", "truncated", "F"])

    # ---- 分割単位の値を1件ごとにまとめる
    if len(chunk_detail):
        g = chunk_detail.groupby(["id", "model"])
        agg = pd.DataFrame({"chunk_F": g["F"].mean(), "chunk_n_empty": g["empty"].sum(),
                            "chunk_n_truncated": g["truncated"].sum()}).reset_index()
        detail = detail.merge(agg, on=["id", "model"], how="left")
    else:
        detail["chunk_F"] = math.nan
        detail["chunk_n_empty"] = math.nan
        detail["chunk_n_truncated"] = math.nan

    # ---- 集計・検定に使う値（切り捨てありは NaN にする）
    for k in KINDS:
        detail[f"{k}_F_valid"] = detail[f"{k}_F"].where(detail[f"{k}_truncated"] == False)  # noqa: E712
    detail["chunk_F_valid"] = detail["chunk_F"].where(detail["chunk_n_truncated"] == 0)
    return detail, chunk_detail


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """データセット × モデルごとの平均（切り捨てなしの件だけ）と、使った件数・除いた件数。"""
    both = pd.concat([detail, detail.assign(set="ALL")])
    g = both.groupby(["set", "model"])
    out = pd.DataFrame(index=g.size().index)
    for k in KINDS:
        out[f"{k}_F"] = g[f"{k}_F_valid"].mean()
        out[f"{k}_n_used"] = g[f"{k}_F_valid"].count()
        out[f"{k}_n_truncated"] = g[f"{k}_truncated"].apply(lambda s: int((s == True).sum()))  # noqa: E712
    out["chunk_F"] = g["chunk_F_valid"].mean()
    out["chunk_n_used"] = g["chunk_F_valid"].count()
    out["chunk_n_truncated"] = g["chunk_n_truncated"].apply(lambda s: int((s > 0).sum()))
    out["chunk_n_no_info"] = g["chunk_F"].apply(lambda s: int(s.isna().sum()))
    return out.round(4).reset_index()
