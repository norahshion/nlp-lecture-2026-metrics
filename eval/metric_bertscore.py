"""指標1：BERTScore（意味的類似度）

  vs_ref : 生成文 と 正解対話（reference）を比較 … データA（逆生成データ）のみ
  vs_src : 生成文 と 入力文（source）を比較     … 全データ。情報保持の目安

どちらも話者ラベル「A：」「B：」を外した発話内容だけで比較する。
"""
from __future__ import annotations

import math

import pandas as pd

from parse_dialogue import to_plain


def _make_scorer(cfg: dict):
    import torch
    from bert_score import BERTScorer

    bcfg = cfg["bertscore"]
    device = bcfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return BERTScorer(model_type=bcfg["model_type"], num_layers=int(bcfg["num_layers"]),
                      batch_size=int(bcfg["batch_size"]), device=device)


def evaluate(records: list[dict], cfg: dict, scorer=None) -> pd.DataFrame:
    scorer = scorer or _make_scorer(cfg)

    # (レコード番号, 種類, 生成文, 比較対象) のペアをまとめて一度に計算する
    pairs = []
    for i, r in enumerate(records):
        cand = to_plain(r["output"])
        pairs.append((i, "vs_src", cand, r["source"].strip()))
        if r["reference"]:
            pairs.append((i, "vs_ref", cand, to_plain(r["reference"])))

    rows = [{"id": r["id"], "set": r["set"], "model": r["model"],
             **{f"{k}_{m}": math.nan for k in ("vs_ref", "vs_src") for m in "PRF"}}
            for r in records]

    # 空の生成文は BERTScore が計算できないので 0 点扱い
    valid = [p for p in pairs if p[2] and p[3]]
    for i, kind, cand, _ in pairs:
        if not cand:
            for m in "PRF":
                rows[i][f"{kind}_{m}"] = 0.0

    if valid:
        P, R, F = scorer.score([p[2] for p in valid], [p[3] for p in valid])
        for (i, kind, _, _), p, r_, f in zip(valid, P.tolist(), R.tolist(), F.tolist()):
            rows[i][f"{kind}_P"], rows[i][f"{kind}_R"], rows[i][f"{kind}_F"] = p, r_, f
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in detail.columns if c.startswith(("vs_ref_", "vs_src_"))]
    both = pd.concat([detail, detail.assign(set="ALL")])
    return both.groupby(["set", "model"])[cols].mean().round(4).reset_index()
