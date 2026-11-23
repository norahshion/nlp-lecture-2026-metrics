"""指標3：形式遵守率

1件ごとに以下をチェックし、すべて満たせば「遵守」とする。
  speakers_valid  : 許可された話者（A, B）以外のラベルがない
  speaker_count   : 許可された話者が全員登場している（2人対話になっている）
  min_turns       : 発話数が最低ターン数以上
  alternation     : 同じ話者が連続しない（設定で無効化可）
  first_speaker   : 指定した話者から始まる（設定で null なら常に True）
  no_empty        : 内容が空の発話がない
  no_extra_lines  : 話者ラベルのない行（前置き・注釈など）がない
"""
from __future__ import annotations

import pandas as pd

from parse_dialogue import parse

CHECK_LABELS = {
    "speakers_valid": "想定外の話者なし",
    "speaker_count": "2人とも登場",
    "min_turns": "最低ターン数",
    "alternation": "交互に発話",
    "first_speaker": "開始話者",
    "no_empty": "空の発話なし",
    "no_extra_lines": "余計な行なし",
}


def check(text: str, cfg: dict) -> dict:
    fcfg = cfg["format"]
    allowed = list(fcfg["speakers"])
    dlg = parse(text)
    spk = dlg.speakers

    res = {
        "n_turns": len(dlg.turns),
        "speakers_valid": len(spk) > 0 and all(s in allowed for s in spk),
        "speaker_count": set(allowed) <= set(spk),
        "min_turns": len(dlg.turns) >= int(fcfg["min_turns"]),
        "alternation": (not fcfg.get("require_alternation", True))
                       or (len(spk) > 0 and all(a != b for a, b in zip(spk, spk[1:]))),
        "first_speaker": (fcfg.get("first_speaker") is None)
                         or (len(spk) > 0 and spk[0] == fcfg["first_speaker"]),
        "no_empty": len(dlg.turns) > 0 and all(t.text for t in dlg.turns),
        "no_extra_lines": len(dlg.extra_lines) == 0,
    }
    res["all_pass"] = all(res[k] for k in CHECK_LABELS)
    return res


def evaluate(records: list[dict], cfg: dict) -> pd.DataFrame:
    """1件ごとの判定結果（詳細表）。"""
    rows = []
    for r in records:
        rows.append({"id": r["id"], "set": r["set"], "model": r["model"],
                     **check(r["output"], cfg)})
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """データセット × モデルごとの遵守率（%）。set='ALL' は全データ。"""
    cols = ["all_pass", *CHECK_LABELS]
    both = pd.concat([detail, detail.assign(set="ALL")])
    out = both.groupby(["set", "model"])[cols].mean().mul(100).round(1)
    out["n"] = both.groupby(["set", "model"]).size()
    return out.reset_index()
