"""生成された対話文を発話（話者・内容）に分解する。全指標で共通に使う。

対応する書式: 「A：こんにちは」「A: こんにちは」「Ａ：こんにちは」など
話者ラベルの付いていない行は extra_lines（形式違反の候補）として別に保持する。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# 行頭の「話者ラベル＋コロン」。ラベルは空白・コロンを含まない1〜10文字
LINE_RE = re.compile(r"^\s*([^\s:：]{1,10})\s*[:：]\s*(.*)$")


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Dialogue:
    turns: list[Turn] = field(default_factory=list)
    extra_lines: list[str] = field(default_factory=list)  # ラベルなしの行

    @property
    def speakers(self) -> list[str]:
        return [t.speaker for t in self.turns]


def normalize_speaker(label: str) -> str:
    """全角英字などをそろえる（Ａ→A）。"""
    return unicodedata.normalize("NFKC", label).strip()


def parse(text: str) -> Dialogue:
    dlg = Dialogue()
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            dlg.turns.append(Turn(normalize_speaker(m.group(1)), m.group(2).strip()))
        else:
            dlg.extra_lines.append(line)
    return dlg


def to_plain(text: str) -> str:
    """話者ラベルを外し、発話内容だけを改行でつなげる（BERTScore 用）。
    ラベル付きの行が1つもなければ、元の文章をそのまま返す。"""
    dlg = parse(text)
    if not dlg.turns:
        return (text or "").strip()
    return "\n".join(t.text for t in dlg.turns if t.text)


if __name__ == "__main__":
    sample = "A：明日、急に寒くなるらしいですね。\nＢ: そうなんです。\n（以上）"
    d = parse(sample)
    print(d.turns, d.extra_lines)
    print(to_plain(sample))
