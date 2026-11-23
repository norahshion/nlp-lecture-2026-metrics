"""指標3：形式遵守率 ＋ 日本語判定 ＋ 全分割合格（3種類は別々に集計する）

(1) 形式遵守（従来どおり）: 以下をすべて満たせば all_pass=True
  speakers_valid  : 許可された話者（A, B）以外のラベルがない
  speaker_count   : 許可された話者が全員登場している（2人対話になっている）
  min_turns       : 発話数が最低ターン数以上
  alternation     : 同じ話者が連続しない（設定で無効化可）
  first_speaker   : 指定した話者から始まる（設定で null なら常に True）
  no_empty        : 内容が空の発話がない
  no_extra_lines  : 話者ラベルのない行（前置き・注釈など）がない

(2) 日本語判定 ja_status: 合格 / 要確認 / 不合格 / 出力なし
  発話内容（話者ラベルを除く）の文字のうち、
  日本語の文字数（ひらがな・カタカナ・漢字）÷（日本語の文字数＋英単語の数）を ja_ratio とする。
  英語は文字ではなく単語単位で数える（日本語文中の少数の英単語で不合格にしないため）。
  設定の許可リスト（allowed_abbreviations: URL, PDF など）にある略語だけは数えない。
  長音記号ー・中黒・などの記号は、日本語の文字にも「かな」にも数えない。
    ja_ratio <  fail_ratio                                      → 不合格
    ja_ratio >= pass_ratio かつ かなを含む かつ 英語中心の発話がない → 合格
    それ以外（混在・かなを含まない・英語中心の発話がある等）      → 要確認（人が確認する）
  英語中心の発話 ＝ 英単語を含み、その発話単独の ja_ratio が fail_ratio 未満（ja_mixed_turns に件数）
  この判定は補助的なもので、日本語として自然かどうかは人的評価で確認する。
  集計では「合格」だけを合格率の分子に入れる。要確認は別の列で率を示す。

(3) 全分割合格 chunks_all_ok: True / False / None（分割情報なし＝対象外）
  生成側が分割単位ごとのチェック結果（gen_issues）を残している場合だけ判定する。

失敗理由 output_status（1件ごと）:
  出力あり / 出力なし（生出力なし）/ 出力なし（抽出不可）/ 出力なし（生出力の記録なし）
  抽出不可の推定手がかりとして、生出力の行頭にある想定外の話者表記（例: Person A）を
  extract_hint に記録する。手がかりがなければ空欄（原因は断定しない）。
"""
from __future__ import annotations

import math
import re
import unicodedata

import pandas as pd

from parse_dialogue import LINE_RE, normalize_speaker, parse

CHECK_LABELS = {
    "speakers_valid": "想定外の話者なし",
    "speaker_count": "2人とも登場",
    "min_turns": "最低ターン数",
    "alternation": "交互に発話",
    "first_speaker": "開始話者",
    "no_empty": "空の発話なし",
    "no_extra_lines": "余計な行なし",
}
JA_STATUSES = ["合格", "要確認", "不合格", "出力なし"]

# かなは「文字」だけ（ぁ〜ゖ, ァ〜ヺ）。長音記号ー・中黒・・踊り字ゝゞヽヾは含めない
KANA_RE = re.compile(r"[ぁ-ゖァ-ヺ]")
JA_RE = re.compile(r"[ぁ-ゖァ-ヺ㐀-䶿一-鿿々]")
LATIN_RE = re.compile(r"[A-Za-z]+")  # 英語は「単語」単位で数える（1語を漢字1文字と同じ重みにする）
DEFAULT_ABBREVIATIONS = ["URL", "PDF", "AI", "IT", "OK", "PC", "SNS", "TV", "CM", "ID", "QR", "NHK", "JR", "GDP"]


# ---------------------------------------------------------------- (1) 形式
def check_dialogue(text: str, cfg: dict) -> dict:
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


# 互換のため旧名も残す
check = check_dialogue


# ---------------------------------------------------------------- (2) 日本語
def check_japanese(text: str, cfg: dict) -> dict:
    jcfg = cfg.get("japanese", {})
    pass_ratio = float(jcfg.get("pass_ratio", 0.9))
    fail_ratio = float(jcfg.get("fail_ratio", 0.5))

    allowed = set(jcfg.get("allowed_abbreviations", DEFAULT_ABBREVIATIONS))

    def counts(s: str) -> tuple[int, int, int]:
        s = unicodedata.normalize("NFKC", s)
        # 許可リストの略語（URL・PDF など、大文字小文字は区別）だけは英単語として数えない
        words = [w for w in LATIN_RE.findall(s) if w not in allowed]
        return len(JA_RE.findall(s)), len(words), len(KANA_RE.findall(s))

    if not (text or "").strip():
        return {"ja_ratio": math.nan, "ja_status": "出力なし", "ja_mixed_turns": 0}

    dlg = parse(text)
    segments = [t.text for t in dlg.turns] + dlg.extra_lines
    n_ja, n_latin, n_kana = counts("\n".join(segments))

    # 発話ごとの混在チェック：全体では日本語が多くても、英語中心の発話が混ざっていれば要確認にする
    mixed = 0
    for seg in segments:
        sj, sl, _ = counts(seg)
        if sl > 0 and sj / (sj + sl) < fail_ratio:
            mixed += 1

    if n_ja + n_latin == 0:
        return {"ja_ratio": math.nan, "ja_status": "要確認", "ja_mixed_turns": mixed}  # 記号・数字だけ等
    ratio = n_ja / (n_ja + n_latin)
    if ratio < fail_ratio:
        status = "不合格"
    elif ratio >= pass_ratio and n_kana > 0 and mixed == 0:
        status = "合格"
    else:
        status = "要確認"
    return {"ja_ratio": round(ratio, 4), "ja_status": status, "ja_mixed_turns": mixed}


# ---------------------------------------------------------------- (3) 分割と失敗理由
def _unexpected_labels(raw: str, allowed: list[str]) -> list[str]:
    labels = []
    for line in raw.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            lab = normalize_speaker(m.group(1))
            if lab not in allowed and lab not in labels:
                labels.append(lab)
    # 「Person A:」のように空白を含む表記は LINE_RE に合わないので別に拾う
    for m in re.finditer(r"(?m)^\s*([A-Za-z]+\s+[A-Za-z0-9]{1,3})\s*[:：]", raw):
        lab = m.group(1)
        if lab not in labels:
            labels.append(lab)
    return labels


def check_chunks(rec: dict, cfg: dict) -> dict:
    allowed = list(cfg["format"]["speakers"])
    raw = rec.get("raw_outputs")
    issues = rec.get("gen_issues")
    has_output = bool((rec.get("output") or "").strip())

    res = {"n_chunks": math.nan, "n_chunks_gen_ok": math.nan, "n_chunks_raw_empty": math.nan,
           "n_chunks_gen_ng": math.nan, "n_chunks_unrecorded": math.nan,
           "chunks_all_ok": None, "gen_issue_kinds": "",
           "extract_hint": ""}

    if isinstance(issues, list):
        # 合格＝空リスト []、不合格＝中身のあるリスト、それ以外（null など）＝未記録。未記録を合格に数えない
        recorded = [i for i in issues if isinstance(i, list)]
        n_unknown = len(issues) - len(recorded)
        res["n_chunks"] = len(issues)
        res["n_chunks_gen_ok"] = sum(1 for i in recorded if len(i) == 0)
        res["n_chunks_gen_ng"] = len(recorded) - res["n_chunks_gen_ok"]
        res["n_chunks_unrecorded"] = n_unknown
        if res["n_chunks_gen_ng"] > 0:
            res["chunks_all_ok"] = False           # 1つでも不合格なら、未記録があっても不合格
        elif n_unknown > 0 or len(issues) == 0:
            res["chunks_all_ok"] = None            # 判定できない（対象外として扱う）
        else:
            res["chunks_all_ok"] = True
        kinds = sorted({str(k) for i in recorded for k in i})
        res["gen_issue_kinds"] = ",".join(kinds)

    if isinstance(raw, list):
        res["n_chunks_raw_empty"] = sum(1 for x in raw if not (x or "").strip())
        ng_raws = [x for x, i in zip(raw, issues or [[]] * len(raw))
                   if isinstance(i, list) and i and (x or "").strip()]
        if not isinstance(issues, list) and not has_output:
            ng_raws = [x for x in raw if (x or "").strip()]
        hints = []
        for x in ng_raws:
            for lab in _unexpected_labels(x, allowed):
                if lab not in hints:
                    hints.append(lab)
        if hints:
            res["extract_hint"] = "想定外の話者表記: " + ", ".join(hints[:5])

    if has_output:
        status = "出力あり"
    elif not isinstance(raw, list):
        status = "出力なし（生出力の記録なし）"
    elif all(not (x or "").strip() for x in raw):
        status = "出力なし（生出力なし）"
    else:
        status = "出力なし（抽出不可）"
    res["output_status"] = status
    return res


# ---------------------------------------------------------------- まとめ
def evaluate(records: list[dict], cfg: dict) -> pd.DataFrame:
    """1件ごとの判定結果（詳細表）。"""
    rows = []
    for r in records:
        rows.append({"id": r["id"], "set": r["set"], "model": r["model"],
                     **check_dialogue(r["output"], cfg),
                     **check_japanese(r["output"], cfg),
                     **check_chunks(r, cfg)})
    df = pd.DataFrame(rows)
    df["chunks_all_ok"] = df["chunks_all_ok"].map({True: 1.0, False: 0.0}).astype(float)
    return df


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """データセット × モデルごとの率（%）。set='ALL' は全データ。

    all_pass        : 従来の形式遵守率（分母＝全件）
    ja_pass         : 日本語「合格」の率（分母＝全件。要確認・不合格・出力なしは分子に入れない）
    ja_review       : 日本語「要確認」の率（分母＝全件）
    chunks_all_ok   : 全分割合格率（分母＝分割情報のある件数 n_chunk_known。0件なら NaN＝対象外）
    """
    both = pd.concat([detail, detail.assign(set="ALL")])
    g = both.groupby(["set", "model"])
    out = g[["all_pass", *CHECK_LABELS]].mean().mul(100)
    for s in JA_STATUSES:
        key = {"合格": "ja_pass", "要確認": "ja_review", "不合格": "ja_fail", "出力なし": "ja_no_output"}[s]
        out[key] = g["ja_status"].apply(lambda x, s=s: (x == s).mean() * 100)
    out["chunks_all_ok"] = g["chunks_all_ok"].mean() * 100
    out["n_chunk_known"] = g["chunks_all_ok"].count()
    out["n_output_empty"] = g["output_status"].apply(lambda x: (x != "出力あり").sum())
    out["n"] = g.size()
    return out.round(1).reset_index()
