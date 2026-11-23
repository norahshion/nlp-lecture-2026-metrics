"""評価プログラム共通の処理（設定・データ読み込み・検定）"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent
REQUIRED_FIELDS = ["id", "set", "model", "source", "output"]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_records(path: str | Path) -> list[dict]:
    """outputs.jsonl を読み込み、必須フィールドと重複をチェックする。"""
    path = Path(path)
    records = []
    with open(path, encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{ln} 行目が JSON として読めません: {e}") from e
            missing = [k for k in REQUIRED_FIELDS if k not in r]
            if missing:
                raise ValueError(f"{path.name}:{ln} 行目に必須フィールドがありません: {missing}")
            ref = r.get("reference")
            r["reference"] = ref if isinstance(ref, str) and ref.strip() else None
            r["output"] = r["output"] or ""
            records.append(r)

    dup = [k for k, c in Counter((r["id"], r["model"]) for r in records).items() if c > 1]
    if dup:
        raise ValueError(f"(id, model) の組が重複しています: {dup[:5]}")
    return records


def paired_test(df: pd.DataFrame, value_col: str, base: str, target: str,
                id_col: str = "id") -> dict:
    """同じ入力（id）に対する base と target の値を対応づけて比較する。

    返り値: n, 各平均, 差の平均, 対応のある t 検定の p 値, Wilcoxon 符号順位検定の p 値
    """
    piv = df.pivot_table(index=id_col, columns="model", values=value_col, aggfunc="mean")
    out = {"n": 0, f"mean_{base}": math.nan, f"mean_{target}": math.nan,
           "diff_mean": math.nan, "t_p": math.nan, "wilcoxon_p": math.nan}
    if base not in piv.columns or target not in piv.columns:
        return out
    piv = piv[[base, target]].dropna()
    n = len(piv)
    out["n"] = n
    if n == 0:
        return out
    a, b = piv[base].astype(float), piv[target].astype(float)
    out[f"mean_{base}"] = a.mean()
    out[f"mean_{target}"] = b.mean()
    out["diff_mean"] = (b - a).mean()
    if n >= 2 and (b - a).std() > 0:
        out["t_p"] = stats.ttest_rel(b, a).pvalue
    if n >= 2 and (b - a).abs().sum() > 0:
        try:
            out["wilcoxon_p"] = stats.wilcoxon(b, a).pvalue
        except ValueError:
            pass
    return out
