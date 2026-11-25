"""results_A / results_B から、発表用のグラフと「評価結果.md」を作る（結果は再計算しない）。

使い方（eval/ で実行）:
    python make_report.py
出力先は results_report/（結果を含むため公開しない。.gitignore の対象）。
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1b8a62"
INK, SUB, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
CRIT = ["日本語会話としての自然さ", "情報の網羅性", "対話の一貫性", "内容の正確性"]
SHORT = dict(zip(CRIT, ["自然さ", "網羅性", "一貫性", "正確性"]))
SCORED, ACC = CRIT[:3], CRIT[3]
REF, CHUNK = "BERTScore_F1(正解比)", "BERTScore_F1(入力比・分割単位)"
FMT, JA = "後処理後の形式遵守率(%)", "日本語合格率(%)"


def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig")


def load(folder: Path, s: str) -> dict:
    d = {"s": s}
    d["sum"] = read(folder / "summary.csv").query("set == @s").set_index("model")
    d["stats"] = read(folder / "stats.csv").query("set == @s").set_index("metric")
    d["ja"] = read(folder / "format_detail.csv").query("set == @s").groupby(["model", "ja_status"]).size().unstack(fill_value=0)
    det = read(folder / "human_final_detail.csv").query("set == @s")
    human = det[(det["source"] == "人") & (det["status"] == "数値")]
    d["raters"] = human["rater"].nunique()
    d["items"] = human[human["model"] == "lora"].groupby(["criterion", "id"])["score"].mean()
    h = read(folder / "human_final_summary.csv").query("set == @s")
    d["hs"] = h[h["scope"].str.startswith("全件")].set_index(["model", "criterion"])
    d["cov"] = read(folder / "human_final_coverage.csv").query("set == @s").set_index("model")
    d["alpha"] = read(folder / "human_final_alpha.csv").set_index("criterion")["alpha_ordinal"]
    d["n"] = int(d["sum"].loc["base", "n"])
    d["ref_note"] = "、参考" if s == "B" else ""
    return d


def r2(x) -> str:
    """四捨五入で小数2桁（x.xx5 は切り上げ）。"""
    return "―" if pd.isna(x) else f"{np.floor(float(x) * 100 + 0.5 + 1e-6) / 100:.2f}"


def style(ax, title, ymax, ylabel=""):
    ax.set_title(title, fontsize=14, color=INK, pad=12)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=11, color=SUB)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=11, colors=INK)


def bars(ax, labels, base, lora, fmt, missing=""):
    x, w = np.arange(len(labels)), 0.34
    top = ax.get_ylim()[1]
    for off, vals, color in ((-w / 2 - .01, base, BLUE), (w / 2 + .01, lora, ORANGE)):
        for i, v in enumerate(vals):
            if pd.isna(v):
                ax.text(x[i] + off, top * .015, missing, ha="center", va="bottom", fontsize=10, color=SUB)
                continue
            ax.bar(x[i] + off, v, w, color=color)
            ax.text(x[i] + off, v + top * .012, fmt(v), ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlim(-.6, len(labels) - .4)


def save(fig, path, legend=True, note=""):
    if legend:
        hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
        fig.legend(hs, ["base（追加学習なし）", "lora（追加学習あり）"], loc="upper center", ncol=2,
                   frameon=False, fontsize=12)
    if note:
        fig.text(.5, .015, note, ha="center", fontsize=10, color=SUB)
    fig.tight_layout(rect=(0, .05 if note else 0, 1, .92 if legend else 1))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def figures(A, B, out):
    # 1. 形式と日本語
    xl = [f"データA\n（n={A['n']}）", f"データB\n（n={B['n']}、参考）"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, c, t in zip(axs, (FMT, JA), ("後処理後の形式遵守率", "日本語合格率")):
        style(ax, t, 115, "割合（%）")
        ax.set_yticks(range(0, 101, 20))
        bars(ax, xl, *[[A["sum"].loc[m, c], B["sum"].loc[m, c]] for m in ("base", "lora")], lambda v: f"{v:.1f}%")
    save(fig, out / "fig1_format_japanese.png")

    # 2. 出力の内訳
    parts = {"合格": ("日本語の対話", GREEN), "要確認": ("要確認（日本語と他言語が混在など）", "#eda100"),
             "不合格": ("日本語以外（英語など）", "#b8b6ab"), "出力なし": ("出力なし", "#5d5c56")}
    parts = {k: v for k, v in parts.items() if any(k in d["ja"] and d["ja"][k].sum() for d in (A, B))}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, d in zip(axs, (A, B)):
        for y, m in enumerate(("lora", "base")):
            left = 0
            for k, (_, col) in parts.items():
                v = int(d["ja"].loc[m].get(k, 0))
                if not v:
                    continue
                ax.barh(y, v, .5, left=left, color=col, edgecolor="white", linewidth=2)
                small = v / d["n"] < .06
                ax.text(left + v / 2, y + (.36 if small else 0), f"{v}件", ha="center", va="center", fontsize=12,
                        color=INK if small or k in ("不合格", "要確認") else "white")
                left += v
        ax.set_title(f"データ{d['s']}（{d['n']}入力{d['ref_note']}）", fontsize=14, color=INK)
        ax.set_yticks([0, 1], ["lora", "base"])
        ax.set_xlim(0, d["n"])
        ax.set_ylim(-.4, 1.5)
        ax.set_xticks([])
        ax.tick_params(length=0, labelsize=13)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in parts.values()], [n for n, _ in parts.values()],
               loc="upper center", ncol=len(parts), frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(out / "fig2_output_types.png", dpi=180, facecolor="white")
    plt.close(fig)

    # 3. BERTScore
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, d, c, key, t in ((axs[0], A, REF, "正解比", "データA：正解の対話との近さ"),
                             (axs[1], B, CHUNK, "分割単位", "データB（参考）：入力との近さ（分割単位）")):
        style(ax, t, 1.0, "BERTScore F1")
        ns = [int(d["sum"].loc[m, f"n_{key}_使用"]) for m in ("base", "lora")]
        bars(ax, [f"base n={ns[0]}　／　lora n={ns[1]}"], [d["sum"].loc["base", c]], [d["sum"].loc["lora", c]],
             lambda v: f"{v:.3f}")
    save(fig, out / "fig3_bertscore.png",
         note="512トークンを超えた件は除外。空の出力は0点として平均に含む。左右は比べる相手が違うため、値を直接比べない。")

    # 4. 人手評価
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, d in zip(axs, (A, B)):
        n = int(d["hs"].loc[("lora", CRIT[0]), "n_items"])
        style(ax, f"データ{d['s']}（{n}入力・評価者{d['raters']}人{d['ref_note']}）", 5.6, "平均点（1〜5）")
        ax.set_yticks(range(0, 6))
        bars(ax, [SHORT[c] for c in CRIT],
             [np.nan if c == ACC else d["hs"].loc[("base", c), "mean"] for c in CRIT],
             [d["hs"].loc[("lora", c), "mean"] for c in CRIT], r2, missing="比較\nしない")
    rule = [int(d["cov"].loc["base", ["n_規則_出力なし", "n_規則_日本語以外"]].sum()) for d in (A, B)]
    save(fig, out / "fig4_human.png",
         note=f"base のうち A {rule[0]}件・B {rule[1]}件は、出力なし・日本語以外のため規則で1点（正確性は評価不能）とした。")

    # 5. lora の入力ごとの平均点
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, d in zip(axs, (A, B)):
        style(ax, f"lora：データ{d['s']}（{d['items'][ACC].size}入力{d['ref_note']}）", 5, "入力ごとの平均点")
        for i, c in enumerate(CRIT):
            v = np.sort(d["items"][c].values)
            for q in np.unique(v):
                k = int((v == q).sum())
                ax.scatter(i + (np.arange(k) - (k - 1) / 2) * .1, [q] * k, s=75, color=ORANGE, edgecolor="white",
                           linewidth=1.5, zorder=3)
            ax.plot([i - .28, i + .28], [v.mean()] * 2, color=INK, linewidth=2, zorder=2)
            ax.text(i + .31, v.mean(), r2(d["hs"].loc[("lora", c), "mean"]), va="center", fontsize=11, color=INK)
        ax.set_xticks(range(4), [SHORT[c] for c in CRIT])
        ax.set_xlim(-.5, 3.7)
        ax.set_ylim(.8, 5.3)
        ax.set_yticks(range(1, 6))
    save(fig, out / "fig5_lora_by_input.png", legend=False, note="点＝1つの入力（評価者の平均）、横線＝全体の平均。縦軸は1点から。")


def report(A, B, out):
    def S(d, m, c):
        v = d["sum"].loc[m, c]
        return "―" if pd.isna(v) else (f"{v:.1f}%" if c.endswith("(%)") else f"{v:.3f}")

    def H(d, m, c):
        return r2(d["hs"].loc[(m, c), "mean"])

    def J(d, m, k):
        return int(d["ja"].loc[m].get(k, 0))

    def cov(d, m, c):
        return int(d["cov"].loc[m, c])

    def rng(d, c):
        return f"{r2(d['items'][c].min())}〜{r2(d['items'][c].max())}"

    na, nb = A["items"][ACC].size, B["items"][ACC].size
    tests = [("形式遵守", "後処理後の形式遵守(0/1)"), ("日本語合格", "日本語合格(0/1)"), ("BERTScore（正解比）", REF)]
    tests += [(f"人手：{SHORT[c]}", f"人的評価(final):{c}") for c in SCORED]
    stat_rows = "\n".join(f"| {t} | {int(A['stats'].loc[m, 'n'])} | {A['stats'].loc[m, 'wilcoxon_p_表示']} |" for t, m in tests)
    hrows = "\n".join(f"| {SHORT[c]} | {H(A, 'base', c)} | **{H(A, 'lora', c)}** | {H(B, 'base', c)} | **{H(B, 'lora', c)}** |"
                      for c in SCORED)
    crows = []
    for d in (A, B):
        for m in ("base", "lora"):
            v = [cov(d, m, k) for k in ("n_人が評価", "n_規則_出力なし", "n_規則_日本語以外")]
            crows.append(f"| データ{d['s']} {m} | {v[0]} | {v[1]} | {v[2]} | {sum(v)} |")
    crows = "\n".join(crows)
    rrows = "\n".join(f"| {SHORT[c]} | {rng(A, c)} | {rng(B, c)} |" for c in CRIT)
    arows = "\n".join(f"| データ{d['s']} | " + " | ".join(f"{d['alpha'][c]:.3f}" for c in CRIT) + " |" for d in (A, B))
    ahead = " | ".join(SHORT[c] for c in CRIT)
    worst = max(A["alpha"].max(), B["alpha"].max())
    atext = "どの項目も目安に届いていない" if worst < .67 else "目安に届かない項目がある"

    md = f"""# 評価結果

追加学習していないモデル（**base**）と、QLoRA で追加学習したモデル（**lora**）を、硬い文章を日本語の2人対話に変換する課題で比べた結果です。
評価のしかたは [評価方法.md](../../評価方法.md) にあります。**評価結果を含むため、このファイルと図は公開しません。**

## 結論

1. **追加学習により、日本語の対話として出力できるようになった。** 日本語合格率は、データAで {S(A, 'base', JA)} → **{S(A, 'lora', JA)}**、データBで {S(B, 'base', JA)} → **{S(B, 'lora', JA)}**。
2. **データAでは、正解の対話との近さ（BERTScore）も上がった。** {S(A, 'base', REF)} → **{S(A, 'lora', REF)}**。
3. **人手評価でも、自然さ・網羅性・一貫性は lora が高い。** ただし差の多くは、base が日本語の対話を出せなかったことによる。
4. **lora にも課題がある。** 正確性の平均はデータA {H(A, 'lora', ACC)}、データB {H(B, 'lora', ACC)} で、入力による差が大きい。

## 評価の対象

| | データA | データB（参考） |
|---|---|---|
| 内容 | 学習に使っていないテスト用データ | 実際のニュース記事 |
| 自動指標 | {A['n']}入力（正解あり） | {B['n']}入力（正解なし） |
| 人手評価 | {na}入力（{A['n']}入力から乱数で抽出） | {nb}入力すべて |
| 評価者 | {A['raters']}人 | {B['raters']}人 |

## 1. 形式と日本語

![形式遵守率と日本語合格率](fig1_format_japanese.png)

- lora は、形式遵守率がA {S(A, 'lora', FMT)}・B {S(B, 'lora', FMT)}、日本語合格率がA {S(A, 'lora', JA)}・B {S(B, 'lora', JA)}。
- base は形式だけ整った英語などの対話も出すため、形式遵守率より日本語合格率のほうが低い。

### 出力の内訳

![出力の内訳](fig2_output_types.png)

| | 日本語の対話 | 日本語以外 | 出力なし |
|---|---|---|---|
| データA base | {J(A, 'base', '合格')} | {J(A, 'base', '不合格')} | {J(A, 'base', '出力なし')} |
| データA lora | {J(A, 'lora', '合格')} | {J(A, 'lora', '不合格')} | {J(A, 'lora', '出力なし')} |
| データB base | {J(B, 'base', '合格')} | {J(B, 'base', '不合格')} | {J(B, 'base', '出力なし')} |
| データB lora | {J(B, 'lora', '合格')} | {J(B, 'lora', '不合格')} | {J(B, 'lora', '出力なし')} |

- 「出力なし」は後処理後の出力が空という意味で、モデルが何も生成しなかったとは限らない。「Person A:」のような話者表記は後処理で取り出せず、空になる。

## 2. BERTScore

![BERTScore](fig3_bertscore.png)

| | base | lora |
|---|---|---|
| データA：正解比 | {S(A, 'base', REF)} | **{S(A, 'lora', REF)}** |
| データB：入力比・分割単位（参考） | {S(B, 'base', CHUNK)} | **{S(B, 'lora', CHUNK)}** |

- base が低い主な理由は、空の出力が0点になること。「対話の内容が正解と違う」ことだけを示す値ではない。
- BERTScore は文章の意味の近さを測るもので、事実の正しさは表さない。

## 3. 人手評価（1〜5点）

![人手評価](fig4_human.png)

| 項目 | A base | A lora | B base | B lora |
|---|---|---|---|---|
{hrows}
| 正確性 | 比較しない | {H(A, 'lora', ACC)} | 比較しない | {H(B, 'lora', ACC)} |

採点した件の内訳：

| | 人が評価 | 規則（出力なし） | 規則（日本語以外） | 合計 |
|---|---|---|---|---|
{crows}

- 出力なし・日本語以外の件は、規則で自然さ・網羅性・一貫性を1点、正確性を評価不能とした。
- **正確性を base と比べない理由**：base で人が採点できたのはデータA {cov(A, 'base', 'n_人が評価')}件・データB {cov(B, 'base', 'n_人が評価')}件だけで、平均を比べると誤解を招くため。

### lora の入力ごとの差

![lora のばらつき](fig5_lora_by_input.png)

| 項目 | データA（最低〜最高） | データB（最低〜最高） |
|---|---|---|
{rrows}

- 形式と日本語の条件を満たしていても、内容の質は入力によって差がある。

### 評価者間の一致度（Krippendorff's α）

| | {ahead} |
|---|---|---|---|---|
{arows}

- 目安は0.67以上（1に近いほど一致）。{atext}ため、**平均点の小さな差は結論に使わない**。
- データBは{nb}入力だけなので、αも参考値。

## 4. 検定（データA、base と lora の対応のある比較）

| 指標 | 組の数 | p値（Wilcoxon） |
|---|---|---|
{stat_rows}

- BERTScore の組の数は、base・lora のどちらかが512トークンを超えた入力を除いた数。
- 人手評価の p値には、規則で1点とした件が含まれる。「日本語の対話を出せたか」を含む差であり、対話の質だけの差ではない。
- データBは{B['n']}入力と少ないため、検定は示さない。

## 5. 読むときの注意

- **現在のプロンプト・後処理の条件での比較**である。日本語で書くことや A/B の表記はプロンプトで明示していないため、条件を変えれば base の結果は変わりうる。
- 人手評価の件数は少ない（データA {na}入力、データB {nb}入力）。
- base で人が評価できた件が少ないため、**日本語の対話を出せた場合の質で base と lora を比べることはできない**。

## 6. スライドに載せるなら

| 優先 | 図 | 伝えること |
|---|---|---|
| 1 | fig1 | 追加学習で日本語の2人対話を出せるようになった |
| 2 | fig4 | 人手評価でも lora が高い（規則で採点した件数の注記も載せる） |
| 3 | fig3 の左 | データAで正解に近づいた（空の出力が0点であることも書く） |
| 余裕があれば | fig2・fig5 | base が低い理由、lora の課題 |

---
作り直すとき（`eval/` で、再集計のあとに実行）：`python make_report.py`
"""
    (out / "評価結果.md").write_text(md, encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-a", default="results_A")
    ap.add_argument("--results-b", default="results_B")
    ap.add_argument("--out", default="results_report")
    a = ap.parse_args()
    inst = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP") if f in inst][:1] or ["sans-serif"]
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    A, B = load(Path(a.results_a), "A"), load(Path(a.results_b), "B")
    figures(A, B, out)
    report(A, B, out)
    print(f"保存しました: {out}")


if __name__ == "__main__":
    main()
