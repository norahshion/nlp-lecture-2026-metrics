# 評価プログラム（group-f）

未学習モデル（base）と QLoRA 追加学習済みモデル（lora）を、3つの指標で比較します。

| 指標 | ファイル | 内容 |
|---|---|---|
| 1. BERTScore | `metric_bertscore.py` | 生成文と正解対話（データAのみ）・入力文（全データ）の意味的類似度 |
| 2. 人的評価 | `human_eval_sheet.py` | 自然さ・内容の正確性・一貫性を1〜5で評価（モデル名は伏せる） |
| 3. 形式遵守率 | `metric_format.py` | 話者がA/Bのみ・最低ターン数・交互発話・空の発話なし・余計な行なし |

## 準備
```
pip install -r requirements.txt
```
BERTScore は初回に `cl-tohoku/bert-base-japanese-v3`（約450MB）をダウンロードします。GPU があれば自動で使います。

## 評価対象出力ファイルの形式
`data/outputs.jsonl` に、1行1件の JSON で保存してください（見本: `data/sample_outputs.jsonl`）。

```json
{"id": "jdd_001", "set": "A", "model": "base", "source": "入力した硬い文章", "reference": "A：…\nB：…", "output": "A：…\nB：…"}
```
- `id`：入力ごとに一意。**base と lora で同じ入力には同じ id** をつける（対応づけて比較するため）
- `set`：`A`＝Japanese-Daily-Dialogue からの逆生成データ、`B`＝実際のニュース・専門文書
- `model`：`base` か `lora`（名前は `config.yaml` の `compare` で変更可能）
- `reference`：データAの正解対話。データBは `null`
- `output`：後処理済みの対話文（「A：」「B：」形式）

## 実行手順
```
# 1) 人的評価シートを作る → human/sheet_<名前>.xlsx を各評価者に配る
python human_eval_sheet.py make --raters member1 member2 member3 member4 member5

# 2) 記入済みシートを human/ に戻したら、全指標をまとめて計算
python run_eval.py
```
`human/key.csv` には、どの項目がどのモデルの出力かが書かれています。評価が終わるまで評価者には見せないでください。

形式遵守率だけを素早く確認したいときは `python run_eval.py --no-bertscore` を使います。

## サンプルデータでの試行方法

以下は PowerShell で `eval/` フォルダに移動して実行します。
`data/sample_outputs.jsonl` は、入力4件 × 2モデルの計8件からなる動作確認用の見本です。
本番用の `data/outputs.jsonl` がなくても試せます。拡張子は `.jsonl` です。

### 1. 自動評価を試す

```powershell
python run_eval.py --outputs data/sample_outputs.jsonl --human-dir human_sample_unused --results results_sample_auto
```

`human_sample_unused/` は評価シートのない保存先として指定します。このフォルダにシートを置かないでください。
人的評価はスキップされ、形式遵守率と BERTScore が `results_sample_auto/` に出力されます。
初回の BERTScore はモデルのダウンロードに時間がかかります。
まず形式チェックだけを試すなら、末尾に `--no-bertscore` を付けます。

### 2. 仮点数入りの人的評価シートを作り、集計を試す

通常のシートは空欄です。`--dummy-scores` を付けた場合のみ、1〜5の仮点数を乱数で埋めます。
これは採点の代行ではなく、集計処理を試すための機能です。設定の seed が同じなら再現できます。

```powershell
python human_eval_sheet.py make --outputs data/sample_outputs.jsonl --raters テスト1 テスト2 テスト3 --dummy-scores
python run_eval.py --outputs data/sample_outputs.jsonl --human-dir human_sample --results results_sample --no-bertscore
```

BERTScore も試す場合は `--no-bertscore` を外します。
仮点数入りシートは既定で `human_sample/`、上のコマンドの集計結果は `results_sample/` に保存します。
各行のコメントと説明シートに仮点数であることを表示します。
本番用 `human/` への保存と、見本以外の入力への `--dummy-scores` の使用は禁止しています。
同じ保存先・評価者名で作成し直すとシートは上書きされます。

### 3. 自分で採点する場合

仮点数入りシートとは別のフォルダに、空欄のシートを作ります。
`--human-dir` は `make` より前に指定してください。

```powershell
python human_eval_sheet.py --human-dir human_sample_manual make --outputs data/sample_outputs.jsonl --raters member1 member2 member3 member4 member5
```

各評価者は `human_sample_manual/sheet_<名前>.xlsx` の「説明」を読み、
「評価」シートの評価欄に1〜5の整数を記入して保存します。
`key.csv` はモデル名との対応表なので、評価者には渡しません。
採点後は次で集計します。

```powershell
python run_eval.py --outputs data/sample_outputs.jsonl --human-dir human_sample_manual --results results_sample_manual
```

評価対象や評価項目を確定してからシートを配布してください。
配布後に入力データや設定を変更したり、同じ保存先でシートを再作成したりしないでください。

### 4. サンプル結果の読み方

- まず `summary.csv` と `summary.png` で指標ごとの傾向を確認し、`format_detail.csv` で形式違反の原因を確認します。
- 現在の見本・既定設定では、全体の形式遵守率は base が50%（4件中2件）、lora が75%（4件中3件）になります。
- `summary.csv` の `n` は各モデルの出力件数です。`set=ALL` は全体を表します。
- BERTScore は意味的な類似度です。例えば0.8でも「80%の事実が正しい」という意味ではありません。
- データBは正解対話がないため、正解比は `NaN`（未計算）です。正解比の `ALL` も正解のあるデータAだけを集計します。
- `stats.csv` の `n` は両モデルの値が揃った入力の組数、`diff_mean` は lora − base です。形式遵守の差が0.25なら25パーセントポイントの差です。
- 仮点数から得た人的評価の平均・p値・評価者間一致度は、集計が動くことを確認するためだけに使います。
- この見本は実モデルの優劣を判断するためのデータではありません。本番評価では、実際の生成結果と人による採点を使ってください。

### ファイルが見つからない場合

`--outputs` を省略すると `data/outputs.jsonl` が必要です。
`FileNotFoundError` が出た場合、サンプル試行ではシート作成・集計の両方に
`--outputs data/sample_outputs.jsonl` を指定しているか確認してください。

## 出力（results/）
- `summary.csv`：データセット × モデルごとの主要指標（`set=ALL` は全体）
- `stats.csv`：base と lora を同じ入力どうしで比べた検定（対応のある t検定・Wilcoxon）
- `summary.png`：スライド用のグラフ
- `*_detail.csv`：1件ごとの結果（失敗例を探すときに使う）
- `human_alpha.csv`：評価者間の一致度（Krippendorff's α、順序尺度）。0.67 以上が目安

## 設定（config.yaml）
最低ターン数・開始話者・評価項目の説明などは `config.yaml` で変えられます。

## 結果を読むときの注意
- **BERTScore（入力比）は、文体の近さにも反応します。** 硬い入力文に文体が近い base のほうが高く出やすいため、「情報保持」の根拠には人的評価の「内容の正確性」も合わせて示してください。
- データ数が20〜30件程度なので、p値は参考値として扱ってください。
