# 文章から日常対話への変換：評価プログラム

「自然言語処理システム特論」の講義で、Group-f が取り組む文章生成実験の評価用プログラムです。
硬い解説文やニュースを自然な日本語の2人対話に変換するタスクについて、追加学習前のモデル（`base`）と QLoRA による追加学習後のモデル（`lora`）を比較します。

本リポジトリは生成済みの文章を評価します。モデルの学習・推論を実行するプログラムや、学習済みの重みは含みません。
サンプルデータによる動作確認と、実際の生成結果を用いた性能評価を区別して扱います。

## 評価する内容

- **形式遵守率**：話者が A・B のみで両者が登場する、最低発話数を満たす、交互に話す、空の発話や余計な行がない、などの条件をすべて満たした出力の割合。
- **BERTScore**：生成文と正解対話（データA）、生成文と入力文（全データ）の意味的な類似度。
- **人的評価**：モデル名を伏せた Excel シートで、自然さ・内容の正確性・一貫性を1〜5で採点。平均値と評価者間一致度（Krippendorff's α）を集計します。

評価条件は [eval/config.yaml](eval/config.yaml) で設定します。既定の最低発話数は4（A→B→A→Bで4発話）、開始話者の指定はありません。
数値・固有名詞の保持率や、事実単位の保持率を計算する機能は現時点ではありません。

## ファイル構成

主要なファイルは `eval/` にあります。

```text
eval/
  README.md                  詳細な使い方
  requirements.txt           依存ライブラリ
  config.yaml                評価条件
  run_eval.py                一括評価・表とグラフの出力
  human_eval_sheet.py         人的評価シートの作成・集計
  metric_format.py           形式遵守率
  metric_bertscore.py         BERTScore
  common.py                  データ読み込み・検定などの共通処理
  parse_dialogue.py          対話文の解析
  data/sample_outputs.jsonl  動作確認用のサンプル
```

実行時には、評価シートや集計結果の保存先も作成されます。
詳しいオプションとサンプル結果の読み方は [評価プログラムの使い方](eval/README.md) を参照してください。

## 準備

Python 環境を用意し、リポジトリのルートから次を実行します。以下のコマンド例は PowerShell 用です。

```powershell
cd eval
python -m pip install -r requirements.txt
```

BERTScore には PyTorch（`torch`）も必要です。使用する Python・OS・CUDA に適合する環境を用意してください。
GPU は必須ではなく、設定が `device: auto` の場合は利用可能な CUDA GPU、なければ CPU を使います。
初回の BERTScore 実行時は、評価用モデル `cl-tohoku/bert-base-japanese-v3` のダウンロードに通信と時間が必要です。

以降のコマンドはすべて `eval/` 内で実行します。

## サンプルですぐに試す

サンプルは入力4件に対する2モデル分、計8件です。本番用の入力ファイルがなくても試せます。

まず、仮点数入りの人的評価シートを作成し、BERTScore を省いて集計します。

```powershell
python human_eval_sheet.py make --outputs data/sample_outputs.jsonl --raters member1 member2 member3 --dummy-scores
python run_eval.py --outputs data/sample_outputs.jsonl --human-dir human_sample --results results_sample --no-bertscore
```

評価シートは `human_sample/`、集計結果は `results_sample/` に保存されます。
BERTScore も試す場合は、2行目の `--no-bertscore` を外してください。

**仮点数は乱数で作った動作確認用の値です。モデルの品質を示す採点結果ではありません。**
`--dummy-scores` は指定したサンプルファイル専用で、通常の評価シートは空欄で作成されます。
同じ保存先・評価者名でシートを再作成すると上書きされます。

## 本番データを評価する

### 1. 生成結果を用意する

`eval/data/outputs.jsonl` を UTF-8 で保存します。1行につき1件の JSON とし、同じ入力に対する base・lora に同じ `id` を付けます。

```jsonl
{"id":"news_001","set":"B","model":"base","source":"変換対象の入力本文","reference":null,"output":"A：生成された発話\nB：生成された返答"}
{"id":"news_001","set":"B","model":"lora","source":"変換対象の入力本文","reference":null,"output":"A：生成された発話\nB：生成された返答"}
```

これは項目の説明用で、発話数などの合格例ではありません。

- `set`：`A` は Japanese-Daily-Dialogue からの逆生成データ、`B` はニュース・専門文書などの実データ。
- `source`：変換対象の入力本文。出典名ではありません。
- `reference`：データAでは正解対話、データBでは `null`。
- `output`：評価対象の生成対話。対話中の改行は JSON 内で `\n` と記述します。

同じ `(id, model)` を重複させず、両モデルで入力・評価条件・後処理を統一してください。
学習担当から届く `pred_base.jsonl`・`pred_lora.jsonl` は、そのままでは評価用形式として読み込めません。
今回の受領形式では `input` を評価用の `source`、`prediction` を `output` に対応させ、`set` と `model` を付けてまとめる必要があります。自動変換機能はありません。

### 2. 人的評価シートを配布する

```powershell
python human_eval_sheet.py make --raters member1 member2 member3
```

各評価者に `human/sheet_<評価者ID>.xlsx` を配布し、「説明」を読んで「評価」シートを採点してもらいます。
評価者IDには個人名を使わず、`member1` などの識別子を推奨します。
モデルとの対応表 `human/key.csv` は評価終了まで評価者に見せないでください。
採点済みシートを同じファイル名で `human/` に戻します。配布後のシート再作成や評価対象の差し替えは避けてください。

### 3. 一括集計する

```powershell
python run_eval.py
```

人的評価シートがない場合は、その集計をスキップします。
`data/outputs.jsonl` がない場合は `FileNotFoundError` になります。別のファイルを使うときは、シート作成と集計の両方に `--outputs <ファイルのパス>` を指定してください。

## 出力と解釈

本番の既定出力先は `eval/results/` です。

- `summary.csv`：データ群・モデル別の主要指標。`ALL` は全体。
- `summary.png`：base（青）と lora（オレンジ）の比較グラフ。
- `format_detail.csv` / `format_summary.csv`：形式チェックの詳細と集計。
- `bertscore_detail.csv`：各出力の BERTScore。計算した場合のみ出力。
- `human_detail.csv` / `human_summary.csv` / `human_alpha.csv`：人的評価の詳細・平均・一致度。シートを集計した場合のみ出力。
- `stats.csv`：同じ入力どうしの比較。`diff_mean` は lora − base。

BERTScore は事実の正しさや情報保持率を直接表しません。正解のないデータBの正解比は `NaN`（未計算）であり、0点ではありません。
人的評価が1〜5の評点である場合も、「何%の事実が保持されたか」という割合とは区別してください。
少数の見本や仮点数から性能の優劣を判断せず、本番データの内容確認・人的評価と合わせて解釈します。
試行条件ごとに `--results` で保存先を分けると、以前の出力との混同を防げます。

## 共有するファイルについて

実データ、評価者の情報・コメント、モデルとの対応表、ローカル環境のパスが入るファイルは、公開前に内容と Git の追加対象を確認してください。
生成された Excel・CSV・画像や Python のキャッシュが、現在の `.gitignore` ですべて除外されるとは限りません。
外部のモデル・データセット・入力文を再配布する場合は、それぞれの利用条件を確認してください。
