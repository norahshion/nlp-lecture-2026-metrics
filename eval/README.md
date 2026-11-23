# 評価プログラム（group-f）

未学習モデル（base）と QLoRA 追加学習済みモデル（lora）を比較します。
コマンドはすべて PowerShell で `eval/` フォルダに移動して実行します。

## 2種類の評価

生成側の処理は「モデルが文章を生成する → 後処理で話者付きの発話を抽出・整形・結合する」の2段階です。
この評価プログラムでは、どちらを評価するかを区別します。

| 評価 | 対象 | 使う項目 | 何がわかるか |
|---|---|---|---|
| **final**（システム全体） | 後処理後の最終出力 | `output`（pred の `prediction`） | 利用者が実際に受け取る対話の品質。後処理の失敗も含む |
| **raw**（生成文） | モデルが生成した文章 | `raw_outputs` | 後処理の前に、モデルがどんな文章を書いたか |

- `raw_outputs` は、入力を分割した単位ごとにモデルが新しく生成した文章です。モデルへの入力ではありません。
- `prediction` は、`raw_outputs` から話者付きの発話を抽出・整形・結合した最終結果です。変換後は `output` になります。
- 自動指標（形式遵守率・日本語判定・全分割合格・BERTScore）と `run_eval.py` の集計は、すべて **final** が対象です。
- raw の人的評価は別のシート・別の集計先で扱い、`summary.csv` や `stats.csv` には混ぜません。
- 空の出力は削除せず、`raw_outputs` で補うこともしません。

### 今回の結果の前提
- 今回の pred ファイルは、**現在の生成プロンプトの条件で比べた結果**です。プロンプトでは「A＝聞き手・B＝説明役」の役割、日本語で書くこと、A/B の話者表記は、どれも明示されていません。
- 生成側の後処理は `Person A:` / `Person B:` という表記に対応していません。そのため base の5件中3件は、英文の対話を生成していても `prediction` が空になっています（共有コードの後処理で、全10件の `prediction` と `issues` を再現できることを確認済み）。
- **final の Excel で base の一部が空欄になるのは、このためです。** 生成された英文は raw のシートで確認できます。
- プロンプトや後処理を変えて生成し直す場合は、**別の実験**として扱い、今回の結果と混ぜないでください。

## 指標

| 指標 | ファイル | 対象 | 内容 |
|---|---|---|---|
| 後処理後の形式遵守率 | `metric_format.py` | final | 話者がA/Bのみ・最低ターン数・交互発話・空の発話なし・余計な行なし |
| 日本語判定（補助） | `metric_format.py` | final | 出力が日本語か（合格／要確認／不合格／出力なし） |
| 全分割合格 | `metric_format.py` | final | 生成側の形式チェックに全分割単位が合格したか。分割情報がなければ対象外 |
| BERTScore | `metric_bertscore.py` | final | 生成文と正解対話（データAのみ）・入力文（全文／分割単位）の意味的類似度 |
| 人的評価 | `human_eval_sheet.py` | final / raw | 4項目を1〜5で評価（モデル名は伏せる） |

## 準備
```powershell
python -m pip install -r requirements.txt
```
BERTScore は初回に `cl-tohoku/bert-base-japanese-v3`（約450MB）をダウンロードします。GPU があれば自動で使います。

## 評価用ファイルの形式
`data/outputs.jsonl` に、1行1件の JSON で保存します（見本: `data/sample_outputs.jsonl`）。

```json
{"id": "jdd_001", "set": "A", "model": "base", "source": "入力した硬い文章", "reference": "A：…\nB：…", "output": "A：…\nB：…"}
```
- `id`：入力ごとに一意。**base と lora で同じ入力には同じ id** をつける（対応づけて比較するため）
- `set`：`A`＝Japanese-Daily-Dialogue からの逆生成データ、`B`＝実際のニュース・専門文書
- `model`：`base` か `lora`（名前は `config.yaml` の `compare` で変更可能）
- `reference`：データAの正解対話。データBは `null`
- `output`：後処理後の対話文
- `raw_outputs`（任意）：分割ごとの生出力。raw の人的評価に必要

## 生成結果（pred_*.jsonl）の変換
```powershell
# データB（ニュース）。分割ごとの入力・出力も付ける（分割単位の BERTScore に必要）
python convert_pred.py --base ../pred_base.jsonl --target ../pred_lora.jsonl --pipeline ../member_program/dialogue_pipeline.py

# データA（test.jsonl の生成結果）。id がなければ index から test_000 のような id を作る
python convert_pred.py --base <base のファイル> --target <lora のファイル> --set A --id-from-index --pipeline ../member_program/dialogue_pipeline.py --out data/outputs_A.jsonl
```
データA と データB は別のファイルに変換し、`run_eval.py --outputs` と `--results` も分けて実行します。

| pred_*.jsonl | 変換後 | 備考 |
|---|---|---|
| `input` | `source` | 入力した文章 |
| `prediction` | `output` | 空でも削除しない（失敗を除くと評価が過大になるため） |
| `source` | `publisher` | 発信元の名前 |
| `format_ok` / `issues` | `gen_format_ok` / `gen_issues` | 生成側の形式チェック結果（分割単位ごと） |
| `raw_outputs` | `raw_outputs` | 分割単位ごとの生出力。そのまま引き継ぐ |
| `adapter` | `uses_adapter` | ローカルパスは引き継がず、使用の有無だけ残す |
| （なし） | `model` | `--base` 側を base、`--target` 側を lora とする |
| （なし） | `set` | id が `news_` なら B、`jdd_` なら A。それ以外は `--set` で指定 |
| `id` がない | `id` | `--id-from-index` を付けたときだけ、`index` から `test_000` のように作る（接頭辞は `--id-prefix`） |
| （なし） | `source_chunks` / `final_chunks` | `--pipeline` を付けたときだけ。分割ごとの入力と、分割ごとの後処理後の出力 |

- 次の場合は変換せずに停止します：id の重複、id のない行がある（`--id-from-index` なし）、両モデルの id の不一致、同じ id で入力文や正解（`reference`）が違う、`raw_outputs` と `issues` の数が違う、`raw_outputs` が文字列のリストでない、`issues` が「文字列のリスト」のリストでない（`null` を含む場合など）。
- **分割情報（`--pipeline`）**：生成に使った共有コード `dialogue_pipeline.py` の前処理（`preprocess`）で、入力を生成時と同じ分割に分けます。各 `raw_outputs` を同じコードの後処理で発話に分けたものが `final_chunks` です。分割数が `raw_outputs` と一致し、`final_chunks` を結合すると `prediction` が再現できることを確かめ、合わなければ停止します（別の版のコードや、別の `--max-chunk-chars` で生成された可能性があるため）。`member_program/` は Git の管理外なので、評価コードは `--pipeline` で指定されたファイルだけを読み込みます。
- 変換先（既定は `data/outputs.jsonl`）が既にあると停止します。上書きするときだけ `--overwrite` を付けます。
- 書き出しは、他と重ならない名前の一時ファイルを新しく作ってから置き換えます。元の pred ファイルは変更しません。

## 人的評価

### シートの作成・集計

| | final（既定） | raw |
|---|---|---|
| 作成 | `python human_eval_sheet.py make --raters member1 member2 member3` | `python human_eval_sheet.py make --target raw --raters member1 member2 member3` |
| シート・対応表 | `human/final/` | `human/raw/` |
| 集計 | `python human_eval_sheet.py aggregate`（または `python run_eval.py`） | `python human_eval_sheet.py aggregate --target raw` |
| 集計結果 | `results/human_final/`（`run_eval.py` では `results/human_final_*.csv`） | `results/human_raw/` |

- 評価者名には個人名ではなく `member1` などの ID を使います。
- 各評価者に `sheet_<ID>.xlsx` を配り、採点後に同じフォルダへ同じ名前で戻します。
- `key.csv`（モデルとの対応表）と `meta.json` は、評価が終わるまで評価者に見せないでください。
- 既にシートや対応表があるフォルダには作成しません。作り直すときだけ `--overwrite` を付けます。配布後の作り直しは避けてください。
- 別のフォルダを使うときは `--human-dir` を `make` / `aggregate` より前に書きます。

**raw のシート**
- 1入力・1モデルを1行にし、生出力を `【分割1/3】` のような境界つきで順番どおりに並べます。本文は翻訳・修正・抽出をしません（保存後に読み直し、1文字も変わっていないことを確認します）。
- 空の分割も削除せず、`【分割2/3】（空の生出力）` と表示し、「表示上の注意」に空の分割番号を出します。
- `raw_outputs` がないデータ（`sample_outputs.jsonl` など）では、final に切り替えずに、対象の id を示して停止します。

**共通**
- モデル名は伏せ、順番はシャッフルします（final と raw で順番は別）。
- Excel の1セルの上限（32,767文字）を超える入力・出力があれば、切り捨てずに対象を示して停止します。
- 旧形式（自然さ・内容の正確性・一貫性の3項目）のシートとは互換性がありません。`human/` 直下にある旧シートは、そのまま残し、集計には使いません。

### 評価項目（`config.yaml` の `human_eval`）

| 項目 | 定義 |
|---|---|
| 日本語会話としての自然さ | 想定する日本語の日常会話として自然か。英文のみの出力は日本語条件を満たさない。丁寧語であること自体は減点しない |
| 内容の正確性 | 入力の事実・数値・対象・条件を変えていないか。入力にない事実を断定していないか。欠落はここでは減点せず、網羅性で評価する |
| 情報の網羅性 | 入力の重要な情報が含まれているか。言い換えは許容する |
| 対話の一貫性 | 発話の流れ、対話内の矛盾、不自然な会話の再開始や役割の変化。A が説明すること自体は減点しない |

- 各項目の 1・3・5 点の目安はシートの「説明」に載せています。2・4 点はその中間です。
- 項目や定義を変えたら `criteria_version` を上げてください。版の違うシートは混ぜて集計できません。

### 空出力・評価不能の扱い
- **空出力**（評価対象の欄が空、または raw で全分割が空）：自然さ・網羅性・一貫性は 1。正確性は「評価不能」とし、理由に「空出力」を選びます（述べている内容がないため。欠落は網羅性で評価します）。
- **raw で一部の分割だけが空**：その分割の内容がないものとして、通常どおり評価します。
- **英語など日本語以外の出力**：自然さは 1。内容が読み取れる項目は通常どおり採点し、言語のために判断できない項目だけ「評価不能」にして、理由に「言語のため判断できない」を選びます。
- 「評価不能」は点数ではなく、**欠測の理由**として記録します。空欄のままの「未記入」とは別に数えます。
- 1〜5 の整数・「評価不能」・空欄以外の値があると、集計は停止します。

### 集計の出力
- `summary`：項目ごとの平均・標準偏差と、`n_valid`（平均に使った評価数）・`n_unable`（評価不能）・`n_blank`（未記入）・`n_total`
- `alpha`：評価者間の一致度（Krippendorff's α、順序尺度）。点数の入った評価だけで計算し、有効件数（`n_valid_ratings`、2人以上が採点した項目数）を併記します。0.67 以上が目安です。
- `tests`：評価者の平均（点数のみ）を、同じ入力どうしで base と lora 比べた検定。`n` は有効な組の数です。
- `reasons`：評価不能の理由別の件数
- 集計時に、評価対象（final/raw）・評価項目・基準の版・対応表がシートと一致するかを確かめ、違えば停止します（final に raw のシートや旧シートが混ざった場合など）。

## 一括評価
```powershell
python run_eval.py                 # BERTScore を含む
python run_eval.py --no-bertscore  # 形式チェックだけ素早く確認
```
- `human/final/` に final のシートがあれば、人的評価（final）も集計します。raw のフォルダや旧形式のシートを `--human-dir` に渡すと停止します。
- 出力先は `results/` です。条件ごとに `--results` で分けると、以前の結果と混同しません。

### 出力（results/）
- `summary.csv`：データセット × モデルごとの主要指標（`set=ALL` は全体）
- `stats.csv`：base と lora を同じ入力どうしで比べた検定（対応のある t検定・Wilcoxon）
- `summary.png`：スライド用のグラフ
- `format_detail.csv` / `bertscore_detail.csv`：1件ごとの結果（失敗例を探すときに使う）
- `human_final_*.csv`：人的評価（final）の詳細・平均・一致度・評価不能の理由

## 形式・日本語・分割の3つの判定（final が対象）
3つは原因が違うので、表（`summary.csv`）でもグラフでも別々に出します。

| 列 | 意味 | 分母 |
|---|---|---|
| 後処理後の形式遵守率(%) | 後処理後の出力が、対話形式のチェックに全項目合格 | 全件 |
| 日本語合格率(%) | 日本語判定が「合格」 | 全件 |
| 日本語要確認率(%) | 日本語判定が「要確認」 | 全件 |
| 全分割合格率(%) | 生成側の形式チェックに全分割単位が合格 | 分割情報のある件数（`n_分割情報あり`） |

**日本語判定の基準**（しきい値は `config.yaml` の `japanese`）
- **この判定は補助的なものです。** 日本語として自然かどうかは人的評価で確認します。
- 日本語の文字数 ÷（日本語の文字数＋英単語の数）を `ja_ratio` とします。話者ラベルは除きます。
- 英単語として数えないのは、`config.yaml` の許可リスト（`allowed_abbreviations`：URL・PDF・AI など）にある略語だけです。それ以外の英単語は、大文字だけでもすべて数えます。
- 長音記号「ー」や中黒「・」などの記号は、日本語の文字にも、かなにも数えません。
- `ja_ratio` が 0.5 未満は「不合格」です。0.9 以上で、かなを含み、英語中心の発話（その発話だけの `ja_ratio` が 0.5 未満）がなければ「合格」です。それ以外は「要確認」です。英語中心の発話の数は `ja_mixed_turns` に出ます。
- **「要確認」は日本語合格率の分子に入れません。** グラフでは日本語合格率の棒に「(+要確認 ○%)」と添えます。
- 出力が空のものは「出力なし」として、分母に含めます。

**全分割合格**
- 分割情報（`gen_issues`）のないデータ（`sample_outputs.jsonl` など）は、不合格ではなく「対象外」とします。表では `NaN`、グラフでは「対象外」と表示します。
- 合格＝`[]`、不合格＝問題名の入ったリストです。`null` などの未記録は合格に数えません。不合格がなく未記録がある場合は「対象外」です（件数は `n_chunks_unrecorded`）。
- 「全分割合格」は生成側の形式チェックに合格したという意味で、日本語で正しく変換できたかどうかは別です。

**失敗理由（`format_detail.csv`）**
- `output_status`：出力あり／出力なし（生出力なし）／出力なし（抽出不可＝生出力はあるが最終出力にならなかった）／出力なし（生出力の記録なし）
- `extract_hint`：抽出できなかった生出力に想定外の話者表記（例: `Person A`）があれば記録します。原因の手がかりで、断定ではありません。
- `gen_issue_kinds`：生成側が記録した問題の種類（例: `no_dialogue`）。将来の `no_dialogue` が、すべて話者表記による抽出失敗とは限りません。

## サンプルデータでの試行
`data/sample_outputs.jsonl` は、入力4件 × 2モデルの計8件からなる動作確認用の見本です。`raw_outputs` を含まないので、raw のシートは作れません。

```powershell
# 自動評価だけ（人的評価はスキップ）
python run_eval.py --outputs data/sample_outputs.jsonl --human-dir human_sample_unused --results results_sample_auto --no-bertscore

# 仮点数入りの final シートを作り、集計を試す（human_sample/final/ に作成）
python human_eval_sheet.py make --outputs data/sample_outputs.jsonl --raters member1 member2 member3 --dummy-scores
python run_eval.py --outputs data/sample_outputs.jsonl --human-dir human_sample/final --results results_sample --no-bertscore
```
- `--dummy-scores` は1〜5の仮点数を乱数で埋める動作確認専用の機能です。サンプル以外の入力と、本番用 `human/` への保存には使えません。
- 仮点数から得た平均・p値・一致度は、集計が動くことを確認するためだけのものです。
- 見本での後処理後の形式遵守率は、base が50%（4件中2件）、lora が75%（4件中3件）です。

## BERTScore の3つの値（final が対象）

| 列（summary.csv） | 比べるもの | 使えるデータ | 位置づけ |
|---|---|---|---|
| BERTScore_F1(正解比) | 出力全体 ↔ 正解対話 | データA（`reference` あり） | **主な自動指標** |
| BERTScore_F1(入力比・全文) | 出力全体 ↔ 入力全体 | 全データ | 補助。長い出力は上限を超えやすい |
| BERTScore_F1(入力比・分割単位) | 分割 i の出力 ↔ 分割 i の入力 | 分割情報（`--pipeline`）があるデータ | 補助。長文でも上限に収まりやすい |

- **512トークンの上限**：BERT は特殊トークンを含めて512トークンまでしか読めず、超えた部分は切り捨てられます。比べる2つの文章のどちらかが上限を超えた件は「切り捨てあり」とし、**平均・検定から除きます**。値自体は `bertscore_detail.csv` に残します。表には使った件数（`n_…_使用`）と除いた件数（`n_…_切り捨て除外`）を出し、グラフには「(除外n)」と表示します。全件が除かれた場合は値を出しません。
- **分割単位の値**：1件の値は、分割ごとの F1 の単純平均です。後処理後の出力が空の分割は、削除せず0点として平均に含めます（`chunk_n_empty`）。分割のどれか1つでも上限を超えた件は、集計から除きます。分割ごとの値は `bertscore_chunk_detail.csv` に出ます。分割情報のないデータは「対象外」です（`n_分割単位_分割情報なし`）。
- 出力が空の件は、全文の値も0点として数えます（削除しません）。
- 今回のデータB（ニュース5件）では、lora の出力は全文だと5件すべてが上限を超えるため、入力比・全文の lora の値は出ません。分割単位では lora の5件すべてを計算でき、base は英文の分割が上限を超えた1件だけが除かれます。
- base の英文を日本語用の BERT で比べた値は、意味のある比較になりません。日本語判定と並べて示してください。

## 結果を読むときの注意
- **形式遵守率だけでは日本語かどうかはわかりません。** 英語でも「A：」「B：」形式なら合格するので、日本語合格率と並べて示してください。
- **BERTScore（入力比）は、文体の近さにも反応します。** 硬い入力文に近い出力ほど高く出やすいため、情報保持の根拠には人的評価の正確性・網羅性も合わせて示してください。
- **BERTScore は最大512トークン（特殊トークンを含む）までしか読めません。** 上限を超えた件は集計から自動で除き、件数を表とグラフに示します（「BERTScore の3つの値」を参照）。
- データ数が少ないため、p値は参考値として扱ってください。

## 今後の実装方針（未実装）
- **数値保持率・入力にない数値の出現率**（補助指標）：英語出力などで比べられない件は黙って除外せず、対象外の件数をモデルごとに示します。結果は「日本語出力に限定した補助分析」と明記します。
- **raw を使った BERTScore**：base の英文を日本語用の BERT で比べても意味のある値にならないため、入れていません。
