# EV較正データセット

`tools/calibrate_ev.py` が、読み取り専用のMリーグ原牌譜から生成する。
現在はフェーズC.1まで実装済み。観測結果モデルは確認評価で採用基準を満たさなかったため、アプリへ接続しない。
フェーズD.1、D.2、D.3.0、D.3.1とD.3.2の相手行動モデル学習まで実装済み。詳細は [PHASE_D_DESIGN.md](PHASE_D_DESIGN.md)、[D.1レポート](PHASE_D1_REPORT.md)、[D.2cレポート](PHASE_D2C_REPORT.md)、[D.3.0レポート](PHASE_D30_REPORT.md)、[D.3.1レポート](PHASE_D31_REPORT.md)、[D.3.2レポート](PHASE_D32_REPORT.md) を参照する。
[D.3設計](PHASE_D3_DESIGN.md) と [開発計画](d3-development-plan.json) に従い、[D.3.2a](PHASE_D32A_DESIGN.md) の実装を開始した。厳密な候補別受け入れ、特徴cache、性能probe、v2再学習経路を実装済みであり、次は全件cache生成、検証、再学習を行う。未識別成分などの採用holdは別に扱い、相手モデルはまだD.3.3へ接続しない。
D.3.2aの固定20,000窓probeはcold 137.84秒、warm 97.53秒、永続読込3.06秒、peak working set 1.42 GiBで設計予算内だった。全件生成時間の外挿は約4時間だが、この外挿は合格判定に使わず、全2,085,155窓の実測で8時間上限を判定する。
D.2の [ルール解釈と依存選定](PHASE_D2_RULES_AND_DEPENDENCIES.md)、[D.2aの得点と精算](PHASE_D2A_REPORT.md)、[D.2bの一局状態機械](PHASE_D2B_REPORT.md)、[D.2cの決定的方策、合成局、牌譜prefix再検証](PHASE_D2C_REPORT.md) は完了した。

2026-27の [将来評価台帳](future-evaluation-reservation.json) は予約のみで、データ取得やモデル凍結を示さない。
対象が未開催の間も、旧期間でDの実装と検証を進める。
Dの開発入力は予約台帳の許可リストを必須とする。既存のC用CLIはこの台帳を使わない。

## 生成

```powershell
python -m pip install --require-hashes --only-binary=:all: -r calibration/requirements-scoring.txt
python tools/calibrate_ev.py audit
python tools/calibrate_ev.py extract
python tools/calibrate_ev.py fit
python tools/calibrate_ev.py evaluate
python tools/calibrate_ev.py extract-observed
python tools/calibrate_ev.py fit-ron
python tools/calibrate_ev.py evaluate-ron
python tools/calibrate_ev.py validate-policy-input
python tools/calibrate_ev.py extract-policy
python tools/calibrate_ev.py verify-policy-dataset
python -m tools.calibrate_ev verify-policy-replay
python -m tools.calibrate_ev simulate-policy-debug --decision-id '<decisionId>' --trials 4
python -m tools.calibrate_ev audit-policy-runtime --runtime-limit 256
python tools/calibrate_ev.py extract-opponent-events
python tools/calibrate_ev.py verify-opponent-dataset
python tools/calibrate_ev.py probe-opponent-features
python tools/calibrate_ev.py build-opponent-features
python tools/calibrate_ev.py verify-opponent-features
python tools/calibrate_ev.py fit-opponent --feature-dir calibration/features-opponent-v2
python tools/calibrate_ev.py evaluate-opponent --feature-dir calibration/features-opponent-v2
```

得点依存をネット接続なしで導入するときは、検証済みwheelを `calibration/.dependency-cache/` に置き、`--no-index --find-links calibration/.dependency-cache` を追加する。

`extract` の既定単位は、各局各家の最初の適格判断である。
`extract-observed` は全適格打牌について実選択だけを軽量な1行へ投影する。

`extract-policy`は実打牌の結果ではなく、選択前の手牌に0/1シャンテン候補がある判断を抽出する。
出力は`dataset-policy/policy-decisions.jsonl`と`policy-actions.jsonl`に分け、実打牌、選手名、局末結果を含めない。
JSONLは再生成物としてGit管理外にし、`extraction-summary.json`と`verification.json`を管理する。

`verify-policy-replay`はD.1と同じ抽出器で記録済みprefixを再構築し、`dataset-policy/replay-verification.json`へ全件照合結果を保存する。
`audit-policy-runtime`は別のイベントアダプターでD.1入口を全件復元し、指定数の参照局を一局実行器へ通して`policy-runtime-audit.json`へ保存する。
原本にない牌位置は検証用に補完するが、未記録の裏ドラや未来を正解として採用しない。
`simulate-policy-debug`は一様未知牌と単純相手で全候補を一局実行する。出力は常に`synthetic_debug`であり、予測値やアプリ採点には使わない。

`extract-opponent-events`はD.3.1の相手モデル用データを`dataset-opponent/`へ生成する。
全局の公開イベント、各家の配牌と自摸だけを持つ私有イベント、自己行動・捨牌応答・搶槓応答の教師窓、D.1判断の推論prefix、局末結果を別ファイルにする。
無鳴きは全応答家の正確なpassとし、成立したポンなどで下位希望が隠れる家は具体行動を補わず`censored`にする。
`verify-opponent-dataset`は生成ファイルhash、参照、件数、禁止期間、観測質量、censoredラベル、推論prefixの3種のhashを保存済みファイルから再検証する。
検証が`pass`になるまで`eligibleForModelFit`はfalseである。

`probe-opponent-features`は固定seedのtrain標本でcold、warm、永続cache読込を測る。
`build-opponent-features`は候補別の厳密受け入れを`features-opponent-v2/`へ最大10,000窓のgzip JSONL shardとして保存し、`--resume`ではhashが一致する完了shardだけを再利用する。
`verify-opponent-features`は入力、schema、計算版、shard hash、窓順、候補数を検査する。
`fit-opponent`は検証済みv2特徴cacheから階層付き相手行動モデルを学習し、2023-24で正則化、2024-25でphase別温度を選ぶ。
`evaluate-opponent`は保存モデル、D.3.1 manifest、v2特徴cacheのhashを照合し、全期間を再評価する。2025-26は開発確認であり独立テストではない。
出力先は`model-opponent-v2/`である。厳密受け入れ以外の未実装特徴、未識別成分、応答率の較正不一致は独立したholdとして残し、`eligibleForD33`をfitとevaluateで同じ関数から判定する。

```powershell
python tools/calibrate_ev.py extract-observed --output-dir calibration/dataset-observed
```

`data-audit.json` と現在の入力ファイルの集約SHA-256が違う場合、抽出は失敗する。
大きいJSONLは再生成物としてGit管理対象外にし、`extraction-summary.json` は管理する。

## レコード

### decisions.jsonl

主キーは `decisionId`。
判断直前にプレイヤーが知り得る情報だけを持つ。

- 自分のツモ前13枚、ツモ牌、行動前14枚。赤5は物理牌コードで区別する。
- 四者の河、リーチ宣言位置、初期ドラ表示牌、点数、本場、供託、親、自風。
- 判断時点までのツモ数と残り山枚数、ツモ前13枚・打牌前14枚・打牌後13枚のシャンテン。
- 期間分割と、各局各家の最初の適格判断かどうか。

裏ドラ、未来のツモ、局末結果、最終待ちは含めない。

### candidates.jsonl

主キーは `decisionId + actionId`。
同じ牌種でも赤5と通常5を分け、テンパイする打牌はリーチ宣言あり／なしを別行動にする。
各候補に打牌後シャンテン、公開牌を引いた受け入れ、フリテン、対リーチ危険度を持つ。
実際に選ばれた候補だけ `isActual=true` かつ `observedOutcomeId` が入る。

### outcomes.jsonl

`decisionId` と1対1。
実際に選ばれた行動の直後ロン、局末分類、局末点、観測収支を持つ。
未選択候補の結果は作らない。

観測収支は次で計算する。

```text
R_t = 復元した局末点 - 判断直前点
```

### rejections.jsonl

適用範囲外の判断と、既知または追加検出した異常局を理由付きで記録する。
異常局を並べ替えや欠損値0で修復しない。

### extraction-summary.json

シーズン別件数、除外理由、品質検査、生成ファイルのSHA-256を持つ。
採用判断は既存シャンテン派生行へ照合し、前後シャンテンと実打牌の一致率100%を必須にする。

## フェーズCの出力

`fit` は2018-19〜2022-23を学習、2023-24を方式選択、2024-25を確率較正に使う。
この処理では2025-26の評価値をモデル選択へ使わない。現在の2025-26はフェーズCで既に開封済みなので、
`evaluate` の既定は `reused-confirmatory` とし、新しい採用判定には使わない。
対局単位ブートストラップで現行危険度表および縮約層別平均と比較する。

- `model/model.json`: 特徴量変換、係数、正則化強度、較正パラメータ
- `model/fit-summary.json`: 分割台帳と方式選択結果。最終評価値は含まない
- `model/evaluation.json`: 開封済み確認データの指標、95%区間、採否。新しい封印テストとしては扱わない
- `model/reliability.csv`, `model/reliability.svg`: 直後放銃の信頼度曲線
- `model/segment-errors.csv`: 主要層の観測収支誤差
- `PHASE_C_REPORT.md`: 結論、解釈、利用制限
- `dataset-observed/`: C.1の全適格打牌・実選択のみの軽量JSONL
- `model-c1/`: C.1直後放銃モデルと確認評価。局収支は対象外
- `PHASE_C1_REPORT.md`: C.1の抽出修正、比較結果、利用制限

`evaluation.json` の `status=hold` は実装失敗ではなく、事前に固定した統計基準を満たさないという判定である。
