# 麻雀なに切るアプリ

vault「麻雀強者の考え方」の知見（受け入れ・押し引き・危険度分類）にもとづく、ブラウザで動く何切る練習アプリ。
`index.html` をブラウザで開くだけで動く（ビルド・サーバー・AI APIは不要）。問題生成と正解判定はすべてローカルのJavaScriptで行う。
Tailwind CSSのみCDNから読み込むため、完全オフライン時も問題機能は動くが、CSSがキャッシュされていなければ表示が崩れる場合がある。

牌効率モードの出題改善に関する条件と検証結果は [改善計画と引き継ぎ](docs/EFFICIENCY_IMPROVEMENT_PLAN.md) を参照。

## モード

### 牌効率モード（平面何切る）

- ツモ前13枚が**2シャンテン**で、実際のツモ牌を加えた14枚から**1シャンテン**に進める問題を出題。
- **正解 = 1シャンテンに進む打牌のうち、テンパイへの受け入れ枚数が最大の打牌**。受け入れ枚数が同数の場合は**変化（改良ツモ）の枚数が多い方だけが正解**。変化まで同数の手は出題せず、正解を必ず1種類に絞る。
- 1シャンテンを維持する候補に字牌切りが1種類でもある手と、同種4枚を含む手は問題全体を除外する。字牌の対子や刻子があり、それを切るとシャンテンが悪化する手は出題できる。
- 受け入れと変化が競合する手は、全候補について2ツモ以内のテンパイ到達率も調べ、正解候補より高い候補があれば問題自体を除く。
- 2ツモの到達率は、手牌と打牌以外の未知牌から重複なしで2枚引く前提。1枚目で進まなければ最善の打牌を選び直す。相手の行動や3ツモ以降の価値、打点の評価は含まない。
- 回答後、打牌候補ごとの受け入れ枚数・受け入れ牌・変化の一覧を表示する。
- 初心者が間違えやすい形を「罠型」として判定する。検出機能は孤立字牌、浮き牌の質、唯一の対子、二度受けの4種に対応するが、現在の牌効率問題では字牌候補を除外するため孤立字牌型は出題しない。半数の問題で残る罠型を優先的に探し、見つからない場合は有効な通常問題へフォールバックする。実際の罠型割合は出題条件と乱数によって変わる。回答後は該当する考え方を解説カードに表示する。
- 罠型の判定根拠は vault の `wiki/資料要約/牌効率_note_youtube_外部資料_要約.md` と、その原典（新谷note・まるなおN note・となかいnote・Mahjong Watch）。
- 最大受け入れが2種以下の局面を出題する。通常問題（最大の不正解と3枚以上差）と高難度問題（最大の不正解と1〜2枚差）を約50%ずつ混ぜる。変化タイブレークで決まる問題（受け入れ枚数差0）は高難度でのみ出題する。

### 清一色モード（萬子の何切る）

- ツモ前13枚がすべて萬子の**1シャンテン**で、実際のツモ牌を加えた14枚から**テンパイ**に進める問題を出題。
- 四枚使いがある手では、牌を切るほかに**暗槓**も選択できる。
- **正解 = テンパイに進む打牌・暗槓のうち、和了牌の受け入れ枚数が最大の行動**（同数の場合はすべて正解扱い）。暗槓による打点上昇は評価しない。
- 暗槓は、4枚を手牌から除いて槓子1面子を固定した形から受け入れを計算する。槓した4枚は使用済みなので残り枚数には数えず、同じ牌を1枚切る場合とは独立に評価する。
- 暗槓の受け入れは、固定面子と残り10枚から嶺上牌で和了する枚数を表す。嶺上牌を引いた後の打牌や新しいドラによる価値は評価しない。
- 清一色モードの制約として、受け入れは萬子のみを数える。
- 回答後、打牌・暗槓候補ごとの受け入れ枚数と受け入れ牌の一覧を表示する。
- 清一色は受け入れ同率の行動が出やすいため、出題の明確さ条件は牌効率モードより緩め（同率3種以下・2位と2枚以上の差）。

### 押し引きモード（立体何切る）

- 他家リーチの捨て牌・巡目・ドラ・親子を表示。**全問、ツモ前13枚が1シャンテンの危険牌勝負問題**。
- ツモ後の全打牌から、テンパイまたは1シャンテンを保つ候補を列挙。候補ごとに速度（打牌後シャンテン・受け入れ）、打点、打牌危険度をEVへ換算し、最大の一打で「押す」か「オリる（現物切り）」かを選ぶ。
- 1シャンテン→1シャンテンと、1シャンテン→テンパイの問題が混在する。2シャンテンへ戻る打牌は押し候補にしない。
- **正解 = 現在の局収支概算モデル内でEVが最大の方**。押し候補の1位と2位が75点未満、または押しとオリが300点未満の微妙な局面は出題しない。
- 勝負牌と同じか、よりよいシャンテン数を放銃率3%未満で保てる別候補がある手は出題しない。テンパイを崩す現物は撤退の選択肢として扱う。
- 回答後、両選択のEVと内訳（選択牌固有の放銃率・和了率・打牌後に残る打点）を表示する。

## 押し引きEVモデルの前提（概算）

- 危険牌の放銃率: 危険度カテゴリ別の統計的近似値（現物0% / スジ2.2〜3.8% / 無スジ3.4〜5.7% / 字牌は見え枚数依存 など）。
  カテゴリ体系は vault の `data/mleague/danger/danger_summary.json`（現物・スジ・ワンチャンス・ノーチャンス・字牌見え枚数）に対応。
- リーチの平均打点: 子5,300点 / 親7,700点。被ツモの支払い概算は、相手が親なら2,300点、相手が子なら自分が子で1,400点、自分が親で2,800点。
- 自分の和了率: 打牌後テンパイと1シャンテンを分け、巡目と受け入れ枚数から概算（現在牌が通る条件でテンパイ上限58% / 1シャンテン上限34%）。受け入れ0枚では0%とする。手変わりによる復活はモデル外。
- 打点: 打牌後に残る通常ドラと赤5（各色1枚）を候補ごとに数える。同じ5が複数ある場合は通常5を切り、赤5を残す。
- 現在牌が通る確率、押し続けた場合の追加放銃リスク、どちらも和了しない間のツモられ失点を、重複しない確率分岐で織り込む。
- 回答後は和了収入、放銃失点、被ツモ失点、撤退コスト300点の内訳を表示。EV差は画面に表示する丸め済みの押しEVとオリEVから計算する。内訳は各項を丸めるため、合計とEVに1点程度の差が出る場合がある。
- 完全な役・符・待ち別打点や全員の河・副露を使う麻雀AIではないため、「実戦の真値」ではなく、表示した前提を使う概算モデル内の最大値として扱う。
- 参考値として、Mリーグ牌譜集計の対リーチ押し率（テンパイ88% / イーシャンテン78%、vault研究ノート「Mリーグ対リーチ押し率」）を解説に併記。

## 実牌譜によるEV較正

較正は段階的に実装する。フェーズAの監査、フェーズBの学習用レコード抽出、フェーズCの観測値較正、フェーズC.1の全適格打牌による軽量較正まで実装済み。C.1も採用基準を満たさなかったため、アプリへの反映は行わない。

フェーズDはD.1の状態管理・選択前抽出・入力検査、D.2の一局シミュレーション基盤、D.3.0の独立牌譜イベント監査、D.3.1の相手モデル用観測データ、D.3.2の相手行動モデル学習まで実装した。D.3.2aでは候補別の厳密受け入れ、特徴cache、全2,085,155教師窓の生成・検証、v2再学習と保存モデル再評価まで完了した。未実装特徴、未識別成分、応答率の較正不一致が残るため、D.3.3への接続はholdしている。
D.2の [ルール解釈と依存選定](calibration/PHASE_D2_RULES_AND_DEPENDENCIES.md)、[D.2aの得点と精算](calibration/PHASE_D2A_REPORT.md)、[D.2bの一局状態機械](calibration/PHASE_D2B_REPORT.md)、[D.2cの方策と合成局](calibration/PHASE_D2C_REPORT.md) はデバッグ基盤として実装済み。
[D.3の設計](calibration/PHASE_D3_DESIGN.md) に基づく[D.3.0](calibration/PHASE_D30_REPORT.md)では、35,218判断の独立入口復元と7,849局のresult差分を全件照合し、局実行器は決定的標本256局のうち254局が一致した。残る2局は原牌譜の牌在庫不整合としてholdにした。[D.3.1](calibration/PHASE_D31_REPORT.md)では23,358局から2,085,155教師窓と32,019推論prefixを生成し、公開・私有情報の遮断と共同応答ラベルを全件検証した。[D.3.2](calibration/PHASE_D32_REPORT.md)では階層付き相手行動モデルを全件学習・較正した。[D.3.2a](calibration/PHASE_D32A_REPORT.md)では厳密な候補別受け入れを全件計算してv2を再学習したが、4件のholdが残り、`eligibleForD33=false`である。
2026-27は [将来評価枠](calibration/future-evaluation-reservation.json) として予約している。牌譜は未取得、モデル凍結は未了であり、それまでの開発には2025-26以前を使う。

```powershell
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
python tests/test_calibrate_ev.py
```

監査結果は `calibration/data-audit.json` に出力される。収録期間と件数、入力ファイルのSHA-256、手牌復元例外ID、得点台帳の連続性、v1条件の暫定適格件数、次フェーズで復元する項目を含む。同じ入力からは同じJSONを生成する。

得点台帳は全局で候補式を比較した結果、`局末点 = 局開始点 + result精算差分 - 和了者自身の当該局リーチ供託` とする。台帳異常の両側にある局は、順序を推測して修復せず隔離対象として記録する。適格件数は、自家リーチ前状態と完全な公開局面を既存派生表だけでは確定できないため、フェーズBまでは暫定値として扱う。

設計全体と段階ごとの合否条件は [`docs/EV_CALIBRATION_DESIGN.md`](docs/EV_CALIBRATION_DESIGN.md) を参照。フェーズAの監査結果は既存の押し引き問題や表示EVを変更しない。

抽出の既定単位は各局各家の最初の適格判断。`calibration/dataset/` に判断直前の公開状態、全打牌候補、実選択の局末結果、理由付き除外、品質集計を分離して生成する。詳細なスキーマと全判断抽出の方法は `calibration/README.md` を参照。

## ファイル構成

- `index.html` — 画面。牌は自前SVGで描画（画像素材は不要。画面レイアウト用のTailwind CSSのみCDN読み込み）
  - 絵柄: 筒子=円柄、索子=竹柄（1索は鳥、8索は両端の垂直竹＋中央2本を傾けた八の字組み。上段Λ形・下段V形のW/M配置）、萬子=漢数字＋萬、字牌=漢字（白は枠のみ）
  - 赤5（5萬・5筒・5索）は全体を赤く塗った赤ドラ柄で描画。押し引きモードの打点計算にも通常ドラと同様に加算される
  - `index.html?gallery=1` で全34種＋赤5の牌デザインを一覧確認できる
- `engine.js` — 牌表現・シャンテン計算（一般手/七対子/国士、暗槓の固定面子）・受け入れ計算・問題生成・押し引きEVモデル
- `app.js` — UI描画・回答判定・成績表示（成績はlocalStorageに保存）
- `tests/engine.test.js` — エンジンの検証テスト。`node tests/engine.test.js` で実行
- `tests/app.test.js` — 清一色の打牌・暗槓UIテスト。`node tests/app.test.js` で実行
- `tools/calibrate_ev.py` — 実牌譜EV較正パイプライン。監査、抽出、学習、評価、C.1軽量較正を実装
- `tests/test_calibrate_ev.py`, `tests/test_ev_calibration_model.py` — 監査、抽出、得点台帳、モデル学習・評価、再現性の固定テスト
- `calibration/data-audit.json` — フェーズAの実データ監査結果
- `calibration/README.md` — フェーズBの生成方法、レコード境界、未来情報の禁止事項
- `calibration/dataset/extraction-summary.json` — フェーズBの件数、品質検査、生成ファイルハッシュ
- `calibration/PHASE_C_REPORT.md` — フェーズCの最終評価、採用保留の根拠、利用制限
- `calibration/model/` — 係数、分割台帳、最終評価、信頼度曲線、層別誤差
- `calibration/PHASE_C1_REPORT.md` — 全適格打牌による軽量較正、抽出条件修正、確認評価
- `calibration/dataset-observed/`, `calibration/model-c1/` — C.1の再生成データsummaryとモデル成果物
- `calibration/dataset-policy/` — D.1の選択前局面、全行動、抽出summary、全件検証結果
- `tools/ev_policy_state.py`, `tests/test_ev_policy_state.py` — Dの状態境界、入力ガード、D.1固定テスト
- `tools/ev_policy_scoring.py`, `tests/test_ev_policy_scoring.py` — Mリーグ用得点アダプター、通常精算、責任払い、D.2a固定テスト
- `tools/ev_policy_round.py`, `tests/test_ev_policy_round.py` — 一局状態機械、応答、フリテン、槓、流局、D.2b固定テスト
- `tools/ev_policy_simulation.py`, `tests/test_ev_policy_simulation.py` — 決定的push/fold、合成局、情報遮断、牌譜prefix照合、D.2c固定テスト
- `tools/ev_policy_replay.py`, `tests/test_ev_policy_replay.py` — 独立牌譜イベント、局実行器adapter、D.3.0入口監査
- `tools/ev_policy_observation.py`, `tests/test_ev_policy_observation.py` — D.3.1の公開・私有イベント、意味上の合法行動、共同応答ラベル、推論prefix
- `tools/ev_policy_opponent.py`, `tests/test_ev_policy_opponent.py` — D.3.2の階層付きsoftmax、共同応答尤度、期間分割学習、有限差分検査
- `tools/ev_policy_features.py`, `tests/test_ev_policy_features.py` — D.3.2aの厳密受け入れ、行動投影、有界shape cache、永続特徴shard、再開検証
- `calibration/PHASE_D2C_REPORT.md` — D.2cの全件牌譜照合、合成局処理量、利用制限
- `calibration/PHASE_D30_REPORT.md`, `calibration/policy-runtime-audit.json` — D.3.0の全件入口、result差分、局実行器標本の照合結果
- `calibration/PHASE_D31_REPORT.md`, `calibration/dataset-opponent/` — D.3.1の全件抽出・検証結果と再生成データmanifest
- `calibration/PHASE_D32_REPORT.md`, `calibration/model-opponent/` — D.3.2の全件学習・較正、期間別評価、未識別成分と特徴hold
- `calibration/PHASE_D32A_DESIGN.md`, `calibration/PHASE_D32A_REPORT.md`, `calibration/probes/opponent-features-v2/` — D.3.2aの計算仕様、全件再学習結果、固定性能標本、cold／warm／永続読込測定
- `docs/` — 改善計画、EV較正の設計、リファクタリング記録
- `scratch/` — 調査や性能測定で作る一時作業物。用途は `scratch/README.md` を参照

計算では今切った牌も場に見えている枚数として扱う。変化後に切る牌も山へ戻さない。
生成上限に達した場合は条件を緩めず再試行を案内する。問題取得と先読みは全モードで共用し、モード変更時は古い先読みを取り消す。
テストの乱数は固定。別の乱数列は PowerShell で `$env:TEST_SEED='20260906'; node tests/engine.test.js` のように指定できる。

## 制限事項

- チー・ポン・大明槓・加槓などの副露手と、点棒状況（着順条件）は未対応。清一色モードの暗槓だけを選択肢として扱う。
- 赤5は「各スート4枚中1枚」という前提で、手牌中の枚数に応じた確率（k/4）で割り当てる近似実装。山や河に残っている赤5の追跡はしない。
- 牌効率・清一色の正解は受け入れ枚数基準であり、打点までは評価しない。牌効率モードは受け入れ同数時のみ変化で優劣を付け、それでも単一正解にならない手は出題しない。清一色は変化を評価しない（同枚数なら複数正解）。
- EVモデルの定数は概算値。厳密なシミュレーション値ではない。
