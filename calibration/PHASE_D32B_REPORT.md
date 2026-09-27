# フェーズD.3.2b 全件実行・hold解消レポート

実行日：2026-09-27〜2026-09-28。
設計：[PHASE_D32B_DESIGN.md](PHASE_D32B_DESIGN.md)（2026-09-27改訂第2版、SHA-256 `bde6c13861eca73f7e29cc16c902216284b3b6254a770aee7764e3069b18dd4d`）。
反証：[第1回](PHASE_D32B_ADVERSARIAL_REVIEW.md)・[第2回](PHASE_D32B_ADVERSARIAL_REVIEW_2.md)（Astra/high）。
実装：Claude（工程1・3はOpus 5.5/high、工程2・4・5はSonnet 5）。

## 1. 結論

D.3.2aで残った4件のhold（`full_multi_riichi_danger_class`、`explicit_yaku_shape_features`、`opponent_component_unidentified`、`response_rate_miscalibration`）と、工程1で新たに見つかった`win_legality_unresolved`を合わせた5件すべてが、全件データで解消した。

```json
"eligibleForD33": true, "eligibleForAdoption": false, "holds": [],
"d33Conditions": ["fixed_component_sensitivity"]
```

fitと再読込evaluateは、holds・eligibleForD33・d33Conditions・metricsのすべてが一致した（D32B-10の「fitと再評価で結果が一致する」を実データで確認）。

ただし、全体の応答率診断は合格したが、リーチ人数で層別すると、リーチ中の窓でポンの予測確率が観測の2〜4倍に過大という偏りが残っている（6.4節）。全体診断は該当窓が2.6%しかないため、この偏りを検出できない。これは新しいholdの追加理由にはしないが、D.4での採否判断に持ち越す重要な限界として記録する。

`eligibleForAdoption`は設計どおり常にfalseである。この結果はD.3.3の実装可否をAstraの反証と合わせてユーザーが判断する材料であり、アプリの採点への接続を意味しない。

## 2. 全件処理

| 工程 | 結果 | 実測 |
|---|---:|---|
| D.3.1再抽出（`dataset-opponent-v3`） | 成功 | 27分、教師窓2,085,155件、`verify-opponent-dataset` pass |
| v3特徴生成（`features-opponent-v3`） | `complete` | 181分46秒（3時間02分）、209 shard、760 MiB |
| v3特徴検証 | `pass` | 2,085,155窓、14,534,711候補 |
| v3 fit（初期定数→θ学習→定数推定→θ再学習→定数再推定） | `complete_with_fixed_components` | 95分39秒 |
| 保存モデルevaluate | `confirmation_only_not_independent_test` | 14分49秒 |

特徴生成はD.3.2a（v2、137.8秒cold・97.5秒warm）の約1.6〜2倍の時間だった（v3のprobe実測は227.1秒cold・201.4秒warm、[工程2の記録](d3-development-plan.json)参照）。全件では3時間02分で、設計予算8時間の約38%だった。危険度と役・形の特徴を追加した分の増加であり、性能の高速化は不要だった。

fit＋evaluate合計は1時間50分28秒で、設計予算8時間の約23%だった。

特徴cacheは760 MiB（v2は約690 MiB）で、8 GiB予算内だった。候補総数はv2の14,533,857からv3で14,534,711へ854件増えた。工程1の和了判定修正により、実際に和了した窓の合法候補へ`ron`・`tsumo`が正しく加わったことによる増分であり、想定どおりである。

## 3. 工程1：和了判定の不一致925件

原因は2種類だった。

- **チー面子の牌順（924件）**：得点計算ライブラリ（mahjong 2.0.0）へ渡す面子牌を、実行器は鳴いた牌を末尾に置いたまま渡していた。ライブラリは面子の先頭牌を順子の開始牌とみなすため、例えば4萬を5・6萬で鳴いた`[5m, 6m, 4m]`の並びを「5-6-7萬」と誤読し、和了形を認めなかった。`tools/ev_policy_scoring.py`の`score_hand`で、ライブラリへ渡す面子牌をtile34昇順にそろえて修正した。
- **打ち切りprefixの末尾ラベル（1件）**：実行prefixが自摸直後で切れた局で、最後の自摸窓に局末のツモ和了を誤ってラベルしていた。`tools/ev_policy_observation.py`の`_self_teacher_window`で、ラベルの参照元を打ち切り前の牌譜へ変更した。

修正後、D.3.1を再抽出し、全教師窓で「観測された和了行動が合法候補にある」ことを照合した。ロン8,827件・ツモ7,446件のすべてが合法候補に含まれ、和了headのholdは0件だった。残存例外（検出条件が必要な未対応ルール）は0件のため、4.2〜4.3節の検出器は実装していない。

詳細は[分類記録](probes/win-legality-d32b/)（`summary.json`、`cases.jsonl`、`verification.json`）を参照。

## 4. 工程2：特徴v3

### 4.1 危険度

家別の安全牌集合`G_r`（本人の河の全牌＋リーチ後に見逃された牌）から、既存の危険度分類を複数リーチ分まとめた。旧`genbutsu`はリーチ前の本人の捨牌を現物に数えていなかったため、この修正でも危険度の精度が上がっている。

### 4.2 役・形の手掛かり

得点器を使わず、固定面子と手中牌の牌数だけから、役牌・食いタン・混一色・対々・七対子の経路と、どの経路にも当たらない副露（`open_no_listed_yaku_cue`）を数える。混一色距離は、固定面子の色が決まっていればその色に合わせる（第2回反証レビューS7の修正）。

## 5. 工程3：未識別成分の固定

### 5.1 定数と推定

| 定数 | 初期値（件数比） | 1巡目θ後 | 最終値 | 95%区間 |
|---|---:|---:|---:|---|
| ε_ron | 0.0019967 | 0.0021513 | 0.0021513 | [0.0012134, 0.0033570] |
| ε_tsumo | 0.0022189 | 0.0022189 | 0.0022189 | [0.0011984, 0.0035478] |
| ρ | 0.0069460 | 0.0070374 | 0.0070374 | [0.0047653, 0.0097468] |
| ε_chankan | （ε_ronに追従） | | 0.0021513 | 未推定（モデル仮定） |

θとの往復は1回で、1巡目と最終値の差は浮動小数点誤差の範囲だった（θの再学習で定数がほぼ動かなかった）。ε_tsumoはBeta厳密事後、ε_ronとρは変数変換`p=sin²(u)`によるu上の中点則格子で推定した。格子は区間数を倍にしても平均・2.5%点・97.5%点の相対差が1%以内になるまで細かくした（`fixedComponents.roundTrips.final.posteriors.*.detail.cells`、`convergence`に記録）。

推定に使った窓は、応答窓11,537件、自己行動（ツモ）窓6,083件（学習・選択・較正の3期間）。開発確認期間は推定に使っていない。

### 5.2 層別結果

30層（4次元×最大3値×3定数）のうち、difference_detectedが2件、unsupportedが1件だった。

| 層 | 定数 | 機会数 | 事後平均 | 95%区間 | 判定 |
|---|---|---:|---:|---|---|
| 巡目：早い（6巡以内） | ρ | 1,819 | 0.00193 | [0.00047, 0.00440] | difference_detected |
| 巡目：遅い（13巡以降） | ρ | 706 | 0.01871 | [0.00987, 0.03027] | difference_detected |
| 自家リーチ中 | ρ | 0 | 0.5（事前分布） | [0.00154, 0.99846] | unsupported |

早い巡目は全体よりも大明槓率が低く、遅い巡目は高い。実戦感覚（早い巡目の大明槓は手が整っていないことが多く、遅い巡目ほど押し切る場面が増える）と整合する。自家リーチ中は、大明槓の機会自体が0件だった（リーチ後は手替わりできないため大明槓を選ばない設計上の制約と整合するが、単なる無観測の可能性も残る）。

### 5.3 感度シナリオ

D.3.3用20シナリオ、D.3.4用21シナリオを生成した（`base` 1、`oat_*` 6、`corner_*` 8、`chankan_assumption` 1、層別展開`K=6`、D.3.4限定の`zero` 1）。展開したのは上記3層×2側（低・高）。シナリオ一覧は[fixed-components.json](model-opponent-v3/fixed-components.json)の`scenarios`に固定した。

## 6. 工程4：診断と採用判定

### 6.1 支持件数

| 成分 | exactラベルの見送り | 公開結果から確定する見送り | 合計 |
|---|---:|---:|---|
| ロン見送り | 17件 | 1件（calibration期間） | 18件 |
| ツモ見送り | 14件 | （自己行動には該当なし） | 14件 |
| 大明槓選択 | 32件（観測） | — | 32件 |

いずれも識別可能とみなす閾値（見送り30件、観測100件）を下回るため、`unidentifiedComponents`は引き続き4件（ron_pass、tsumo_pass、chankan_response_policy、daiminkan_policy）報告される。ただし、この4件すべてに対応する固定成分（定数・推定方法・事後区間・シナリオ一覧）が宣言されているため、`opponent_component_unidentified`のholdは解除される。「未識別」という状態自体は変わっておらず、モデル仮定で扱っていることを明示する設計どおりの結果である。

観測が合法機会を上回らないことを`_support_diagnostics`内で検査し、全期間で違反なしを確認した。

### 6.2 応答率診断（全体）

| 期間 | 状態 | 最大絶対誤差 | 希少率の最大倍率 |
|---|---|---:|---:|
| calibration | pass | 0.003423（pass） | daiminkan 2.0倍（0.000072 vs 0.000036） |
| developmentConfirmation | pass | 0.003704（pass） | daiminkan 3.3倍（0.000011 vs 0.000036） |

v2との比較：

| kind | v2 developmentConfirmation絶対誤差 | v3 developmentConfirmation絶対誤差 |
|---|---:|---:|
| chi | 0.002283 | 0.003237 |
| pon | 0.003610 | 0.000445 |
| ron | 0.002111 | 0.000003 |
| daiminkan | 0.000731（倍率約68倍） | 0.000025（倍率3.3倍） |
| pass | 0.007274 | 0.003704 |

閾値内に収まったため、チー・ポンのバイアス補正（設計8.2節）は実施していない。ronとdaiminkanは固定成分の導入により誤差が大きく縮小した。chiはv2よりわずかに悪化したが、閾値0.005は下回る。

### 6.3 採用判定

`opponent_adoption_holds`の入力は、工程1の分類記録（`win_legality`）、特徴manifest、支持件数、上記2期間の率診断、固定成分ファイルの5点。結果は holds=[]、eligibleForD33=true、eligibleForAdoption=false、d33Conditions=["fixed_component_sensitivity"]。残存例外が0件のため`residual_win_legality_detector`は付かない。

## 7. 危険度特徴の評価：リーチ人数・親リーチ有無で層別した自己行動NLL

[stratified-self-nll.json](model-opponent-v3/stratified-self-nll.json)。全自己行動窓（打牌候補を持つ）を対象とし、スキップ0件。

| 期間 | リーチ人数 | 親リーチを含む | 窓数 | 平均自己行動NLL |
|---|---|---|---:|---:|
| calibration | 0 | — | 109,693 | 1.4700 |
| calibration | 1 | いいえ | 19,906 | 1.1597 |
| calibration | 1 | はい | 7,515 | 1.1616 |
| calibration | 2+ | いいえ | 1,820 | 1.0940 |
| calibration | 2+ | はい | 1,474 | 1.0589 |
| developmentConfirmation | 0 | — | 144,081 | 1.4863 |
| developmentConfirmation | 1 | いいえ | 26,256 | 1.1668 |
| developmentConfirmation | 1 | はい | 9,777 | 1.1343 |
| developmentConfirmation | 2+ | いいえ | 2,388 | 1.1011 |
| developmentConfirmation | 2+ | はい | 1,933 | 1.0771 |

リーチ人数が増えるほど自己行動NLLは下がる（予測しやすくなる）。リーチ圧力下では押し・オリの選択肢が絞られ、方策が読みやすくなるという実戦感覚と整合する。親リーチを含む場合はわずかに低い（より読みやすい）が、リーチ1人の場合はほぼ差がない。

## 8. 応答率の限界：リーチ人数で層別すると偏りが残る

[stratified-response-rate.json](model-opponent-v3/stratified-response-rate.json)。危険度特徴はdiscard系候補にしか付かないため、親リーチの有無ではなくリーチ人数（0/1/2+）だけで層別した。

| 期間 | リーチ人数 | 窓数（割合） | pass絶対誤差 | pon絶対誤差 |
|---|---|---:|---:|---:|
| calibration | 0 | 135,619（97.4%） | 0.002696 | 0.000098 |
| calibration | 1 | 3,151（2.3%） | 0.029117 | 0.020039 |
| calibration | 2+ | 442（0.3%） | 0.043596 | 0.040168 |
| developmentConfirmation | 0 | 178,020（97.4%） | 0.003042 | 0.000221 |
| developmentConfirmation | 1 | 4,144（2.3%） | 0.029117 | 0.024845 |
| developmentConfirmation | 2+ | 627（0.3%） | 0.024158 | 0.028349 |

リーチが1人以上いる窓（全体の2.6%）では、ポンの予測確率が観測の2〜4倍に過大で、その分パスが過小になる。全体診断（6.2節）はこの2.6%が希釈されるため閾値内に収まり、holdを出さない。

この偏りは新しいholdの追加理由にしない。理由は次の3点である。

1. 設計8.3節の応答率診断は、固定成分を含む公開結果の率を対象とし、層別の診断基準はD.4で固定するとされている（本書は診断のみ）。
2. リーチ中の窓はまさにD.3以降が対象とする局面であり、この偏りが押し引きの前向きシミュレーションへ与える影響は、D.3.4の感度シナリオ（7節）と別に、D.3.3実装時に個別評価が必要である。
3. 偏りの方向（ポン過大・パス過小）は、危険度特徴がまだ「ポンして守備を崩すリスク」を明示的に表現していないことを示唆する。これは設計6節の役・形特徴や、7節の固定成分では捕捉していない残差であり、特徴設計の見直し候補としてD.4へ引き継ぐ。

## 9. 受入試験

| ID | 結果 | 根拠 |
|---|---|---|
| D32B-01 | pass | 925件全件分類、未分類0件、修正前後のfixture、全教師窓の和了照合pass（3節） |
| D32B-02〜05 | pass | 固定例テスト11件（`tests/test_ev_policy_features.py`）、全件特徴検証`complete` |
| D32B-06〜08 | pass | 固定例テスト14件（`tests/test_ev_policy_fixed.py`）、格子とBeta厳密解の相対誤差0.5%以内、感度シナリオの反例検出 |
| D32B-09〜10 | pass | 固定例テスト25件（`tests/test_ev_policy_opponent.py`）、実データでfit/evaluate一致 |
| D32B-11 | pass | 本レポートの全件実測（本節上表）。ただし8節の層別偏りは限界として記録 |

Python全テストは156件（工程1〜4での新規追加を含む）、JS全テストは92件で、いずれもexit 0だった。

## 10. 残る作業とD.4への引き継ぎ

- **8節の層別偏り**：リーチ中のポン予測過大／パス予測過小。D.3.3実装時に、リーチ中局面での前向きシミュレーション結果への影響を個別評価する。
- **5.2節の層別difference_detected**：巡目による大明槓率の差。層別の定数はモデルへ採用していないが、D.3.4の感度シナリオ（`stratum_turn-early_rho_*`、`stratum_turn-late_rho_*`）で影響を確認できる。
- **`opponent_component_unidentified`の解除は「識別できた」ことを意味しない**：4成分は引き続き未識別で、固定成分というモデル仮定に置き換えただけである。D.4では、この仮定を採用可否の判断に含める必要がある。
- **D.3.3・D.3.4は未実装**：本書はD.3.2bの範囲（相手モデルの学習側）を完了しただけであり、粒子法（D.3.3）と前向きシミュレーション（D.3.4）は次の設計・実装が必要。

## 11. 実行環境と検証

- Python: `C:\Users\maste\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`（3.12.14）
- NumPy: 2.3.5
- mahjong: 2.0.0（`calibration/requirements-scoring.txt`のhashロック済みwheel）
- `node tests/engine.test.js`、`node tests/app.test.js`：exit 0（アプリ側は無変更、既存92件）
- `python -B -m unittest discover -s tests`：exit 0（156件）

本レポートの数値はすべて上記環境での実測であり、設計書（PHASE_D32B_DESIGN.md）の見込み値をこの実測で置き換える。
