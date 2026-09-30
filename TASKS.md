# 作業リスト：D.3.3 工程4（MCMCの移動M1〜M4と受理計算）※完了

設計：[calibration/PHASE_D33_DESIGN.md](calibration/PHASE_D33_DESIGN.md) 4.2節、7節、14節のD33-01・D33-02、D33-03の積み残し。
実行：Claude（Opus 5.5）。再開時はこのファイルを先に読む。
完了条件：D33-01（a〜e）とD33-02（a〜d）のテスト、D33-03の系統をまたぐ遷移のテストが合格し、既存テストも合格する。

- [x] 汎用の核：位置の構造、牌種水準の重み`w(x)`、選択器（乱数／全分岐の列挙）
- [x] テンパイ生成器の密度（全除去牌の経路の和、赤の超幾何）。麻雀用と小例用の規則
- [x] M1、M2（`|V|`補正つき）、M3、M4、1反復の合成。M3・M4は「提案→受理比→復元」の3段に分けた（小例の行列を現実的な時間で組むため。実装も同じ合成）
- [x] 実局面への接続（初期化の割当⇔鎖の状態、評価器による因子、牌種水準の硬い制約が評価器と一致）
- [x] D33-01：4つの小例で核ごとの詳細釣合い・1反復の不変性（1e-12）、1反復の行列が全要素正で定常分布が正確な事後分布、プールの並びを固定した復元の検出、M3・M4を外した分断、系統の確率0の拒否
- [x] D33-02：事前分布（超幾何）・制約付き（p0×H）の検査、8種の変異の検出、168の固定例、独立な経路列挙、生成器10^6回のカイ二乗
- [x] D33-03の積み残し：小例で両方向、実局面で国士→面子手をM3・M4が受理
- [x] 全Python 201件、JS 2本合格。開発計画の更新
- [x] コミット（pushは未実施）
- メモ：`|V|`補正を省く変異は、他家に打牌制約がない小例では原理的に検出できない（交換の前後で`|V|`が等しい）。専用の小例`v_asymmetry_example`で検出
- メモ：面子手・七対子の個々の手の生成確率は1e-8程度で、10^6回の頻度検定には向かない。経路の和は独立な列挙で照合し、頻度検定は国士157形と在庫のセルで行った
- メモ：実局面1判断（baseシナリオ）で1反復2.7〜5.4秒、評価器呼び出し49〜59回。予算約1.1ミリ秒/反復とは3桁以上の差。工程5のキャッシュと工程7のベンチマークで判定する
- 次：工程5（キャッシュ、シナリオの並列実行、診断、hold。D33-04、D33-05、D33-07）

# 完了済み：D.3.3 工程3（構成的初期化）

設計：[calibration/PHASE_D33_DESIGN.md](calibration/PHASE_D33_DESIGN.md) 6節、7.3節の生成器`g_r`、D33-03。
完了条件：D33-03のテストが合格し、pilot 16判断で初期化の成否・試行回数・時間・系統を記録する。

- [x] テンパイ形の生成器`g_r`（系統の確率0.90/0.05/0.05、面子手・七対子・国士、赤は同じ牌種の物理牌から一様）
- [x] 構成的初期化（リーチ者→他2家、逆算、整合と尤度の検査、試行と時間の上限、`init_failed`）
- [x] D33-03のテスト5件＋生成器2件。全Python 179件、JS 2本合格
- [x] pilot 16判断×4鎖の初期化：64件すべて成功（面子手53、七対子6、国士5、最大307試行・1.6秒）。`calibration/probes/d33-pilot/initialization.json`
- [x] 開発計画・TASKSの更新
- [x] コミット、push
- メモ：D33-03のうち「国士と面子手の両方に支持がある例で、M3・M4が系統をまたぐ遷移を受理する」は、M3・M4を作る工程4で追加する

# 完了済み：D.3.3 工程2（基準SMCの縮小実装と測定）

設計：[calibration/PHASE_D33_DESIGN.md](calibration/PHASE_D33_DESIGN.md) 3.2〜3.3節、12節、D33-09。上位設計7.1〜7.3節。
完了条件：D33-09が合格し、pilot 16判断の測定から3.3節の関門の判定が記録される（予算超過なら測定前に`resource_budget_exceeded`として報告）。

- [x] SMC本体（一段先の提案、G_eによる重み、ESS<N/2でsystematic resampling、log正規化定数、診断）
- [x] 有限小例と全列挙の正解、D33-09のテスト（3件）と関門の判定関数のテスト（3件）
- [x] 麻雀への適用（遅延割当、対象家の既知自摸の残数比、他家の自摸＋打牌の列挙、応答窓のpass確率）
- [x] 適用部分の検査（真の経路の行動確率の和が評価器と一致、公開特徴状態の一致）。全Python 172件、JS 2本合格
- [x] pilot 16判断のmanifest固定（`calibration/probes/d33-pilot/manifest.json`。9層から1件＋一様7件、同じ局なし）
- [x] 1判断の速度測定：N=256で26秒（15番目のイベントで全滅）、N=4,096で481秒（リーチ宣言でG>0の粒子0）。全192回の見積もり約13時間（3並列で約4.5時間）で予算内
- [x] 測定：192回すべて`posterior_zero_mass`（5時間30分、3並列）。N=4,096の64回のうち48回はリーチ宣言より前、16回は宣言時。1回平均：N=256で27秒、1,024で141秒、4,096で774秒
- [x] 関門の判定：`proceed_to_mcmc_implementation`（N=4,096で64/64が全粒子消失）。開発計画・TASKSの更新、コミット
- 次：工程3（構成的初期化、D33-03）

# 完了済み：D.3.3 工程1（家ごとの履歴評価器）

設計：[calibration/PHASE_D33_DESIGN.md](calibration/PHASE_D33_DESIGN.md)（5節、13節の順序1、14節のD33-06・D33-07）。
実行：Claude（Opus 5.5、2026-09-29にユーザーが選択）。再開時はこのファイルを先に読む。
完了条件：`tests/test_ev_policy_belief.py`のD33-06・D33-07が合格し、既存テストも合格。1,000割当の照合probeが全件一致。

- [x] 判断文脈（公開履歴＋対象家の私有履歴だけ）と家ごとの仮説（配牌・自摸の物理ID）の型
- [x] 規則側：自己行動窓・応答窓の合法集合、フリテン3種、リーチ・一発、和了可否（`RoundState`と同じ規則）
- [x] 特徴側：その家だけを追う特徴状態で、観測行動の確率（固定定数・シナリオ対応）
- [x] 硬い制約の違反（打牌整合、リーチ時テンパイ）と得点器のhold（`rule_unresolved`の元）の報告
- [x] 参照経路：同じ割当で`RoundState`と全家の`RoundFeatureState`を進めた結果
- [x] D33-06（情報境界）3件、D33-07（無作為割当・違反・教師窓・シナリオ・特徴状態）5件のテスト。全Python 164件、JS 2本合格
- [x] 1,000割当の照合probe：pass（不一致0、違反100/100一致、教師窓300家一致）。評価器66秒、参照293秒
- [x] 開発計画の更新
- [x] コミット、push
- メモ：D33-07のうちキャッシュの有無の一致とθ変種の交互評価は、キャッシュを作る工程5で追加する
- メモ：評価器1家1回は約22ミリ秒（モデル確率込み）。12.3節の予算（約1.1ミリ秒/反復）とは桁が違うため、工程5のキャッシュと工程7のベンチマークで判定する
- メモ：mahjong 2.0.0は同梱Pythonに未導入。`PYTHONPATH=calibration/.dependency-cache/mahjong-2.0.0-py3-none-any.whl`で実行（テストは自動でwheelを読む）

---

# 完了済み：D.3.2b（D.3.3接続前hold解消）

設計：[calibration/PHASE_D32B_DESIGN.md](calibration/PHASE_D32B_DESIGN.md)（SHA-256 `bde6c138…18dd4d`）。
実行：Claude。工程ごとの推奨モデルは開発計画 `models.d32bImplementation` を参照。
再開時はこのファイルを先に読み、完了・未完了を更新する。

## 工程1：和了判定の不一致925件（Opus 5.5/high）

- [x] 925件の抽出と記録項目の収集スクリプト（`scratch/d32b/investigate_win_legality.py`）
- [x] 原因分類：チー面子の牌順（アダプター）924件、打ち切りprefix末尾のラベル1件。残存例外0件
- [x] 修正とfixture（`ev_policy_scoring.score_hand`で面子牌を昇順化、`_self_teacher_window`のラベル元を元の牌譜へ）。新テスト2件は修正前に失敗・修正後に合格。既存Python 105件、JS 92件も合格
- [x] 修正後の得点が牌譜記録と一致（実和了843件：符翻一致638、満貫以上の区分一致205）
- [ ] ~~残存例外の`residual_detector`~~ → 残存例外0件のため不要（4.4節の条件4は該当なし）
- [x] D.3.1再抽出（`calibration/dataset-opponent-v3/`、27分、2,085,155窓）と`verify-opponent-dataset` pass
- [x] 分類記録（`calibration/probes/win-legality-d32b/`：cases.jsonl、summary.json、verification.json）
- [x] 全教師窓の和了照合：ロン8,827件・ツモ7,446件すべて合法候補に含まれる。和了headのholdは0件
- [x] D32B-01合格（残存例外0件のため検出条件と3経路の項目は該当なし）
- 未対応メモ：和了と無関係な自己行動hold 4件（打牌1、リーチ打牌3）は範囲外として残る

## 工程2：特徴v3（Sonnet 5）

- [x] 危険度（`G_r`、`classify_danger_v3`、集約、親リーチの率）。旧`genbutsu`のリーチ前河の欠落も解消
- [x] 役・形の手掛かり（`yaku_shape_features`、`melds_after_action`）と種別集約2件
- [x] スキーマv3化、CLI既定をv3のdataset/features/modelへ変更
- [x] D32B-02〜05の固定例テスト11件合格。全Python 116件、JS 92件合格
- [x] 性能probe（20,000窓）：全体8分25秒、cold 227.1秒、warm 201.4秒、peak 1.80 GiB。予算（1時間、4 GiB）内
- メモ：warmがv2の約2倍（97.5→201.4秒）。全件生成の単純外挿は約6.6時間（v2は外挿4.0時間に対し実測1.9時間）。8時間予算の余裕が小さいため、工程5で超過したら特徴計算の高速化を検討

## 工程3：固定成分（Opus 5.5/high）

- [x] 合成規則（`HierarchicalSoftmax.fixed`、`compose_probabilities`）。固定種別の勾配0、学習・評価・率診断が自動で合成方策を使う
- [x] 尤度、ツモのBeta厳密事後、ε_ron・ρの2次元u中点格子（`tools/ev_policy_fixed.py`）
- [x] 層別推定（difference_detected/unsupported/consistent）とシナリオ一覧（16+K / 17+K、ε_chankan追従、zero別種別）
- [x] fitを「初期定数→θ学習→定数推定→θ再学習→定数再推定」へ組替え。`fixed-components.json`を出力
- [x] D32B-06〜08のテスト14件合格。全Python 130件合格
- メモ：全件での格子推定の計算時間は未測定（工程5で記録）

## 工程4：診断と採用判定（Sonnet 5）

- [x] 支持件数：学習マスクで観測を数え直し、ロン見送りの公開結果確定分（confirmedRonSkips）を別欄に追加。観測が合法機会を超えないことを関数内で検査
- [x] 公開結果の率診断をperiod引数化し、較正期間と開発確認期間の両方で実行。developmentConfirmationReuseの区分を明記
- [x] 採用判定関数を設計9節どおり全面書換え（win_legality_satisfied、fixed_components_declared、opponent_adoption_holds）。eligibleForD33/eligibleForAdoption/d33Conditionsを返す
- [x] fit/evaluateへwin-legality-dir引数を追加。CLI既定はcalibration/probes/win-legality-d32b
- [x] D32B-09〜10のテスト25件合格。全Python 156件、JS 92件合格
- [x] 実データ（300窓デバッグ）でfit→evaluateのholds/eligibleForD33/d33Conditionsが完全一致することを確認。win_legality/danger/yaku/unidentifiedの4holdは実データで解消済み、残るのはfeature cache未完成とresponse_rate_miscalibration（300窓では想定どおり）

## 工程5：全件実行とレポート（Sonnet 5、判定はOpus 5.5）

- [x] 全件特徴生成（3時間02分、760MiB、予算8時間・8GiB内）、検証pass
- [x] 全件fit（初期定数→θ→定数→θ再学習→定数再推定、95分39秒）＋evaluate（14分49秒）、予算8時間内
- [x] 応答率は閾値内に収まったため補正なし。v2比較：daiminkan誤差0.000731→0.000025、ron誤差0.002111→0.000003
- [x] リーチ人数・親リーチ有無で層別した自己行動NLL、リーチ人数で層別した応答率を追加集計（設計5.3節・8.3節）
- [x] D32B-11合格。holds=[]、eligibleForD33=true、eligibleForAdoption=false、d33Conditions=["fixed_component_sensitivity"]。fit/evaluate完全一致
- [x] `PHASE_D32B_REPORT.md`作成。開発計画を更新
- 発見：リーチ中の窓（全体の2.6%）でポン予測が観測の2〜4倍に過大。全体診断では希釈されて検出されない。新holdにはせずD.4へ引き継ぎ（レポート8節）
- 全Python 156件、JS 92件合格

## メモ

- 2026-09-27：開発計画にD.3.2bを登録し、工程1に着手。
- 2026-09-27：工程1完了。次は工程2（Sonnet 5推奨）。以後の特徴・学習は`dataset-opponent-v3`を入力にする。
- 2026-09-27：工程2完了。次は工程3（Opus 5.5/high）。
- 2026-09-27：工程3完了。次は工程4（Sonnet 5推奨）。
- 2026-09-27：工程4完了。次は工程5（全件実行、Sonnet 5＋レポート判定はOpus 5.5）。
- 2026-09-28：工程5完了。D.3.2b全体が完了（hold 0件、eligibleForD33=true）。レポート: calibration/PHASE_D32B_REPORT.md。次はD.3.3の設計判断（ユーザー確認事項あり、レポート10節）。
