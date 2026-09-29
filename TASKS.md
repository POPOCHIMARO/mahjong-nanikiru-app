# 作業リスト：D.3.3 工程1（家ごとの履歴評価器）

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
- [ ] コミット（ユーザーの指示待ち）
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
