# 作業リスト：D.3.2b（D.3.3接続前hold解消）

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

- [ ] cache生成、再学習、定数推定の往復、補正、再評価
- [ ] D32B-11、`PHASE_D32B_REPORT.md`、開発計画の更新

## メモ

- 2026-09-27：開発計画にD.3.2bを登録し、工程1に着手。
- 2026-09-27：工程1完了。次は工程2（Sonnet 5推奨）。以後の特徴・学習は`dataset-opponent-v3`を入力にする。
- 2026-09-27：工程2完了。次は工程3（Opus 5.5/high）。
- 2026-09-27：工程3完了。次は工程4（Sonnet 5推奨）。
- 2026-09-27：工程4完了。次は工程5（全件実行、Sonnet 5＋レポート判定はOpus 5.5）。
