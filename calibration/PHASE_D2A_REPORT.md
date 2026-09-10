# フェーズD.2a 得点計算と精算

実行日：2026-09-07
判定：D.2a完了。D.2bの一局状態遷移は未実装

## 実装した範囲

- `mahjong==2.0.0`をSHA-256付きrequirementsへ固定した。
- Dの物理牌IDから依存ライブラリの136形式へ、赤5を保持して決定的に変換する。
- 役、符、切り上げ満貫、数え上限、複合役満をMリーグ設定で計算する。
- 連風牌の雀頭を2符とするため、完成手に連風牌がちょうど2枚ある場合だけ計算用の場風を省く。
- 和了点から本場と供託を分離し、通常ロン、通常ツモ、責任払いを一つの精算器で処理する。
- 大三元、大四喜、四槓子の責任発生を、成立済み面子と新しいポン／大明槓から判定する。
- 複数の責任者がいて本場配分を確定できない場合は `hold: multiple_pao_honba_unresolved` として停止する。

得点計算依存へは本場数と供託数を常に0で渡す。
精算後は `sum(scores) + 1000 * kyoutaku` の保存を検査する。
通常5を赤牌IDへ割り当てる誤り、同じ物理牌を手牌とドラ表示牌へ重複指定する誤り、同じ牌を複数副露へ使う誤りを入力境界で拒否する。

## 公式責任払い例

南家が大三元と字一色の2倍役満をツモ和了し、西家が大三元の責任者、東家が親、1本場の場合を固定例にした。

| 席 | 点差 |
|---|---:|
| 東 | −16,000 |
| 南 | +64,300 |
| 西 | −40,300 |
| 北 | −8,000 |

4家の点差合計は0となる。
大三元分32,000点と本場300点を西家が負担し、字一色分は通常の子ツモとして東家16,000点、北家8,000点、西家8,000点を負担する。

## 検証結果

`tests/test_ev_policy_scoring.py`の13件が合格した。
得点境界、連風雀頭、七対子、国士、実際の大三元＋字一色、食いタン、通常精算、責任発生、責任ツモ、第三者ロン折半、複数責任者保留、牌ID重複拒否を確認した。

```powershell
python -m unittest tests.test_ev_policy_scoring
# Ran 13 tests ... OK

python -m unittest tests.test_calibrate_ev tests.test_ev_calibration_model tests.test_ev_policy_state tests.test_ev_policy_scoring
# Ran 36 tests ... OK

python -m pip install --dry-run --no-index --find-links calibration/.dependency-cache --require-hashes --only-binary=:all: -r calibration/requirements-scoring.txt
# Would install mahjong-2.0.0

python calibration/probes/probe_mahjong_2.py calibration/.dependency-cache/mahjong-2.0.0-py3-none-any.whl --output calibration/probes/probe-mahjong-2-result.json
# 16 checks passed / status: pass
```

共有のPython実行環境へパッケージをインストールしていない。
テストはハッシュ検証済みwheelを直接読み込んだ。
実装環境では[固定requirements](requirements-scoring.txt)から導入する。

## 未実装の範囲

D.2aは完成した和了入力の得点と精算を扱う。
ツモ、打牌、応答窓、フリテン、槓、嶺上、ドラ公開、流局まで一局を進める処理と、責任発生をイベント列へ保存する接続は [D.2b](PHASE_D2B_REPORT.md) で実装した。
原牌譜再現、決定的なpush/fold、情報遮断の全件確認はD.2cで扱う。

過去の公式規定、複数責任者の本場配分、同巡内フリテンなど、[D.2設計](PHASE_D2_RULES_AND_DEPENDENCIES.md)で未確認とした規定は解消していない。
これらを推測値で埋めてD.2完了とはしない。
