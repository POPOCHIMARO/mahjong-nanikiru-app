# PHASE_D33_DESIGN 反証レビュー（第5回、改訂4.1案に対するレビュー）

レビュー日：2026-10-11。レビュー担当：Codex GPT-6 Astra/high（読み取り専用、codex-companion task、ジョブ `task-mv3t0rwd-vffjj8`）。
以下はレビュー結果の原文である。対象は [PHASE_D33_DESIGN.md](PHASE_D33_DESIGN.md) の18節（改訂4.1案、コミット 78b6f36）。
本文中のリンクの行番号は、レビュー当時の版を指す。

**判定：needs-attention**

第4回の6件は、段階Aの用途を限定した**設計としては解消**されています。P1・P2相当の未解消事項は見つかりませんでした。第18節に追加された記述について、次のP3を修正してください。

1. **［P3］`family_mass_unknown`だけのholdを数える規則は、段階Aでは成立しません。**

   **根拠：** [設計853行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:853)では全件に`fixed_component_sensitivity_missing`が付きます。一方、[866行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:866)は「`family_mass_unknown`だけが理由」の判断を数えるため、文字どおり実装すると常に0件になります。未訪問系統の扱いを見直すための報告として機能しません。

   **修正案：** `family_mass_unknown`を含む判断数を集計し、併存するhold理由を別に示してください。「それだけが原因」の件数も必要なら、共通の感度診断不足と、同じ未訪問に由来するESS未定義をどう扱うか明記してください。

2. **［P3］固定反復数の式と、使用するrの桁数が一致していません。**

   **根拠：** [設計813行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:813)は`r＝82.1`としていますが、[825行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:825)の式をその値で計算すると、`ceil(400 × 82.1 × 1.5 / 4)＝12,315`です。記載された12,314は、[保存値82.09205769338926](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/probes/d33-pilot/mixing/pooled-ess.json:449)を使った結果と一致します。

   **修正案：** 「表示は82.1、計算とmanifestには丸め前の値を使う」と明記するか、82.1を固定値としてN＝12,315、総反復＝13,315に統一してください。精度への影響は微小ですが、固定長の再現条件は一意にする必要があります。

第4回の6件の個別判定は次のとおりです。「解消」は設計上の判定であり、本体実装の完了を意味しません。

| 前回の指摘 | 判定 | 根拠 |
|---|---|---|
| 1［P1］短い出発検査による質量の過小認定 | 解消 | 未訪問は`family_mass_unknown`となり、出発検査からの解除を禁止。[18.2節](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:763) |
| 2［P2］対象系統から出発する保証と提案確率の混同 | 解消 | 実際の所属を確認し、不一致は検査未成立。初期化確率とM3/M4の確率も分離。[778行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:778) |
| 3［P1］ESS単純合計と統合平均の精度の不一致 | 解消 | 標本数の重みに応じた分散へ修正。`pooled_mean_ess`も式に一致。[設計786行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:786)、[実装3622行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/tools/ev_policy_belief.py:3622) |
| 4［P2］r・安全係数・burn-inの根拠不足 | 段階Aの用途に限り解消 | R̂の誤記を撤回し、各値を暫定値と明示。新しい乱数系列と固定長を使い、用途も限定。[18.4節](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:813) |
| 5［P2］未訪問・ESS未定義の合否が未定義 | 解消 | 支持から系統を列挙し、構造的一定を区別。主要統計のESS未定義はheld。[763行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:763)、[796行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:796) |
| 6［P2］開発利用の例外が広すぎる | 解消 | 接続・動作確認・性能測定に限定。他のholdがある標本は返さず、heldを下流へ伝播。結果を見て選ぶ段階Bも探索扱い。[18.6節](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:853) |

指定の観点について、補足します。

- **分散とNoneの扱い：** 独立な鎖なら、統合平均の分散と「各鎖平均−統合平均」の分散は、ともに正しい式です。batch meansによる値は推定値であり、非定常性による偏りは含みません。バッチ平均が一定でも元の系列は動き得ますが、その場合に精度を認定せずNoneにする扱いは保守的で妥当です。[テスト1874行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/tests/test_ev_policy_belief.py:1874)も、この区別を含んでいます。
- **rと追加校正：** 82.1は、ESSを計算できた統計についての暫定値です。実際に[未定義の統計も残っています](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/probes/d33-pilot/mixing/pooled-ess.json:221)。混合の十分性を保証する値ではありませんが、18.6節の用途に限れば、別の校正鎖を必須とする理由はありません。
- **先頭9判断の選択：** 既存manifestの先頭9件は9層を1件ずつ含み、manifestは段階2より前のコミットで固定されています。[選択規則](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/probes/d33-pilot/manifest.json:6)から、結果の良かった判断への差替えは認められません。ただし「約5日の規則を再計算結果を見る前に決めた」という時系列は、[18.1節の記録](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D33_DESIGN.md:754)以上には独立確認していません。
- **d33Conditionsとの整合：** D.3.2bは、条件未充足の実行をheldで終えることを要求しています。[D.3.2b設計420行](C:/AI_Workspace/CodexProjects/麻雀なにきるアプリ/calibration/PHASE_D32B_DESIGN.md:420)。18.6節はheldを維持・伝播するため、この条件と整合します。

保存32鎖をNode.jsでメモリ上に展開し、3バッチ幅のrを最大絶対誤差約`2.2×10⁻¹²`以内で再現しました。関連ハッシュも一致しています。`MixingAnalysisTest`はコード確認のみで、PythonがPATH上にないため実行していません。ファイル変更・コミット・外部送信は行っていません。

