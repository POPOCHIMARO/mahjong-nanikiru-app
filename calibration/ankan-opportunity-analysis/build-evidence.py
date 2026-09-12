"""検証済み集計を読み、説明資料と実行済みノートブックを作る（標準ライブラリのみ）。"""
import contextlib
import io
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
s = json.loads((HERE / 'full/summary.json').read_text(encoding='utf-8'))
g = json.loads((HERE / 'generated-sample.json').read_text(encoding='utf-8'))
v = json.loads((HERE / 'verification.json').read_text(encoding='utf-8'))
t = s['totals']
draws = t['self_action_after_live'] + t['self_action_after_rinshan']

def rate(n, d):
    return f'{n / d * 100:.3f}%'

metrics = [
    ('通常／嶺上ツモ後に合法暗槓がある', t['ankan_windows'], draws),
    ('1人1局で合法暗槓を1回以上検討できる', t['unique_ankan_seat_rounds'], t['seat_rounds']),
    ('現モードに近い2→1の局面に合法暗槓がある', t['app_structural_ankan'], t['app_structural_windows']),
    ('字牌打牌候補を除外した2→1の局面に合法暗槓がある', t['app_no_honor_candidate_ankan'], t['app_no_honor_candidate_windows']),
]
table = '\n'.join(f'| {label} | {n:,} | {d:,} | {rate(n,d)} |' for label,n,d in metrics)
season_table = '\n'.join(f"| {name} | {r['self_action_after_live']+r['self_action_after_rinshan']:,} | {r['ankan_windows']:,} | {rate(r['ankan_windows'],r['self_action_after_live']+r['self_action_after_rinshan'])} | {r['observed_ankan']:,} |" for name,r in s['seasons'].items())
labels = {'own_riichi':'自家リーチ後','not_own_riichi':'自家リーチ前','opponent_riichi':'他家リーチあり','no_opponent_riichi':'他家リーチなし','plain_live':'固定面子なし／自家リーチ前の通常ツモ','app_structural':'現モードに近い2→1','app_no_honor_candidate':'2→1から字牌候補を除外'}
strata_table = '\n'.join(f"| {labels[name]} | {r.get('windows',0):,} | {r.get('ankan',0):,} | {r.get('discard',0)+r.get('riichi_discard',0):,} | {r.get('tsumo',0):,} | {r.get('unknown',0):,} |" for name,r in s['strata'].items() if name in labels)
episode = s['episodeChoices']
notes = f'''# 暗槓判断を牌効率練習へ追加する優先度の調査

調査日：2026-09-11。対象：手元のMリーグ2018–19〜2025–26牌譜。アプリ変更なし。

## 判断

暗槓判断を通常の牌効率問題より優先して増やすことは勧めない。まず、字牌切りが自明な問題や現在の採点で説明しきれない四枚使いを通常出題から除き、通常の比較問題を改善する方が、今回確認した問題数に対する効果が大きい。

暗槓そのものを学習対象から捨てる必要はない。1人1局の遭遇率は{rate(t['unique_ankan_seat_rounds'],t['seat_rounds'])}で、約{t['seat_rounds']/t['unique_ankan_seat_rounds']:.1f}局に1回。任意のテーマ練習に代表例を少数用意する候補にはなる。ただし正解を付けるには個々の局面の評価が別途必要である。

これは出現頻度と実装範囲からの優先度判断である。学習者の誤答率、1問あたりの学習効果、暗槓を誤ったときの平均損失は測っていない。出現率をそのまま最適な教材配分にはしない。

## 作業範囲と完了条件

- 正規化済みデータの合法手から暗槓機会を数え、実暗槓と区別する。
- ツモ後の判断、局家、局家と牌種の3単位を使い、反復機会を示す。
- 現モードの構造条件と字牌候補除外後の部分集合を別集計する。
- 別集計による件数一致と、JavaScriptによる固定標本のシャンテン照合を通す。
- 分析スクリプト、個票、JSON集計、実行済みノートブックをこのフォルダへ保存する。原牌譜、学習済みモデル、engine.js、app.jsは変更しない。

## 対象範囲

元牌譜は1,916対局、23,358局。8シーズンの原ファイルを保存manifestのSHA-256と照合した。外部の公式全試合一覧との網羅性照合は行っていない。

合法手のある教師データを使用した局は{t['rounds']:,}局。初期復元不能20局、南4局以降2,891局、自己行動の教師窓がない局などは対象外である。南4局以降の機会頻度を推定した値ではない。後半の復元に問題がある局は、その問題以前の正常な判断までを含む。

2026–27は将来評価用の予約に従い読み込んでいない。元データの範囲、除外理由、シーズン別母数は `full/summary.json` に記録した。

## 暗槓の判断機会

| 指標 | 分子 | 分母 | 割合 |
|---|---:|---:|---:|
{table}

「暗槓機会」は、判断前の合法候補集合に暗槓が1種類以上ある窓である。単に同じ牌が4枚あることでは判定しない。リーチ後の合法性、山と嶺上牌の残り、槓数の上限は既存の合法手生成に従う。鳴き直後の25,137窓は暗槓できないため、ツモ後の分母には入れない。

暗槓機会6,868窓のうち、和了も合法候補にある窓が25、観測ラベルがholdの窓が2ある。holdのうち1窓は実選択がnull、もう1窓はツモ和了の記録があるが和了判定側がholdである。実選択nullを「見送り」として埋めていない。和了判定自体の状態もsummaryの `ankan_win_action_*` に残した。

## 繰り返しと実選択

- 暗槓を選べた局：{t['unique_ankan_rounds']:,}局。
- 暗槓を選べた局家：{t['unique_ankan_seat_rounds']:,}件。
- 同じ局家と牌種をまとめた機会：{t['unique_opportunity_episodes']:,}件。
- その最初の機会で暗槓した件数：{episode['first_chosen']:,}件。
- 保留後、観測範囲内で暗槓した件数：{episode['deferred_then_chosen']:,}件。
- 観測範囲内で暗槓実行の記録がなかった件数：{episode['never_chosen_in_observed_prefix']:,}件。選択不明や観測終了後に実行した可能性は残る。
- 実暗槓：{t['observed_ankan']:,}回。判断窓単位の実行率は{rate(t['observed_ankan'],t['ankan_windows'])}。

「選べたのに切った」は観測行動であり、「暗槓しない方が正しい」という教師ラベルではない。実行率には複合形、打点、安全度、リーチ、巡目などが混在する。同じ四枚を数巡残すだけでも判断窓が増えるため、6,868種類の独立教材があるとは言えない。

## 現行モードに近い局面

通常ツモ後、既存の副露と暗槓がともになく、自家リーチ前、実際のツモ前13枚が2シャンテンで、ツモ後の最善打牌で1シャンテンになる局面を数えた。実際に何を切ったかによる選別はしていない。

この条件に合う{t['app_structural_windows']:,}窓中、合法暗槓は{t['app_structural_ankan']:,}窓。四枚使い自体は{t['app_quad_windows']:,}窓であり、合法暗槓との差はルール上の制約による。字牌切りが1シャンテン維持候補に入る問題を仮除外しても、合法暗槓は{t['app_no_honor_candidate_ankan']:,}/{t['app_no_honor_candidate_windows']:,}窓にとどまる。

他家リーチがない部分集合では{t['app_no_opponent_riichi_ankan']:,}/{t['app_no_opponent_riichi_windows']:,}窓（{rate(t['app_no_opponent_riichi_ankan'],t['app_no_opponent_riichi_windows'])}）。したがって、他家リーチの有無だけで全体の結論を作ってはいない。

この部分集合は構造条件の一致であり、受け入れ差、単一正解、変化、2ツモ逆転排除など、現アプリの全品質条件を通した問題数ではない。それらの高価な全件検査は今回行っていない。

| 部分集合（重複あり） | 暗槓機会 | 実暗槓 | 打牌／リーチ打牌 | ツモ和了の記録 | 選択不明 |
|---|---:|---:|---:|---:|---:|
{strata_table}

ツモ和了の記録には和了判定側がholdの1窓を含む。実行と保留の両方を学べる例は存在するが、この表からどちらが正しいかは判定しない。

合法候補と判断窓の組合せ6,877件を牌の形で分けると、字牌726件、周辺に数牌がない数牌1,533件、前後2つ以内に別の数牌を持つ候補4,618件だった。最後の群は複合形を検討するための粗い候補群であり、難しい判断と認定した件数ではない。近接牌があるだけでは、暗槓による形の損失も正解も確定しない。

## シーズンごとの頻度

| シーズン | ツモ後の判断 | 暗槓機会 | 頻度 | 実暗槓 |
|---|---:|---:|---:|---:|
{season_table}

## 現在の出題器との比較

固定seed 20260911、通常100問と高難度100問を生成した。これは構成済みの手や罠型を優先する人工分布であり、実戦の発生率とは別の調査である。

| 200問中の特徴 | 件数 | 割合 |
|---|---:|---:|
| 字牌がシャンテン維持の打牌候補 | 70 | 35.0% |
| 字牌切りが正解 | 67 | 33.5% |
| 四枚使い | 12 | 6.0% |
| 字牌候補または四枚使い | 77 | 38.5% |

字牌候補70問を除くと130問残り、そのうち四枚使いは7問（5.38%）。この固定標本では、両方を除いても123問が残る。除外後の生成待ち時間や難度別の供給量は未測定であり、現在の生成器を修正済みという意味ではない。

実戦の類似局面で四枚使いは{rate(t['app_quad_windows'],t['app_structural_windows'])}だった一方、この生成標本では6.0%だった。出題器の分布が異なるため比率の倍率を一般化しないが、画面で四枚使いに出会う頻度をそのまま実戦の学習優先度には使えない。

## 検証と限界

1. 3本の入力gzipと8本の原牌譜のSHA-256が保存manifestと一致。
2. 全2,085,155教師窓を走査し、主集計の暗槓6,868窓／実行1,015回が、別エージェントによる教師窓だけの独立集計と一致。
3. 公開暗槓イベントとの照合は、各局の最終教師時点までに限定し1,015回一致。初回検査では終了直後の1イベントを余分に含めて失敗した。対象は `mleague:2018-19:L001_S001_0036_02A:2:0`、最終教師140、後続暗槓141。境界を訂正し全件を再実行した。
4. Python内の全打牌列挙確認{len(s['discardShantenChecks'])}件に加え、別実装のJavaScriptで構造条件と字牌候補を{v['checkedDistinctHands']}個の固定標本で照合しpass。全手牌の独立照合ではない。
5. 保存済み200問標本の個票から旗と合計を再計算し、エンジンのhash一致を確認。
6. 原JSONLの行数は対局数ではなくレコード数だったため、集計後にsourceFilesのメタデータ名をgamesからrecordsへ訂正した。主集計の数値は変更せず、対局数はroundId内のgameIdを重複除去した1,916を使う。実行時コードhashとメタデータ訂正後のhashをsummaryに区別して保存した。

個々の暗槓の最適性、打点込みEV、学習効果、南4以降の頻度、実ブラウザ動作は今回の検証対象外。特に「難しい／誤りやすい暗槓判断」の正確な件数は、この頻度調査だけでは確定しない。合法機会はその上限となる母集団である。

## 再実行

Python 3.12とnumpyを使う。既存のmahjong 2.0.0 wheelはスクリプトが読み取り専用で参照する。PowerShellではPythonの実行パスを指定し、プロジェクト直下から実行する。

```text
python calibration/ankan-opportunity-analysis/analyze.py
node calibration/ankan-opportunity-analysis/sample-generator.js
node calibration/ankan-opportunity-analysis/verify.js
python calibration/ankan-opportunity-analysis/build-evidence.py
```

`analysis.ipynb` は保存した集計の読み直しと計算を行う軽量な再実行用ノートブック。原牌譜からの全件再構築は上記analyze.pyで行う。ノートブックは標準ライブラリの同一プロセス内で先頭から実行し、出力を保存した。Jupyterカーネルによる実行と画面の描画確認はしていない。
'''
(HERE / 'FINDINGS.md').write_text(notes, encoding='utf-8')

def md(text):
    return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}

def code(text):
    return {'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'execution_count':None,'outputs':[]}

cells = [md('# 暗槓判断の頻度と教材の優先度\n\n## tl;dr\n通常の牌効率問題を優先する。暗槓は任意のテーマ練習候補。合法機会はツモ後判断の約0.67%、現モードに近い局面の約0.94%。'),
    md('## Context & Methods\n判断前の合法候補を数える。実行率は最適率ではない。2018–19〜2025–26の保存牌譜のうち、南4以降などを除いた20,446局。\n\n### Key Assumptions\n局家牌種の重複除去は同じ四枚の反復保有をまとめる。構造条件一致は全出題品質条件の合格ではない。元データ再構築はanalyze.pyで行う。'),
    md('## Data\n3本のgzipのhashと原牌譜8本を主集計で照合済み。ここではfull/summary.jsonとgenerated-sample.json、verification.jsonを読み直す。'),
    code("import json\nfrom pathlib import Path\nbase = Path.cwd()\nif not (base / 'full/summary.json').exists():\n    base = base / 'calibration/ankan-opportunity-analysis'\nsummary = json.loads((base / 'full/summary.json').read_text(encoding='utf-8'))\ngenerated = json.loads((base / 'generated-sample.json').read_text(encoding='utf-8'))\nvalidation = json.loads((base / 'verification.json').read_text(encoding='utf-8'))\nassert validation['status'] == 'pass'\nprint('教師窓:', summary['totals']['teacher_windows'])\nprint('局:', summary['totals']['rounds'])\nprint('保存生成問題:', len(generated['records']))"),
    md('## Results\n### 判断機会の割合\n分母が異なるため率を直接足さない。小さな比較なので、図の代わりに正確な分子と分母の表を使う。'),
    code("t = summary['totals']\ncomparisons = [\n ('ツモ後', t['ankan_windows'], t['self_action_after_live']+t['self_action_after_rinshan']),\n ('局家', t['unique_ankan_seat_rounds'], t['seat_rounds']),\n ('2→1の構造条件', t['app_structural_ankan'], t['app_structural_windows']),\n ('字牌候補除外後', t['app_no_honor_candidate_ankan'], t['app_no_honor_candidate_windows'])]\nfor label,n,d in comparisons:\n    print(f'{label}: {n:,} / {d:,} = {n/d*100:.3f}%')\nassert t['ankan_windows'] == 6868 and t['observed_ankan'] == 1015"),
    md('### 出題器の200問標本\n人工的な出題分布であり、実戦頻度の推定ではない。'),
    code("for key in ['honorDiscardCandidate','correctHonorDiscard','anyQuad','quadOrHonorCandidate']:\n    count = sum(r['flags'][key] for r in generated['records'])\n    assert count == generated['aggregate']['counts'][key]\n    print(f'{key}: {count}/200 = {count/2:.1f}%')\nprint('JavaScriptとの独立照合標本:', validation['checkedDistinctHands'])"),
    md('## Takeaways\n字牌候補を含む問題の整理を先に行い、暗槓を頻繁に通常枠へ混ぜる開発は後回しにする。代表例のテーマ練習は残す。誤答率や打点込み損失は未測定であり、最適な教材比率や個々の暗槓の正解を示す調査ではない。詳細と再現手順はFINDINGS.md。')]

# nbclientが同梱されていないため、標準Pythonでセルを順に実行する。
import os
os.chdir(HERE)
namespace = {}
count = 0
for cell in cells:
    if cell['cell_type'] != 'code':
        continue
    count += 1
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(''.join(cell['source']), f'analysis.ipynb:cell{count}', 'exec'), namespace)
    cell['execution_count'] = count
    cell['outputs'] = [{'output_type':'stream','name':'stdout','text':output.getvalue().splitlines(keepends=True)}]
notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.12'},'execution_method':'sequential Python exec; not a Jupyter kernel'},'cells':cells}
for i,cell in enumerate(cells): cell['id'] = f'cell-{i:02d}'
(HERE / 'analysis.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')

items = []
for ident,title,rows,files,caveats in [
    ('real-opportunities','Mリーグで暗槓を判断できる頻度',[{'対象':label,'機会数':n,'母数':d,'割合':rate(n,d)} for label,n,d in metrics],
     ['teacher-windows.jsonl.gz','public-events.jsonl.gz','private-events.jsonl.gz','full/summary.json'],
     ['南4局以降と初期復元不能局などは対象外。公式の全試合一覧との網羅性は未照合。','合法機会は最適性や難しさの判定ではない。2→1の集計は全出題品質条件を通した問題数ではない。']),
    ('generated-distribution','現在の出題器で整理できる問題',[{'特徴':label,'件数':n,'母数':200} for label,n in [('字牌候補',70),('字牌正解',67),('四枚使い',12),('字牌候補または四枚使い',77)]],
     ['sample-generator.js','generated-sample.json','verification.json'],
     ['固定seedの通常100問と高難度100問。人工的な出題分布であり実戦頻度ではない。','誤答率や学習効果は測定していない。'])]:
    items.append({'id':ident,'title':title,'queries':[{'id':ident+'-query','source':{'label':'ローカルMリーグ牌譜と出題器の分析',
        'files':[{'label':name} for name in files], 'caveats':caveats,
        'evidenceFlow':[{'kind':'calculation','title':'割合','detail':'機会数を対応する母数で割り、100を掛ける。'},
                        {'kind':'validation','title':'照合','detail':'暗槓6,868窓と実行1,015回は独立集計と一致。生成200問の個票から合計を再計算。'}]},
        'reportingPeriod':'保存済み2018–19〜2025–26牌譜／生成標本seed 20260911',
        'columns':list(rows[0]),'rows':rows} ]})
(HERE / 'sources.json').write_text(json.dumps({'schemaVersion':1,'items':items}, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps({'notebookCodeCellsExecuted':count,'findings':'FINDINGS.md','sources':'sources.json'},ensure_ascii=False))
