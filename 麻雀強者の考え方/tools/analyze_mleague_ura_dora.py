#!/usr/bin/env python3
"""公開牌譜からリーチ和了の裏ドラ率を再計算する。

実行: python -m pip install --target .analysis_deps -r tools/ura_dora_requirements.txt
      python tools/analyze_mleague_ura_dora.py
役名は原資料にないため、状況役を除いた最高得点分解による再判定。
裏表示欠損は0枚に置き換えず、率の分母から除く。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.analysis_deps'))
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules, HandConstants
from mahjong.hand_calculating.divider import HandDivider
from mahjong.hand_calculating.fu import FuCalculator
import mahjong.hand_calculating.hand as hand_module
from mahjong.meld import Meld
from mahjong.tile import TilesConverter

OUT = ROOT / 'data/mleague/ura_dora'

class MLeagueFuCalculator(FuCalculator):
    """公式第6章3条: 連風牌の対子も2符（通常ライブラリは4符）。"""
    @staticmethod
    def calculate_fu(hand, win_tile, win_group, config, valued_tiles=None, melds=None):
        valued = list(dict.fromkeys(valued_tiles)) if valued_tiles else valued_tiles
        return FuCalculator.calculate_fu(hand, win_tile, win_group, config, valued, melds)

# 外部パッケージのファイルは変えず、この実行だけ公式ルールに適応する。
hand_module.FuCalculator = MLeagueFuCalculator
YAKU = {'Pinfu':'平和','Tanyao':'断么九','Iipeiko':'一盃口',
        'Ryanpeikou':'二盃口','Chiitoitsu':'七対子','Sanshoku Doujun':'三色同順',
        'Ittsu':'一気通貫','Chantai':'混全帯么九','Junchan':'純全帯么九',
        'Honitsu':'混一色','Chinitsu':'清一色','San Ankou':'三暗刻',
        'Toitoi':'対々和','Honroutou':'混老頭','Shou Sangen':'小三元',
        'Sanshoku Doukou':'三色同刻','San Kantsu':'三槓子','Menzen Tsumo':'門前清自摸和'}

def tile34(raw):
    """天鳳の赤5を通常5と同じ種類にする。"""
    raw = {51:15,52:25,53:35}.get(raw, raw)
    if not isinstance(raw, int):
        raise ValueError(f'not a numeric tile: {raw}')
    if 11 <= raw <= 19: return raw - 11
    if 21 <= raw <= 29: return raw - 12
    if 31 <= raw <= 39: return raw - 13
    if 41 <= raw <= 47: return raw - 14
    raise ValueError(f'invalid tile: {raw}')

def successor(t):
    if t < 27: return (t//9)*9+(t%9+1)%9
    if t < 31: return 27+(t-27+1)%4
    return 31+(t-31+1)%3

def reconstruct(log, seat):
    """座ごとの摸打列を処理。暗槓は捨牌列の行為で、実捨牌ではない。"""
    hand = Counter(log[4+3*seat])
    draws, discards = log[5+3*seat], log[6+3*seat]
    kans, last_discard, riichi_index = [], None, None
    j = 0
    for i, draw in enumerate(draws):
        if isinstance(draw, str):
            # リーチ者には副露がない。放銃者の末尾ツモ切り解決は別関数。
            raise ValueError('open call in riichi winner')
        hand[draw] += 1
        if j >= len(discards): continue
        discard = discards[j]; j += 1
        if isinstance(discard, str) and 'a' in discard:
            quad = [int(x) for x in re.findall(r'\d{2}', discard)]
            if len(quad) != 4: raise ValueError('invalid closed kan')
            hand.subtract(quad); kans.append(quad)
        else:
            if isinstance(discard, str) and discard.startswith('r'):
                riichi_index = j-1
                discard = int(discard[1:])
            last_discard = draw if discard == 60 else discard
            hand[last_discard] -= 1
        if any(v<0 for v in hand.values()): raise ValueError('negative hand count')
    if j != len(discards): raise ValueError('unprocessed discards')
    return list(hand.elements()), kans, riichi_index

def ron_tile(log, loser):
    """放銃座の最終行為。大明槓の嶺上ツモによる添字ずれを吸収。"""
    draws, discards = log[5+3*loser], log[6+3*loser]
    d = discards[-1]
    if isinstance(d,str) and ('k' in d or 'a' in d):
        return int(re.findall(r'\d{2}',d)[-1]) # 槍槓
    if isinstance(d,str) and d.startswith('r'): d=int(d[1:])
    if d == 60:
        draw = draws[-1]
        if not isinstance(draw,int): raise ValueError('last tsumogiri without numeric draw')
        return draw
    return d

def physical_tiles(raw_tiles):
    """重複しない136形式IDを与え、赤5のIDを予約する。"""
    ids, used = [], set()
    for raw in sorted(raw_tiles, key=lambda x: x not in (51,52,53)):
        t=tile34(raw); pool=list(range(t*4,t*4+4))
        if t in (4,13,22) and raw not in (51,52,53): pool=pool[1:]
        if raw in (51,52,53): pool=pool[:1]
        available=[x for x in pool if x not in used]
        if not available: raise ValueError('too many copies or red fives')
        used.add(available[0]); ids.append(available[0])
    return ids

def evaluate(raw, kans, win_raw, log, seat, tsumo, ura):
    tiles=physical_tiles(raw)
    remaining=tiles.copy(); melds=[]
    for quad in kans:
        q=[]
        for r in quad:
            t=tile34(r)
            candidates=[x for x in remaining if x//4==t and (x%4==0 if r in (51,52,53) else (t not in (4,13,22) or x%4!=0))]
            if not candidates: raise ValueError('kan allocation error')
            x=candidates[0]; remaining.remove(x); q.append(x)
        melds.append(Meld(Meld.KAN,q,opened=False))
    winning=[x for x in remaining if x//4==tile34(win_raw) and (x%4==0 if win_raw in (51,52,53) else (tile34(win_raw) not in (4,13,22) or x%4!=0))]
    if not winning: raise ValueError('winning tile missing')
    cfg=HandConfig(is_tsumo=tsumo,is_riichi=True,
        player_wind=27+(seat-log[0][0]%4)%4, round_wind=27+log[0][0]//4,
        options=OptionalRules(has_aka_dora=True,has_open_tanyao=True,
            has_double_yakuman=False,kazoe_limit=HandConstants.KAZOE_SANBAIMAN,kiriage=True))
    value=HandCalculator.estimate_hand_value(tiles,winning[0],melds=melds,
        dora_indicators=[tile34(x)*4 for x in log[2]],
        ura_dora_indicators=[tile34(x)*4 for x in log[3]],config=cfg)
    if value.error: raise ValueError(value.error)
    names=[y.name for y in value.yaku or []]
    yakuman=any(y.is_yakuman for y in value.yaku or [])
    roles=set(YAKU[x] for x in names if x in YAKU)
    if any(x.startswith('Yakuhai') for x in names): roles.add('役牌')
    if yakuman: roles.add('役満')
    lib_ura=sum(y.han_closed for y in value.yaku or [] if y.name=='Ura Dora')
    if ura is not None and not yakuman and lib_ura != ura: raise ValueError('ura count disagrees with library')
    n_divisions=len(HandDivider.divide_hand(TilesConverter.to_34_array(tiles),melds=melds))
    return sorted(roles), value, yakuman, n_divisions

def rate(rows):
    known=[r for r in rows if r['uraCount'] is not None]
    n=len(known); hit=sum(r['uraCount']>0 for r in known)
    p=hit/n if n else None
    if n:
        z=1.959963984540054; den=1+z*z/n
        c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        ci=[c-h,c+h]
    else: ci=[None,None]
    return {'riichiWins':len(rows),'known':n,'missing':len(rows)-n,'hit':hit,
        'rate':p,'wilson95':ci,'meanUra':sum(r['uraCount'] for r in known)/n if n else None,
        'twoOrMore':sum(r['uraCount']>=2 for r in known),
        'distribution':dict(sorted(Counter(r['uraCount'] for r in known).items()))}

def compare_yaku_by_match(wins, match_keys, repetitions=2000):
    """半荘を丸ごと再抽出し、複合役と半荘内の依存を保った差の感度を出す。"""
    comparisons=[('一気通貫','平和'),('平和','七対子')]
    output=[]
    for first,second in comparisons:
        groups={k:[0,0,0,0] for k in sorted(match_keys)}
        for r in wins:
            if r['uraCount'] is None: continue
            vector=groups[r['matchUrl'].rstrip('/')]
            for i,yaku in enumerate((first,second)):
                if yaku in r['yaku']:
                    vector[i*2]+=1; vector[i*2+1]+=r['uraCount']>0
        vectors=list(groups.values()); rng=random.Random(20261012)
        samples=[]
        for _ in range(repetitions):
            n1=h1=n2=h2=0
            for v in rng.choices(vectors,k=len(vectors)):
                n1+=v[0]; h1+=v[1]; n2+=v[2]; h2+=v[3]
            if n1 and n2: samples.append(h1/n1-h2/n2)
        samples.sort(); total=[sum(v[i] for v in vectors) for i in range(4)]
        output.append({'first':first,'second':second,'difference':total[1]/total[0]-total[3]/total[2],
            'clusterBootstrap95':[samples[int(len(samples)*.025)],samples[int(len(samples)*.975)-1]],
            'repetitions':repetitions,'seed':20261012,'cluster':'matchUrl'})
    return output

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    wins, issues, inputs, seen = [], [], [], set()
    correction_path=ROOT/'data/mleague/ingest/2026-10-12.paifu-corrections.jsonl'
    corrections={}
    if correction_path.exists():
        for line in correction_path.open(encoding='utf-8'):
            row=json.loads(line)
            key=(row['matchUrl'].rstrip('/'),row['roundIndex'])
            if key in corrections: raise ValueError('duplicate correction')
            corrections[key]=row
        inputs.append({'path':correction_path.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(correction_path.read_bytes()).hexdigest()})
    used_corrections=set()
    round_count=0; all_wins=0; match_keys=set(); dates=[]
    for path in sorted((ROOT/'data/mleague/paifu').glob('*.jsonl')):
        inputs.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.open(encoding='utf-8'):
            row=json.loads(line); url=row['matchUrl'].rstrip('/')
            correction_key=(url,row['roundIndex'])
            if correction_key in corrections:
                row=corrections[correction_key]; used_corrections.add(correction_key)
            match_keys.add(url); dates.append(row['date'])
            for li,log in enumerate(row['paifu']['log']):
                key=(url,row['roundIndex'],li)
                if key in seen: raise ValueError(f'duplicate round: {key}')
                seen.add(key); round_count+=1
                result=log[-1]
                if result[0]!='和了': continue
                for detail in result[2:]:
                    if not isinstance(detail,list): continue
                    all_wins+=1; seat,loser=detail[:2]
                    if not any(isinstance(x,str) and x.startswith('r') for x in log[6+3*seat]): continue
                    base={'season':row['season'],'gameId':row['gameId'],'roundIndex':row['roundIndex'],
                        'logIndex':li,'matchUrl':row['matchUrl'],'date':row['date'],'winner':seat,
                        'player':row['paifu']['name'][seat],'tsumo':seat==loser,'scoreText':detail[3]}
                    try:
                        concealed,kans,ri=reconstruct(log,seat)
                        win=log[5+3*seat][-1] if seat==loser else ron_tile(log,loser)
                        if seat!=loser: concealed.append(win)
                        raw=concealed+[x for q in kans for x in q]
                        counts=Counter(tile34(x) for x in raw)
                        if len(raw)!=14+len(kans) or max(counts.values())>4: raise ValueError('illegal final tile counts')
                        indicators=log[3]
                        ura=sum(counts[successor(tile34(x))] for x in indicators) if indicators else None
                        roles,val,yakuman,divs=evaluate(raw,kans,win,log,seat,seat==loser,ura)
                        original_fu=re.search(r'(\d+)符',detail[3])
                        fu_match=not original_fu or int(original_fu[1])==val.fu
                        rec={**base,'handRaw':sorted(raw),'hand34':[counts[i] for i in range(34)],
                            'winTile':win,'closedKans':kans,'uraIndicators':indicators,
                            'indicatorCount':len(indicators),'doraIndicatorCount':len(log[2]),
                            'uraCount':ura,'yaku':roles,'yakuman':yakuman,'decompositions':divs,
                            'calculatedFu':val.fu,'calculatedHanWithoutSituationalYaku':val.han,
                            'originalFuMatches':bool(fu_match),'riichiDiscardIndex':ri,
                            'distinctTileTypes':len(counts)}
                        wins.append(rec)
                    except (ValueError,IndexError,TypeError) as exc:
                        issues.append({**base,'error':str(exc)})
    overall=rate(wins); no_kan=rate([r for r in wins if r['indicatorCount']==1])
    if used_corrections!=set(corrections): raise ValueError('unmatched correction rows')
    seasons={s:rate([r for r in wins if r['season']==s]) for s in sorted({r['season'] for r in wins})}
    roles=[]; all_hit=overall['hit']
    for y in sorted({y for r in wins for y in r['yaku']}):
        subset=[r for r in wins if y in r['yaku']]
        q=rate(subset)
        roles.append({'yaku':y,**q,'noKan':rate([r for r in subset if r['indicatorCount']==1]),
            'shareAmongHits':q['hit']/all_hit if all_hit else None,
            'meanDistinctTileTypes':sum(r['distinctTileTypes'] for r in subset)/len(subset)})
    summary={'acquired':'2026-10-12','definition':'P(uraCount >= 1 | riichi win, ura indicator disclosed)',
        'yakuMethod':'mahjong 2.0.0 maximum score decomposition, with riichi/tsumo and no ippatsu/haitei/houtei/rinshan/chankan/double-riichi flags; structural roles reconstructed, not source labels',
        'inputs':inputs,'correctionsUsed':len(used_corrections),'matches':len(match_keys),'rounds':round_count,'allWins':all_wins,
        'dateRange':[min(dates),max(dates)],'overall':overall,'oneIndicator':no_kan,
        'twoOrMoreIndicators':rate([r for r in wins if r['indicatorCount']>1]),
        'nonYakuman':rate([r for r in wins if not r['yakuman']]),
        'unknownBounds':[overall['hit']/len(wins),(overall['hit']+overall['missing'])/len(wins)],
        'bySeason':seasons,'byYaku':roles,
        'yakuComparisons':compare_yaku_by_match(wins,match_keys),
        'byDistinctTypes':{str(k):rate([r for r in wins if r['distinctTileTypes']==k and r['indicatorCount']==1]) for k in sorted({r['distinctTileTypes'] for r in wins})},
        'quality':{'reconstructionFailures':len(issues),'multiDecompositionWins':sum(r['decompositions']>1 for r in wins),
            'fuMismatches':sum(not r['originalFuMatches'] for r in wins),
            'missingUraYakuman':sum(r['uraCount'] is None and r['yakuman'] for r in wins),
            'indicatorCountMismatch':sum(r['uraCount'] is not None and r['indicatorCount']!=r['doraIndicatorCount'] for r in wins)}}
    (OUT/'wins.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in wins),encoding='utf-8')
    for name,obj in [('summary.json',summary),('issues.json',issues)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['matches','rounds','overall','oneIndicator','quality']},ensure_ascii=False,indent=2))
    return 1 if issues else 0

if __name__=='__main__': raise SystemExit(main())
