"""裏表示の循環、赤牌、特殊形、Mリーグ固有符の境界検証。"""
import unittest
from collections import Counter
from analyze_mleague_ura_dora import tile34, successor, physical_tiles, evaluate, rate

class UraDoraTests(unittest.TestCase):
    def test_indicator_cycles(self):
        for raw, expected in [(19,11),(29,21),(39,31),(44,41),(47,45),(51,16)]:
            self.assertEqual(successor(tile34(raw)), tile34(expected))

    def test_red_is_same_type_but_distinct_physical_tile(self):
        self.assertEqual(tile34(51),tile34(15))
        self.assertEqual(len(set(physical_tiles([15,15,15,51]))),4)
        with self.assertRaises(ValueError): physical_tiles([15]*4)

    def test_missing_is_not_zero(self):
        r=rate([{'uraCount':None},{'uraCount':0},{'uraCount':2}])
        self.assertEqual((r['known'],r['missing'],r['hit'],r['rate']),(2,1,1,0.5))

    def test_chiitoitsu_ura_two(self):
        hand=[11,11,14,14,18,18,23,23,27,27,36,36,45,45]
        log=[[0,0,0],[],[12],[19]]
        roles,v,_,_=evaluate(hand,[],45,log,1,False,2)
        self.assertIn('七対子',roles); self.assertEqual(v.fu,25)

    def test_ryanpeikou_beats_seven_pairs(self):
        hand=[12,12,13,13,14,14,26,26,27,27,28,28,33,33]
        log=[[0,0,0],[],[19],[19]]
        roles,_,_,n=evaluate(hand,[],14,log,1,False,0)
        self.assertGreater(n,1); self.assertIn('二盃口',roles); self.assertNotIn('七対子',roles)

    def test_double_wind_pair_is_two_fu(self):
        # 2022-11-25 A-2 南2局。南家の連風対子を2符にして40符。
        hand=[11,11,11,16,17,24,25,26,27,28,29,42,42,51]
        log=[[5,0,0],[],[19],[35]]
        _,v,_,_=evaluate(hand,[],29,log,2,False,0)
        self.assertEqual(v.fu,40)

    def test_closed_kan_counts_four_ura_tiles(self):
        hand=[11,11,11,11,22,23,24,26,27,28,35,36,37,45,45]
        log=[[0,0,0],[],[15,25],[19,16]]
        roles,_,_,_=evaluate(hand,[[11]*4],37,log,1,True,4)
        self.assertIn('門前清自摸和',roles)

if __name__=='__main__': unittest.main()
