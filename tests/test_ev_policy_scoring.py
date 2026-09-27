from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "calibration" / ".dependency-cache" / "mahjong-2.0.0-py3-none-any.whl"
if WHEEL.is_file():
    sys.path.insert(0, str(WHEEL))
sys.path.insert(0, str(ROOT))

from mahjong.constants import EAST, SOUTH, WEST  # noqa: E402
from mahjong.tile import TilesConverter  # noqa: E402

from tools.ev_policy_scoring import (  # noqa: E402
    BaseScore,
    HandFlags,
    HandScoreRequest,
    HandScoringError,
    MeldInput,
    MultiplePaoHonbaUnresolved,
    PaoMeld,
    PaoResponsibility,
    calculate_base_score,
    detect_pao,
    score_hand,
    settle_win,
    to_mahjong_136,
)
from tools.ev_policy_state import TileInstance, canonical_tile_set  # noqa: E402


class TilePool:
    def __init__(self) -> None:
        self.available = list(canonical_tile_set())

    def hand(self, *, man: str = "", pin: str = "", sou: str = "", honors: str = "") -> tuple[TileInstance, ...]:
        library_tiles = TilesConverter.string_to_136_array(
            man=man, pin=pin, sou=sou, honors=honors, has_aka_dora=True
        )
        result = []
        for library_id in library_tiles:
            tile34 = library_id // 4
            is_red = library_id in (16, 52, 88)
            tile = next(item for item in self.available if item.tile34 == tile34 and item.is_red == is_red)
            self.available.remove(tile)
            result.append(tile)
        return tuple(result)

    def one(self, tile34: int, *, red: bool = False) -> TileInstance:
        tile = next(item for item in self.available if item.tile34 == tile34 and item.is_red == red)
        self.available.remove(tile)
        return tile


class PolicyScoringTest(unittest.TestCase):
    def test_dependency_lock_matches_verified_wheel(self) -> None:
        self.assertTrue(WHEEL.is_file(), "選定済みwheelがない")
        digest = hashlib.sha256(WHEEL.read_bytes()).hexdigest()
        self.assertEqual(digest, "52f63a8dfda5a9e81542642ed315f0e1e8626c0d324c8d8eabb3418ae7d43870")
        requirements = (ROOT / "calibration" / "requirements-scoring.txt").read_text(encoding="utf-8")
        self.assertIn("mahjong==2.0.0", requirements)
        self.assertIn(f"sha256:{digest}", requirements)

    def test_tile_conversion_keeps_normal_and_red_fives_distinct(self) -> None:
        pool = TilePool()
        tiles = pool.hand(man="50", pin="50", sou="50")
        converted = to_mahjong_136(tiles)
        self.assertEqual(sum(tile in (16, 52, 88) for tile in converted), 3)
        for tile34, red_id in ((4, 16), (13, 52), (22, 88)):
            matching = [tile for source, tile in zip(tiles, converted) if source.tile34 == tile34]
            self.assertEqual(len(matching), 2)
            self.assertIn(red_id, matching)
            self.assertEqual(sum(tile == red_id for tile in matching), 1)

        invalid = tuple(TileInstance(index, 4, False) for index in range(4))
        with self.assertRaisesRegex(HandScoringError, "red_five_composition_invalid"):
            to_mahjong_136(invalid)

    def test_official_limit_boundaries(self) -> None:
        self.assertEqual(calculate_base_score(han=4, fu=30, is_tsumo=False, is_dealer=False).main, 8000)
        self.assertEqual(calculate_base_score(han=3, fu=60, is_tsumo=False, is_dealer=False).main, 8000)
        self.assertEqual(calculate_base_score(han=13, fu=30, is_tsumo=False, is_dealer=False).main, 24000)
        pinfu_tsumo = calculate_base_score(han=4, fu=20, is_tsumo=True, is_dealer=False)
        self.assertEqual((pinfu_tsumo.main, pinfu_tsumo.additional), (2600, 1300))

    def test_double_wind_pair_is_two_fu_before_rounding(self) -> None:
        pool = TilePool()
        tiles = pool.hand(man="111234", pin="567", sou="789", honors="11")
        win = next(tile for tile in tiles if tile.tile34 == 26)
        result = score_hand(
            HandScoreRequest(
                tiles=tiles,
                win_tile_id=win.tile_id,
                seat_wind_tile34=EAST,
                round_wind_tile34=EAST,
                flags=HandFlags(is_riichi=True),
            )
        )
        self.assertEqual((result.base.han, result.base.fu, result.base.main), (1, 40, 2000))
        self.assertTrue(result.double_wind_pair_normalized)
        self.assertIsNone(result.scoring_round_wind_tile34)

    def test_seven_pairs_and_single_yakuman(self) -> None:
        pool = TilePool()
        tiles = pool.hand(man="1122", pin="3344", sou="5566", honors="11")
        result = score_hand(
            HandScoreRequest(tiles, tiles[-1].tile_id, SOUTH, EAST)
        )
        self.assertEqual((result.base.han, result.base.fu, result.base.main), (2, 25, 1600))

        pool = TilePool()
        tiles = pool.hand(man="19", pin="19", sou="19", honors="11234567")
        duplicate = next(tile for tile in tiles if sum(other.tile34 == tile.tile34 for other in tiles) == 2)
        result = score_hand(HandScoreRequest(tiles, duplicate.tile_id, SOUTH, EAST))
        self.assertEqual((result.base.yakuman_multiplier, result.base.main), (1, 32000))

        pool = TilePool()
        tiles = pool.hand(honors="11122555666777")
        result = score_hand(HandScoreRequest(tiles, tiles[-1].tile_id, SOUTH, EAST))
        self.assertEqual(result.base.yakuman_multiplier, 2)
        self.assertEqual(result.base.main, 64000)
        self.assertEqual({item.name for item in result.yaku}, {"Daisangen", "Tsuu Iisou"})

    def test_open_tanyao_meld_is_scored_through_adapter(self) -> None:
        pool = TilePool()
        tiles = pool.hand(man="234", pin="22234", sou="345678")
        meld_tiles = tuple(tile.tile_id for tile in tiles if tile.tile34 in (1, 2, 3))
        win = next(tile for tile in tiles if tile.tile34 == 25)
        result = score_hand(
            HandScoreRequest(
                tiles,
                win.tile_id,
                SOUTH,
                EAST,
                melds=(MeldInput("chi", meld_tiles, True, meld_tiles[0]),),
            )
        )
        self.assertTrue(result.is_open_hand)
        self.assertEqual({item.name for item in result.yaku}, {"Tanyao"})
        self.assertEqual((result.base.han, result.base.fu, result.base.main), (1, 30, 1000))

    def test_chi_meld_is_scored_regardless_of_called_tile_position(self) -> None:
        # D.3.2b：実行器は鳴いた牌を面子の末尾に置く。mahjong 2.0.0は面子の先頭牌を
        # 順子の開始牌とみなすため、4萬を56萬で鳴いた[5,6,4]の順では和了形を認めなかった。
        for order in ((0, 1, 2), (1, 2, 0), (0, 2, 1), (2, 1, 0)):
            with self.subTest(order=order):
                pool = TilePool()
                tiles = pool.hand(man="456", pin="789", sou="11678", honors="777")
                chi = [tile.tile_id for tile in tiles if tile.tile34 in (3, 4, 5)]
                chi_ids = tuple(chi[index] for index in order)
                pon_ids = tuple(tile.tile_id for tile in tiles if tile.tile34 == 33)
                win = next(tile for tile in tiles if tile.tile34 == 23)
                result = score_hand(
                    HandScoreRequest(
                        tiles,
                        win.tile_id,
                        SOUTH,
                        EAST,
                        melds=(
                            MeldInput("chi", chi_ids, True, chi_ids[-1]),
                            MeldInput("pon", pon_ids, True, pon_ids[-1]),
                        ),
                    )
                )
                self.assertEqual({item.name for item in result.yaku}, {"Yakuhai (chun)"})
                self.assertEqual((result.base.han, result.base.fu, result.base.main), (1, 30, 1000))

    def test_no_yaku_is_not_silently_scored(self) -> None:
        pool = TilePool()
        tiles = pool.hand(man="123456", pin="789", sou="345", honors="11")
        win = next(tile for tile in tiles if tile.tile34 == 22)
        with self.assertRaisesRegex(HandScoringError, "no_yaku"):
            score_hand(HandScoreRequest(tiles, win.tile_id, SOUTH, EAST))

    def test_normal_ron_and_tsumo_settlement_add_bonus_once(self) -> None:
        ron = calculate_base_score(han=4, fu=30, is_tsumo=False, is_dealer=False)
        settlement = settle_win(ron, winner=1, dealer=0, loser=2, honba=2, kyoutaku=3)
        self.assertEqual(settlement.deltas, (0, 11600, -8600, 0))
        self.assertEqual((settlement.kyoutaku_before, settlement.kyoutaku_after), (3, 0))

        tsumo = calculate_base_score(han=5, fu=30, is_tsumo=True, is_dealer=False)
        settlement = settle_win(tsumo, winner=1, dealer=0, honba=1)
        self.assertEqual(settlement.deltas, (-4100, 8300, -2100, -2100))

        forged = BaseScore(1, 30, 1000, 0, 2000, "", False, False)
        with self.assertRaisesRegex(ValueError, "合計"):
            settle_win(forged, winner=1, dealer=0, loser=2)

    def test_scoring_rejects_one_physical_tile_in_two_zones(self) -> None:
        pool = TilePool()
        tiles = pool.hand(man="111234", pin="567", sou="789", honors="11")
        win = next(tile for tile in tiles if tile.tile34 == 26)
        with self.assertRaisesRegex(HandScoringError, "indicator_tile_in_hand"):
            score_hand(
                HandScoreRequest(
                    tiles,
                    win.tile_id,
                    EAST,
                    EAST,
                    dora_indicators=(tiles[0],),
                    flags=HandFlags(is_riichi=True),
                )
            )

    def test_pao_detection_uses_the_confirming_call(self) -> None:
        dragons = (PaoMeld("pon", 31, 0), PaoMeld("ankan", 32))
        self.assertEqual(
            detect_pao(dragons, PaoMeld("daiminkan", 33, 3)),
            (PaoResponsibility("Daisangen", 3),),
        )
        winds = (PaoMeld("ankan", 27), PaoMeld("pon", 28, 0), PaoMeld("shouminkan", 29))
        self.assertEqual(
            detect_pao(winds, PaoMeld("pon", 30, 2)),
            (PaoResponsibility("Daisuushi", 2),),
        )
        kans = (PaoMeld("ankan", 1), PaoMeld("shouminkan", 2), PaoMeld("daiminkan", 3, 0))
        self.assertEqual(
            detect_pao(kans, PaoMeld("daiminkan", 4, 2)),
            (PaoResponsibility("Suukantsu", 2),),
        )
        self.assertEqual(detect_pao(dragons, PaoMeld("ankan", 33)), ())
        with self.assertRaisesRegex(ValueError, "重複"):
            detect_pao((PaoMeld("pon", 31, 0),), PaoMeld("daiminkan", 31, 2))

    def test_official_pao_compound_yakuman_example(self) -> None:
        score = calculate_base_score(han=26, fu=0, is_tsumo=True, is_dealer=False, is_yakuman=True)
        settlement = settle_win(
            score,
            winner=1,
            dealer=0,
            honba=1,
            pao=(PaoResponsibility("Daisangen", 2),),
        )
        self.assertEqual(settlement.deltas, (-16000, 64300, -40300, -8000))
        self.assertEqual(sum(settlement.deltas), 0)

    def test_third_party_ron_splits_pao_component(self) -> None:
        score = calculate_base_score(han=13, fu=0, is_tsumo=False, is_dealer=False, is_yakuman=True)
        settlement = settle_win(
            score,
            winner=1,
            dealer=0,
            loser=3,
            pao=(PaoResponsibility("Daisangen", 2),),
        )
        self.assertEqual(settlement.deltas, (0, 32000, -16000, -16000))

    def test_multiple_pao_honba_is_held(self) -> None:
        score = calculate_base_score(han=26, fu=0, is_tsumo=True, is_dealer=False, is_yakuman=True)
        with self.assertRaisesRegex(MultiplePaoHonbaUnresolved, "multiple_pao_honba_unresolved"):
            settle_win(
                score,
                winner=1,
                dealer=0,
                honba=1,
                pao=(PaoResponsibility("Daisangen", 2), PaoResponsibility("Daisuushi", 3)),
            )


if __name__ == "__main__":
    unittest.main()
