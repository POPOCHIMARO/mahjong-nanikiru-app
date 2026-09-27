from __future__ import annotations

import random
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ev_calibration_state import shanten  # noqa: E402
from tools.ev_policy_features import (  # noqa: E402
    SAFETY_GROUPS,
    MeldView,
    RiichiOpponentView,
    classify_danger_v3,
    clear_shape_cache,
    danger_features,
    exact_shape,
    make_action_view,
    melds_after_action,
    project_action,
    sha256_file,
    shape_cache_info,
    yaku_shape_features,
)
from tools.ev_policy_opponent import (  # noqa: E402
    DETAIL_FEATURE_NAMES,
    RoundFeatureState,
    build_opponent_feature_cache,
    evaluate_opponent_model,
    fit_opponent_model,
    iter_cached_windows,
    iter_encoded_windows,
    verify_opponent_feature_cache,
)


def counts(*tiles: int) -> tuple[int, ...]:
    result = [0] * 34
    for tile34 in tiles:
        result[tile34] += 1
    return tuple(result)


def tile(tile34: int, red: bool = False) -> dict[str, object]:
    return {"tile34": tile34, "isRed": red}


TENPAI_13 = counts(0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28)
OPEN_TENPAI_10 = counts(0, 1, 2, 9, 10, 11, 18, 27, 27, 27)


class ExactShapeTests(unittest.TestCase):
    def test_d32a_01_fixed_normal_chiitoi_kokushi_and_open_shapes(self) -> None:
        normal = exact_shape(TENPAI_13, 0)
        self.assertEqual((normal.shanten, normal.improving_mask), (0, 1 << 28))

        chiitoi = exact_shape(counts(0, 0, 1, 1, 2, 2, 9, 9, 10, 10, 11, 11, 18), 0)
        self.assertEqual((chiitoi.shanten, chiitoi.improving_mask), (0, 1 << 18))

        kokushi_tiles = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
        kokushi = exact_shape(counts(*kokushi_tiles), 0)
        expected_mask = sum(1 << value for value in kokushi_tiles)
        self.assertEqual((kokushi.shanten, kokushi.improving_mask), (0, expected_mask))

        open_shape = exact_shape(OPEN_TENPAI_10, 1)
        self.assertEqual((open_shape.shanten, open_shape.improving_mask), (0, 1 << 18))
        four_melds = exact_shape(counts(31), 4)
        self.assertEqual((four_melds.shanten, four_melds.improving_mask), (0, 1 << 31))

    def test_d32a_01_seeded_reference_enumeration_matches_mask(self) -> None:
        rng = random.Random(20260909)
        for _ in range(10_000):
            fixed_melds = rng.randrange(5)
            target = 13 - 3 * fixed_melds
            work = [0] * 34
            while sum(work) < target:
                index = rng.randrange(34)
                if work[index] < 4:
                    work[index] += 1
            value = exact_shape(tuple(work), fixed_melds)
            self.assertEqual(value.shanten, shanten(tuple(work), fixed_melds))
            expected = 0
            for index in range(34):
                if work[index] == 4:
                    continue
                added = work.copy()
                added[index] += 1
                if shanten(tuple(added), fixed_melds) < value.shanten:
                    expected |= 1 << index
            self.assertEqual(value.improving_mask, expected)

    def test_d32a_01_shape_cache_is_bounded_and_reused(self) -> None:
        clear_shape_cache()
        exact_shape(TENPAI_13, 0)
        first = shape_cache_info()
        exact_shape(TENPAI_13, 0)
        second = shape_cache_info()
        self.assertEqual(first.misses, 1)
        self.assertEqual(second.hits, 1)
        self.assertEqual(second.maxsize, 100_000)


class ActionProjectionTests(unittest.TestCase):
    def test_d32a_02_discard_does_not_restore_remaining_and_zero_visible_is_removed(self) -> None:
        hand14 = list(TENPAI_13)
        hand14[29] += 1
        visible = [0] * 34
        visible[28] = 2
        result = project_action(
            make_action_view(hand14, 0, visible),
            {"kind": "discard", "tile34": 29, "isRed": False, "origin": "drawn"},
        )
        self.assertEqual((result["shanten"], result["ukeireKinds"], result["ukeireCount"]), (0, 1, 1))
        self.assertEqual(result["visible34"][29], 1)

        visible[28] = 3
        no_remaining = project_action(
            make_action_view(hand14, 0, visible),
            {"kind": "discard", "tile34": 29, "isRed": False, "origin": "drawn"},
        )
        self.assertEqual((no_remaining["ukeireKinds"], no_remaining["ukeireCount"]), (0, 0))

    def test_d32a_02_rejects_invalid_inventory_counts_and_consumption(self) -> None:
        with self.assertRaisesRegex(ValueError, "34要素"):
            exact_shape((0,) * 33, 0)
        with self.assertRaisesRegex(ValueError, "牌数不一致"):
            exact_shape((0,) * 34, 0)
        hand14 = list(TENPAI_13)
        hand14[29] += 1
        visible = [0] * 34
        visible[0] = 4
        with self.assertRaisesRegex(ValueError, "4枚を超える"):
            make_action_view(hand14, 0, visible)
        with self.assertRaisesRegex(ValueError, "存在しない牌"):
            project_action(
                make_action_view(hand14, 0, [0] * 34),
                {"kind": "discard", "tile34": 33, "isRed": False, "origin": "concealed"},
            )

    def test_d32a_03_call_projection_counts_only_consumed_tiles_once(self) -> None:
        pre_pon = list(OPEN_TENPAI_10)
        pre_pon[4] += 2
        pre_pon[30] += 1
        visible = [0] * 34
        visible[4] = 1  # 鳴かれる捨牌はすでに公開済み。
        pon = project_action(
            make_action_view(pre_pon, 0, visible, called_tile34=4),
            {"kind": "pon", "consumed": [tile(4), tile(4)]},
        )
        self.assertEqual(sum(pon["counts34"]), 10)
        self.assertEqual(pon["visible34"][4], 3)
        self.assertNotEqual(pon["followupTile34"], 4)

        pre_chi = list(OPEN_TENPAI_10)
        pre_chi[1] += 1
        pre_chi[2] += 1
        pre_chi[30] += 1
        chi = project_action(
            make_action_view(pre_chi, 0, [0] * 34, called_tile34=0),
            {"kind": "chi", "consumed": [tile(1), tile(2)]},
        )
        self.assertNotIn(chi["followupTile34"], {0, 3})

    def test_d32a_03_pass_kans_and_terminal_actions_follow_contract(self) -> None:
        passed = project_action(make_action_view(TENPAI_13, 0, [0] * 34), {"kind": "pass"})
        self.assertEqual((passed["shanten"], passed["applicable"]), (0, 1))

        daiminkan_hand = list(OPEN_TENPAI_10)
        daiminkan_hand[33] += 3
        daiminkan_visible = [0] * 34
        daiminkan_visible[33] = 1
        daiminkan = project_action(
            make_action_view(daiminkan_hand, 0, daiminkan_visible, called_tile34=33),
            {"kind": "daiminkan", "consumed": [tile(33), tile(33), tile(33)]},
        )
        self.assertEqual((sum(daiminkan["counts34"]), daiminkan["visible34"][33]), (10, 4))

        ankan_hand = list(OPEN_TENPAI_10)
        ankan_hand[33] += 4
        ankan = project_action(
            make_action_view(ankan_hand, 0, [0] * 34),
            {"kind": "ankan", "tile34": 33, "redCount": 0},
        )
        self.assertEqual((sum(ankan["counts34"]), ankan["visible34"][33]), (10, 4))

        kakan_hand = list(OPEN_TENPAI_10)
        kakan_hand[33] += 1
        kakan = project_action(
            make_action_view(kakan_hand, 1, [0] * 34),
            {"kind": "kakan", "tile34": 33, "addedIsRed": False},
        )
        self.assertEqual((sum(kakan["counts34"]), kakan["visible34"][33]), (10, 1))

        for kind in ("ron", "tsumo"):
            terminal = project_action(make_action_view(TENPAI_13, 0, [0] * 34), {"kind": kind})
            self.assertEqual(
                (terminal["shanten"], terminal["ukeireCount"], terminal["ukeireKinds"], terminal["applicable"]),
                (-1, 0, 0, 0),
            )

    def test_d32a_03_empty_legal_followup_is_reported_as_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "合法な打牌がない"):
            project_action(
                make_action_view(counts(0, 1, 2, 3), 3, [0] * 34, called_tile34=0),
                {"kind": "chi", "consumed": [tile(1), tile(2)]},
            )


class FeatureAdapterTests(unittest.TestCase):
    def test_d32a_04_other_private_hands_and_future_events_do_not_change_features(self) -> None:
        actor_hand = [tile(index) for index, amount in enumerate(TENPAI_13) for _ in range(amount)]
        future_a = tile(5)
        future_b = tile(6)
        public_a = {
            "initial": {
                "dealerSeat": 0,
                "doraIndicator": tile(33),
                "honba": 0,
                "kyoku": 0,
                "riichiSticks": 0,
                "scores": [25_000] * 4,
            },
            "events": [
                {"type": "draw", "seat": 0, "rawEventIndex": 0, "source": "live"},
                {"type": "draw", "seat": 1, "rawEventIndex": 1, "source": "live"},
            ],
        }
        public_b = {**public_a, "events": [public_a["events"][0], {**public_a["events"][1], "source": "rinshan"}]}
        private_a = {
            seat: {
                "initialHand": actor_hand if seat == 0 else [tile((seat * 7 + index) % 33) for index in range(13)],
                "events": ([{"type": "draw_observation", "rawEventIndex": 0, "tile": tile(29)}]
                           if seat == 0 else [{"type": "draw_observation", "rawEventIndex": 1, "tile": future_a}]),
            }
            for seat in range(4)
        }
        private_b = {
            seat: {
                "initialHand": private_a[seat]["initialHand"] if seat == 0 else list(reversed(private_a[seat]["initialHand"])),
                "events": ([{"type": "draw_observation", "rawEventIndex": 0, "tile": tile(29)}]
                           if seat == 0 else [{"type": "draw_observation", "rawEventIndex": 1, "tile": future_b}]),
            }
            for seat in range(4)
        }
        action = {"kind": "discard", "tile34": 29, "isRed": False, "origin": "drawn"}
        left = RoundFeatureState(public_a, private_a)
        right = RoundFeatureState(public_b, private_b)
        left.advance(1)
        right.advance(1)
        left_candidate = left.encode(0, [action])[0]
        right_candidate = right.encode(0, [action])[0]
        self.assertEqual(left_candidate.kind_features.tolist(), right_candidate.kind_features.tolist())
        self.assertEqual(left_candidate.detail_features.tolist(), right_candidate.detail_features.tolist())


def _public(events: list[dict[str, object]], dealer: int = 0) -> dict[str, object]:
    return {
        "initial": {
            "dealerSeat": dealer,
            "doraIndicator": tile(33),
            "honba": 0,
            "kyoku": 0,
            "riichiSticks": 0,
            "scores": [25_000] * 4,
        },
        "events": events,
    }


def _private(hands: dict[int, list[int]], draws: dict[int, list[tuple[int, int]]] | None = None) -> dict[int, dict]:
    draws = draws or {}
    return {
        seat: {
            "initialHand": [tile(value) for value in hands[seat]],
            "events": [
                {"type": "draw_observation", "rawEventIndex": index, "tile": tile(value)}
                for index, value in draws.get(seat, [])
            ],
        }
        for seat in range(4)
    }


def _discard(seat: int, tile34: int, riichi: bool = False) -> dict[str, object]:
    return {"type": "discard", "seat": seat, "tile": tile(tile34), "riichiDeclaration": riichi}


PASS = {"type": "response_resolution", "resolution": {"kind": "pass"}}


class DangerFeatureTests(unittest.TestCase):
    def test_d32b_02_safe_tiles_include_own_river_declaration_and_passed_tiles(self) -> None:
        # 席3が1萬、席1が2萬（リーチ前）と3萬（宣言牌）、席2が4萬（リーチ後）を捨てる。
        hands = {seat: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] for seat in range(4)}
        hands[1] = [1, 2] + hands[1][2:]
        hands[2] = [3] + hands[2][1:]
        hands[3] = [0] + hands[3][1:]
        events = [
            _discard(3, 0), PASS,
            _discard(1, 1), PASS,
            _discard(1, 2, riichi=True), PASS,
            _discard(2, 3), PASS,
        ]
        state = RoundFeatureState(_public(events), _private(hands))
        state.advance(len(events))
        self.assertEqual(state.safe_tiles(1), frozenset({1, 2, 3}))
        # リーチ前に他家だけが捨てた1萬は、席1に対する安全牌ではない。
        self.assertNotIn(0, state.safe_tiles(1))

    def test_d32b_02_called_discard_counts_as_passed_but_pending_one_does_not(self) -> None:
        hands = {seat: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] for seat in range(4)}
        hands[1] = [1] + hands[1][1:]
        hands[2] = [5] + hands[2][1:]
        hands[3] = [3, 4] + hands[3][2:]
        chi = {"type": "chi", "seat": 3, "fromSeat": 2, "tiles": [tile(3), tile(4), tile(5)]}
        events = [_discard(1, 1, riichi=True), PASS, _discard(2, 5), chi]
        state = RoundFeatureState(_public(events), _private(hands))
        state.advance(3)
        # 応答が解決する前の捨牌は、まだ見逃されていない。
        self.assertNotIn(5, state.safe_tiles(1))
        state.advance(4)
        self.assertIn(5, state.safe_tiles(1))
        self.assertEqual(state.meld_kinds[3], ["chi"])

    def test_d32b_03_two_riichi_aggregation_and_dealer_swap(self) -> None:
        seen = [0] * 34
        # 4萬（tile34=3）：席1には現物、席2には1萬スジ（suji_456、4.1%）。
        dealer_safe = RiichiOpponentView(1, True, frozenset({3}))
        child_suji = RiichiOpponentView(2, False, frozenset({0}))
        values = danger_features(3, [child_suji, dealer_safe], seen)
        self.assertEqual(values["danger_applicable"], 1.0)
        self.assertAlmostEqual(values["danger_max_rate"], 0.041 / 0.057)
        self.assertAlmostEqual(values["danger_sum_rate"], 0.041 / 0.171)
        self.assertEqual(values["danger_dealer_rate"], 0.0)
        self.assertEqual((values["genbutsu_all"], values["genbutsu_any"]), (0.0, 1.0))
        self.assertEqual(values["danger_group_guarded"], 1.0)
        self.assertEqual(sum(values[f"danger_group_{g}"] for g in SAFETY_GROUPS), 1.0)

        swapped = danger_features(
            3,
            [RiichiOpponentView(1, False, frozenset({3})), RiichiOpponentView(2, True, frozenset({0}))],
            seen,
        )
        self.assertAlmostEqual(swapped["danger_dealer_rate"], 0.041 / 0.057)
        for name in values:
            if name != "danger_dealer_rate":
                self.assertEqual(swapped[name], values[name], name)

        empty = danger_features(3, [], seen)
        self.assertTrue(all(value == 0.0 for value in empty.values()))
        self.assertTrue(all(value == 0.0 for value in danger_features(None, [dealer_safe], seen).values()))

    def test_d32b_03_classification_uses_safe_set_for_suji_and_seen_for_chance(self) -> None:
        seen = [0] * 34
        self.assertEqual(classify_danger_v3(6, frozenset({3}), seen), "suji_37")
        self.assertEqual(classify_danger_v3(4, frozenset({1, 7}), seen), "double_suji_middle")
        seen[1] = 4  # 2萬が4枚見えていれば1萬はノーチャンス
        self.assertEqual(classify_danger_v3(0, frozenset(), seen), "no_chance_19")
        seen[29] = 2
        self.assertEqual(classify_danger_v3(29, frozenset(), seen), "honor_2_visible")


def _yaku(concealed: tuple[int, ...], melds: list[MeldView], seat_wind: int = 28, round_wind: int = 27) -> dict[str, float]:
    return yaku_shape_features(concealed, melds, seat_wind, round_wind)


class YakuShapeFeatureTests(unittest.TestCase):
    def test_d32b_04_listed_cues_match_hand_calculation(self) -> None:
        # 役牌ポン済み（中）
        values = _yaku(counts(0, 1, 2, 9, 10, 11, 18, 19, 20, 30), [MeldView("pon", (33, 33, 33))])
        self.assertEqual((values["yakuhai_secured"], values["open_no_listed_yaku_cue"]), (1.0, 0.0))
        # 食いタン形：234萬チー、手中は中張牌だけ
        tanyao = _yaku(counts(10, 11, 12, 13, 14, 15, 19, 20, 21, 22), [MeldView("chi", (1, 2, 3))])
        self.assertEqual((tanyao["tanyao_path"], tanyao["tanyao_distance"]), (1.0, 0.0))
        # 混一色形：123萬チー、手中は萬子と字牌
        flush = _yaku(counts(3, 4, 5, 6, 7, 8, 27, 27, 29, 29), [MeldView("chi", (0, 1, 2))])
        self.assertEqual((flush["flush_path"], flush["flush_distance"]), (1.0, 0.0))

    def test_d32b_04_flush_distance_follows_fixed_meld_suit(self) -> None:
        # 固定面子123萬、手中22334455筒＋西西。手中の最多色（筒子）ではなく萬子を対象にする。
        values = _yaku(counts(10, 10, 11, 11, 12, 12, 13, 13, 29, 29), [MeldView("chi", (0, 1, 2))])
        self.assertEqual(values["flush_path"], 1.0)
        self.assertAlmostEqual(values["flush_distance"], 8 / 14)
        two_suits = _yaku(counts(10, 11, 12, 13), [MeldView("chi", (0, 1, 2)), MeldView("chi", (18, 19, 20)), MeldView("pon", (29, 29, 29))])
        self.assertEqual((two_suits["flush_path"], two_suits["flush_distance"]), (0.0, 1.0))

    def test_d32b_04_toitoi_boundary_and_chi_breaks_path(self) -> None:
        pon = [MeldView("pon", (3, 3, 3))]
        # m=1：刻子と対子の合計が 4-1=3 なら経路あり、2 なら経路なし。
        self.assertEqual(_yaku(counts(9, 9, 18, 18, 20, 20, 22, 24, 26, 30), pon)["toitoi_path"], 1.0)
        self.assertEqual(_yaku(counts(9, 9, 18, 18, 19, 20, 22, 24, 26, 30), pon)["toitoi_path"], 0.0)
        with_chi = pon + [MeldView("chi", (12, 13, 14))]
        self.assertEqual(_yaku(counts(9, 9, 18, 18, 20, 20, 30), with_chi)["toitoi_path"], 0.0)

    def test_d32b_04_open_without_listed_cue_and_each_path_clears_it(self) -> None:
        # 123萬チーと789筒チー：么九牌を含み二色。手中に役牌対子も刻子・対子の組もない。
        melds = [MeldView("chi", (0, 1, 2)), MeldView("chi", (15, 16, 17))]
        base = _yaku(counts(18, 19, 21, 23, 25, 26, 30), melds)
        self.assertEqual(base["open_no_listed_yaku_cue"], 1.0)
        # 役牌の対子（中）があれば手掛かりあり。
        self.assertEqual(_yaku(counts(18, 19, 21, 23, 25, 33, 33), melds)["open_no_listed_yaku_cue"], 0.0)
        # 么九牌を含まない二色の副露なら食いタンの経路あり。
        simple_melds = [MeldView("chi", (1, 2, 3)), MeldView("chi", (12, 13, 14))]
        self.assertEqual(_yaku(counts(18, 19, 21, 23, 25, 26, 30), simple_melds)["open_no_listed_yaku_cue"], 0.0)
        # 一色の副露なら混一色の経路あり。
        one_suit = [MeldView("chi", (0, 1, 2)), MeldView("chi", (6, 7, 8))]
        self.assertEqual(_yaku(counts(18, 19, 21, 23, 25, 26, 30), one_suit)["open_no_listed_yaku_cue"], 0.0)
        # ポンだけの副露で刻子・対子が揃えば対々の経路あり。
        pons = [MeldView("pon", (0, 0, 0)), MeldView("pon", (15, 15, 15))]
        self.assertEqual(_yaku(counts(18, 18, 21, 21, 26, 26, 30), pons)["open_no_listed_yaku_cue"], 0.0)

    def test_d32b_04_ankan_keeps_menzen_but_disables_chiitoi(self) -> None:
        values = _yaku(counts(9, 10, 11, 18, 19, 20, 22, 22, 30, 30), [MeldView("ankan", (4, 4, 4, 4))])
        self.assertEqual((values["menzen_after"], values["chiitoi_applicable"], values["chiitoi_shanten"]), (1.0, 0.0, 0.0))
        self.assertEqual(values["open_no_listed_yaku_cue"], 0.0)
        closed = _yaku(counts(0, 0, 1, 1, 2, 2, 9, 9, 10, 10, 11, 11, 18), [])
        self.assertEqual((closed["chiitoi_applicable"], closed["chiitoi_shanten"]), (1.0, 0.0))

    def test_d32b_04_melds_after_call_and_kakan(self) -> None:
        melds = (MeldView("pon", (5, 5, 5)),)
        after_chi = melds_after_action(melds, {"kind": "chi", "consumed": [tile(1), tile(2)]}, 0)
        self.assertEqual(after_chi[-1], MeldView("chi", (0, 1, 2)))
        after_kakan = melds_after_action(melds, {"kind": "kakan", "tile34": 5}, None)
        self.assertEqual(after_kakan, (MeldView("kakan", (5, 5, 5, 5)),))
        self.assertEqual(melds, (MeldView("pon", (5, 5, 5)),))


class V3InformationBoundaryTests(unittest.TestCase):
    def test_d32b_05_riichi_danger_and_yaku_ignore_hidden_and_future_information(self) -> None:
        actor = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28]
        base_hands = {0: actor, 1: [3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 30, 31, 32],
                      2: [6, 7, 8, 15, 16, 17, 24, 25, 26, 29, 29, 31, 32],
                      3: [3, 4, 5, 12, 13, 14, 21, 22, 23, 29, 30, 31, 33]}
        other_hands = {**base_hands, 2: list(reversed(base_hands[2])), 3: [6, 7, 8, 15, 16, 17, 24, 25, 26, 29, 30, 31, 33]}
        events = [
            _discard(1, 3, riichi=True), PASS,
            {"type": "draw", "seat": 0, "rawEventIndex": 2, "source": "live"},
            {"type": "draw", "seat": 2, "rawEventIndex": 3, "source": "live"},
        ]
        left = RoundFeatureState(_public(events), _private(base_hands, {0: [(2, 29)], 2: [(3, 5)]}))
        right = RoundFeatureState(_public(events), _private(other_hands, {0: [(2, 29)], 2: [(3, 6)]}))
        left.advance(3)
        right.advance(3)
        actions = [{"kind": "discard", "tile34": value, "isRed": False, "origin": "concealed"} for value in (0, 9, 28)]
        actions.append({"kind": "discard", "tile34": 29, "isRed": False, "origin": "drawn"})
        for a, b in zip(left.encode(0, actions), right.encode(0, actions)):
            self.assertEqual(a.kind_features.tolist(), b.kind_features.tolist())
            self.assertEqual(a.detail_features.tolist(), b.detail_features.tolist())
        # リーチ者がいるので打牌候補の危険度は適用される。
        detail = dict(zip(DETAIL_FEATURE_NAMES, left.encode(0, actions)[0].detail_features))
        self.assertEqual(detail["danger_applicable"], 1.0)


class FeatureCacheTests(unittest.TestCase):
    def _write_gzip_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        public_rows = []
        private_rows = []
        teacher_rows = []
        splits = ("train", "selection", "calibration", "developmentConfirmation")
        actor_hand = [tile(index) for index, amount in enumerate(TENPAI_13) for _ in range(amount)]
        action = {"kind": "discard", "tile34": 29, "isRed": False, "origin": "drawn"}
        for index, split in enumerate(splits):
            round_id = f"fixture:{index}"
            public_rows.append({
                "roundId": round_id,
                "initial": {
                    "dealerSeat": 0,
                    "doraIndicator": tile(33),
                    "honba": 0,
                    "kyoku": index,
                    "riichiSticks": 0,
                    "scores": [25_000] * 4,
                },
                "events": [{"type": "draw", "seat": 0, "rawEventIndex": 0, "source": "live"}],
            })
            for seat in range(4):
                private_rows.append({
                    "roundId": round_id,
                    "seat": seat,
                    "initialHand": actor_hand if seat == 0 else [tile(value) for value in range(13)],
                    "events": ([{"type": "draw_observation", "rawEventIndex": 0, "tile": tile(29)}]
                               if seat == 0 else []),
                })
            teacher_rows.append({
                "windowId": f"{round_id}:window",
                "roundId": round_id,
                "developmentSplit": split,
                "phase": "self_action_after_live",
                "actorSeat": 0,
                "publicEventCount": 1,
                "legalActions": [action],
                "observation": {"status": "exact", "mass": 1.0, "action": action},
                "learningMask": {"kind": True, "conditionalDetail": True},
            })
        files = {
            "public-events.jsonl.gz": public_rows,
            "private-events.jsonl.gz": private_rows,
            "teacher-windows.jsonl.gz": teacher_rows,
        }
        generated = []
        for name, rows in files.items():
            path = dataset / name
            self._write_gzip_rows(path, rows)
            generated.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        summary = {
            "labelDefinition": {"version": "joint-public-resolution-v1"},
            "input": {
                "selectedSeasons": ["fixture"],
                "allowedSeasons": ["fixture"],
                "forbiddenSeasons": ["future"],
            },
            "generatedFiles": {"files": generated},
            "totals": {"teacherWindows": len(teacher_rows)},
        }
        (dataset / "extraction-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (dataset / "verification.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
        return dataset

    def test_d32a_05_cache_round_trip_matches_live_features(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            feature_dir = root / "features"
            manifest = build_opponent_feature_cache(dataset, feature_dir, windows_per_shard=2)
            self.assertEqual((manifest["status"], manifest["totalWindows"]), ("complete", 4))
            live = list(iter_encoded_windows(dataset))
            cached = list(iter_cached_windows(dataset, feature_dir))
            self.assertEqual([item["window"]["windowId"] for item in live], [item["window"]["windowId"] for item in cached])
            for left, right in zip(live, cached):
                self.assertEqual([value.action for value in left["candidates"]], [value.action for value in right["candidates"]])
                self.assertEqual(left["candidates"][0].kind_features.tolist(), right["candidates"][0].kind_features.tolist())
                self.assertEqual(left["candidates"][0].detail_features.tolist(), right["candidates"][0].detail_features.tolist())

    def test_d32a_06_interrupted_shards_resume_and_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            feature_dir = root / "features"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                build_opponent_feature_cache(
                    dataset, feature_dir, windows_per_shard=1, stop_after_shards=1
                )
            first_elapsed = json.loads((feature_dir / "build-state.json").read_text(encoding="utf-8"))["elapsedSeconds"]
            resumed = build_opponent_feature_cache(
                dataset, feature_dir, windows_per_shard=1, resume=True
            )
            self.assertEqual((resumed["status"], resumed["totalWindows"]), ("complete", 4))
            self.assertGreaterEqual(resumed["elapsedSeconds"], first_elapsed)
            verify_opponent_feature_cache(dataset, feature_dir)

            manifest_path = feature_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["calculationVersion"] = "wrong-version"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calculationVersion"):
                verify_opponent_feature_cache(dataset, feature_dir)

    def test_d32a_08_debug_fit_and_reload_evaluation_keep_metrics_and_holds(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            feature_dir = root / "features"
            model_dir = root / "model"
            build_opponent_feature_cache(dataset, feature_dir, maximum_windows=1, windows_per_shard=4)
            fitted = fit_opponent_model(dataset, feature_dir, model_dir, maximum_windows=1)
            evaluated = evaluate_opponent_model(dataset, feature_dir, model_dir, maximum_windows=1)
            self.assertEqual(fitted["metrics"], evaluated["metrics"])
            self.assertEqual(fitted["holds"], evaluated["holds"])
            self.assertFalse(fitted["eligibleForD33"])
            self.assertFalse(evaluated["eligibleForD33"])
            # D.3.2b：固定定数の往復結果とシナリオ一覧が保存され、モデルへ最終定数が入る。
            fixed = json.loads((model_dir / "fixed-components.json").read_text(encoding="utf-8"))
            self.assertEqual(set(fixed["roundTrips"]), {"initial", "afterFirstFit", "final"})
            self.assertEqual(fixed["estimationSplits"], ["calibration", "selection", "train"])
            self.assertGreaterEqual(fixed["scenarioCounts"]["D.3.3"], 16)
            self.assertEqual(fixed["scenarioCounts"]["D.3.4"], fixed["scenarioCounts"]["D.3.3"] + 1)
            model = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model["fixedConstants"], fixed["roundTrips"]["final"]["constants"])
            for value in model["fixedConstants"].values():
                self.assertTrue(0.0 < value < 1.0)


if __name__ == "__main__":
    unittest.main()
