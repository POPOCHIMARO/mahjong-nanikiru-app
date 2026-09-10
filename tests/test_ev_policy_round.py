from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "calibration" / ".dependency-cache" / "mahjong-2.0.0-py3-none-any.whl"
if WHEEL.is_file():
    sys.path.insert(0, str(WHEEL))
sys.path.insert(0, str(ROOT))

from mahjong.tile import TilesConverter  # noqa: E402

from tools.ev_policy_round import (  # noqa: E402
    PHASE_CALL_DISCARD,
    PHASE_CHANKAN_RESPONSES,
    PHASE_DRAW,
    PHASE_ENDED,
    PHASE_SELF_ACTION,
    IllegalAction,
    ResponseClaim,
    RiichiStatus,
    RiverTile,
    RoundMeld,
    RoundState,
    kuikae_forbidden_tile34,
    noten_deltas,
    riichi_ankan_legal,
    riichi_timing_legal,
    waiting_tile34,
)
from tools.ev_policy_scoring import HandScoringError, ScoringDependencyError  # noqa: E402
from tools.ev_policy_state import TileInstance, canonical_tile_set  # noqa: E402


class TilePool:
    def __init__(self) -> None:
        self.available = list(canonical_tile_set())

    def hand(self, *, man: str = "", pin: str = "", sou: str = "", honors: str = "") -> list[int]:
        library_tiles = TilesConverter.string_to_136_array(
            man=man, pin=pin, sou=sou, honors=honors, has_aka_dora=True
        )
        result = []
        for library_id in library_tiles:
            tile34 = library_id // 4
            is_red = library_id in (16, 52, 88)
            tile = next(item for item in self.available if item.tile34 == tile34 and item.is_red == is_red)
            self.available.remove(tile)
            result.append(tile.tile_id)
        return result

    def one(self, tile34: int, *, red: bool | None = None) -> int:
        tile = next(
            item
            for item in self.available
            if item.tile34 == tile34 and (red is None or item.is_red == red)
        )
        self.available.remove(tile)
        return tile.tile_id

    def fill(self, count: int) -> list[int]:
        result = [tile.tile_id for tile in self.available[:count]]
        del self.available[:count]
        return result


def make_state(
    specified_hands: dict[int, list[int]],
    pool: TilePool,
    *,
    live_front: tuple[int, ...] = (),
    rinshan: tuple[int, ...] = (),
    dealer_seat: int = 0,
    scores: tuple[int, int, int, int] = (25000, 25000, 25000, 25000),
) -> RoundState:
    hands = []
    for seat in range(4):
        hand = list(specified_hands.get(seat, []))
        hand.extend(pool.fill(13 - len(hand)))
        hands.append(hand)
    if len(rinshan) > 4:
        raise ValueError("rinshan fixtureは4枚まで")
    rinshan_ids = list(rinshan) + pool.fill(4 - len(rinshan))
    dead_wall = rinshan_ids + pool.fill(10)
    live_wall = list(live_front) + pool.fill(70 - len(live_front))
    if pool.available:
        raise AssertionError("fixtureに未配置牌がある")
    return RoundState.from_parts(
        hands,
        live_wall,
        dead_wall,
        scores=scores,
        dealer_seat=dealer_seat,
    )


def head_bump_state() -> tuple[RoundState, int]:
    pool = TilePool()
    hands = {
        0: pool.hand(man="123", pin="123", sou="123", honors="1112"),
        1: pool.hand(man="5", pin="123456789", honors="222"),
        2: pool.hand(man="5", sou="123456789", honors="333"),
        3: pool.hand(man="5123678789", honors="444"),
    }
    discard = pool.one(4)
    state = make_state(hands, pool, live_front=(discard,))
    self_draw = state.draw()
    assert self_draw == discard
    state.discard(discard, declare_riichi=True)
    return state, discard


def advance_without_claim(state: RoundState) -> None:
    if state.phase == PHASE_DRAW:
        state.draw()
    tile_id = state.drawn_tile_id
    if tile_id is None:
        tile_id = state.hands[state.turn_seat][0]
    state.discard(tile_id)
    state.resolve_discard_responses([])


class PolicyRoundTest(unittest.TestCase):
    def test_shouminkan_does_not_retain_pon_source_in_pao_scan(self) -> None:
        state = RoundState.shuffled(901)
        tile_ids = tuple(tile.tile_id for tile in state.tiles if tile.tile34 == 31)
        state.melds[0].append(RoundMeld("shouminkan", tile_ids, tile_ids[0], 2))

        pao_meld = state._pao_melds(0)[0]
        self.assertEqual((pao_meld.kind, pao_meld.tile34, pao_meld.from_seat), ("shouminkan", 31, None))

    def test_riichi_declaration_ron_does_not_take_deposit(self) -> None:
        state, _ = head_bump_state()
        result = state.resolve_discard_responses([ResponseClaim("ron", 1)])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual((result.kind, result.winner, result.loser), ("ron", 1, 0))
        self.assertEqual(state.kyoutaku, 0)
        self.assertNotIn(0, state.active_riichi)
        self.assertEqual(sum(state.scores), 100000)
        self.assertEqual(state.phase, PHASE_ENDED)

    def test_tsumo_uses_scoring_adapter_and_preserves_ledger(self) -> None:
        pool = TilePool()
        hand = pool.hand(man="5", pin="123456789", honors="111")
        winning_tile = pool.one(4)
        state = make_state({0: hand}, pool, live_front=(winning_tile,))
        state.draw()
        result = state.declare_tsumo()
        self.assertEqual((result.kind, result.winner, result.loser), ("tsumo", 0, None))
        self.assertGreater(result.score_deltas[0], 0)
        self.assertEqual(sum(state.scores) + 1000 * state.kyoutaku, 100000)
        self.assertEqual(state.phase, PHASE_ENDED)

    def test_head_bump_is_independent_of_claim_order(self) -> None:
        first, _ = head_bump_state()
        second, _ = head_bump_state()
        self.assertIn(ResponseClaim("ron", 1), first.legal_response_claims(1))
        result1 = first.resolve_discard_responses([ResponseClaim("ron", 3), ResponseClaim("ron", 1)])
        result2 = second.resolve_discard_responses([ResponseClaim("ron", 1), ResponseClaim("ron", 3)])
        assert result1 is not None and result2 is not None
        self.assertEqual(result1.winner, 1)
        self.assertEqual(result1.score_deltas, result2.score_deltas)
        self.assertNotIn(3, first.temporary_furiten)

    def test_riichi_commits_before_pon_and_call_clears_ippatsu(self) -> None:
        pool = TilePool()
        hands = {
            0: pool.hand(man="123", pin="123", sou="123", honors="1112"),
            2: pool.hand(man="55", pin="123456789", honors="22"),
        }
        discard = pool.one(4)
        state = make_state(hands, pool, live_front=(discard,))
        state.draw()
        state.discard(discard, declare_riichi=True)
        pon = next(claim for claim in state.legal_call_claims(2) if claim.kind == "pon")
        state.resolve_discard_responses([pon])
        self.assertEqual(state.scores[0], 24000)
        self.assertEqual(state.kyoutaku, 1)
        self.assertIn(0, state.active_riichi)
        self.assertEqual(state.ippatsu_eligible, set())
        self.assertEqual((state.turn_seat, state.phase), (2, PHASE_CALL_DISCARD))

    def test_riichi_timing_and_negative_score_follow_round_rules(self) -> None:
        self.assertTrue(riichi_timing_legal("live", False))
        self.assertFalse(riichi_timing_legal("live", True))
        self.assertTrue(riichi_timing_legal("rinshan", False))
        self.assertTrue(riichi_timing_legal("rinshan", True))

        pool = TilePool()
        hand = pool.hand(man="123", pin="123", sou="123", honors="1112")
        discard = pool.one(4)
        state = make_state(
            {0: hand},
            pool,
            live_front=(discard,),
            scores=(500, 33100, 33200, 33200),
        )
        state.draw()
        state.discard(discard, declare_riichi=True)
        state.resolve_discard_responses([])
        self.assertEqual((state.scores[0], state.kyoutaku), (-500, 1))
        self.assertEqual(sum(state.scores) + 1000 * state.kyoutaku, 100000)

    def test_call_priority_and_kuikae_use_tile_types(self) -> None:
        pool = TilePool()
        hands = {
            1: pool.hand(man="12", pin="123456789", honors="11"),
            2: pool.hand(man="333", sou="12345678", honors="22"),
        }
        discard = pool.one(2)
        state = make_state(hands, pool, live_front=(discard,))
        state.draw()
        state.discard(discard)
        chi = next(claim for claim in state.legal_call_claims(1) if claim.kind == "chi")
        pon = next(claim for claim in state.legal_call_claims(2) if claim.kind == "pon")
        state.resolve_discard_responses([chi, pon])
        self.assertEqual((state.turn_seat, state.melds[2][-1].kind), (2, "pon"))
        forbidden = next(iter(state.kuikae_forbidden))
        prohibited = next(tile_id for tile_id in state.hands[2] if state.tile_by_id[tile_id].tile34 == forbidden)
        with self.assertRaisesRegex(IllegalAction, "食い換え"):
            state.discard(prohibited)
        self.assertEqual(kuikae_forbidden_tile34("chi", 0, (1, 2)), frozenset({0, 3}))

    def test_temporary_furiten_survives_draw_and_clears_after_discard(self) -> None:
        state, _ = head_bump_state()
        state.resolve_discard_responses([])
        self.assertIn(1, state.temporary_furiten)
        state.draw()
        self.assertIn(1, state.temporary_furiten)
        state.discard(state.drawn_tile_id)  # type: ignore[arg-type]
        self.assertNotIn(1, state.temporary_furiten)

    def test_riichi_pass_furiten_does_not_clear_after_own_discard(self) -> None:
        state, _ = head_bump_state()
        state.active_riichi[1] = RiichiStatus(0, False, state._counts_in_hand(1))
        state.resolve_discard_responses([])
        self.assertIn(1, state.riichi_furiten)
        state.draw()
        state.discard(state.drawn_tile_id)  # type: ignore[arg-type]
        self.assertIn(1, state.riichi_furiten)

    def test_called_own_discard_still_causes_discard_furiten(self) -> None:
        pool = TilePool()
        waiting_hand = pool.hand(man="34", pin="123", sou="123", honors="11222")
        caller_hand = pool.hand(man="13", pin="123456789", honors="11")
        past_discard = pool.one(1)
        state = make_state({1: waiting_hand, 2: caller_hand}, pool, live_front=(past_discard,))
        state.live_wall.remove(past_discard)
        consumed = tuple(
            tile_id for tile_id in state.hands[2] if state.tile_by_id[tile_id].tile34 in (0, 2)
        )
        for tile_id in consumed:
            state.hands[2].remove(tile_id)
        caller_discard = state.hands[2].pop()
        state.rivers[1].append(RiverTile(past_discard, False, False, 0, called=True))
        state.rivers[2].append(RiverTile(caller_discard, False, False, 1))
        state.melds[2].append(RoundMeld("chi", (*consumed, past_discard), past_discard, 1))
        state.validate()
        self.assertEqual(waiting_tile34(state._counts_in_hand(1)), (1, 4))
        self.assertIn("own_discard", state.furiten_reasons(1))

    def test_waits_and_riichi_ankan_compare_all_decompositions(self) -> None:
        legal = tuple(TilesConverter.string_to_34_array(man="111", pin="234456", sou="789", honors="1"))
        self.assertEqual(waiting_tile34(legal), (27,))
        self.assertTrue(riichi_ankan_legal(legal, 0))

        chuuren = tuple(TilesConverter.string_to_34_array(man="1112345678999"))
        self.assertFalse(riichi_ankan_legal(chuuren, 0))

    def test_four_consecutive_kans_keep_dead_wall_and_inventory(self) -> None:
        pool = TilePool()
        hand = pool.hand(man="111222333444", pin="5")
        live_draw = pool.one(0)
        rinshan = (pool.one(1), pool.one(2), pool.one(3), pool.one(9))
        state = make_state({0: hand}, pool, live_front=(live_draw,), rinshan=rinshan)
        state.draw()
        for tile_type in range(4):
            ids = tuple(tile_id for tile_id in state.hands[0] if state.tile_by_id[tile_id].tile34 == tile_type)
            state.declare_ankan(ids)
            state.validate()
        self.assertEqual((state.total_kans, state.revealed_dora_count, len(state.dead_wall)), (4, 5, 14))
        self.assertFalse(any(action.kind in ("ankan", "kakan") for action in state.legal_self_actions()))

    def test_ankan_reveals_dora_before_rinshan_and_clears_ippatsu(self) -> None:
        pool = TilePool()
        hand = pool.hand(man="111", pin="123456789", honors="1")
        live_draw = pool.one(0)
        state = make_state({0: hand}, pool, live_front=(live_draw,))
        state.ippatsu_eligible.add(2)
        state.draw()
        ids = tuple(tile_id for tile_id in state.hands[0] if state.tile_by_id[tile_id].tile34 == 0)
        state.declare_ankan(ids)
        self.assertEqual((state.total_kans, state.revealed_dora_count), (1, 2))
        self.assertEqual((state.phase, state.draw_source), (PHASE_SELF_ACTION, "rinshan"))
        self.assertEqual(state.ippatsu_eligible, set())
        self.assertEqual([event["type"] for event in state.events[-2:]], ["kan_committed", "draw"])

    def test_chankan_cancels_kakan_without_revealing_dora(self) -> None:
        pool = TilePool()
        hands = {
            0: pool.hand(man="55", pin="123456789", honors="11"),
            1: pool.hand(man="34", pin="123", sou="123", honors="11222"),
        }
        first_five = pool.one(4)
        middle = (pool.one(8), pool.one(17), pool.one(26))
        fourth_five = pool.one(4)
        state = make_state(
            hands,
            pool,
            live_front=(first_five, *middle, fourth_five),
            dealer_seat=3,
        )
        state.draw()
        state.discard(first_five)
        pon = next(claim for claim in state.legal_call_claims(0) if claim.kind == "pon")
        state.resolve_discard_responses([pon])
        state.discard(state.hands[0][0])
        state.resolve_discard_responses([])
        advance_without_claim(state)
        advance_without_claim(state)
        advance_without_claim(state)
        state.draw()
        self.assertEqual((state.turn_seat, state.drawn_tile_id), (0, fourth_five))
        self.assertTrue(
            any(action.kind == "kakan" and action.tile_id == fourth_five for action in state.legal_self_actions())
        )
        state.propose_kakan(0, fourth_five)
        self.assertEqual(state.phase, PHASE_CHANKAN_RESPONSES)
        self.assertEqual(state.legal_chankan_claims(1), (ResponseClaim("ron", 1),))
        result = state.resolve_chankan_responses([ResponseClaim("ron", 1)])
        assert result is not None
        self.assertEqual((result.kind, result.winner, result.loser), ("ron", 1, 0))
        self.assertEqual((state.total_kans, state.revealed_dora_count), (0, 1))
        self.assertEqual(state.melds[0][0].kind, "pon")

    def test_kakan_pass_commits_then_draws_rinshan(self) -> None:
        pool = TilePool()
        hands = {0: pool.hand(man="55", pin="123456789", honors="11")}
        first_five = pool.one(4)
        middle = (pool.one(8), pool.one(17), pool.one(26))
        fourth_five = pool.one(4)
        state = make_state(
            hands,
            pool,
            live_front=(first_five, *middle, fourth_five),
            dealer_seat=3,
        )
        state.draw()
        state.discard(first_five)
        pon = next(claim for claim in state.legal_call_claims(0) if claim.kind == "pon")
        state.resolve_discard_responses([pon])
        state.discard(state.hands[0][0])
        state.resolve_discard_responses([])
        advance_without_claim(state)
        advance_without_claim(state)
        advance_without_claim(state)
        state.draw()
        state.propose_kakan(0, fourth_five)
        state.resolve_chankan_responses([])
        self.assertEqual((state.melds[0][0].kind, state.total_kans, state.revealed_dora_count), ("shouminkan", 1, 2))
        self.assertEqual((state.phase, state.draw_source), (PHASE_SELF_ACTION, "rinshan"))
        state.validate()

    def test_confirming_dragon_pon_records_pao_from_event_history(self) -> None:
        pool = TilePool()
        white = tuple(pool.one(31) for _ in range(3))
        green = tuple(pool.one(32) for _ in range(3))
        seat0 = [pool.one(33), pool.one(33), *pool.fill(5)]
        seat1 = pool.fill(13)
        seat2 = pool.fill(13)
        discarded_red = pool.one(33)
        seat3 = [*pool.fill(13), discarded_red]
        dead = pool.fill(14)
        live = pool.fill(69)
        state = RoundState(
            tiles=canonical_tile_set(),
            hands=[seat0, seat1, seat2, seat3],
            live_wall=live,
            dead_wall=dead,
            rinshan_tiles=list(dead[:4]),
            dora_indicator_slots=tuple(dead[4:9]),
            ura_indicator_slots=tuple(dead[9:14]),
            scores=[25000] * 4,
            dealer_seat=3,
            turn_seat=3,
            phase=PHASE_SELF_ACTION,
            melds=[
                [
                    RoundMeld("pon", white, white[0], 1),
                    RoundMeld("pon", green, green[0], 2),
                ],
                [],
                [],
                [],
            ],
            rivers=[
                [],
                [RiverTile(white[0], False, False, 0, called=True)],
                [RiverTile(green[0], False, False, 1, called=True)],
                [],
            ],
            drawn_tile_id=discarded_red,
            draw_source="live",
            calls_occurred=True,
        )
        state.discard(discarded_red)
        pon = next(claim for claim in state.legal_call_claims(0) if claim.kind == "pon")
        state.resolve_discard_responses([pon])
        self.assertEqual(len(state.pao[0]), 1)
        self.assertEqual((state.pao[0][0].yaku, state.pao[0][0].liable_seat), ("Daisangen", 3))
        state.validate()

    def test_exhaustive_draw_and_noten_table_preserve_ledger(self) -> None:
        self.assertEqual(noten_deltas(()), (0, 0, 0, 0))
        self.assertEqual(noten_deltas((0,)), (3000, -1000, -1000, -1000))
        self.assertEqual(noten_deltas((0, 2)), (1500, -1500, 1500, -1500))
        self.assertEqual(noten_deltas((0, 1, 3)), (1000, 1000, -3000, 1000))
        self.assertEqual(noten_deltas((0, 1, 2, 3)), (0, 0, 0, 0))

        state = RoundState.shuffled(20260907, honba=2, kyoutaku=1, scores=(24000, 25000, 25000, 25000))
        while state.phase != PHASE_ENDED:
            advance_without_claim(state)
        assert state.result is not None
        self.assertEqual(state.result.kind, "exhaustive_draw")
        self.assertEqual(state.result.next_honba, 3)
        self.assertEqual(state.kyoutaku, 1)
        self.assertEqual(sum(state.scores) + 1000 * state.kyoutaku, 100000)

    def test_seeded_round_progression_is_deterministic(self) -> None:
        states = [RoundState.shuffled(2026090701) for _ in range(2)]
        for state in states:
            while state.phase != PHASE_ENDED:
                advance_without_claim(state)
        self.assertEqual(states[0].events, states[1].events)
        self.assertEqual(states[0].result, states[1].result)
        self.assertEqual(states[0].scores, states[1].scores)


class ScoringErrorBoundaryTest(unittest.TestCase):
    def test_no_yaku_is_the_only_scoring_error_treated_as_illegal_win(self) -> None:
        state, _ = head_bump_state()
        with patch("tools.ev_policy_round.score_hand", side_effect=HandScoringError("no_yaku")):
            self.assertNotIn(ResponseClaim("ron", 1), state.legal_response_claims(1))

    def test_dependency_failure_propagates_out_of_legal_action_generation(self) -> None:
        state, _ = head_bump_state()
        error = ScoringDependencyError("fixture_dependency_failure")
        with patch("tools.ev_policy_round.score_hand", side_effect=error):
            with self.assertRaisesRegex(ScoringDependencyError, "fixture_dependency_failure"):
                state.legal_response_claims(1)

    def test_state_or_adapter_scoring_failure_is_not_converted_to_illegal_action(self) -> None:
        state, _ = head_bump_state()
        error = HandScoringError("invalid_complete_hand_size")
        with patch("tools.ev_policy_round.score_hand", side_effect=error):
            with self.assertRaisesRegex(HandScoringError, "invalid_complete_hand_size"):
                state.legal_response_claims(1)


if __name__ == "__main__":
    unittest.main()
