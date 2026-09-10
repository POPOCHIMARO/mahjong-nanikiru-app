from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "calibration" / ".dependency-cache" / "mahjong-2.0.0-py3-none-any.whl"
if WHEEL.is_file():
    sys.path.insert(0, str(WHEEL))
sys.path.insert(0, str(ROOT))

from tests.test_calibrate_ev import extraction_paifu_row  # noqa: E402
from tests.test_ev_policy_round import TilePool, make_state  # noqa: E402
from tools.calibrate_ev import extract_round  # noqa: E402
from tools.ev_policy_replay import (  # noqa: E402
    RecordedEvent,
    apply_recorded_event,
    build_runtime_world,
    compare_policy_decision,
    parse_recorded_round,
    score_delta_check,
)
from tools.ev_policy_round import PHASE_CALL_DISCARD, PHASE_SELF_ACTION  # noqa: E402
from tools.ev_policy_state import canonical_tile_set, policy_decision_record  # noqa: E402


def raw_for_tile(state, tile_id: int) -> int:
    tile = state.tile_by_id[tile_id]
    if tile.is_red:
        return {4: 51, 13: 52, 22: 53}[tile.tile34]
    if tile.tile34 < 27:
        return (tile.tile34 // 9 + 1) * 10 + tile.tile34 % 9 + 1
    return 41 + tile.tile34 - 27


def valid_four_turn_log() -> list[object]:
    tiles = canonical_tile_set()
    ids = list(range(136))
    # 固定した別順序にし、配牌・通常自摸・表示牌が物理的に重複しないようにする。
    ids = ids[37:] + ids[:37]

    def raw(tile_id: int) -> int:
        tile = tiles[tile_id]
        if tile.is_red:
            return {4: 51, 13: 52, 22: 53}[tile.tile34]
        if tile.tile34 < 27:
            return (tile.tile34 // 9 + 1) * 10 + tile.tile34 % 9 + 1
        return 41 + tile.tile34 - 27

    hands = [ids[seat * 13:(seat + 1) * 13] for seat in range(4)]
    draws = ids[52:56]
    dora = ids[126]
    log: list[object] = [[0, 0, 0], [25000] * 4, [raw(dora)], []]
    for seat in range(4):
        log.extend([[raw(tile_id) for tile_id in hands[seat]], [raw(draws[seat])], [60]])
    log.append(["流局", [0, 0, 0, 0]])
    return log


class RecordedEventAdapterTest(unittest.TestCase):
    def test_independent_snapshot_matches_fixed_policy_input(self) -> None:
        row = extraction_paifu_row()
        log = row["paifu"]["log"][0]
        parsed = parse_recorded_round(log)
        extracted = extract_round(row, log, 0, "fixture.jsonl", 1, "all", "all", "pre_action")
        decision = next(item for item in extracted["decisions"] if item["seat"] == 1)
        policy = policy_decision_record(decision)

        self.assertEqual(compare_policy_decision(policy, parsed), ())
        snapshot = next(item for item in parsed.snapshots if item.seat == 1)
        self.assertEqual(snapshot.event_index, 3)
        self.assertEqual(snapshot.active_riichi, ((0, 1),))
        self.assertEqual(snapshot.scores, (24000, 25000, 25000, 25000))
        self.assertEqual(snapshot.total_draws, 2)
        self.assertEqual(score_delta_check(parsed), ("pass", None))

    def test_two_unknown_completions_preserve_recorded_normal_events(self) -> None:
        parsed = parse_recorded_round(valid_four_turn_log())
        signatures = []
        for reverse in (False, True):
            world = build_runtime_world(parsed, reverse_completion=reverse)
            for event in parsed.events:
                apply_recorded_event(world.state, event, world.draw_tile_ids)
            state = world.state
            signatures.append(
                (
                    tuple(state.scores),
                    state.kyoutaku,
                    tuple(
                        tuple((state.tile_by_id[item.tile_id].tile34, item.is_riichi_declaration) for item in river)
                        for river in state.rivers
                    ),
                )
            )
        self.assertEqual(signatures[0], signatures[1])

    def test_adapter_applies_chi_and_pon_as_recorded_responses(self) -> None:
        # チー
        pool = TilePool()
        chi_hand = pool.hand(man="12", pin="123456789", honors="11")
        discard = pool.one(2)
        chi_state = make_state({1: chi_hand}, pool, live_front=(discard,))
        chi_state.draw()
        chi_state.discard(discard)
        chi_ids = next(claim.consumed_tile_ids for claim in chi_state.legal_call_claims(1) if claim.kind == "chi")
        apply_recorded_event(
            chi_state,
            RecordedEvent(
                2,
                "chi",
                1,
                from_seat=0,
                consumed_raw=tuple(raw_for_tile(chi_state, tile_id) for tile_id in chi_ids),
            ),
            {},
        )
        self.assertEqual((chi_state.phase, chi_state.melds[1][-1].kind), (PHASE_CALL_DISCARD, "chi"))

        # ポン
        pool = TilePool()
        pon_hand = pool.hand(man="55", pin="123456789", honors="22")
        discard = pool.one(4)
        pon_state = make_state({2: pon_hand}, pool, live_front=(discard,))
        pon_state.draw()
        pon_state.discard(discard)
        pon_ids = next(claim.consumed_tile_ids for claim in pon_state.legal_call_claims(2) if claim.kind == "pon")
        apply_recorded_event(
            pon_state,
            RecordedEvent(
                2,
                "pon",
                2,
                from_seat=0,
                consumed_raw=tuple(raw_for_tile(pon_state, tile_id) for tile_id in pon_ids),
            ),
            {},
        )
        self.assertEqual((pon_state.phase, pon_state.melds[2][-1].kind), (PHASE_CALL_DISCARD, "pon"))


class RoundRuntimeFixtureTest(unittest.TestCase):
    def test_ankan_event_enters_rinshan_self_action(self) -> None:
        pool = TilePool()
        hand = pool.hand(man="111", pin="123456789", honors="1")
        fourth = pool.one(0)
        state = make_state({0: hand}, pool, live_front=(fourth,))
        state.draw()
        ids = tuple(tile_id for tile_id in state.hands[0] if state.tile_by_id[tile_id].tile34 == 0)
        apply_recorded_event(
            state,
            RecordedEvent(1, "ankan", 0, consumed_raw=tuple(raw_for_tile(state, tile_id) for tile_id in ids)),
            {},
        )
        self.assertEqual((state.phase, state.draw_source, state.total_kans), (PHASE_SELF_ACTION, "rinshan", 1))
