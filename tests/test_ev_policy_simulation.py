from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "calibration" / ".dependency-cache" / "mahjong-2.0.0-py3-none-any.whl"
if WHEEL.is_file():
    sys.path.insert(0, str(WHEEL))
sys.path.insert(0, str(ROOT))

from tests.test_calibrate_ev import extraction_paifu_row  # noqa: E402
from tools.calibrate_ev import extract_round  # noqa: E402
from tools.ev_policy_simulation import (  # noqa: E402
    FOLD_POLICY,
    PUSH_POLICY,
    PolicyActionView,
    SimulationPlayerView,
    choose_policy_action,
    evaluate_synthetic_debug,
    initial_player_view,
    replay_policy_decision,
    round_player_view,
    run_synthetic_round,
    synthetic_world_from_decision,
)
from tools.ev_policy_state import policy_action_from_candidate, policy_decision_record  # noqa: E402


def policy_fixture() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    row = extraction_paifu_row()
    extracted = extract_round(
        row,
        row["paifu"]["log"][0],
        0,
        "data/mleague/paifu/2024-25.jsonl",
        1,
        "all",
        "all",
        "pre_action",
    )
    decision = policy_decision_record(extracted["decisions"][0])
    actions = [policy_action_from_candidate(item) for item in extracted["candidates"]]
    return decision, actions, row


class PolicySimulationTest(unittest.TestCase):
    def test_push_uses_shanten_then_ukeire_and_declares_riichi_on_tie(self) -> None:
        actions = (
            PolicyActionView("discard", 1, False, 1, 20, 0, (2,)),
            PolicyActionView("discard", 2, False, 0, 8, 0, (9,)),
            PolicyActionView("discard", 3, False, 0, 12, 0, (9,)),
            PolicyActionView("riichi_discard", 3, False, 0, 12, 0, (9,)),
        )
        view = SimulationPlayerView(0, (1,), (), None, actions)
        choice = choose_policy_action(PUSH_POLICY, view)
        self.assertEqual((choice.kind, choice.tile34), ("riichi_discard", 3))

    def test_fold_rejects_riichi_and_uses_max_then_sum_danger(self) -> None:
        actions = (
            PolicyActionView("riichi_discard", 0, False, 0, 20, 0, (0, 0)),
            PolicyActionView("discard", 1, False, 2, 1, 0, (3, 3)),
            PolicyActionView("discard", 2, False, 0, 40, 0, (3, 2)),
        )
        view = SimulationPlayerView(0, (1, 2), (), None, actions)
        choice = choose_policy_action(FOLD_POLICY, view)
        self.assertEqual((choice.kind, choice.tile34), ("discard", 2))

    def test_legal_win_is_selected_by_both_policies(self) -> None:
        actions = (PolicyActionView("discard", 1), PolicyActionView("tsumo"))
        view = SimulationPlayerView(0, (), (), None, actions)
        self.assertEqual(choose_policy_action(PUSH_POLICY, view).kind, "tsumo")
        self.assertEqual(choose_policy_action(FOLD_POLICY, view).kind, "tsumo")

    def test_initial_push_excludes_two_shanten_and_fold_excludes_riichi(self) -> None:
        decision, actions, _ = policy_fixture()
        extra = copy.deepcopy(actions[0])
        extra["actionId"] = "extra"
        extra["shantenAfterDiscard"] = 2
        push = initial_player_view(decision, [*actions, extra], PUSH_POLICY)
        fold = initial_player_view(decision, actions, FOLD_POLICY)
        self.assertTrue(all(item.shanten_after_discard in (0, 1) for item in push.actions))
        self.assertTrue(all(item.kind == "discard" for item in fold.actions))

    def test_uniform_world_preserves_inventory_and_seed(self) -> None:
        decision, _, _ = policy_fixture()
        first = synthetic_world_from_decision(decision, 17)
        second = synthetic_world_from_decision(decision, 17)
        first.validate()
        self.assertEqual(first.hands, second.hands)
        self.assertEqual(first.live_wall, second.live_wall)
        self.assertEqual(len(first.live_wall), decision["publicState"]["remaining_live_wall_tiles"])

    def test_hidden_hands_and_future_wall_do_not_change_target_choice(self) -> None:
        decision, _, _ = policy_fixture()
        first = synthetic_world_from_decision(decision, 1)
        second = synthetic_world_from_decision(decision, 2)
        first_view = round_player_view(first, decision["playerView"]["seat"])
        second_view = round_player_view(second, decision["playerView"]["seat"])
        self.assertEqual(first_view, second_view)
        self.assertEqual(
            choose_policy_action(PUSH_POLICY, first_view),
            choose_policy_action(PUSH_POLICY, second_view),
        )

    def test_equivalent_physical_tile_ids_do_not_change_policy_view(self) -> None:
        decision, _, _ = policy_fixture()
        first = synthetic_world_from_decision(decision, 7)
        second = copy.deepcopy(first)
        target = decision["playerView"]["seat"]
        by_id = second.tile_by_id
        swap = next(
            (hand_id, wall_id)
            for hand_id in second.hands[target]
            for wall_id in second.live_wall
            if (by_id[hand_id].tile34, by_id[hand_id].is_red)
            == (by_id[wall_id].tile34, by_id[wall_id].is_red)
        )
        hand_id, wall_id = swap
        hand_index = second.hands[target].index(hand_id)
        wall_index = second.live_wall.index(wall_id)
        second.hands[target][hand_index] = wall_id
        second.live_wall[wall_index] = hand_id
        if second.drawn_tile_id == hand_id:
            second.drawn_tile_id = wall_id
        second.validate()
        first_view = round_player_view(first, target)
        second_view = round_player_view(second, target)
        self.assertEqual(first_view, second_view)
        self.assertNotIn("tile_id", repr(first_view))

    def test_same_world_and_policy_reproduce_the_round(self) -> None:
        decision, actions, _ = policy_fixture()
        world = synthetic_world_from_decision(decision, 29)
        initial = choose_policy_action(PUSH_POLICY, initial_player_view(decision, actions, PUSH_POLICY))
        first = run_synthetic_round(
            world,
            target_seat=decision["playerView"]["seat"],
            target_policy=PUSH_POLICY,
            forced_initial=initial,
        )
        second = run_synthetic_round(
            world,
            target_seat=decision["playerView"]["seat"],
            target_policy=PUSH_POLICY,
            forced_initial=initial,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "synthetic_debug")

    def test_debug_evaluation_keeps_all_failures_visible(self) -> None:
        decision, actions, _ = policy_fixture()
        report = evaluate_synthetic_debug(decision, actions, seeds=(3, 5))
        self.assertEqual(report["scopeStatus"], "synthetic_debug")
        self.assertIn("not_eligible_for_app_scoring", report["holdReasons"])
        self.assertEqual(
            report["runtime"]["attemptedTrials"],
            2 * len(report["candidates"]),
        )
        self.assertTrue(all("unscorableTrials" in item for item in report["candidates"]))

    def test_candidate_order_does_not_change_seed_results(self) -> None:
        decision, actions, _ = policy_fixture()
        first = evaluate_synthetic_debug(decision, actions, seeds=(11,))
        second = evaluate_synthetic_debug(decision, list(reversed(actions)), seeds=(11,))

        def keyed(report: dict[str, object]) -> dict[tuple[str, str], tuple[object, object, object]]:
            return {
                (
                    item["policyVersion"],
                    json.dumps(item["initialAction"], ensure_ascii=False, sort_keys=True),
                ): (item["meanTargetPoints"], item["outcomeCounts"], item["unscorableReasons"])
                for item in report["candidates"]
            }

        self.assertEqual(keyed(first), keyed(second))

    def test_recorded_prefix_replay_matches_and_ignores_future_result(self) -> None:
        decision, _, row = policy_fixture()
        loaded_decision = json.loads(json.dumps(decision, ensure_ascii=False))
        self.assertEqual(replay_policy_decision(loaded_decision, row)["status"], "pass")
        changed = copy.deepcopy(row)
        changed["paifu"]["log"][0][-1] = ["和了", [1000, -1000, 0, 0]]
        self.assertEqual(replay_policy_decision(loaded_decision, changed)["status"], "pass")


if __name__ == "__main__":
    unittest.main()
