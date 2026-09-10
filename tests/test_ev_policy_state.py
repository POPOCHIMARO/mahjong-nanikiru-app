from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_calibrate_ev import extraction_paifu_row, write_jsonl  # noqa: E402
from tools.calibrate_ev import extract_policy_dataset, extract_round, main  # noqa: E402
from tools.ev_policy_state import (  # noqa: E402
    RuleProfile,
    TileLocation,
    WorldState,
    canonical_tile_set,
    policy_action_from_candidate,
    policy_decision_record,
    select_development_inputs,
    verify_policy_dataset,
    validate_policy_payload,
)


RESERVATION = ROOT / "calibration" / "future-evaluation-reservation.json"


def policy_bundle(row: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    log = row["paifu"]["log"][0]  # type: ignore[index]
    extracted = extract_round(row, log, 0, "fixture.jsonl", 1, "all", "all", "pre_action")
    decision = extracted["decisions"][0]
    actions = [
        policy_action_from_candidate(item)
        for item in extracted["candidates"]
        if item["decisionId"] == decision["decisionId"]
    ]
    return policy_decision_record(decision), sorted(actions, key=lambda item: item["actionId"])


class PolicyStateTest(unittest.TestCase):
    def test_mleague_riichi_boundary_allows_no_next_draw_but_not_haitei(self) -> None:
        profile = RuleProfile()
        for remaining in (1, 2, 3):
            self.assertTrue(
                profile.can_declare_riichi(
                    closed=True,
                    shanten_after_discard=0,
                    after_normal_draw=True,
                    live_wall_tiles_after_draw=remaining,
                )
            )
        self.assertFalse(
            profile.can_declare_riichi(
                closed=True,
                shanten_after_discard=0,
                after_normal_draw=True,
                live_wall_tiles_after_draw=0,
            )
        )
        self.assertFalse(
            profile.can_declare_riichi(
                closed=True,
                shanten_after_discard=1,
                after_normal_draw=True,
                live_wall_tiles_after_draw=3,
            )
        )

    def test_world_state_requires_each_of_136_tiles_in_one_location(self) -> None:
        tiles = canonical_tile_set()
        locations = tuple(TileLocation(tile.tile_id, "live_wall") for tile in tiles)
        WorldState(tiles, locations).validate()
        broken = list(locations)
        broken[-1] = TileLocation(0, "dead_wall")
        with self.assertRaisesRegex(ValueError, "所在はちょうど一つ"):
            WorldState(tiles, tuple(broken)).validate()

    def test_policy_payload_rejects_observed_action_and_future_result(self) -> None:
        for key in ("actualActionId", "rewardPoints", "futureDraws"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "混入"):
                validate_policy_payload({"playerView": {key: "leak"}})

    def test_policy_input_is_independent_of_actual_action_and_round_result(self) -> None:
        original = extraction_paifu_row()
        changed = copy.deepcopy(original)
        changed_log = changed["paifu"]["log"][0]  # type: ignore[index]
        changed_log[9] = [45]  # type: ignore[index]
        changed_log[-1] = ["破損した未来結果"]  # type: ignore[index]

        original_policy = policy_bundle(original)
        changed_policy = policy_bundle(changed)
        self.assertEqual(original_policy, changed_policy)

        changed_c = extract_round(changed, changed_log, 0, "fixture.jsonl", 1, "all", "all")
        self.assertEqual(changed_c["decisions"], [])
        self.assertTrue(
            any("actual_shanten_after_out_of_scope" in row["reasons"] for row in changed_c["rejections"])
        )

    def test_policy_extraction_keeps_valid_prefix_after_future_chronology_failure(self) -> None:
        row = extraction_paifu_row()
        broken = copy.deepcopy(row)
        broken_log = broken["paifu"]["log"][0]  # type: ignore[index]
        broken_log[11] = ["invalid future draw"]  # type: ignore[index]
        bundle = extract_round(broken, broken_log, 0, "fixture.jsonl", 1, "all", "all", "pre_action")
        self.assertGreaterEqual(len(bundle["decisions"]), 1)
        self.assertTrue(any(item["level"] == "round_suffix" for item in bundle["rejections"]))
        self.assertEqual(bundle["stats"]["discarded_accepted_decisions_due_to_round_chronology"], 0)

    def test_policy_extraction_rejects_invalid_initial_hand(self) -> None:
        row = extraction_paifu_row()
        log = row["paifu"]["log"][0]  # type: ignore[index]
        log[10] = log[10][:-1]  # type: ignore[index]
        bundle = extract_round(row, log, 0, "fixture.jsonl", 1, "all", "all", "pre_action")
        self.assertEqual(bundle["decisions"], [])
        self.assertEqual(bundle["rejections"][0]["reasons"], ["initial_hand_invalid"])

    def test_policy_extraction_rejects_broken_score_conservation(self) -> None:
        row = extraction_paifu_row()
        log = row["paifu"]["log"][0]  # type: ignore[index]
        log[1] = [24900, 25000, 25000, 25000]  # type: ignore[index]
        bundle = extract_round(row, log, 0, "fixture.jsonl", 1, "all", "all", "pre_action")
        self.assertEqual(bundle["decisions"], [])
        self.assertTrue(
            any("score_conservation_violation" in item["reasons"] for item in bundle["rejections"])
        )

    def test_reservation_selects_only_development_seasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / "vault"
            paifu = vault / "data" / "mleague" / "paifu"
            paifu.mkdir(parents=True)
            (paifu / "2024-25.jsonl").write_text("{}\n", encoding="utf-8")
            (paifu / "2026-27.jsonl").write_text("{}\n", encoding="utf-8")
            report = select_development_inputs(vault, RESERVATION, {"2024-25"})
            self.assertEqual(report["selectedSeasons"], ["2024-25"])
            self.assertEqual(report["availableForbiddenSeasons"], ["2026-27"])
            with self.assertRaisesRegex(ValueError, "牌譜がない"):
                select_development_inputs(vault, RESERVATION)
            with self.assertRaisesRegex(ValueError, "許可されていない"):
                select_development_inputs(vault, RESERVATION, {"2026-27"})

    def test_policy_dataset_contains_no_observed_action_or_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vault = root / "vault"
            output = root / "out"
            row = extraction_paifu_row()
            log = row["paifu"]["log"][0]  # type: ignore[index]
            log[9] = [45]  # type: ignore[index]
            write_jsonl(vault / "data" / "mleague" / "paifu" / "2024-25.jsonl", [row])
            summary = extract_policy_dataset(vault, RESERVATION, output, {"2024-25"})
            self.assertEqual(summary["quality"]["status"], "pass_with_exclusions")
            self.assertGreaterEqual(summary["records"]["decisions"], 1)
            verification = verify_policy_dataset(output)
            self.assertEqual(verification["status"], "pass")
            payload = (output / "policy-decisions.jsonl").read_text(encoding="utf-8")
            actions = (output / "policy-actions.jsonl").read_text(encoding="utf-8")
            for forbidden in ("actor", "actualActionId", "isActual", "rewardPoints", "roundEndScores"):
                self.assertNotIn(f'"{forbidden}"', payload + actions)

    def test_cli_rejects_future_evaluation_season(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vault = root / "vault"
            paifu = vault / "data" / "mleague" / "paifu"
            paifu.mkdir(parents=True)
            (paifu / "2026-27.jsonl").write_text("{}\n", encoding="utf-8")
            code = main(
                [
                    "validate-policy-input",
                    "--vault-root",
                    str(vault),
                    "--reservation",
                    str(RESERVATION),
                    "--output",
                    str(root / "report.json"),
                    "--season",
                    "2026-27",
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
