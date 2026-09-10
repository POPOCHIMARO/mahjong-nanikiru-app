from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "calibration" / ".dependency-cache" / "mahjong-2.0.0-py3-none-any.whl"
if WHEEL.is_file():
    sys.path.insert(0, str(WHEEL))
sys.path.insert(0, str(ROOT))

from tests.test_ev_policy_replay import valid_four_turn_log  # noqa: E402
from tests.test_ev_policy_round import TilePool, make_state  # noqa: E402
from tools.calibrate_ev import main  # noqa: E402
from tools.ev_calibration_state import is_red, tile34  # noqa: E402
from tools.ev_policy_observation import (  # noqa: E402
    build_inference_prefix,
    compatible_joint_responses,
    extract_opponent_dataset,
    extract_teacher_windows,
    project_round,
    runtime_prefix,
    semantic_self_actions,
)
from tools.ev_policy_replay import (  # noqa: E402
    ParsedRound,
    RecordedChronologyError,
    RecordedEvent,
    parse_recorded_round,
)


class ObservationBoundaryTest(unittest.TestCase):
    def test_d31_01_future_and_teacher_only_mutations_do_not_change_prefix(self) -> None:
        baseline = valid_four_turn_log()
        variants = []

        actual_action = copy.deepcopy(baseline)
        actual_action[6][0] = actual_action[5][0]  # 60を同じ自摸牌の明示tokenへ変える。
        variants.append(actual_action)

        hidden_hand = copy.deepcopy(baseline)
        hidden_hand[7][0], hidden_hand[8][0] = hidden_hand[8][0], hidden_hand[7][0]
        variants.append(hidden_hand)

        future_draw = copy.deepcopy(baseline)
        future_draw[11][0], future_draw[14][0] = future_draw[14][0], future_draw[11][0]
        variants.append(future_draw)

        ura = copy.deepcopy(baseline)
        ura[3] = [11, 22]
        variants.append(ura)

        result = copy.deepcopy(baseline)
        result[-1] = ["和了", [1000, -1000, 0, 0], [0, 1, 0, "fixture"]]
        variants.append(result)

        expected = build_inference_prefix(
            project_round(parse_recorded_round(baseline), "same-round"),
            seat=0,
            raw_event_cutoff=1,
        )
        expected.pop("payload")
        for changed in variants:
            actual = build_inference_prefix(
                project_round(parse_recorded_round(changed), "same-round"),
                seat=0,
                raw_event_cutoff=1,
            )
            actual.pop("payload")
            self.assertEqual(actual, expected)

    def test_pass_resolution_is_visible_before_the_next_draw(self) -> None:
        projection = project_round(parse_recorded_round(valid_four_turn_log()), "round")
        prefix = build_inference_prefix(projection, seat=1, raw_event_cutoff=2)
        self.assertEqual(prefix["payload"]["public"]["events"][-1]["resolution"], {"kind": "pass"})

    def test_missing_initial_dora_is_a_reasoned_round_rejection(self) -> None:
        parsed = parse_recorded_round(valid_four_turn_log())
        with self.assertRaisesRegex(RecordedChronologyError, "initial_dora_missing"):
            project_round(replace(parsed, dora_raw=()), "round")

    def test_late_inventory_error_keeps_earlier_teacher_prefix(self) -> None:
        parsed = parse_recorded_round(valid_four_turn_log())
        counts = Counter()
        representative = {}
        for hand in parsed.initial_hands:
            for raw in hand:
                key = (tile34(raw), is_red(raw))
                counts[key] += 1
                representative[key] = raw
        first_dora = parsed.dora_raw[0]
        counts[(tile34(first_dora), is_red(first_dora))] += 1
        key = next(pair for pair, count in counts.items() if count == (1 if pair[0] in {4, 13, 22} and pair[1] else 4))
        bad_raw = representative[key]
        events = list(parsed.events)
        events[6] = replace(events[6], raw=bad_raw)
        events[7] = replace(events[7], raw=bad_raw)
        changed = replace(parsed, events=tuple(events))

        prefix, rejection = runtime_prefix(changed)
        self.assertEqual((len(prefix.events), rejection["rawEventIndex"]), (6, 6))
        windows, suffix = extract_teacher_windows(changed, project_round(changed, "round"), "round")
        self.assertEqual((len(windows), suffix["rawEventIndex"]), (6, 6))


class SemanticActionTest(unittest.TestCase):
    def test_d31_02_physical_copies_collapse_but_red_and_origin_remain(self) -> None:
        pool = TilePool()
        normal_a = pool.one(4, red=False)
        normal_b = pool.one(4, red=False)
        red = pool.one(4, red=True)
        draw = pool.one(4, red=False)
        state = make_state({0: [normal_a, normal_b, red]}, pool, live_front=(draw,))
        state.draw()
        actions, _, _ = semantic_self_actions(state)
        fives = [action for action in actions if action["kind"] == "discard" and action["tile34"] == 4]

        self.assertEqual(
            {(item["isRed"], item["origin"]) for item in fives},
            {(False, "concealed"), (True, "concealed"), (False, "drawn")},
        )
        self.assertEqual(len(fives), 3)

    def test_d31_03_pon_resolution_censors_a_lower_priority_chi_seat(self) -> None:
        template = parse_recorded_round(valid_four_turn_log())
        parsed = ParsedRound(
            events=(
                RecordedEvent(0, "discard", 0, raw=15),
                RecordedEvent(1, "pon", 2, token="p151515", from_seat=0, consumed_raw=(15, 15)),
            ),
            snapshots=(),
            initial_hands=template.initial_hands,
            dealer_seat=0,
            kyoku=0,
            honba=0,
            kyoutaku=0,
            start_scores=(25000, 25000, 25000, 25000),
            dora_raw=(template.dora_raw[0],),
            ura_raw=(),
            result=["流局", [0, 0, 0, 0]],
        )
        projection = project_round(parsed, "round")
        seat1_label = next(
            item for item in projection.private_records[1]["events"] if item["type"] == "response_observation"
        )
        self.assertEqual(
            (seat1_label["status"], seat1_label["action"], seat1_label["constraint"]),
            ("censored", None, "joint_public_resolution"),
        )

        chi = {"kind": "chi", "consumed": [{"tile34": 3, "isRed": False}, {"tile34": 4, "isRed": False}]}
        pon = {"kind": "pon", "consumed": [{"tile34": 4, "isRed": False}] * 2}
        legal = {1: ({"kind": "pass"}, chi), 2: ({"kind": "pass"}, pon), 3: ({"kind": "pass"},)}
        observed = {"kind": "pon", "seat": 2, "action": pon}
        compatible = compatible_joint_responses(0, legal, observed)
        self.assertEqual({choice[1]["kind"] for choice in compatible}, {"pass", "chi"})
        conditional_mass = sum(1 / len(compatible) for _ in compatible)
        self.assertAlmostEqual(conditional_mass, 1.0, places=12)


class InputGuardTest(unittest.TestCase):
    def test_d31_04_cli_rejects_reserved_future_season(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            code = main(
                [
                    "extract-opponent-events",
                    "--vault-root",
                    str(ROOT / "麻雀強者の考え方"),
                    "--reservation",
                    str(ROOT / "calibration" / "future-evaluation-reservation.json"),
                    "--d1-dataset-dir",
                    str(ROOT / "calibration" / "dataset-policy"),
                    "--output-dir",
                    temp,
                    "--season",
                    "2026-27",
                ]
            )
        self.assertEqual(code, 2)

    def test_d31_04_d1_manifest_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            d1 = root / "d1"
            d1.mkdir()
            (d1 / "extraction-summary.json").write_text(
                json.dumps({"input": {"files": [], "selectedSeasons": []}}), encoding="utf-8"
            )
            selected = {
                "files": [{"season": "2018-19", "path": "x.jsonl", "bytes": 1, "sha256": "x"}],
                "selectedSeasons": ["2018-19"],
            }
            with (
                patch("tools.ev_policy_observation.select_development_inputs", return_value=selected),
                patch(
                    "tools.ev_policy_observation.verify_policy_dataset",
                    return_value={"status": "pass", "aggregateSha256": "d1"},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    extract_opponent_dataset(root, root / "reservation.json", d1, root / "out")


if __name__ == "__main__":
    unittest.main()
