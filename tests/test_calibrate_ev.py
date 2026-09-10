from __future__ import annotations

import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.calibrate_ev import (  # noqa: E402
    build_audit,
    extract_dataset,
    extract_observed_dataset,
    extract_round,
    main,
    write_json,
)


def round_log(scores: list[int], result: list[object], *, riichi_seat: int | None = None) -> list[object]:
    value: list[object] = [[0, 0, 0], scores, [11], []]
    for seat in range(4):
        discards: list[object] = ["r11"] if seat == riichi_seat else [11]
        value.extend([[11] * 13, [12], discards])
    value.append(result)
    return value


def paifu_row(round_index: int, scores: list[int], result: list[object], *, riichi_seat: int | None = None) -> dict[str, object]:
    return {
        "season": "2024-25",
        "stage": "regular",
        "date": f"2024-10-{round_index + 1:02d}",
        "gameId": "fixture-game",
        "roundIndex": round_index,
        "roundName": f"東{round_index + 1}局",
        "paifu": {"rule": {"aka": 1}, "log": [round_log(scores, result, riichi_seat=riichi_seat)]},
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def decision_row(action_index: int, *, eligible: bool) -> dict[str, object]:
    return {
        "season": "2024-25",
        "gameId": "fixture-game",
        "roundIndex": 0,
        "logIndex": 0,
        "actionIndex": action_index,
        "seat": action_index,
        "opponentRiichiCount": 1,
        "openMelds": 0 if eligible else 1,
        "shantenBeforeDiscard": 1,
        "shantenAfterDiscard": 0,
        "isRiichiDeclaration": False,
    }


def extraction_paifu_row() -> dict[str, object]:
    hands = [
        [12, 13, 14, 15, 16, 17, 21, 22, 23, 31, 32, 33, 44],
        [11, 12, 13, 14, 15, 16, 21, 22, 24, 25, 45, 45, 37],
        [11, 11, 11, 22, 22, 22, 33, 33, 33, 41, 42, 43, 44],
        [19, 19, 19, 28, 28, 28, 36, 36, 36, 45, 46, 47, 41],
    ]
    draws = [[19], [47], [34], [42]]
    discards: list[list[object]] = [["r19"], [47], [34], [42]]
    log: list[object] = [[0, 0, 0], [25000, 25000, 25000, 25000], [41], [45]]
    for seat in range(4):
        log.extend([hands[seat], draws[seat], discards[seat]])
    log.append(["流局", [-1000, 0, 0, 0]])
    return {
        "season": "2024-25",
        "stage": "regular",
        "date": "2024-10-01",
        "gameId": "extract-fixture",
        "matchNumber": 1,
        "roundIndex": 0,
        "roundName": "東1局",
        "paifu": {
            "name": ["A", "B", "C", "D"],
            "rule": {"aka": 1},
            "log": [log],
        },
    }


class CalibrationAuditTest(unittest.TestCase):
    def make_fixture(self, root: Path, *, duplicate_round: bool = False) -> Path:
        vault = root / "vault"
        rows = [
            paifu_row(
                0,
                [25000, 25000, 25000, 25000],
                ["和了", [2000, -1000, 0, 0], [0, 1, 0, "fixture"]],
                riichi_seat=0,
            ),
            paifu_row(1, [26000, 24000, 25000, 25000], ["流局", [0, 0, 0, 0]]),
        ]
        if duplicate_round:
            rows.append(rows[0])
        write_jsonl(vault / "data" / "mleague" / "paifu" / "2024-25.jsonl", rows)
        write_jsonl(
            vault / "data" / "mleague" / "danger" / "against_riichi_decisions_with_danger.jsonl",
            [decision_row(0, eligible=True), decision_row(1, eligible=False)],
        )
        write_jsonl(
            vault / "data" / "mleague" / "validation" / "hand_reconstruction_issues" / "2024-25.jsonl",
            [
                {
                    "season": "2024-25",
                    "gameId": "fixture-game",
                    "roundIndex": 1,
                    "logIndex": 0,
                    "turnIndex": 3,
                    "seat": 2,
                    "issue": "discard_tile_missing",
                }
            ],
        )
        return vault

    def test_audit_counts_and_winning_riichi_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_fixture(Path(temp))
            report = build_audit(vault)

        self.assertEqual(report["coverage"]["totals"]["roundLogs"], 2)
        self.assertEqual(report["coverage"]["totals"]["matches"], 1)
        continuity = report["scoreLedger"]["betweenRoundContinuity"]
        self.assertEqual(continuity["checked"], 1)
        self.assertEqual(continuity["exactMatches"], 1)
        self.assertEqual(continuity["mismatches"], 0)
        hypotheses = report["scoreLedger"]["riichiDepositHypothesisCheck"]
        self.assertEqual(hypotheses["primary"]["exactMatches"], 1)
        self.assertEqual(hypotheses["alternatives"]["result精算差分をそのまま使う"], 0)
        self.assertEqual(report["exceptions"]["issueEvents"], 1)
        self.assertEqual(report["exceptions"]["affectedRounds"], 1)
        self.assertEqual(report["eligibility"]["totals"]["provisionalEligibleDecisionRows"], 1)
        self.assertEqual(report["eligibility"]["totals"]["provisionalEligibleIndependentRounds"], 1)
        self.assertEqual(report["eligibility"]["totals"]["knownExcludedIndependentRounds"], 0)
        self.assertEqual(report["eligibility"]["totals"]["provisionalEligibleAfterKnownExclusions"], 1)
        self.assertEqual(report["knownExclusions"]["uniqueRounds"], 1)
        self.assertEqual(report["qualityGate"]["status"], "pass_with_exclusions")

    def test_report_is_deterministic_and_cli_writes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            vault = self.make_fixture(temp_root)
            first = build_audit(vault)
            second = build_audit(vault)
            self.assertEqual(first, second)

            output = temp_root / "out" / "audit.json"
            exit_code = main(["audit", "--vault-root", str(vault), "--output", str(output)])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)

    def test_duplicate_round_key_fails_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_fixture(Path(temp), duplicate_round=True)
            report = build_audit(vault)

        self.assertEqual(report["qualityGate"]["status"], "fail")
        self.assertEqual(len(report["coverage"]["duplicateRoundKeys"]), 1)
        self.assertIn("牌譜に重複局キーがある", report["qualityGate"]["errors"])


class CalibrationExtractionTest(unittest.TestCase):
    def target_records(self, row: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        log = row["paifu"]["log"][0]  # type: ignore[index]
        bundle = extract_round(row, log, 0, "fixture.jsonl", 1)  # type: ignore[arg-type]
        decision = next(item for item in bundle["decisions"] if item["seat"] == 1)
        candidates = [item for item in bundle["candidates"] if item["decisionId"] == decision["decisionId"]]
        outcome = next(item for item in bundle["outcomes"] if item["decisionId"] == decision["decisionId"])
        return decision, candidates, outcome

    def test_extracts_public_state_candidates_and_observed_reward(self) -> None:
        decision, candidates, outcome = self.target_records(extraction_paifu_row())

        self.assertEqual(decision["opponentRiichiSeats"], [0])
        self.assertEqual(decision["shantenBeforeDiscard"], 1)
        self.assertEqual(decision["scoresAtDecision"], [24000, 25000, 25000, 25000])
        self.assertEqual(decision["riichiSticksAtDecision"], 1)
        self.assertEqual([item["raw"] for item in decision["publicDoraIndicators"]], [41])
        self.assertNotIn("result", decision)
        self.assertNotIn("uraDoraIndicators", decision)
        actual = [item for item in candidates if item["isActual"]]
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["discardRaw"], 47)
        self.assertEqual(actual[0]["shantenAfterDiscard"], 1)
        self.assertTrue(all("ukeireCount" in item and "dangerClass" in item for item in candidates))
        self.assertEqual(outcome["resultClass"], "draw")
        self.assertEqual(outcome["rewardPoints"], 0)

    def test_future_result_ura_and_draw_do_not_change_features(self) -> None:
        original = extraction_paifu_row()
        changed = copy.deepcopy(original)
        changed_log = changed["paifu"]["log"][0]  # type: ignore[index]
        changed_log[3] = [47]
        changed_log[11][0] = 35
        changed_log[12][0] = 35
        changed_log[-1] = ["和了", [-1000, -1000, 3000, -1000], [2, 2, 2, "fixture"]]

        decision_before, candidates_before, outcome_before = self.target_records(original)
        decision_after, candidates_after, outcome_after = self.target_records(changed)
        self.assertEqual(decision_before, decision_after)
        self.assertEqual(candidates_before, candidates_after)
        self.assertNotEqual(outcome_before, outcome_after)
        self.assertEqual(outcome_after["resultClass"], "opponent_tsumo")
        self.assertEqual(outcome_after["rewardPoints"], -1000)

    def test_chronology_failure_quarantines_the_whole_round_with_an_id(self) -> None:
        row = extraction_paifu_row()
        log = row["paifu"]["log"][0]  # type: ignore[index]
        log[11][0] = "zz"
        bundle = extract_round(row, log, 0, "fixture.jsonl", 7)  # type: ignore[arg-type]

        self.assertEqual(bundle["decisions"], [])
        round_rejections = [item for item in bundle["rejections"] if item.get("level") == "round"]
        self.assertEqual(len(round_rejections), 1)
        self.assertEqual(round_rejections[0]["source"]["line"], 7)
        self.assertEqual(round_rejections[0]["reasons"], ["chronology_unparseable_draw"])

    def test_dataset_writer_checks_audit_hash_and_writes_one_to_one_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "vault"
            source_row = extraction_paifu_row()
            write_jsonl(vault / "data" / "mleague" / "paifu" / "2024-25.jsonl", [source_row])
            write_jsonl(
                vault / "data" / "mleague" / "danger" / "against_riichi_decisions_with_danger.jsonl",
                [decision_row(0, eligible=True)],
            )
            bundle = extract_round(source_row, source_row["paifu"]["log"][0], 0, "fixture.jsonl", 1)  # type: ignore[index]
            shanten_rows = [
                {
                    "season": decision["source"]["season"],
                    "gameId": decision["source"]["gameId"],
                    "roundIndex": decision["source"]["roundIndex"],
                    "logIndex": decision["source"]["logIndex"],
                    "seat": decision["seat"],
                    "discardIndex": decision["source"]["discardIndex"],
                    "shantenBeforeDiscard": decision["shantenBeforeDiscard"],
                    "shantenAfterDiscard": decision["actualShantenAfterDiscard"],
                    "discardTile34": decision["actualDiscardTile34"],
                }
                for decision in bundle["decisions"]
            ]
            write_jsonl(vault / "data" / "mleague" / "shanten" / "2024-25.jsonl", shanten_rows)
            audit_path = root / "audit.json"
            write_json(audit_path, build_audit(vault))
            output_dir = root / "dataset"
            summary = extract_dataset(vault, audit_path, output_dir)

            self.assertEqual(summary["quality"]["status"], "pass_with_exclusions")
            self.assertGreaterEqual(summary["records"]["decisions"], 1)
            self.assertEqual(summary["records"]["decisions"], summary["records"]["outcomes"])
            self.assertTrue((output_dir / "decisions.jsonl").exists())
            self.assertTrue((output_dir / "candidates.jsonl").exists())
            self.assertTrue((output_dir / "outcomes.jsonl").exists())

    def test_actual_projection_matches_full_candidate_and_writes_one_compact_row(self) -> None:
        source_row = extraction_paifu_row()
        log = source_row["paifu"]["log"][0]  # type: ignore[index]
        full = extract_round(source_row, log, 0, "fixture.jsonl", 1, "all", "all")  # type: ignore[arg-type]
        actual = extract_round(source_row, log, 0, "fixture.jsonl", 1, "all", "actual")  # type: ignore[arg-type]
        self.assertEqual(full["decisions"], actual["decisions"])
        self.assertEqual(full["outcomes"], actual["outcomes"])
        self.assertEqual(actual["candidates"], [row for row in full["candidates"] if row["isActual"]])
        self.assertEqual(len(actual["candidates"]), len(actual["decisions"]))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "vault"
            write_jsonl(vault / "data" / "mleague" / "paifu" / "2024-25.jsonl", [source_row])
            write_jsonl(
                vault / "data" / "mleague" / "danger" / "against_riichi_decisions_with_danger.jsonl",
                [decision_row(0, eligible=True)],
            )
            shanten_rows = [
                {
                    "season": decision["source"]["season"],
                    "gameId": decision["source"]["gameId"],
                    "roundIndex": decision["source"]["roundIndex"],
                    "logIndex": decision["source"]["logIndex"],
                    "seat": decision["seat"],
                    "discardIndex": decision["source"]["discardIndex"],
                    "shantenBeforeDiscard": decision["shantenBeforeDiscard"],
                    "shantenAfterDiscard": decision["actualShantenAfterDiscard"],
                    "discardTile34": decision["actualDiscardTile34"],
                }
                for decision in actual["decisions"]
            ]
            write_jsonl(vault / "data" / "mleague" / "shanten" / "2024-25.jsonl", shanten_rows)
            audit_path = root / "audit.json"
            write_json(audit_path, build_audit(vault))
            output_dir = root / "observed"
            summary = extract_observed_dataset(vault, audit_path, output_dir)
            rows = [json.loads(line) for line in (output_dir / "observed-actions.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["quality"]["status"], "pass_with_exclusions")
            self.assertEqual(len(rows), summary["records"]["observedActions"])
            self.assertEqual(len(rows), len(actual["decisions"]))
            self.assertEqual(set(rows[0]), {"schemaVersion", "decisionId", "split", "source", "features", "label"})
            self.assertNotIn("rewardPoints", rows[0]["features"])
            self.assertNotIn("rewardPoints", rows[0]["label"])


if __name__ == "__main__":
    unittest.main()
