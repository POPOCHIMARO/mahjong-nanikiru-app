import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.ev_calibration_model import (
    COMPACT_FEATURES,
    _cluster_bootstrap_difference,
    _make_row,
    evaluate_immediate_ron_model,
    fit_binary_logistic,
    fit_immediate_ron_model,
    verify_dataset,
)


class CalibrationModelTest(unittest.TestCase):
    @staticmethod
    def _write_c1_dataset(root: Path) -> None:
        split_seasons = {
            "train": "2022-23",
            "selection": "2023-24",
            "calibration": "2024-25",
            "finalTest": "2025-26",
        }
        records = []
        for split_index, (split, season) in enumerate(split_seasons.items()):
            for index in range(20):
                danger = index % 5
                features = {
                    "turn": 5 + index % 10,
                    "remaining_wall": 55 - index,
                    "shanten_after": index % 2,
                    "ukeire_count": 4 + index % 12,
                    "ukeire_kinds": 1 + index % 4,
                    "visible_discard_count": index % 4,
                    "danger_level": danger,
                    "old_danger_probability": 0.01 + danger * 0.01,
                    "self_dealer": index % 2,
                    "opponent_dealer": (index + 1) % 2,
                    "score": 2.5 + index / 100,
                    "score_lead": (index - 10) / 100,
                    "honba": index % 3,
                    "riichi_sticks": 1,
                    "discards_red": 0,
                    "declares_riichi": index % 7 == 0,
                    "dora_count_after": index % 3,
                    "aka_count_after": index % 2,
                    "tsumogiri": index % 4 == 0,
                    "danger_class": "genbutsu" if danger == 0 else "non_suji_456",
                    "safety_group": "safe" if danger == 0 else "high_risk",
                    "turn_bucket": "early" if index < 7 else "middle",
                    "tile_band": "outer" if index % 2 else "middle",
                    "seat_wind": str(index % 4),
                }
                assert set(features) == set(COMPACT_FEATURES)
                records.append(
                    {
                        "schemaVersion": "ev-calibration-immediate-ron-action/v1",
                        "decisionId": f"{split}-{index}",
                        "split": split,
                        "source": {
                            "season": season,
                            "gameId": f"game-{split_index}-{index}",
                            "roundKey": f"round-{split_index}-{index}",
                        },
                        "features": features,
                        "label": {"immediateRonByActiveRiichi": danger == 4},
                    }
                )

        observed = root / "observed-actions.jsonl"
        observed.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        excluded = root / "excluded-rounds.jsonl"
        excluded.write_text("", encoding="utf-8")
        files = []
        aggregate = hashlib.sha256()
        for path in (excluded, observed):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"path": path.name, "bytes": path.stat().st_size, "lines": 0 if path == excluded else 80, "sha256": digest})
            aggregate.update(path.name.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\n")
        summary = {
            "phase": "C1-all-observed-actions",
            "records": {"decisions": 80, "observedActions": 80},
            "splitCounts": {name: 20 for name in split_seasons},
            "generatedFiles": {"files": files, "aggregateSha256": aggregate.hexdigest()},
        }
        (root / "extraction-summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def test_regularized_logistic_learns_signal(self):
        x = np.asarray([[1, -2], [1, -1], [1, -0.5], [1, 0.5], [1, 1], [1, 2]], dtype=float)
        y = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
        beta = fit_binary_logistic(x, y, l2=1.0)
        probability = 1.0 / (1.0 + np.exp(-(x @ beta)))
        self.assertLess(float(np.max(probability[:3])), float(np.min(probability[3:])))
        self.assertTrue(np.all(np.isfinite(beta)))

    def test_physical_discard_removes_only_one_equal_normal_tile(self):
        hand = [
            {"raw": 15, "tile34": 4, "isRed": False},
            {"raw": 15, "tile34": 4, "isRed": False},
            {"raw": 11, "tile34": 0, "isRed": False},
        ]
        decision = {
            "decisionId": "d1",
            "split": "train",
            "source": {"season": "2018-19", "gameId": "g1", "roundIndex": 0, "logIndex": 0},
            "seat": 1,
            "dealerSeat": 0,
            "seatWindIndex": 1,
            "ownTurnIndex": 7,
            "remainingWallTiles": 39,
            "scoresAtDecision": [25000] * 4,
            "honba": 0,
            "riichiSticksAtDecision": 1,
            "handBeforeAction": hand,
            "drawnTile": hand[2],
            "publicDoraIndicators": [{"tile34": 3}],
            "visibleCounts": [0] * 34,
        }
        candidate = {
            "riichiSeat": 0,
            "discardTile34": 4,
            "discardRaw": 15,
            "shantenAfterDiscard": 1,
            "ukeireCount": 8,
            "ukeireKinds": 2,
            "dangerLevel": 9,
            "dangerClass": "terminal_non_suji",
            "safetyGroup": "high_risk",
            "discardsRed": False,
            "riichiDeclaration": False,
        }
        outcome = {
            "immediateRonByActiveRiichi": False,
            "rewardPoints": 0,
            "resultClass": "draw",
        }
        row = _make_row(decision, candidate, outcome)
        self.assertEqual(row["dora_count_after"], 1)
        self.assertAlmostEqual(row["old_danger_probability"], 0.034)

    def test_dataset_manifest_detects_content_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            aggregate = hashlib.sha256()
            for name in ("decisions.jsonl", "candidates.jsonl", "outcomes.jsonl", "rejections.jsonl"):
                path = root / name
                path.write_text("{}\n", encoding="utf-8")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                files.append({"path": name, "bytes": path.stat().st_size, "lines": 1, "sha256": digest})
                aggregate.update(name.encode("utf-8"))
                aggregate.update(b"\0")
                aggregate.update(digest.encode("ascii"))
                aggregate.update(b"\n")
            (root / "extraction-summary.json").write_text(
                json.dumps({"generatedFiles": {"files": files, "aggregateSha256": aggregate.hexdigest()}}),
                encoding="utf-8",
            )
            verify_dataset(root)
            (root / "outcomes.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "整合性検査"):
                verify_dataset(root)

    def test_cluster_bootstrap_is_deterministic_and_uses_matches(self):
        truth = np.asarray([0, 1, 0, 1, 0, 1], dtype=float)
        new = np.asarray([0.1, 0.8, 0.2, 0.9, 0.3, 0.7], dtype=float)
        old = np.full(6, 0.5)
        groups = ["a", "a", "b", "b", "c", "c"]
        first = _cluster_bootstrap_difference(truth, new, old, groups, "brier", 200, 17)
        second = _cluster_bootstrap_difference(truth, new, old, groups, "brier", 200, 17)
        self.assertEqual(first, second)
        self.assertEqual(first["matches"], 3)
        self.assertEqual(first["status"], "pass")

    def test_c1_fit_and_evaluate_are_immediate_ron_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            model = root / "model"
            dataset.mkdir()
            self._write_c1_dataset(dataset)

            fit = fit_immediate_ron_model(dataset, model)
            evaluation = evaluate_immediate_ron_model(dataset, model, replicates=100, seed=17)

            self.assertEqual(fit["phase"], "C1-all-observed-actions-immediate-ron")
            self.assertEqual(evaluation["status"], "hold")
            self.assertNotIn("reward", evaluation)
            self.assertIn("rewardPoints", evaluation["excludedTargets"])
            self.assertNotIn("rewardPoints", (model / "model.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
