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
    clear_shape_cache,
    exact_shape,
    make_action_view,
    project_action,
    sha256_file,
    shape_cache_info,
)
from tools.ev_policy_opponent import (  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
