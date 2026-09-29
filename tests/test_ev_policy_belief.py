"""D.3.3 工程1：家ごとの履歴評価器の受入試験（PHASE_D33_DESIGN.md 14節のD33-06、D33-07）。

実データ（calibration/dataset-opponent-v3/*.jsonl.gz、Git管理外）がない環境では飛ばす。
1,000割当の照合は `python tools/calibrate_ev.py probe-policy-belief` で行い、
ここでは同じ照合を少数の判断で固定seedで行う。
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import random
import sys
import unittest
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "calibration" / ".dependency-cache" / "mahjong-2.0.0-py3-none-any.whl"
if WHEEL.is_file():
    sys.path.insert(0, str(WHEEL))
sys.path.insert(0, str(ROOT))

from tools.ev_policy_belief import (  # noqa: E402
    ContextError,
    DecisionContext,
    SeatFeatureState,
    compare_evaluations,
    evaluate_seat,
    model_resolver,
    random_world,
    reference_evaluations,
    teacher_window_differences,
    world_from_teacher,
)
from tools.ev_policy_fixed import constants_for_seat  # noqa: E402
from tools.ev_policy_opponent import HierarchicalSoftmax, RoundFeatureState, _stratum_attributes_of  # noqa: E402

DATASET = ROOT / "calibration" / "dataset-opponent-v3"
MODEL_DIR = ROOT / "calibration" / "model-opponent-v3"
DECISIONS = 12
HAS_DATA = all((DATASET / name).is_file() for name in (
    "inference-prefixes.jsonl.gz", "public-events.jsonl.gz", "private-events.jsonl.gz", "teacher-windows.jsonl.gz"
))


def _rows(name: str):
    with gzip.open(DATASET / name, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


@lru_cache(maxsize=1)
def fixture() -> dict:
    """先頭から固定件数の判断と、その局の公開・私有・教師窓を読む（列はroundId順）。"""
    prefixes = []
    for row in _rows("inference-prefixes.jsonl.gz"):
        prefixes.append(row)
        if len(prefixes) >= DECISIONS:
            break
    rounds = {row["roundId"] for row in prefixes}

    def collect(name: str) -> dict:
        found: dict = {}
        for row in _rows(name):
            if row["roundId"] in rounds:
                found.setdefault(row["roundId"], []).append(row)
            elif len(found) == len(rounds):
                break
        return found

    public = {key: value[0] for key, value in collect("public-events.jsonl.gz").items()}
    private = {key: {int(r["seat"]): r for r in value} for key, value in collect("private-events.jsonl.gz").items()}
    teacher = collect("teacher-windows.jsonl.gz")
    model = HierarchicalSoftmax.from_dict(json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8")))
    scenarios = json.loads((MODEL_DIR / "fixed-components.json").read_text(encoding="utf-8"))["scenarios"]
    return {"prefixes": prefixes, "public": public, "private": private, "teacher": teacher, "model": model, "scenarios": scenarios}


def context_of(prefix: dict) -> DecisionContext:
    data = fixture()
    round_id = prefix["roundId"]
    return DecisionContext.from_records(prefix, data["public"][round_id], data["private"][round_id][int(prefix["seat"])])


@unittest.skipUnless(HAS_DATA, "D.3.1の実データ（Git管理外）がない")
class D33_06InformationBoundaryTest(unittest.TestCase):
    def test_context_ignores_future_public_events_and_future_own_draws(self) -> None:
        prefix = fixture()["prefixes"][0]
        round_id = prefix["roundId"]
        public = fixture()["public"][round_id]
        target = fixture()["private"][round_id][int(prefix["seat"])]
        original = DecisionContext.from_records(prefix, public, target)
        count = int(prefix["publicEventCount"])

        changed_public = copy.deepcopy(public)
        changed_public["events"] = changed_public["events"][:count] + [{"type": "draw", "seat": 0, "rawEventIndex": 999, "source": "live"}]
        changed_target = copy.deepcopy(target)
        for item in changed_target["events"]:
            if item["type"] == "draw_observation" and int(item["availableAtPublicEventCount"]) > count:
                item["tile"] = {"tile34": (int(item["tile"]["tile34"]) + 1) % 34, "isRed": False}
        changed = DecisionContext.from_records(prefix, changed_public, changed_target)
        self.assertEqual(original.input_hash(), changed.input_hash())

        world = random_world(original, random.Random(5))
        for seat, hypothesis in world.hypotheses.items():
            before = evaluate_seat(original, hypothesis, None)
            after = evaluate_seat(changed, hypothesis, None)
            self.assertEqual(
                [(w.kind, w.event_position, w.legal_actions, w.furiten) for w in before.windows],
                [(w.kind, w.event_position, w.legal_actions, w.furiten) for w in after.windows],
            )

    def test_context_rejects_private_rows_of_other_seats(self) -> None:
        prefix = fixture()["prefixes"][0]
        round_id = prefix["roundId"]
        other = (int(prefix["seat"]) + 1) % 4
        with self.assertRaises(ContextError):
            DecisionContext.from_records(prefix, fixture()["public"][round_id], fixture()["private"][round_id][other])

    def test_other_seat_teacher_hands_do_not_enter_the_context(self) -> None:
        # 文脈は他家の私有列を受け取らないので、教師データを差し替えても入力は同じ。
        prefix = fixture()["prefixes"][1]
        context = context_of(prefix)
        swapped = copy.deepcopy(fixture()["private"][prefix["roundId"]])
        for seat, row in swapped.items():
            if seat != int(prefix["seat"]):
                row["initialHand"] = list(reversed(row["initialHand"]))
        again = DecisionContext.from_records(
            prefix, fixture()["public"][prefix["roundId"]], swapped[int(prefix["seat"])]
        )
        self.assertEqual(context.input_hash(), again.input_hash())


@unittest.skipUnless(HAS_DATA, "D.3.1の実データ（Git管理外）がない")
class D33_07EvaluatorMatchesRoundStateTest(unittest.TestCase):
    def test_random_consistent_assignments_match_reference_with_scenarios(self) -> None:
        data = fixture()
        base = next(s for s in data["scenarios"] if s["id"] == "base")
        stratified = next(s for s in data["scenarios"] if s.get("usableInD33") and s["stratumOverrides"])
        rng = random.Random(20260929)
        compared = 0
        for index, prefix in enumerate(data["prefixes"]):
            context = context_of(prefix)
            resolver = model_resolver(data["model"], base if index % 2 == 0 else stratified)
            world = random_world(context, rng)
            reference = reference_evaluations(context, world, resolver)
            for seat, hypothesis in world.hypotheses.items():
                with self.subTest(decision=prefix["decisionId"], seat=seat):
                    fast = evaluate_seat(context, hypothesis, resolver)
                    self.assertIsNone(fast.violation)
                    self.assertTrue(math.isfinite(fast.log_likelihood))
                    self.assertEqual(compare_evaluations(fast, reference[seat]), [])
                    compared += 1
        self.assertEqual(compared, 3 * len(data["prefixes"]))

    def test_non_tenpai_riichi_hand_is_a_violation_in_both_paths(self) -> None:
        rng = random.Random(7)
        for prefix in fixture()["prefixes"][:6]:
            context = context_of(prefix)
            world = random_world(context, rng, riichi_tenpai=False)
            riichi = context.riichi_seat
            fast = evaluate_seat(context, world.hypotheses[riichi], None)
            reference = reference_evaluations(context, world, None)
            with self.subTest(decision=prefix["decisionId"]):
                self.assertIsNotNone(fast.violation)
                self.assertEqual(fast.log_likelihood, -math.inf)
                self.assertIsNotNone(reference[riichi].violation)

    def test_teacher_hands_reproduce_d31_teacher_windows(self) -> None:
        data = fixture()
        rng = random.Random(11)
        for prefix in data["prefixes"]:
            context = context_of(prefix)
            world = world_from_teacher(context, data["private"][prefix["roundId"]], rng)
            for seat, hypothesis in world.hypotheses.items():
                with self.subTest(decision=prefix["decisionId"], seat=seat):
                    evaluation = evaluate_seat(context, hypothesis, None)
                    self.assertIsNone(evaluation.violation)
                    self.assertEqual(
                        teacher_window_differences(evaluation, data["teacher"][prefix["roundId"]], int(prefix["publicEventCount"])),
                        [],
                    )

    def test_scenario_resolver_equals_direct_with_fixed(self) -> None:
        data = fixture()
        stratified = next(s for s in data["scenarios"] if s.get("usableInD33") and s["stratumOverrides"])
        prefix = data["prefixes"][0]
        context = context_of(prefix)
        world = random_world(context, random.Random(13))
        seat = next(s for s in context.other_seats if s != context.riichi_seat)
        hypothesis = world.hypotheses[seat]
        evaluation = evaluate_seat(context, hypothesis, None)
        window = next(w for w in evaluation.windows if w.kind == "self")
        features = SeatFeatureState(context.public_row(), seat, hypothesis.private_row(context))
        features.advance(window.event_position + 1)
        candidates = features.encode(seat, window.legal_actions)
        direct = data["model"].with_fixed(constants_for_seat(stratified, _stratum_attributes_of(candidates[0])))
        resolved = model_resolver(data["model"], stratified)(candidates)
        self.assertEqual(
            direct.probabilities(candidates, "self_action_after_live").tolist(),
            resolved.probabilities(candidates, "self_action_after_live").tolist(),
        )

    def test_seat_feature_state_matches_full_feature_state(self) -> None:
        # その家だけを追う特徴状態は、全家の手牌を持つ状態と同じ特徴を返す（家ごとの因子分解、設計5.4節）。
        data = fixture()
        prefix = data["prefixes"][2]
        context = context_of(prefix)
        world = random_world(context, random.Random(17))
        private = {context.target_seat: context.target_private_row()}
        private.update({seat: h.private_row(context) for seat, h in world.hypotheses.items()})
        full = RoundFeatureState(context.public_row(), private)
        for seat, hypothesis in world.hypotheses.items():
            evaluation = evaluate_seat(context, hypothesis, None)
            focus = SeatFeatureState(context.public_row(), seat, hypothesis.private_row(context))
            full_state = RoundFeatureState(context.public_row(), private)
            for window in evaluation.windows:
                if len(window.legal_actions) == 1:
                    continue
                focus.advance(window.event_position + 1)
                full_state.advance(window.event_position + 1)
                mine = focus.encode(seat, window.legal_actions)
                theirs = full_state.encode(seat, window.legal_actions)
                for left, right in zip(mine, theirs):
                    self.assertEqual(left.kind_features.tolist(), right.kind_features.tolist())
                    self.assertEqual(left.detail_features.tolist(), right.detail_features.tolist())
        del full


if __name__ == "__main__":
    unittest.main()
