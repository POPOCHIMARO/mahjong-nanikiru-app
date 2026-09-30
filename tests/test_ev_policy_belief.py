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
    PublicFeatureState,
    SeatFeatureState,
    build_smc_events,
    YAOCHUU,
    compare_evaluations,
    construct_initial_world,
    counts34,
    evaluate_seat,
    sample_tenpai_shape,
    validate_family_probabilities,
    id_key,
    initial_mahjong_particles,
    model_resolver,
    random_world,
    reference_evaluations,
    reference_smc_gate,
    run_smc,
    teacher_window_differences,
    toy_exact,
    toy_initial_particles,
    toy_problem,
    world_from_teacher,
)
from tools.ev_calibration_state import shanten  # noqa: E402
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


class D33_09ReferenceSmcFiniteExampleTest(unittest.TestCase):
    """基準SMCを全列挙できる有限小例で走らせ、正規化定数と事後分布を正解と照合する。"""

    NON_DEGENERATE = [
        ("opp", (1, "concealed", False)), ("target_draw", 2), ("opp", (0, "drawn", False)), ("opp", (2, "concealed", False)),
    ]
    WITH_RIICHI = [("opp", (1, "concealed", False)), ("opp", (2, "drawn", False)), ("opp", (0, "concealed", True))]

    def _check(self, observations: list, seed: int) -> None:
        pool, events = toy_problem(0, observations)
        exact_z, exact_posterior = toy_exact(pool, events)
        self.assertGreater(exact_z, 0.0)
        rng = random.Random(seed)
        result = run_smc(toy_initial_particles(pool, 20_000, rng), events, rng)
        self.assertEqual(result.status, "ok")
        # 粒子数20,000での正規化定数の相対誤差と、事後確率の絶対誤差（固定seed）。
        self.assertLess(abs(math.exp(result.log_normalizer) / exact_z - 1.0), 0.05)
        estimate: dict = {}
        for particle, weight in zip(result.particles, result.normalized_weights()):
            key = tuple(sorted(particle.hand))
            estimate[key] = estimate.get(key, 0.0) + weight
        for key in set(exact_posterior) | set(estimate):
            self.assertLess(abs(exact_posterior.get(key, 0.0) - estimate.get(key, 0.0)), 0.02, key)
        self.assertGreater(result.resampling_count, 0)
        for record in result.event_diagnostics:
            for field in ("essBefore", "essAfter", "maxWeight", "initialAncestors", "previousAncestors", "positiveG"):
                self.assertIn(field, record)

    def test_normalizer_and_posterior_match_exact_enumeration(self) -> None:
        self._check(self.NON_DEGENERATE, seed=1)

    def test_riichi_hard_constraint_matches_exact_enumeration(self) -> None:
        self._check(self.WITH_RIICHI, seed=2)

    def test_all_zero_weights_return_posterior_zero_mass_without_numbers(self) -> None:
        # 3枚目の1を2回切らせる：プールに1が残らない観測。
        observations = [("opp", (1, "drawn", False)), ("opp", (1, "drawn", False)), ("opp", (1, "drawn", False))]
        pool, events = toy_problem(1, observations)
        self.assertEqual(toy_exact(pool, events)[0], 0.0)
        rng = random.Random(3)
        result = run_smc(toy_initial_particles(pool, 500, rng), events, rng)
        self.assertEqual(result.status, "posterior_zero_mass")
        self.assertIsNone(result.log_normalizer)
        self.assertIsNotNone(result.zero_mass_event)
        self.assertTrue(all(value == -math.inf for value in result.log_weights))


class ReferenceSmcGateTest(unittest.TestCase):
    """3.3節の関門：N=4,096の64反復だけで判定し、低い粒子数の失敗は分岐に使わない。"""

    def runs(self, zero_at: dict) -> list:
        return [
            {"particleCount": n, "status": "posterior_zero_mass" if zero_at.get(n, 0) > i else "ok"}
            for n in (256, 1024, 4096) for i in range(64)
        ]

    def test_any_zero_mass_at_4096_proceeds_to_mcmc(self) -> None:
        self.assertEqual(reference_smc_gate(self.runs({4096: 1}))["verdict"], "proceed_to_mcmc_implementation")

    def test_failures_only_at_lower_counts_return_to_design_as_undetermined(self) -> None:
        self.assertEqual(reference_smc_gate(self.runs({256: 64, 1024: 10}))["verdict"], "return_to_design_as_undetermined")

    def test_incomplete_4096_runs_are_not_judged(self) -> None:
        # N=4,096が63反復しかなく、失敗もない場合は判定しない。
        self.assertEqual(reference_smc_gate(self.runs({})[:-1])["verdict"], "incomplete")


@unittest.skipUnless(HAS_DATA, "D.3.1の実データ（Git管理外）がない")
class ReferenceSmcAdapterTest(unittest.TestCase):
    """麻雀への適用部分：真の経路をたどったときの行動確率の和が、工程1の評価器と一致する。"""

    def test_forced_true_path_matches_seat_evaluator(self) -> None:
        data = fixture()
        base = next(s for s in data["scenarios"] if s["id"] == "base")
        resolver = model_resolver(data["model"], base)
        for index in (0, 3):
            prefix = data["prefixes"][index]
            context = context_of(prefix)
            world = random_world(context, random.Random(40 + index))
            events, _ = build_smc_events(context, resolver)
            particle = initial_mahjong_particles(context, 1, random.Random(0))[0]
            # 粒子を割当の配牌で置き換え、残りをプールにする。
            hands = {seat: h.initial for seat, h in world.hypotheses.items()}
            used = set(i for hand in hands.values() for i in hand)
            particle.rules = {seat: particle.rules[seat].__class__(context, seat, list(hand)) for seat, hand in hands.items()}
            particle.initial = dict(hands)
            known = list(world.target_initial) + [world.dora_indicator]
            particle.pool = {}
            for tile_id in range(136):
                if tile_id in used or tile_id in known:
                    continue
                particle.pool.setdefault(id_key(tile_id), []).append(tile_id)
            particle.pool_size = sum(len(v) for v in particle.pool.values())
            for event in events:
                terms = event.expand(particle)
                self.assertTrue(terms, event.name)
                if hasattr(event, "turn"):
                    truth = id_key(world.hypotheses[event.seat].draws[event.turn.raw_event_index])
                    choice = next(c for _, c in terms if c[0] == truth)
                else:
                    choice = terms[0][1]
                event.apply(particle, choice)
            expected = sum(evaluate_seat(context, h, resolver).log_likelihood for h in world.hypotheses.values())
            with self.subTest(decision=prefix["decisionId"]):
                self.assertAlmostEqual(particle.log_model, expected, places=9)

    def test_public_feature_state_matches_seat_feature_state(self) -> None:
        data = fixture()
        context = context_of(data["prefixes"][4])
        world = random_world(context, random.Random(21))
        public = PublicFeatureState(context.public_row())
        for seat, hypothesis in world.hypotheses.items():
            evaluation = evaluate_seat(context, hypothesis, None)
            focus = SeatFeatureState(context.public_row(), seat, hypothesis.private_row(context))
            for window in evaluation.windows:
                if len(window.legal_actions) == 1:
                    continue
                focus.advance(window.event_position + 1)
                if public.position <= window.event_position + 1:
                    public.advance(window.event_position + 1)
                hand = [id for id in _hand_at(context, hypothesis, window.event_position)]
                mine = public.encode_for(seat, hand, window.legal_actions)
                theirs = focus.encode(seat, window.legal_actions)
                for left, right in zip(mine, theirs):
                    self.assertEqual(left.kind_features.tolist(), right.kind_features.tolist())
                    self.assertEqual(left.detail_features.tolist(), right.detail_features.tolist())
            public = PublicFeatureState(context.public_row())


@unittest.skipUnless(HAS_DATA, "D.3.1の実データ（Git管理外）がない")
class D33_03ConstructiveInitializationTest(unittest.TestCase):
    def _initial_worlds(self, seeds: range, resolver=None):
        for prefix in fixture()["prefixes"]:
            context = context_of(prefix)
            for seed in seeds:
                yield prefix, context, construct_initial_world(context, random.Random(seed), resolver=resolver)

    def test_initial_worlds_satisfy_tenpai_inventory_and_discard_consistency(self) -> None:
        resolver = model_resolver(fixture()["model"])
        for prefix, context, initial in self._initial_worlds(range(2), resolver):
            with self.subTest(decision=prefix["decisionId"]):
                self.assertEqual(initial.status, "ok")
                initial.world.validate()  # 136枚をちょうど1回ずつ使う（牌在庫）
                riichi = initial.world.hypotheses[context.riichi_seat]
                final = _hand_at(context, riichi, len(context.events) - 1)
                self.assertEqual(shanten(counts34(final), 0), 0)
                for seat, hypothesis in initial.world.hypotheses.items():
                    evaluation = evaluate_seat(context, hypothesis, resolver)
                    self.assertIsNone(evaluation.violation)
                    self.assertTrue(math.isfinite(evaluation.log_likelihood))
                    self.assertAlmostEqual(evaluation.log_likelihood, initial.log_likelihood[seat], places=12)

    def test_furiten_riichi_is_accepted(self) -> None:
        # 待ちがリーチ者自身の河にある形（フリテンリーチ）も出発点として受け入れる。
        for prefix, context, initial in self._initial_worlds(range(40)):
            evaluation = evaluate_seat(context, initial.world.hypotheses[context.riichi_seat], None)
            responses = [w for w in evaluation.windows if w.kind == "response"]
            if initial.status == "ok" and responses and "own_discard" in responses[-1].furiten:
                return
        self.fail("フリテンリーチの出発点が見つからない")

    def test_passing_a_winning_tile_after_riichi_sets_riichi_furiten_and_removes_ron(self) -> None:
        found = False
        for prefix, context, initial in self._initial_worlds(range(40)):
            evaluation = evaluate_seat(context, initial.world.hypotheses[context.riichi_seat], None)
            if not evaluation.riichi_furiten:
                continue
            responses = [w for w in evaluation.windows if w.kind == "response"]
            first = next(i for i, w in enumerate(responses) if "riichi_pass" in w.furiten)
            # 見送った窓ではロンが合法だった（またはフリテン前の待ち牌が出た）。以後はロンが合法集合から消える。
            for window in responses[first:]:
                self.assertNotIn("ron", {a["kind"] for a in window.legal_actions})
            found = True
            break
        self.assertTrue(found, "リーチ後の見逃しがある出発点が見つからない")

    def test_initialization_reads_only_the_decision_context(self) -> None:
        prefix = fixture()["prefixes"][1]
        context = context_of(prefix)
        swapped = copy.deepcopy(fixture()["private"][prefix["roundId"]])
        for seat, row in swapped.items():
            if seat != int(prefix["seat"]):
                row["initialHand"] = list(reversed(row["initialHand"]))
                for item in row["events"]:
                    if item["type"] == "draw_observation":
                        item["tile"] = {"tile34": 0, "isRed": False}
        other = DecisionContext.from_records(prefix, fixture()["public"][prefix["roundId"]], swapped[int(prefix["seat"])])
        first = construct_initial_world(context, random.Random(9))
        second = construct_initial_world(other, random.Random(9))
        self.assertEqual(first.world, second.world)

    def test_budget_exhaustion_is_init_failed(self) -> None:
        context = context_of(fixture()["prefixes"][0])
        for options in ({"max_attempts": 0}, {"max_seconds": 0.0}):
            with self.subTest(options=options):
                result = construct_initial_world(context, random.Random(1), **options)
                self.assertEqual(result.status, "init_failed")
                self.assertIsNone(result.world)


class TenpaiGeneratorTest(unittest.TestCase):
    def test_family_probabilities_must_all_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            validate_family_probabilities((("regular", 0.95), ("chiitoi", 0.05), ("kokushi", 0.0)))

    def test_each_family_produces_tenpai_shapes(self) -> None:
        rng = random.Random(4)
        for family in ("regular", "chiitoi", "kokushi"):
            probabilities = tuple((name, 0.98 if name == family else 0.01) for name in ("regular", "chiitoi", "kokushi"))
            seen = 0
            for _ in range(300):
                name, shape = sample_tenpai_shape(rng, probabilities)
                counts = [shape.count(t) for t in range(34)]
                self.assertEqual(len(shape), 13)
                if name != family or max(counts) > 4:
                    continue  # 5枚以上の形は在庫の段で除く
                seen += 1
                self.assertEqual(shanten(tuple(counts), 0), 0)
                if family == "kokushi":
                    self.assertGreaterEqual(sum(counts[t] > 0 for t in YAOCHUU), 12)
            self.assertGreater(seen, 100)


def _hand_at(context: DecisionContext, hypothesis, position: int) -> list[int]:
    """その家の、公開イベントpositionの直後の手牌（物理ID）。"""
    hand = list(hypothesis.initial)
    for index, event in enumerate(context.events[: position + 1]):
        if event["type"] == "draw" and int(event["seat"]) == hypothesis.seat:
            drawn = hypothesis.draws[int(event["rawEventIndex"])]
            hand.append(drawn)
        elif event["type"] == "discard" and int(event["seat"]) == hypothesis.seat:
            key = (int(event["tile"]["tile34"]), bool(event["tile"]["isRed"]))
            if event["origin"] == "drawn":
                hand.remove(drawn)
            else:
                hand.remove(next(i for i in hand if id_key(i) == key and i != drawn))
    return hand


# ===========================================================================
# 工程4：MCMCの移動（D33-01、D33-02、D33-03の積み残し）
# ===========================================================================
#
# 小例では、支持（H=1）の物理状態をすべて列挙し、実装した移動の遷移行列を正確に作る。
# M1・M2は移動そのものを全分岐列挙する。M3・M4は実装と同じ3段（提案→受理比→復元）に分け、
# 提案と復元をそれぞれ全分岐列挙して合成する（regenerate_stepはこの3段の合成）。

import itertools  # noqa: E402
from collections import Counter  # noqa: E402
from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402

import tools.ev_policy_belief as belief  # noqa: E402
from tools.ev_policy_belief import (  # noqa: E402
    BeliefLayout,
    BeliefProblem,
    ExplicitTenpaiRules,
    MahjongTenpaiRules,
    MoveStatistics,
    RandomChooser,
    SeatLayout,
    TurnLayout,
    enumerate_outcomes,
    initial_chain_state,
    mahjong_belief,
    regular_path_numerator,
    seat_type_state,
)
from tools.ev_policy_belief import MENTSU_KINDS  # noqa: E402

TOL = 1e-12


class FlatModel:
    """尤度1（H=1の状態を等しく扱う）。"""

    def log_factor(self, i, initial, draws):
        return 0.0


class ToyModel:
    """牌種水準の状態だけで決まる正の尤度。家ごと・位置ごとに固定の係数を持つ。"""

    def __init__(self, initial_scores, draw_scores):
        self.initial_scores, self.draw_scores = initial_scores, draw_scores

    def log_factor(self, i, initial, draws):
        return math.fsum(self.initial_scores[i].get(t, 0.0) for t in initial) + math.fsum(
            self.draw_scores[i].get((k, t), 0.0) for k, t in enumerate(draws))


def _rules(families, shapes, class_count, hand_size):
    return ExplicitTenpaiRules(tuple(families), tuple(shapes), class_count, hand_size)


def kokushi_example(model=None, families=(("regular", 0.7), ("kokushi", 0.3)), *, discard_type=1, wide=False):
    """国士に相当する系統が、M1・M2では孤立する例。

    牌種（類）：0×2、1（通常と赤11）、2、3。テンパイ形（手2枚＋1枚で完成）は、
    面子手＝{0,1}だけの形、国士＝{2,3}だけの形。面子手の{0,0}は5枚目に当たる000からも生成される。
    リーチ者rは手出し1回（打牌discard_type）とツモ切り1回（牌種4、位置は固定）、もう1家jは配牌2枚、プール1枚。
    discard_type=0なら打牌の牌種が2枚あり、宣言時点で手に残る位置が配置で変わる。
    wide=Trueならテンパイに関わらない牌種5を1枚加えてプールを2枚にする（M3のブロックに国士の牌がそろう）。
    """
    rules = _rules(families, (("regular", ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1))),
                              ("kokushi", ((2, 2, 3), (2, 3, 3)))), 6, 2)
    seats = (SeatLayout(1, True, (TurnLayout(False, discard_type), TurnLayout(True, 4))), SeatLayout(2, False, ()))
    types = [0, 0, 1, 11, 2, 3] + ([5] if wide else [])
    layout = BeliefLayout(2, seats, 2 if wide else 1, types, {0: 0, 1: 1, 11: 1, 2: 2, 3: 3, 4: 4, 5: 5})
    model = model or ToyModel([{0: 0.3, 1: -0.2, 11: 0.5, 2: 0.1, 3: -0.4}, {0: -0.1, 1: 0.2, 11: 0.6, 2: -0.5, 3: 0.3}],
                              [{(0, 0): 0.2, (0, 11): -0.3, (0, 2): 0.4}, {}])
    return BeliefProblem(layout, rules, model, max_generator_attempts=2)


def fixed_tiles_example(model=None):
    """目的の形に必要な牌が別の家に固定され、プールが空の例（M4だけが家の間で牌を移せる）。

    牌種：0×2、1×2、2、5。リーチ者の面子手は{0,0}か{1,1}（国士系統の形は在庫になく常に棄却）。
    rは手出し1回（打牌2）、jは手出し1回（打牌5）。
    """
    rules = _rules((("regular", 0.8), ("kokushi", 0.2)), (("regular", ((0, 0, 0), (1, 1, 1))), ("kokushi", ((3, 3, 4),))), 6, 2)
    seats = (SeatLayout(1, True, (TurnLayout(False, 2),)), SeatLayout(3, False, (TurnLayout(False, 5),)))
    layout = BeliefLayout(2, seats, 0, [0, 0, 1, 1, 2, 5], {t: t for t in range(6)})
    model = model or ToyModel([{0: 0.4, 1: -0.3, 2: 0.2}, {0: -0.2, 1: 0.5, 5: 0.1}],
                              [{(0, 0): 0.3, (0, 1): -0.1}, {(0, 0): -0.4, (0, 1): 0.2, (0, 5): 0.1}])
    return BeliefProblem(layout, rules, model, max_generator_attempts=2)


def v_asymmetry_example():
    """M2の`|V(z,a)|`が交換の前後で変わる例（`|V|`補正を省いた変異の検出用）。

    他家の打牌制約がない例では、aに置ける牌種の集合が交換で変わらず、`|V|`も変わらない。
    ここでは通常の1を1枚足し、リーチ者は赤11、他家jは0を手出しで切る（jの0が1枚だけの状態がある）。
    """
    base = kokushi_example()
    seats = (SeatLayout(1, True, (TurnLayout(False, 11), TurnLayout(True, 4))), SeatLayout(2, False, (TurnLayout(False, 0),)))
    layout = BeliefLayout(2, seats, 1, [0, 0, 1, 11, 2, 3, 1], base.layout.class_of)
    model = ToyModel([{0: 0.3, 1: -0.2, 11: 0.5, 2: 0.1, 3: -0.4}, {0: -0.1, 1: 0.2, 11: 0.6, 2: -0.5, 3: 0.3}],
                     [{(0, 0): 0.2, (0, 11): -0.3, (0, 2): 0.4}, {(0, 0): 0.1, (0, 2): -0.2}])
    return BeliefProblem(layout, base.rules, model, max_generator_attempts=2)


PRIOR_RULES = _rules((("regular", 1.0),), (("regular", ((0, 0, 0),)),), 3, 2)


def prior_pool_example():
    """事前分布の検査用（リーチ者なし、打牌なし）：手1枚の2家とプール3枚。プールの並びと`Π 1/pool!`を検査する。"""
    seats = (SeatLayout(1, False, ()), SeatLayout(2, False, ()))
    layout = BeliefLayout(1, seats, 3, [0, 0, 1, 11, 2], {0: 0, 1: 1, 11: 1, 2: 2})
    return BeliefProblem(layout, PRIOR_RULES, FlatModel())


def prior_hand_example():
    """事前分布の検査用：手2枚の2家とプール1枚。配牌の`Π 1/h0!`を検査する。"""
    seats = (SeatLayout(1, False, ()), SeatLayout(2, False, ()))
    layout = BeliefLayout(2, seats, 1, [0, 0, 1, 1, 2], {0: 0, 1: 1, 2: 2})
    return BeliefProblem(layout, PRIOR_RULES, FlatModel())


class ExactChain:
    """小例の支持状態の全列挙と、移動ごとの正確な遷移行列。"""

    def __init__(self, problem):
        self.problem = problem
        layout = problem.layout
        self.states = [
            perm for perm in itertools.permutations(range(layout.size))
            if all(belief.seat_hard_violation(problem, i, *seat_type_state(layout, perm, i)) is None
                   for i in range(len(layout.seats)))
        ]
        self.index = {state: k for k, state in enumerate(self.states)}
        self.base = {state: initial_chain_state(problem, state) for state in self.states}
        logs = np.array([math.fsum(self.base[state].log_factors) for state in self.states])
        weights = np.exp(logs - logs.max())
        self.target = weights / weights.sum()  # p0（物理配置で一様）× H × L を正規化

    def _row(self, matrix, k, outcomes):
        for outcome, probability in outcomes.items():
            matrix[k, self.index[outcome]] += probability

    def swap_matrix(self, kernel):
        """M1・M2：移動全体を全分岐列挙する。"""
        size = len(self.states)
        matrix = np.zeros((size, size))
        for k, state in enumerate(self.states):
            def step(chooser, state=state):
                current = self.base[state].copy()
                kernel(self.problem, current, chooser, MoveStatistics())
                return tuple(current.ids)
            self._row(matrix, k, enumerate_outcomes(step))
        return matrix

    def m1(self):
        return self.swap_matrix(belief.m1_step)

    def m2(self, a=None):
        return self.swap_matrix(lambda p, s, c, st: belief.m2_step(p, s, c, st, a=a))

    def regeneration(self, seats):
        """M3・M4：提案（牌種状態だけに依存）→受理比→復元を、regenerate_stepと同じ順に合成する。"""
        size = len(self.states)
        matrix = np.zeros((size, size))
        proposals: dict = {}
        for k, state in enumerate(self.states):
            current = self.base[state]
            block = belief.regeneration_block(self.problem, state, seats)
            key = (block.order, tuple(sorted(block.available.items())))
            if key not in proposals:
                def propose(chooser, block=block):
                    proposed = belief.propose_regeneration(self.problem, block, chooser)
                    return None if proposed is None else tuple(sorted(proposed.items()))
                proposals[key] = enumerate_outcomes(propose)
            for proposal, probability in proposals[key].items():
                if proposal is None:
                    matrix[k, k] += probability  # 生成器の上限：現状維持
                    continue
                proposed = dict(proposal)
                log_ratio, _ = belief.regeneration_log_ratio(self.problem, block, current.log_factors, proposed)
                alpha = 1.0 if log_ratio >= 0 else math.exp(log_ratio)
                matrix[k, k] += probability * (1.0 - alpha)
                pool = belief.regeneration_pool(self.problem, block, proposed)

                def restore(chooser, proposed=proposed, pool=pool, block=block, state=state):
                    moved = self.base[state].copy()
                    belief.restore_block(self.problem, moved, block.order, proposed, pool, chooser)
                    return tuple(moved.ids)
                for outcome, share in enumerate_outcomes(restore).items():
                    matrix[k, self.index[outcome]] += probability * alpha * share
        return matrix

    def kernels(self):
        """個々の核（名前、行列）。M2はaを固定した核ごと、M3は家ごと。"""
        layout = self.problem.layout
        result = [("m1", self.m1())]
        if layout.riichi_index is not None:
            result += [(f"m2:a={a}", self.m2(a)) for a in layout.seat_positions(layout.riichi_index)]
        result += [(f"m3:{i}", self.regeneration((i,))) for i in range(len(layout.seats))]
        result.append(("m4", self.regeneration(range(len(layout.seats)))))
        return result

    def iteration(self, *, m3=True, m4=True):
        """7.5節の1反復（順序付き合成）の行列。"""
        layout = self.problem.layout
        result = np.linalg.matrix_power(self.m1(), layout.seat_size)
        if layout.riichi_index is not None:
            result = result @ np.linalg.matrix_power(self.m2(), len(layout.seat_positions(layout.riichi_index)))
        if m3:
            for i in layout.m3_order:
                result = result @ self.regeneration((i,))
        if m4:
            result = result @ self.regeneration(range(len(layout.seats)))
        return result


@lru_cache(maxsize=None)
def exact(name: str) -> ExactChain:
    return ExactChain({
        "kokushi": kokushi_example, "kokushi_flat": lambda: kokushi_example(FlatModel()),
        "kokushi_discard0": lambda: kokushi_example(discard_type=0),
        "v_asymmetry": v_asymmetry_example,
        "kokushi_wide_flat": lambda: kokushi_example(FlatModel(), wide=True),
        "fixed": fixed_tiles_example, "fixed_flat": lambda: fixed_tiles_example(FlatModel()),
        "prior_pool": prior_pool_example, "prior_hand": prior_hand_example,
    }[name]())


def invariance_error(pi, matrix) -> float:
    return float(np.abs(pi @ matrix - pi).max())


def balance_error(pi, matrix) -> float:
    flow = pi[:, None] * matrix
    return float(np.abs(flow - flow.T).max())


def stationary(matrix) -> np.ndarray:
    size = matrix.shape[0]
    system = matrix.T - np.eye(size)
    system[-1, :] = 1.0
    rhs = np.zeros(size)
    rhs[-1] = 1.0
    return np.linalg.solve(system, rhs)


def strongly_connected(matrix) -> bool:
    edges = matrix > 0

    def reach(adjacency) -> int:
        seen, frontier = {0}, [0]
        while frontier:
            node = frontier.pop()
            for nxt in np.nonzero(adjacency[node])[0]:
                if int(nxt) not in seen:
                    seen.add(int(nxt))
                    frontier.append(int(nxt))
        return len(seen)

    return reach(edges) == len(edges) and reach(edges.T) == len(edges)


def family_of(chain: ExactChain, state) -> tuple:
    problem = chain.problem
    r = problem.layout.riichi_index
    final = belief.seat_final_hand(problem.layout, r, *seat_type_state(problem.layout, state, r))
    return belief.tenpai_families(problem.rules, belief.class_counts(problem, final))


EXAMPLES = ("kokushi", "fixed", "prior_pool", "prior_hand")


class D33_01TransitionMatrixTest(unittest.TestCase):
    """有限小例の正確な遷移行列で、個々の核の詳細釣合い、1反復の不変性、既約性を確かめる。"""

    def test_each_kernel_satisfies_detailed_balance(self) -> None:
        for name in EXAMPLES:
            chain = exact(name)
            for kernel, matrix in chain.kernels():
                with self.subTest(example=name, kernel=kernel):
                    self.assertLess(float(np.abs(matrix.sum(axis=1) - 1.0).max()), TOL)
                    self.assertLess(balance_error(chain.target, matrix), TOL)
                    self.assertLess(invariance_error(chain.target, matrix), TOL)

    def test_one_iteration_preserves_the_target(self) -> None:
        for name in EXAMPLES:
            with self.subTest(example=name):
                chain = exact(name)
                self.assertLess(invariance_error(chain.target, chain.iteration()), TOL)

    def test_iteration_is_irreducible_with_the_exact_posterior_as_unique_stationary(self) -> None:
        for name in EXAMPLES:
            with self.subTest(example=name):
                chain = exact(name)
                matrix = chain.iteration()
                # M4を最後に置くので、支持の任意の2状態間で1反復の遷移確率が正（7.4節の論証の実装検査）。
                self.assertTrue(bool((matrix > 0).all()))
                self.assertLess(float(np.abs(stationary(matrix) - chain.target).max()), 1e-10)

    def test_restore_with_fixed_pool_order_is_detected(self) -> None:
        chain = ExactChain(prior_pool_example())
        with mock.patch.object(belief, "arrange_pool_types", lambda items, chooser: list(items)):
            errors = [invariance_error(chain.target, chain.regeneration(seats)) for seats in ((0,), (1,), (0, 1))]
        self.assertGreater(max(errors), 1e-6)

    def test_removing_m3_and_m4_breaks_irreducibility(self) -> None:
        for name in ("kokushi", "fixed"):
            with self.subTest(example=name):
                chain = exact(name)
                self.assertFalse(strongly_connected(chain.iteration(m3=False, m4=False)))
                self.assertTrue(strongly_connected(chain.iteration()))
        # 国士の例：M1・M2だけでは系統をまたぐ遷移の確率が0。
        chain = exact("kokushi")
        local = chain.iteration(m3=False, m4=False)
        families = [family_of(chain, state) for state in chain.states]
        crossing = sum(local[i, j] for i in range(len(families)) for j in range(len(families)) if families[i] != families[j])
        self.assertEqual(crossing, 0.0)
        self.assertEqual(set(families), {("regular",), ("kokushi",)})
        # 牌が別の家に固定され、プールが空の例：M3があってもM4を外すと分断が残る。
        self.assertFalse(strongly_connected(exact("fixed").iteration(m4=False)))

    def test_zero_family_probability_is_rejected_before_start(self) -> None:
        with self.assertRaises(ValueError):
            kokushi_example(families=(("regular", 1.0), ("kokushi", 0.0)))
        layout = kokushi_example().layout
        with self.assertRaises(ValueError):
            BeliefProblem(layout, MahjongTenpaiRules((("regular", 0.95), ("chiitoi", 0.05), ("kokushi", 0.0))), FlatModel())

    def test_iteration_runs_moves_in_the_fixed_order(self) -> None:
        problem = kokushi_example()
        state = initial_chain_state(problem, exact("kokushi").states[0])
        calls = []
        original = belief.regenerate_step

        def record(problem, state, seats, chooser, stats, move):
            calls.append(move)
            return original(problem, state, seats, chooser, stats, move)

        stats = MoveStatistics()
        with mock.patch.object(belief, "regenerate_step", record):
            belief.mcmc_iteration(problem, state, RandomChooser(random.Random(5)), stats)
        self.assertEqual(stats.proposed["m1"], problem.layout.seat_size)
        self.assertEqual(stats.proposed["m2"], 3)
        self.assertEqual(calls, ["m3:1", "m3:2", "m4"])  # リーチ者、他家の順にM3、最後にM4

    def test_random_execution_matches_exact_rows(self) -> None:
        """乱数実行（RandomChooser）の遷移頻度が、正確な行列の行と一致する（固定seed、カイ二乗）。"""
        chain = exact("kokushi")
        start = chain.states[0]
        rng = random.Random(0)
        for label, run, matrix in (
            ("m4", lambda s: belief.m4_step(chain.problem, s, RandomChooser(rng), MoveStatistics()),
             chain.regeneration(range(2))),
            ("iteration", lambda s: belief.mcmc_iteration(chain.problem, s, RandomChooser(rng), MoveStatistics()),
             chain.iteration()),
        ):
            with self.subTest(move=label):
                rng.seed(11 if label == "m4" else 12)
                counts: Counter = Counter()
                trials = 20_000
                for _ in range(trials):
                    state = chain.base[start].copy()
                    run(state)
                    counts[chain.index[tuple(state.ids)]] += 1
                assert_frequencies(self, counts, matrix[chain.index[start]], trials)


def chi_square_critical(df: int, z: float = 3.090232) -> float:
    """カイ二乗分布の上側0.1%点（Wilson–Hilferty近似。有意水準0.001をテスト内に固定）。"""
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def assert_frequencies(case: unittest.TestCase, counts: Counter, probabilities, trials: int) -> None:
    """期待度数5未満のセルをまとめてからカイ二乗検定する。確率0のセルに観測があれば不合格。"""
    statistic, pooled_expected, pooled_observed, cells = 0.0, 0.0, 0, 0
    for index, probability in enumerate(probabilities):
        observed = counts.get(index, 0)
        if probability <= 0:
            case.assertEqual(observed, 0, f"確率0の結果が出た: {index}")
            continue
        expected = probability * trials
        if expected < 5:
            pooled_expected += expected
            pooled_observed += observed
            continue
        statistic += (observed - expected) ** 2 / expected
        cells += 1
    if pooled_expected > 0:
        statistic += (pooled_observed - pooled_expected) ** 2 / pooled_expected
        cells += 1
    case.assertLess(statistic, chi_square_critical(max(cells - 1, 1)))


def remaining_hand_positions(problem, ids):
    """変異用：リーチ者の位置のうち、宣言時点で手中に残る牌の位置（評価器と同じく最初の同種を切る）。"""
    layout = problem.layout
    r = layout.riichi_index
    hand = list(layout.initial_positions(r))
    draws = iter(layout.draw_positions(r))
    for turn in layout.seats[r].turns:
        if turn.tsumogiri:
            continue
        drawn = next(draws)
        hand.append(drawn)
        hand.remove(next(p for p in hand if p != drawn and layout.type_of[ids[p]] == turn.discard_type))
    return hand


def different_type_partner(layout, ids, a, chooser):
    """変異用：M1で同じ牌種どうしの交換を候補から除く。"""
    options = [b for b in range(layout.size) if b != a and layout.type_of[ids[b]] != layout.type_of[ids[a]]]
    return options[chooser.index(len(options))]


def first_path_density(rules, counts):
    """変異用：生成経路の和をとらず、最初に見つかった1経路だけを数える。"""
    for family, probability in rules.families:
        for removed in range(rules.class_count):
            complete = counts[:removed] + (counts[removed] + 1,) + counts[removed + 1:]
            value = rules.complete_probability(family, complete)
            if value > 0:
                return probability * value * complete[removed] / (rules.hand_size + 1)
    return 0.0


def kokushi_hand(rules, chooser) -> tuple:
    """国士の系統だけの生成（完成形から1枚を一様に除く）。"""
    complete = rules.sample_complete("kokushi", chooser)
    complete.pop(chooser.index(len(complete)))
    return tuple(sorted(complete))


# 変異用：麻雀で「すでに4枚持つ牌を待ちから除く」ことに当たる、小例の上限（手に2枚ある類は待ちにしない）。
LEGAL_WAIT_LIMIT = 2


class D33_02PriorAndMutationTest(unittest.TestCase):
    def test_prior_marginals_are_hypergeometric(self) -> None:
        """尤度1、牌在庫だけの例：定常分布の各位置と各家の手の周辺が、残数に対する超幾何分布。"""
        for name in ("prior_pool", "prior_hand"):
            with self.subTest(example=name):
                chain = exact(name)
                layout = chain.problem.layout
                pi = stationary(chain.iteration())
                types = [layout.type_of[tile_id] for tile_id in range(layout.size)]
                supply = Counter(types)
                for position in range(layout.size):
                    marginal: Counter = Counter()
                    for probability, state in zip(pi, chain.states):
                        marginal[layout.type_of[state[position]]] += probability
                    for tile_type, count in supply.items():
                        self.assertAlmostEqual(marginal[tile_type], count / layout.size, places=12)
                for i in range(len(layout.seats)):
                    hands: Counter = Counter()
                    for probability, state in zip(pi, chain.states):
                        hands[seat_type_state(layout, state, i)[0]] += probability
                    for hand, probability in hands.items():
                        expected = math.prod(math.comb(supply[t], n) for t, n in Counter(hand).items()) / math.comb(
                            layout.size, layout.hand_size)
                        self.assertAlmostEqual(probability, expected, places=12)

    def test_constrained_stationary_equals_normalized_p0_times_h(self) -> None:
        """尤度1でHを課した例：定常分布は、p0×Hを全列挙して正規化した分布（物理配置で一様）。"""
        for name in ("kokushi_flat", "fixed_flat"):
            with self.subTest(example=name):
                chain = exact(name)
                uniform = np.full(len(chain.states), 1.0 / len(chain.states))
                self.assertLess(float(np.abs(stationary(chain.iteration()) - uniform).max()), 1e-10)

    def test_mutations_are_detected(self) -> None:
        """設計の誤り方を移植した実装が、核の不変性（πP=π）の不一致で検出される。"""
        kokushi = exact("kokushi")
        cases = [
            ("m2_without_V_correction", "m2_log_correction", lambda before, after: 0.0, exact("v_asymmetry"),
             lambda c: c.m2()),
            ("m1_excludes_same_type", "m1_partner", different_type_partner, exact("prior_pool"), lambda c: c.m1()),
            ("m2_position_in_remaining_hand", "m2_positions", remaining_hand_positions, exact("kokushi_discard0"),
             lambda c: c.m2()),
            ("m3_single_generation_path", "tenpai_class_density", first_path_density, kokushi, lambda c: c.regeneration((0,))),
            ("m3_without_backward_n_over_13", "backward_log_density", lambda *args: 0.0, kokushi, lambda c: c.regeneration((0,))),
            ("regular_paths_only_legal_waits", "tenpai_removal_classes",
             lambda rules, counts: [x for x in range(rules.class_count) if counts[x] < LEGAL_WAIT_LIMIT],
             kokushi, lambda c: c.regeneration((0,))),
            ("without_pool_factorial", "pool_log_weight", lambda pool: 0.0, exact("prior_pool"), lambda c: c.regeneration((0,))),
            ("without_initial_factorial", "initial_log_weight", lambda initial: 0.0, exact("prior_hand"), lambda c: c.regeneration((0,))),
        ]
        for label, attribute, replacement, chain, build in cases:
            with self.subTest(mutation=label):
                self.assertLess(invariance_error(chain.target, build(chain)), TOL)
                with mock.patch.object(belief, attribute, replacement):
                    self.assertGreater(invariance_error(chain.target, build(chain)), 1e-6)

    def test_regular_numerator_of_the_fixed_example_is_168(self) -> None:
        counts = [0] * 34
        for tile, n in ((0, 4), (1, 1), (2, 1), (3, 1), (10, 3), (20, 3)):  # 1111234m 222p 333s
            counts[tile] = n
        self.assertEqual(regular_path_numerator(tuple(counts)), 168)
        # 合法な待ち（4m）だけの和は48＝経路24×4mの枚数2（5枚目の1mを除く経路120が抜ける）
        self.assertEqual(belief.regular_complete_paths(tuple(c + (1 if t == 3 else 0) for t, c in enumerate(counts))) * 2, 48)
        density = belief.tenpai_class_density(MahjongTenpaiRules(), tuple(counts))
        self.assertAlmostEqual(density / (0.9 * 168 / (34 * 55**4 * 14)), 1.0, places=12)

    def test_density_matches_independent_path_enumeration(self) -> None:
        """面子手の経路の和を、雀頭と順序付きの面子4つを直接たどる独立な列挙と照合する。"""
        rng = random.Random(21)
        hands = [
            (0, 0, 0, 0, 1, 2, 3, 10, 10, 10, 20, 20, 20),  # 1111234m 222p 333s（5枚目の除去を含む）
            (0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 24, 25, 26),  # 1111m 123456p 789s（待ちは5枚目の1mだけ）
            (0, 0, 1, 1, 2, 2, 9, 9, 10, 10, 11, 11, 33),  # 七対子と面子手の両方でテンパイ
        ]
        while len(hands) < 8:
            shape = belief.sample_tenpai_classes(MahjongTenpaiRules((("regular", 0.98), ("chiitoi", 0.01), ("kokushi", 0.01))),
                                                 RandomChooser(rng))
            if max(Counter(shape).values()) <= 4 and belief.regular_path_numerator(tuple(shape.count(t) for t in range(34))):
                hands.append(tuple(shape))
        for hand in hands:
            counts = tuple(hand.count(t) for t in range(34))
            with self.subTest(hand=hand):
                self.assertEqual(regular_path_numerator(counts), brute_regular_numerator(counts))
        chiitoi_regular = tuple(hands[2].count(t) for t in range(34))
        rules = MahjongTenpaiRules()
        self.assertEqual(belief.tenpai_families(rules, chiitoi_regular), ("regular", "chiitoi"))
        expected = 0.9 * regular_path_numerator(chiitoi_regular) / (34 * 55**4 * 14) + 0.05 * 2 / (14 * math.comb(34, 7))
        self.assertAlmostEqual(belief.tenpai_class_density(rules, chiitoi_regular) / expected, 1.0, places=12)
        # 国士の系統：生成器を全分岐列挙した分布と密度が一致する。
        outcomes = enumerate_outcomes(lambda chooser: kokushi_hand(rules, chooser))
        self.assertEqual(len(outcomes), 157)
        for hand, probability in outcomes.items():
            counts = tuple(hand.count(t) for t in range(34))
            family = belief._family_densities(rules, counts)[2]
            self.assertAlmostEqual(family, probability, places=15)

    def test_generator_frequencies_match_density_at_fixed_inventory(self) -> None:
        """固定したA_r（么九牌各2枚）で、生成器10^6回の頻度と計算した密度が一致する（固定seed、カイ二乗）。

        在庫に収まる手は国士157形と么九牌の七対子（12,012形、まとめて1セル）だけで、残りは「在庫超過」。
        面子手・七対子の個々の手の確率は1e-8程度で頻度検定に向かないため、経路の和は独立な列挙で照合する。
        """
        rules = MahjongTenpaiRules()
        supply = [2 if t in YAOCHUU else 0 for t in range(34)]
        kokushi = sorted(enumerate_outcomes(lambda chooser: kokushi_hand(rules, chooser)))
        cells = {hand: k for k, hand in enumerate(kokushi)}
        chiitoi_cell, overflow_cell = len(cells), len(cells) + 1
        probabilities = [belief.tenpai_class_density(rules, tuple(hand.count(t) for t in range(34))) for hand in kokushi]
        chiitoi_total = 0.0
        for pairs in itertools.combinations(YAOCHUU, 7):
            for single in pairs:
                counts = [0] * 34
                for t in pairs:
                    counts[t] = 1 if t == single else 2
                chiitoi_total += belief.tenpai_class_density(rules, tuple(counts))
        probabilities += [chiitoi_total, 1.0 - math.fsum(probabilities) - chiitoi_total]
        rng = random.Random(20261001)
        chooser = RandomChooser(rng)
        counts: Counter = Counter()
        for _ in range(1_000_000):
            hand = tuple(belief.sample_tenpai_classes(rules, chooser))
            if hand in cells:
                counts[cells[hand]] += 1
                continue
            shape = Counter(hand)
            feasible = all(supply[t] >= n for t, n in shape.items())
            counts[chiitoi_cell if feasible else overflow_cell] += 1
            if feasible:
                self.assertEqual(belief.tenpai_families(rules, tuple(shape[t] for t in range(34))), ("chiitoi",))
        assert_frequencies(self, counts, probabilities, 1_000_000)

    def test_red_is_hypergeometric_within_the_class(self) -> None:
        """赤：類を決めた後、その類の物理牌から一様（超幾何分布）。全分岐列挙と密度を照合する。"""
        layout = kokushi_example().layout
        problem = BeliefProblem(BeliefLayout(2, layout.seats, 1, TYPE_OF_ID_FOR_TEST, belief.MAHJONG_CLASS_OF),
                                MahjongTenpaiRules(), FlatModel())
        available = Counter({4: 3, 38: 1, 13: 2, 47: 1})  # 5m×3・赤5m、5p×2・赤5p

        def draw(chooser):
            hand: Counter = Counter()
            for tile_class, count in ((4, 2), (13, 2)):
                items = sorted(t for t in available.elements() if belief.MAHJONG_CLASS_OF[t] == tile_class)
                hand.update(belief._draw_without_replacement(items, count, chooser))
            return tuple(sorted(hand.elements()))
        for hand, probability in enumerate_outcomes(draw).items():
            density = math.exp(belief._variants_log_density(problem, available, Counter(hand)))
            self.assertAlmostEqual(density, probability, places=14)

    def test_positive_density_iff_shanten_zero(self) -> None:
        rules = MahjongTenpaiRules()
        rng = random.Random(7)
        tenpai = 0
        for trial in range(3_000):
            if trial % 2:
                hand = belief.sample_tenpai_classes(rules, RandomChooser(rng))
                if max(Counter(hand).values()) > 4:
                    continue
            else:
                hand = [TILE_BY_ID_FOR_TEST[i] for i in rng.sample(range(136), 13)]
            counts = tuple(hand.count(t) for t in range(34))
            positive = belief.tenpai_class_density(rules, counts) > 0
            self.assertEqual(positive, shanten(counts, 0) == 0, hand)
            tenpai += positive
        self.assertGreater(tenpai, 1000)


TYPE_OF_ID_FOR_TEST = belief.TYPE_OF_ID
TILE_BY_ID_FOR_TEST = [belief.TILE_BY_ID[i].tile34 for i in range(136)]


def brute_regular_numerator(counts) -> int:
    """独立な列挙：雀頭と順序付きの面子4つを直接たどり、除去で手に一致する経路×除去位置の数を数える。"""
    def excess(values) -> int:
        return sum(max(0, v - c) for v, c in zip(values, counts))

    def walk(values, depth) -> int:
        if depth == 4:
            extra = [v - c for v, c in zip(values, counts)]
            if min(extra) < 0 or sum(extra) != 1:
                return 0
            return values[extra.index(1)]
        total = 0
        for mentsu in MENTSU_KINDS:
            following = list(values)
            for t in mentsu:
                following[t] += 1
            if excess(following) <= 1:
                total += walk(following, depth + 1)
        return total

    total = 0
    for pair in range(34):
        start = [0] * 34
        start[pair] = 2
        if excess(start) <= 1:
            total += walk(start, 0)
    return total


class D33_03CrossFamilyToyTest(unittest.TestCase):
    def test_m3_and_m4_accept_transitions_across_families_with_unit_likelihood(self) -> None:
        """国士と面子手の両方に支持がある小例で、尤度1のときM3（リーチ者）とM4が系統をまたぐ遷移を受理する。"""
        chain = exact("kokushi_wide_flat")
        families = [family_of(chain, state) for state in chain.states]
        for label, matrix in (("m3", chain.regeneration((0,))), ("m4", chain.regeneration((0, 1)))):
            for source, target in ((("regular",), ("kokushi",)), (("kokushi",), ("regular",))):
                with self.subTest(move=label, source=source):
                    mass = max(sum(matrix[i, j] for j, f in enumerate(families) if f == target)
                               for i, f in enumerate(families) if f == source)
                    self.assertGreater(mass, 0.0)


@unittest.skipUnless(HAS_DATA, "D.3.1の実データ（Git管理外）がない")
class MahjongMcmcTest(unittest.TestCase):
    """実局面への接続：牌種水準の硬い制約が評価器と一致し、移動がH=1を保つ。"""

    def _belief(self, prefix, seed, resolver=None, **options):
        context = context_of(prefix)
        initial = construct_initial_world(context, random.Random(seed), **options)
        self.assertEqual(initial.status, "ok")
        return context, initial.world, mahjong_belief(context, initial.world, resolver)

    def test_type_level_hard_check_matches_the_evaluator(self) -> None:
        for prefix in fixture()["prefixes"][:6]:
            context = context_of(prefix)
            for seed in range(6):
                world = random_world(context, random.Random(seed), riichi_tenpai=seed % 2 == 0)
                layout = _layout_for(context, world)
                ids = _ids_for(context, world, layout)
                for i, seat_layout in enumerate(layout.seats):
                    with self.subTest(decision=prefix["decisionId"], seed=seed, seat=seat_layout.seat):
                        problem = BeliefProblem(layout, MahjongTenpaiRules(), FlatModel())
                        fast = belief.seat_hard_violation(problem, i, *seat_type_state(layout, ids, i))
                        evaluation = evaluate_seat(context, world.hypotheses[seat_layout.seat], None)
                        self.assertEqual(fast is None, evaluation.violation is None)

    def test_state_round_trip_and_factors_match_the_evaluator(self) -> None:
        resolver = model_resolver(fixture()["model"])
        for prefix in fixture()["prefixes"][:4]:
            context, world, (mahjong, state) = self._belief(prefix, 3, resolver)
            with self.subTest(decision=prefix["decisionId"]):
                self.assertEqual(mahjong.world_from_state(state), world)
                for i, seat_layout in enumerate(mahjong.problem.layout.seats):
                    evaluation = evaluate_seat(context, world.hypotheses[seat_layout.seat], resolver)
                    self.assertAlmostEqual(state.log_factors[i], evaluation.log_likelihood, places=10)

    def test_m2_candidates_match_the_evaluator(self) -> None:
        prefix = fixture()["prefixes"][2]
        context, world, (mahjong, state) = self._belief(prefix, 4)
        problem, layout = mahjong.problem, mahjong.problem.layout
        r = layout.riichi_index
        for a in list(layout.seat_positions(r))[:3]:
            candidates = set(belief.m2_candidates(problem, state.ids, a))
            for b in range(layout.size):
                if layout.position_seat[b] == r:
                    continue
                swapped = state.copy()
                swapped.ids[a], swapped.ids[b] = swapped.ids[b], swapped.ids[a]
                trial = mahjong.world_from_state(swapped)
                ok = all(evaluate_seat(context, trial.hypotheses[s], None).violation is None for s in context.other_seats)
                self.assertEqual(b in candidates, ok, (a, b))

    def test_iterations_keep_hard_constraints_and_factors(self) -> None:
        prefix = fixture()["prefixes"][1]
        context, _, (mahjong, state) = self._belief(prefix, 6)
        stats = MoveStatistics()
        chooser = RandomChooser(random.Random(8))
        for _ in range(2):
            belief.mcmc_iteration(mahjong.problem, state, chooser, stats)
            world = mahjong.world_from_state(state)
            world.validate()
            for i, seat in enumerate(context.other_seats):
                evaluation = evaluate_seat(context, world.hypotheses[seat], None)
                self.assertIsNone(evaluation.violation)
                self.assertEqual(state.log_factors[i], 0.0)
        layout = mahjong.problem.layout
        self.assertEqual(stats.proposed["m1"], 2 * layout.seat_size)
        self.assertEqual(stats.proposed["m4"], 2)
        self.assertGreater(sum(stats.accepted.values()), 0)

    def test_m3_and_m4_accept_kokushi_to_regular_with_unit_likelihood(self) -> None:
        """D33-03の積み残し：国士と面子手の両方に支持がある実局面で、尤度1のときM3とM4が系統をまたぐ遷移を受理する。"""
        forced = (("regular", 0.01), ("chiitoi", 0.01), ("kokushi", 0.98))
        for prefix in fixture()["prefixes"]:
            context = context_of(prefix)
            initial = construct_initial_world(context, random.Random(12), probabilities=forced, max_attempts=3_000)
            if initial.status != "ok" or initial.family != "kokushi":
                continue
            for move in ("m3", "m4"):
                with self.subTest(decision=prefix["decisionId"], move=move):
                    mahjong, state = mahjong_belief(context, initial.world, None)
                    stats = MoveStatistics()
                    chooser = RandomChooser(random.Random(13))
                    r = mahjong.problem.layout.riichi_index
                    for _ in range(200):
                        if move == "m3":
                            belief.m3_step(mahjong.problem, state, r, chooser, stats)
                        else:
                            belief.m4_step(mahjong.problem, state, chooser, stats)
                        if sum(stats.cross_family.values()):
                            break
                    self.assertGreater(sum(stats.cross_family.values()), 0, stats.as_dict())
            return
        self.fail("国士の出発点を作れる判断が固定例にない")


def _layout_for(context, world):
    seats = tuple(
        SeatLayout(seat, seat == context.riichi_seat,
                   tuple(TurnLayout(turn.tsumogiri, belief.type_of_key(turn.discard)) for turn in context.turns[seat]))
        for seat in context.other_seats
    )
    return BeliefLayout(13, seats, len(world.pool), belief.TYPE_OF_ID, belief.MAHJONG_CLASS_OF)


def _ids_for(context, world, layout):
    ids = []
    for seat_layout in layout.seats:
        hypothesis = world.hypotheses[seat_layout.seat]
        ids.extend(hypothesis.initial)
        ids.extend(hypothesis.draws[t.raw_event_index] for t in context.turns[seat_layout.seat] if not t.tsumogiri)
    ids.extend(world.pool)
    return ids


# ===========================================================================
# 工程5：診断（D33-04）、hold（D33-05）、窓キャッシュ（D33-07の残り）
# ===========================================================================

import dataclasses  # noqa: E402

from tools.ev_policy_belief import (  # noqa: E402
    ChainSettings,
    ChainStart,
    DisagreementThresholds,
    RuleUnresolvedError,
    WindowCache,
    generic_summary,
    run_chain_group,
)
from tools.ev_policy_belief import ids_of_key  # noqa: E402
from tools.ev_policy_opponent import canonical_action  # noqa: E402


def toy_starts(problem, states, *, fail=()):
    """小例の出発点を鎖ごとに与える（failに含む鎖は初期化の失敗として返す）。"""

    def start(chain, seed):
        if chain in fail:
            return ChainStart(None, None, None, {"chain": chain, "status": "init_failed"})
        state = initial_chain_state(problem, states[chain])
        return ChainStart(problem, state, lambda s: generic_summary(problem, s), {"chain": chain, "status": "ok"})

    return start


class RejectAllChooser(RandomChooser):
    """変異用：MHの提案をすべて棄却する（鎖が出発点に閉じ込められる）。"""

    def accept(self, log_ratio):
        return False


def family_states(chain: ExactChain, family: tuple) -> list:
    return [state for state in chain.states if family_of(chain, state) == family]


class D33_04DiagnosticsTest(unittest.TestCase):
    SETTINGS = ChainSettings(iterations=600, burn_in=100, thin=5, checkpoints=(300,))

    def test_healthy_chains_agree(self) -> None:
        chain = exact("kokushi")
        problem = kokushi_example()
        starts = family_states(chain, ("regular",))[:2] + family_states(chain, ("kokushi",))[:2]
        record = run_chain_group(toy_starts(problem, starts), [1, 2, 3, 4], self.SETTINGS,
                                 DisagreementThresholds(max_rhat=1.2, min_shape_overlap=0.5, require_supported_families_visited=True))
        self.assertEqual(record["status"], "ok", record.get("holdDetails"))
        diagnostics = record["diagnostics"]
        self.assertLess(diagnostics["maxRhat"], 1.2)
        self.assertEqual(diagnostics["families"]["unvisitedSupported"], [])
        self.assertIn("300", record["checkpoints"])
        self.assertEqual(len(record["samples"]), 4 * 100)

    def test_stuck_chains_worsen_rhat_and_shape_overlap(self) -> None:
        chain = exact("kokushi")
        problem = kokushi_example()
        starts = family_states(chain, ("regular",))[:2] + family_states(chain, ("kokushi",))[:2]
        with mock.patch.object(belief, "RandomChooser", RejectAllChooser):
            record = run_chain_group(toy_starts(problem, starts), [1, 2, 3, 4], self.SETTINGS, DisagreementThresholds())
        diagnostics = record["diagnostics"]
        self.assertEqual(record["status"], "ok")  # 閾値がnullの間は数値を報告するだけ
        self.assertEqual(diagnostics["moves"]["accepted"], {})
        self.assertEqual(diagnostics["maxRhat"], math.inf)
        self.assertEqual(diagnostics["shapes"]["minPairwiseOverlap"], 0.0)
        self.assertFalse(diagnostics["chainDisagreement"]["judged"])
        with mock.patch.object(belief, "RandomChooser", RejectAllChooser):
            held = run_chain_group(toy_starts(problem, starts), [1, 2, 3, 4], self.SETTINGS,
                                   DisagreementThresholds(max_rhat=1.2, min_shape_overlap=0.5))
        self.assertEqual(held["status"], "held")
        self.assertEqual(held["holdReasons"], ["chain_disagreement"])
        self.assertIsNone(held["samples"])

    def test_chains_in_different_modes_are_detected(self) -> None:
        """M3・M4を外し、国士と面子手から別々に出発した鎖は別の峰に留まる。"""
        chain = exact("kokushi")
        problem = kokushi_example()
        starts = family_states(chain, ("regular",))[:2] + family_states(chain, ("kokushi",))[:2]
        settings = dataclasses.replace(self.SETTINGS, moves=("m1", "m2"))
        record = run_chain_group(toy_starts(problem, starts), [1, 2, 3, 4], settings, DisagreementThresholds())
        diagnostics = record["diagnostics"]
        self.assertEqual(diagnostics["scalars"]["family:kokushi"]["rhat"], math.inf)
        self.assertEqual(diagnostics["shapes"]["minPairwiseOverlap"], 0.0)
        self.assertEqual(diagnostics["families"]["crossFamilyAccepted"], {})

    def test_unvisited_supported_family_is_reported_even_when_rhat_is_good(self) -> None:
        """M3・M4を外し、全鎖を同じ系統から始めると、R̂は良くても未訪問の系統が報告される。"""
        chain = exact("kokushi")
        problem = kokushi_example()
        starts = family_states(chain, ("regular",))[:4]
        settings = dataclasses.replace(self.SETTINGS, moves=("m1", "m2"))
        record = run_chain_group(toy_starts(problem, starts), [1, 2, 3, 4], settings, DisagreementThresholds())
        diagnostics = record["diagnostics"]
        self.assertLess(diagnostics["maxRhat"], 1.2)
        self.assertEqual(diagnostics["families"]["support"], {"regular": True, "kokushi": True})
        self.assertEqual(diagnostics["families"]["unvisitedSupported"], ["kokushi"])
        held = run_chain_group(toy_starts(problem, starts), [1, 2, 3, 4], settings,
                               DisagreementThresholds(require_supported_families_visited=True))
        self.assertEqual(held["holdReasons"], ["chain_disagreement"])
        self.assertEqual(held["holdDetails"]["chain_disagreement"], ["unvisited_supported_family"])

    def test_rhat_and_batch_means_on_known_series(self) -> None:
        self.assertEqual(belief.split_rhat([[1.0] * 10, [1.0] * 10]), 1.0)
        self.assertEqual(belief.split_rhat([[0.0] * 10, [1.0] * 10]), math.inf)
        rng = random.Random(3)
        independent = [[rng.random() for _ in range(400)] for _ in range(4)]
        self.assertLess(belief.split_rhat(independent), 1.05)
        ess = belief.batch_means_ess(independent[0])
        self.assertGreater(ess, 200)
        sticky = [value for value in independent[0][:40] for _ in range(10)]
        self.assertLess(belief.batch_means_ess(sticky), 100)


class ToyRuleUnresolvedModel(ToyModel):
    """得点器の未知エラーを注入する：ある配牌の評価でRuleUnresolvedErrorを投げる。"""

    def log_factor(self, i, initial, draws):
        if i == 1 and 3 in initial:
            raise RuleUnresolvedError("injected")
        return super().log_factor(i, initial, draws)


class D33_05HoldTest(unittest.TestCase):
    SETTINGS = ChainSettings(iterations=50, burn_in=10, thin=5)

    def _assert_held_without_numbers(self, record, reason) -> None:
        self.assertEqual(record["status"], "held")
        self.assertIn(reason, record["holdReasons"])
        self.assertIsNone(record["samples"])  # 一様配布への退避や、成功した鎖だけの集計をしない
        self.assertNotIn("diagnostics", record)

    def test_init_failed(self) -> None:
        chain = exact("kokushi")
        problem = kokushi_example()
        record = run_chain_group(toy_starts(problem, chain.states[:4], fail=(2,)), [1, 2, 3, 4], self.SETTINGS,
                                 DisagreementThresholds())
        self._assert_held_without_numbers(record, "init_failed")

    def test_resource_budget_exceeded(self) -> None:
        chain = exact("kokushi")
        problem = kokushi_example()
        record = run_chain_group(toy_starts(problem, chain.states[:4]), [1, 2, 3, 4], self.SETTINGS,
                                 DisagreementThresholds(), deadline=0.0)
        self._assert_held_without_numbers(record, "resource_budget_exceeded")
        self.assertEqual(record["holdDetails"]["resource_budget_exceeded"]["completedChains"], 0)

    def test_unknown_scoring_error_is_rule_unresolved(self) -> None:
        base = kokushi_example()
        problem = BeliefProblem(base.layout, base.rules, ToyRuleUnresolvedModel(base.model.initial_scores, base.model.draw_scores),
                                max_generator_attempts=2)
        clean = [s for s in exact("kokushi").states if 3 not in (base.layout.type_of[s[p]] for p in base.layout.initial_positions(1))]
        record = run_chain_group(toy_starts(problem, clean[:4]), [1, 2, 3, 4], self.SETTINGS, DisagreementThresholds())
        self._assert_held_without_numbers(record, "rule_unresolved")

    def test_mahjong_seat_model_raises_on_scorer_hold(self) -> None:
        layout = kokushi_example().layout
        context = mock.Mock(decision_id="d", turns={1: ()})
        model = belief.MahjongSeatModel(context, layout, None)
        evaluation = belief.SeatEvaluation(1, 0.0, holds=["hold:unknown"])
        with mock.patch.object(belief, "evaluate_seat", return_value=evaluation):
            with self.assertRaises(RuleUnresolvedError):
                model.log_factor(0, (0, 0), ())

    def test_chain_disagreement_threshold_holds(self) -> None:
        chain = exact("kokushi")
        problem = kokushi_example()
        with mock.patch.object(belief, "RandomChooser", RejectAllChooser):
            starts = [family_states(chain, ("regular",))[0], family_states(chain, ("kokushi",))[0]]
            record = run_chain_group(toy_starts(problem, starts), [1, 2], self.SETTINGS,
                                     DisagreementThresholds(max_rhat=1.1))
        self.assertEqual(record["holdReasons"], ["chain_disagreement"])
        self.assertIsNone(record["samples"])


@unittest.skipUnless(HAS_DATA, "D.3.1の実データ（Git管理外）がない")
class D33_05DecisionHoldTest(unittest.TestCase):
    """判断単位のhold：シナリオの欠落、状態復元の不一致、初期化の失敗。"""

    def _run(self, context, scenarios, required, **options):
        return belief.run_decision_scenarios(context, fixture()["model"], scenarios, ChainSettings(2, 1, 1),
                                             required_scenarios=required, chains=2, **options)

    def test_missing_scenario_holds_the_decision(self) -> None:
        context = context_of(fixture()["prefixes"][0])
        scenarios = [s for s in fixture()["scenarios"] if s["id"] == "base"]
        fake = {"status": "ok", "holdReasons": [], "samples": [], "diagnostics": {}}
        with mock.patch.object(belief, "run_chain_group", return_value=fake):
            result = self._run(context, scenarios, ["base", "oat_rho_low"], metadata={"fixedComponentsHash": "h"})
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["holdReasons"], ["fixed_component_sensitivity_missing"])
        self.assertEqual(result["missingScenarios"], ["oat_rho_low"])
        record = result["scenarios"][0]
        for key in ("schemaVersion", "decisionId", "informationStateHash", "privatePrefixHash", "thetaId", "scenarioId",
                    "beliefVersion", "fixedComponentsHash"):
            self.assertIn(key, record)
        self.assertEqual(record["thetaId"], belief.model_identity(fixture()["model"]))

    def test_all_d33_scenarios_are_required(self) -> None:
        fixed = json.loads((MODEL_DIR / "fixed-components.json").read_text(encoding="utf-8"))
        ids = [s["id"] for s in belief.belief_scenarios(fixed)]
        self.assertEqual(len(ids), 22)
        self.assertNotIn("zero", ids)

    def test_state_reconstruction_mismatch(self) -> None:
        context = context_of(fixture()["prefixes"][0])
        turns = list(context.turns[context.riichi_seat])
        last = turns[-1]
        broken = dict(context.turns)
        broken[context.riichi_seat] = tuple(turns[:-1]) + (dataclasses.replace(last, tsumogiri=False),)
        if last.riichi_declaration:
            self.skipTest("最後の打牌がリーチ宣言")
        context = dataclasses.replace(context, turns=broken)
        self.assertEqual(belief.reconstruction_mismatches(context), ["tedashi_after_riichi"])
        result = self._run(context, [s for s in fixture()["scenarios"] if s["id"] == "base"], ["base"])
        self.assertIn("state_reconstruction_mismatch", result["holdReasons"])
        self.assertIsNone(result["scenarios"][0]["samples"])

    def test_init_failed_holds_the_decision(self) -> None:
        context = context_of(fixture()["prefixes"][0])
        result = self._run(context, [s for s in fixture()["scenarios"] if s["id"] == "base"], ["base"], init_attempts=0)
        self.assertEqual(result["holdReasons"], ["fixed_component_sensitivity_missing", "init_failed"])
        self.assertIsNone(result["scenarios"][0]["samples"])


@unittest.skipUnless(HAS_DATA, "D.3.1の実データ（Git管理外）がない")
class D33_07WindowCacheTest(unittest.TestCase):
    def test_cache_matches_direct_evaluation_across_scenarios(self) -> None:
        scenarios = [s for s in fixture()["scenarios"] if s["id"] in ("base", "oat_rho_high", "stratum_riichi-riichi_rho_low")]
        for prefix in fixture()["prefixes"][:3]:
            context = context_of(prefix)
            cache = WindowCache()
            for seed in range(4):
                world = random_world(context, random.Random(seed))
                for scenario in scenarios:
                    resolver = model_resolver(fixture()["model"], scenario)
                    for seat in context.other_seats:
                        direct = evaluate_seat(context, world.hypotheses[seat], resolver)
                        cached = evaluate_seat(context, world.hypotheses[seat], resolver, cache)
                        self.assertEqual(direct.log_likelihood, cached.log_likelihood)
                        self.assertEqual([w.probability for w in direct.windows], [w.probability for w in cached.windows])
            self.assertGreater(cache.hits, 0)

    def test_theta_variant_does_not_share_learned_distributions(self) -> None:
        model = fixture()["model"]
        variant = model.copy()
        variant.kind_weights = variant.kind_weights * 1.01
        scenario = next(s for s in fixture()["scenarios"] if s["id"] == "base")
        normal, other = model_resolver(model, scenario), model_resolver(variant, scenario)
        self.assertNotEqual(normal.model_id, other.model_id)
        context = context_of(fixture()["prefixes"][1])
        world = random_world(context, random.Random(5))
        cache = WindowCache()
        for resolver in (normal, other, normal, other):  # 交互に評価しても、キャッシュの有無で一致する
            for seat in context.other_seats:
                direct = evaluate_seat(context, world.hypotheses[seat], resolver)
                cached = evaluate_seat(context, world.hypotheses[seat], resolver, cache)
                self.assertEqual(direct.log_likelihood, cached.log_likelihood)
        ids = {model_id for entry in cache.entries.values() for model_id, _ in entry.base}
        self.assertEqual(ids, {normal.model_id, other.model_id})

    def test_drawn_tile_separates_keys_for_the_same_fourteen_tiles(self) -> None:
        """同じ14枚`112346789m 1247p 東`で自摸牌を2pと1mに替えると、別の鍵になり合法候補数は13と14。"""
        context = context_of(fixture()["prefixes"][0])
        seat = context.other_seats[0]
        tiles = [0, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 15, 27]
        counts = []
        keys = []
        for drawn34 in (10, 0):
            supply = {}
            hand = [supply.setdefault(t, iter(ids_of_key((t, False)))).__next__() for t in tiles]
            drawn = next(i for i in reversed(hand) if belief.TILE_BY_ID[i].tile34 == drawn34)
            rule = belief._SeatRuleState(context, seat, hand)
            rule.drawn = drawn
            actions, status = rule.self_actions()
            self.assertEqual(status, "known")
            counts.append(len(actions))
            names = [canonical_action(action) for action in actions]
            keys.append(belief.window_cache_key(context, "self_action_after_live", 5, seat, hand, ("drawn", id_key(drawn)), names))
        self.assertEqual(counts, [13, 14])
        self.assertNotEqual(keys[0], keys[1])

    def test_lru_capacity_is_enforced(self) -> None:
        cache = WindowCache(capacity=2)
        for key in ("a", "b", "c"):
            cache.put((key,), belief.WindowEntry([], (), {}))
        self.assertEqual(list(cache.entries), [("b",), ("c",)])
        self.assertEqual(cache.statistics()["evictions"], 1)


if __name__ == "__main__":
    unittest.main()
