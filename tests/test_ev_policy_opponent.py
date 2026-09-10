from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ev_policy_opponent import (  # noqa: E402
    DETAIL_FEATURE_NAMES,
    KIND_FEATURE_NAMES,
    EncodedCandidate,
    HierarchicalSoftmax,
    RoundFeatureState,
    action_nll_and_gradient,
    exhaustive_joint_likelihood,
    joint_resolution_likelihood_and_gradient,
    _public_resolution_kind_probabilities,
    response_rate_diagnostic,
)


def candidate(kind: str, marker: float, **action: object) -> EncodedCandidate:
    kind_values = np.linspace(-0.25, 0.35, len(KIND_FEATURE_NAMES)) + marker
    detail_values = np.linspace(0.15, -0.2, len(DETAIL_FEATURE_NAMES)) + marker * 0.7
    return EncodedCandidate({"kind": kind, **action}, kind_values, detail_values)


def seeded_model() -> HierarchicalSoftmax:
    model = HierarchicalSoftmax.zeros()
    rng = np.random.default_rng(20260909)
    model.kind_weights[:] = rng.normal(0.0, 0.08, model.kind_weights.shape)
    model.detail_weights[:] = rng.normal(0.0, 0.08, model.detail_weights.shape)
    return model


class HierarchicalSoftmaxTests(unittest.TestCase):
    def test_d32_01_normalizes_each_legal_set_and_gives_illegal_zero(self) -> None:
        model = seeded_model()
        candidates = [
            candidate("discard", 0.1, tile34=1, isRed=False, origin="concealed"),
            candidate("discard", 0.2, tile34=2, isRed=False, origin="drawn"),
            candidate("riichi_discard", -0.1, tile34=2, isRed=False, origin="drawn"),
            candidate("ankan", 0.3, tile34=27, redCount=0),
        ]
        for phase in model.temperatures:
            probabilities = model.probabilities(candidates, phase)
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
            self.assertTrue(np.all(probabilities > 0.0))
            self.assertEqual(model.action_probability(candidates, {"kind": "tsumo"}, phase), 0.0)
        singleton = [candidate("pass", 0.0)]
        self.assertEqual(model.probabilities(singleton, "discard_response").tolist(), [1.0])

    def test_d32_01_exact_action_gradient_matches_all_parameter_finite_differences(self) -> None:
        model = seeded_model()
        candidates = [
            candidate("discard", 0.1, tile34=3, isRed=False, origin="concealed"),
            candidate("discard", 0.3, tile34=4, isRed=True, origin="drawn"),
            candidate("riichi_discard", -0.2, tile34=3, isRed=False, origin="concealed"),
        ]
        observed = candidates[1].action
        _, analytic = action_nll_and_gradient(model, candidates, observed, "self_action_after_live")
        self._assert_finite_difference(model, analytic, lambda: action_nll_and_gradient(
            model, candidates, observed, "self_action_after_live"
        )[0])

    def test_d32_02_joint_likelihood_matches_independent_exhaustive_sum(self) -> None:
        model = seeded_model()
        per_seat = {
            1: [candidate("pass", 0.0), candidate("chi", 0.1, consumed=[{"tile34": 1, "isRed": False}, {"tile34": 2, "isRed": False}])],
            2: [candidate("pass", -0.1), candidate("pon", 0.2, consumed=[{"tile34": 3, "isRed": False}, {"tile34": 3, "isRed": False}])],
            3: [candidate("pass", 0.3), candidate("ron", -0.2)],
        }
        observed = {"kind": "pon", "seat": 2, "action": dict(per_seat[2][1].action)}
        loss, _, compatible = joint_resolution_likelihood_and_gradient(
            model, 0, per_seat, observed, "discard_response"
        )
        exhaustive = exhaustive_joint_likelihood(model, 0, per_seat, observed, "discard_response")
        self.assertGreater(compatible, 1)
        self.assertAlmostEqual(math.exp(-loss), exhaustive, places=12)

    def test_d32_02_joint_gradient_matches_all_parameter_finite_differences(self) -> None:
        model = seeded_model()
        per_seat = {
            1: [candidate("pass", 0.0), candidate("chi", 0.1, consumed=[{"tile34": 1, "isRed": False}, {"tile34": 2, "isRed": False}])],
            2: [candidate("pass", -0.1), candidate("pon", 0.2, consumed=[{"tile34": 3, "isRed": False}, {"tile34": 3, "isRed": False}])],
            3: [candidate("pass", 0.3)],
        }
        observed = {"kind": "pon", "seat": 2, "action": dict(per_seat[2][1].action)}
        _, analytic, _ = joint_resolution_likelihood_and_gradient(
            model, 0, per_seat, observed, "discard_response"
        )
        self._assert_finite_difference(
            model,
            analytic,
            lambda: joint_resolution_likelihood_and_gradient(
                model, 0, per_seat, observed, "discard_response"
            )[0],
        )

    def test_hidden_lower_priority_chi_is_marginalized_instead_of_labeled_pass(self) -> None:
        model = HierarchicalSoftmax.zeros()
        per_seat = {
            1: [candidate("pass", 0.0), candidate("chi", 0.0, consumed=[{"tile34": 1, "isRed": False}, {"tile34": 2, "isRed": False}])],
            2: [candidate("pass", 0.0), candidate("pon", 0.0, consumed=[{"tile34": 3, "isRed": False}, {"tile34": 3, "isRed": False}])],
            3: [candidate("pass", 0.0)],
        }
        observed = {"kind": "pon", "seat": 2, "action": dict(per_seat[2][1].action)}
        loss, _, compatible = joint_resolution_likelihood_and_gradient(
            model, 0, per_seat, observed, "discard_response"
        )
        # seat 1のpass/chiのどちらも公開ponと両立する。強制passなら尤度は0.25。
        self.assertEqual(compatible, 2)
        self.assertAlmostEqual(math.exp(-loss), 0.5, places=12)

    def test_public_resolution_kind_probabilities_sum_to_one(self) -> None:
        model = seeded_model()
        per_seat = {
            1: [candidate("pass", 0.0), candidate("chi", 0.1, consumed=[{"tile34": 1, "isRed": False}, {"tile34": 2, "isRed": False}])],
            2: [candidate("pass", -0.1), candidate("pon", 0.2, consumed=[{"tile34": 3, "isRed": False}, {"tile34": 3, "isRed": False}])],
            3: [candidate("pass", 0.3), candidate("ron", -0.2)],
        }
        encoded = {
            "window": {"phase": "discard_response", "actorSeat": 0},
            "perSeat": per_seat,
        }
        probabilities = _public_resolution_kind_probabilities(model, encoded)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=12)
        self.assertEqual(set(probabilities), {"pass", "chi", "pon", "ron"})

    def test_response_rate_diagnostic_holds_material_and_rare_miscalibration(self) -> None:
        metrics = {
            "developmentConfirmation": {
                "rates": {
                    "publicResponseResolution": {
                        "observed": {"pass": 0.96, "daiminkan": 0.00001},
                        "predicted": {"pass": 0.97, "daiminkan": 0.0007},
                    }
                }
            }
        }
        diagnostic = response_rate_diagnostic(metrics)
        self.assertEqual(diagnostic["status"], "hold")
        self.assertIn("maximum_absolute_rate_error_above_0.005", diagnostic["reasons"])
        self.assertIn("rare_rate_ratio_above_5:daiminkan", diagnostic["reasons"])

    def test_model_serialization_requires_fixed_label_definition(self) -> None:
        model = seeded_model()
        restored = HierarchicalSoftmax.from_dict(model.to_dict())
        np.testing.assert_array_equal(restored.kind_weights, model.kind_weights)
        payload = model.to_dict()
        del payload["labelDefinitionVersion"]
        with self.assertRaisesRegex(ValueError, "ラベル定義版"):
            HierarchicalSoftmax.from_dict(payload)

        legacy = model.to_dict()
        legacy["schemaVersion"] = "ev-policy-opponent-model/v1"
        legacy["featureSchemaVersion"] = "ev-policy-opponent-features/v1"
        with self.assertRaisesRegex(ValueError, "schema"):
            HierarchicalSoftmax.from_dict(legacy)

    def test_chankan_view_does_not_see_new_dora_before_resolution(self) -> None:
        tile = lambda value: {"tile34": value, "isRed": False}
        initial_hands = {
            0: [tile(4), tile(4), tile(4)] + [tile(value) for value in range(10)],
            1: [tile(4)] + [tile(value) for value in range(1, 13)],
            2: [tile(value) for value in range(13)],
            3: [tile(value) for value in range(13)],
        }
        private = {
            seat: {"initialHand": hand, "events": []}
            for seat, hand in initial_hands.items()
        }
        public = {
            "initial": {
                "dealerSeat": 0,
                "doraIndicator": tile(0),
                "honba": 0,
                "kyoku": 0,
                "riichiSticks": 0,
                "scores": [25000] * 4,
            },
            "events": [
                {"type": "discard", "seat": 1, "rawEventIndex": 0, "tile": tile(4), "riichiDeclaration": False},
                {"type": "response_resolution", "afterRawEventIndex": 0, "resolution": {"kind": "pon", "seat": 0}},
                {"type": "pon", "seat": 0, "fromSeat": 1, "rawEventIndex": 1, "tiles": [tile(4)] * 3},
                {"type": "kakan", "seat": 0, "rawEventIndex": 2, "tiles": [tile(4)] * 4, "revealedDoraIndicator": tile(9)},
                {"type": "chankan_resolution", "afterRawEventIndex": 2, "resolution": {"kind": "pass"}},
            ],
        }
        state = RoundFeatureState(public, private)
        state.advance(4)
        self.assertEqual(state.dora_indicators, [0])
        state.advance(5)
        self.assertEqual(state.dora_indicators, [0, 9])

    def test_duplicate_dora_indicators_multiply_held_dora(self) -> None:
        tile = lambda value: {"tile34": value, "isRed": False}
        private = {
            seat: {"initialHand": [tile(value) for value in range(13)], "events": []}
            for seat in range(4)
        }
        public = {
            "initial": {
                "dealerSeat": 0,
                "doraIndicator": tile(0),
                "honba": 0,
                "kyoku": 0,
                "riichiSticks": 0,
                "scores": [25000] * 4,
            },
            "events": [],
        }
        state = RoundFeatureState(public, private)
        state.dora_indicators.append(0)
        self.assertAlmostEqual(state.context(0)["dora_count"], 2 / 5)

    def _assert_finite_difference(self, model, analytic, objective) -> None:
        epsilon = 1e-6
        for weights, gradient in (
            (model.kind_weights, analytic.kind),
            (model.detail_weights, analytic.detail),
        ):
            for index in np.ndindex(weights.shape):
                original = weights[index]
                weights[index] = original + epsilon
                plus = objective()
                weights[index] = original - epsilon
                minus = objective()
                weights[index] = original
                numerical = (plus - minus) / (2 * epsilon)
                self.assertAlmostEqual(
                    numerical,
                    gradient[index],
                    delta=2e-7,
                    msg=f"index={index}, numerical={numerical}, analytic={gradient[index]}",
                )


if __name__ == "__main__":
    unittest.main()
