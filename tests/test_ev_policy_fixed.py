from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ev_policy_fixed import (  # noqa: E402
    EstimationInputs,
    FixedConstants,
    Posterior,
    ResponseObservation,
    TsumoObservation,
    build_scenarios,
    classify_stratum,
    collect_estimation_inputs,
    constants_for_seat,
    estimate_strata,
    fixed_component_sensitivity,
    grid_posterior,
    jeffreys_beta_posterior,
    regularized_incomplete_beta,
    require_scenario_for_stage,
    response_window_polynomial,
    scenario_count_summary,
    scenarios_for_stage,
    summarize_u_posterior,
    u_grid,
)
from tools.ev_policy_opponent import (  # noqa: E402
    DETAIL_FEATURE_NAMES,
    KIND_FEATURE_NAMES,
    EncodedCandidate,
    HierarchicalSoftmax,
    _log_probability_gradient,
    exhaustive_joint_likelihood,
    joint_resolution_likelihood_and_gradient,
)


def candidate(action: dict, marker: float) -> EncodedCandidate:
    kind_values = np.linspace(-0.3, 0.4, len(KIND_FEATURE_NAMES)) + marker
    detail_values = np.linspace(0.2, -0.1, len(DETAIL_FEATURE_NAMES)) + 0.5 * marker
    return EncodedCandidate(action, kind_values, detail_values)


def random_model(seed: int, fixed: FixedConstants | None) -> HierarchicalSoftmax:
    rng = np.random.default_rng(seed)
    model = HierarchicalSoftmax.zeros(fixed)
    model.kind_weights[:] = rng.normal(scale=0.4, size=model.kind_weights.shape)
    model.detail_weights[:] = rng.normal(scale=0.4, size=model.detail_weights.shape)
    return model


PON = {"kind": "pon", "consumed": [{"tile34": 5, "isRed": False}, {"tile34": 5, "isRed": False}]}
CHI = {"kind": "chi", "consumed": [{"tile34": 3, "isRed": False}, {"tile34": 4, "isRed": False}]}
DMK = {"kind": "daiminkan", "consumed": [{"tile34": 5, "isRed": False}] * 3}
CONSTANTS = FixedConstants(0.1, 0.05, 0.3, 0.2)


def ron_dmk_seat() -> list[EncodedCandidate]:
    # 実例 mleague:2018-19:L001_S001_0009_02A:11:0:e147 の席2と同じ合法集合。
    return [candidate(DMK, 0.1), candidate(PON, 0.2), candidate({"kind": "pass"}, 0.3), candidate({"kind": "ron"}, 0.4)]


class CompositionTests(unittest.TestCase):
    def test_d32b_06_ron_and_daiminkan_sequential_rule(self) -> None:
        model = random_model(1, CONSTANTS)
        seat = ron_dmk_seat()
        probabilities = model.probabilities(seat, "discard_response")
        self.assertAlmostEqual(probabilities[3], 1 - 0.1, places=14)
        self.assertAlmostEqual(probabilities[0], 0.1 * 0.3, places=14)
        self.assertLessEqual(abs(math.fsum(probabilities) - 1.0), 1e-12)
        residual = model.base_probabilities([seat[1], seat[2]], "discard_response")
        self.assertAlmostEqual(probabilities[1] / probabilities[2], residual[0] / residual[1], places=12)
        self.assertAlmostEqual(probabilities[1] + probabilities[2], 0.1 * 0.7, places=14)

    def test_d32b_06_edge_sets_and_seats_without_fixed_kinds(self) -> None:
        model = random_model(2, CONSTANTS)
        self.assertEqual(model.probabilities([candidate({"kind": "pass"}, 0.0)], "discard_response").tolist(), [1.0])
        with self.assertRaises(ValueError):
            model.probabilities([candidate({"kind": "ron"}, 0.0), candidate(DMK, 0.1)], "discard_response")
        plain = [candidate(CHI, 0.1), candidate({"kind": "pass"}, 0.2)]
        np.testing.assert_allclose(
            model.probabilities(plain, "discard_response"), model.base_probabilities(plain, "discard_response"), atol=1e-15
        )
        self_action = [candidate({"kind": "tsumo"}, 0.1), candidate({"kind": "discard", "tile34": 3}, 0.2),
                       candidate({"kind": "discard", "tile34": 9}, 0.3)]
        values = model.probabilities(self_action, "self_action_after_live")
        self.assertAlmostEqual(values[0], 1 - 0.05, places=14)
        self.assertLessEqual(abs(math.fsum(values) - 1.0), 1e-12)
        chankan = model.probabilities([candidate({"kind": "ron"}, 0.1), candidate({"kind": "pass"}, 0.2)], "chankan_response")
        self.assertAlmostEqual(chankan[0], 1 - 0.2, places=14)

    def test_model_round_trip_keeps_fixed_constants_and_requires_the_field(self) -> None:
        model = random_model(3, CONSTANTS)
        restored = HierarchicalSoftmax.from_dict(model.to_dict())
        self.assertEqual(restored.fixed, CONSTANTS)
        payload = model.to_dict()
        payload.pop("fixedConstants")
        with self.assertRaises(ValueError):
            HierarchicalSoftmax.from_dict(payload)


class GradientTests(unittest.TestCase):
    def _finite_difference(self, model: HierarchicalSoftmax, objective, analytic) -> None:
        step = 1e-6
        for name in ("kind_weights", "detail_weights"):
            weights = getattr(model, name)
            analytic_values = analytic.kind if name == "kind_weights" else analytic.detail
            for index in np.ndindex(weights.shape):
                original = weights[index]
                weights[index] = original + step
                upper = objective()
                weights[index] = original - step
                lower = objective()
                weights[index] = original
                self.assertAlmostEqual((upper - lower) / (2 * step), analytic_values[index], places=6)

    def test_d32b_07_fixed_kinds_have_zero_gradient_and_residual_matches_finite_difference(self) -> None:
        model = random_model(4, CONSTANTS)
        seat = ron_dmk_seat()
        for fixed_index in (0, 3):
            gradient = _log_probability_gradient(model, seat, fixed_index, "discard_response")
            self.assertFalse(gradient.kind.any() or gradient.detail.any())
        analytic = _log_probability_gradient(model, seat, 1, "discard_response")
        self._finite_difference(
            model, lambda: math.log(model.probabilities(seat, "discard_response")[1]), analytic
        )

    def test_d32b_07_skip_and_daiminkan_windows_keep_positive_likelihood(self) -> None:
        model = random_model(5, CONSTANTS)
        per_seat = {1: ron_dmk_seat(), 2: [candidate(CHI, 0.5), candidate({"kind": "pass"}, 0.6)], 3: [candidate({"kind": "pass"}, 0.7)]}
        for resolution in ({"kind": "pass"}, {"kind": "daiminkan", "seat": 1, "action": DMK}):
            loss, gradient, _ = joint_resolution_likelihood_and_gradient(model, 0, per_seat, resolution, "discard_response")
            self.assertTrue(math.isfinite(loss))
            self._finite_difference(
                model,
                lambda r=resolution: -math.log(exhaustive_joint_likelihood(model, 0, per_seat, r, "discard_response")),
                gradient,
            )


def jeffreys_windows(skips: int, total: int) -> list[np.ndarray]:
    """ロンだけが合法な一家の窓。見送りはpass、和了はron。"""

    polys = []
    for index in range(total):
        resolution = {"kind": "pass"} if index < skips else {"kind": "ron", "winnerSeats": [1]}
        polys.append(
            response_window_polynomial(
                0, {1: ("pass", "ron")}, {1: ({"kind": "pass"}, {"kind": "ron"})},
                {1: np.asarray([1.0, 0.0])}, resolution,
                variable_seats={1: (True, False)}, constants=CONSTANTS,
            )
        )
    return polys


class EstimationTests(unittest.TestCase):
    def test_d32b_08a_tsumo_posterior_is_exact_beta_and_rises_with_skips(self) -> None:
        exact = jeffreys_beta_posterior(0, 100)
        self.assertAlmostEqual(exact.mean, 0.5 / 101, places=15)
        # 独立な計算：u上の細かい中点格子で (1-p)^100 を積分する。
        cells = 200_000
        _, p = u_grid(cells)
        mean, lower, upper = summarize_u_posterior(100 * np.log1p(-p), cells)
        for approximate, value in ((mean, exact.mean), (lower, exact.lower), (upper, exact.upper)):
            self.assertLess(abs(approximate - value) / value, 0.005)
        self.assertGreater(jeffreys_beta_posterior(3, 100).mean, exact.mean)
        # Beta(0.5, 0.5) は arcsin分布で、CDFの閉じた式と一致する。
        for x in (0.02, 0.4, 0.95):
            self.assertAlmostEqual(regularized_incomplete_beta(0.5, 0.5, x), 2 / math.pi * math.asin(math.sqrt(x)), places=12)

    def test_d32b_08b_grid_matches_independent_incomplete_beta(self) -> None:
        for skips, total in ((0, 100), (5, 5), (1, 10), (3, 10)):
            with self.subTest(skips=skips, total=total):
                grid = grid_posterior(jeffreys_windows(skips, total), variables=("epsRon",), opportunities={"epsRon": total})["epsRon"]
                exact = jeffreys_beta_posterior(skips, total)
                for a, b in ((grid.mean, exact.mean), (grid.lower, exact.lower), (grid.upper, exact.upper)):
                    self.assertLess(abs(a - b) / b, 0.005)
                self.assertTrue(0.0 < grid.lower < grid.upper < 1.0)

    def test_d32b_08c_hidden_daiminkan_wish_matches_exhaustive_enumeration(self) -> None:
        base = random_model(6, None)
        per_seat = {1: [candidate({"kind": "pass"}, 0.1), candidate({"kind": "ron"}, 0.2)],
                    2: [candidate(DMK, 0.3), candidate(PON, 0.4), candidate({"kind": "pass"}, 0.5)],
                    3: [candidate({"kind": "pass"}, 0.6)]}
        residual = {1: np.asarray([1.0, 0.0])}
        residual[2] = np.concatenate([[0.0], base.base_probabilities(per_seat[2][1:], "discard_response")])
        residual[3] = np.asarray([1.0])
        kinds = {seat: tuple(c.kind for c in values) for seat, values in per_seat.items()}
        actions = {seat: tuple(c.action for c in values) for seat, values in per_seat.items()}
        for resolution in ({"kind": "ron", "winnerSeats": [1]}, {"kind": "daiminkan", "seat": 2, "action": DMK}, {"kind": "pass"}):
            poly = response_window_polynomial(
                0, kinds, actions, residual, resolution,
                variable_seats={seat: (True, True) for seat in per_seat}, constants=CONSTANTS,
            )
            for eps, rho in ((0.01, 0.2), (0.3, 0.7), (0.9, 0.05)):
                model = base.with_fixed(FixedConstants(eps, 0.1, rho, eps))
                expected = exhaustive_joint_likelihood(model, 0, per_seat, resolution, "discard_response")
                value = sum(poly[i, j] * eps**i * rho**j for i in range(4) for j in range(4))
                self.assertAlmostEqual(value, expected, places=13)
            if resolution["kind"] == "ron":
                # ロン結果に隠れた大明槓の希望はρに依存しない（負例にしない）。
                self.assertFalse(poly[:, 1:].any())

    def test_d32b_08c_skip_confirmed_by_lower_priority_chi_enters_likelihood(self) -> None:
        base = random_model(7, None)
        per_seat = {1: [candidate({"kind": "pass"}, 0.1), candidate(PON, 0.2), candidate({"kind": "ron"}, 0.3)],
                    3: [candidate(CHI, 0.4), candidate({"kind": "pass"}, 0.5)]}
        residual = {1: np.concatenate([base.base_probabilities(per_seat[1][:2], "discard_response"), [0.0]]),
                    3: base.base_probabilities(per_seat[3], "discard_response")}
        poly = response_window_polynomial(
            2, {s: tuple(c.kind for c in v) for s, v in per_seat.items()},
            {s: tuple(c.action for c in v) for s, v in per_seat.items()},
            residual, {"kind": "chi", "seat": 3, "action": CHI},
            variable_seats={1: (True, False), 3: (True, False)}, constants=CONSTANTS,
        )
        # チー成立 ⇒ 席1はロンもポンもしていない：ε × m(pass) × m(chi)。
        self.assertAlmostEqual(poly[1, 0], residual[1][0] * residual[3][0], places=14)
        self.assertEqual(poly[0, 0], 0.0)

    def test_d32b_08d_estimation_rejects_development_confirmation(self) -> None:
        window = {"developmentSplit": "developmentConfirmation", "phase": "discard_response"}
        with self.assertRaises(ValueError):
            collect_estimation_inputs(random_model(8, None), [{"window": window, "perSeat": {}}], lambda c: {})

    def test_d32b_08b_zero_opportunity_stratum_is_unsupported(self) -> None:
        attributes = {"dealer": "nondealer", "riichi": "no_riichi", "gap": "top", "turn": "mid"}
        inputs = EstimationInputs(
            tsumo=[TsumoObservation(False, attributes) for _ in range(20)],
            response=[
                ResponseObservation(0, {1: ("pass", "ron")}, {1: ({"kind": "pass"}, {"kind": "ron"})},
                                    {1: np.asarray([1.0, 0.0])}, {"kind": "ron", "winnerSeats": [1]}, {1: attributes})
                for _ in range(20)
            ],
            excluded={},
        )
        overall = {name: jeffreys_beta_posterior(0, 20) for name in ("epsRon", "epsTsumo", "rho")}
        strata = estimate_strata(inputs, CONSTANTS, overall)
        dealer = next(item for item in strata if item["value"] == "dealer" and item["constant"] == "epsRon")
        self.assertEqual((dealer["opportunities"], dealer["status"]), (0, "unsupported"))
        self.assertEqual(dealer["posterior"]["method"], "jeffreys_prior_only")


def posterior(mean: float, lower: float, upper: float) -> Posterior:
    return Posterior(mean, lower, upper, "test", 100, {})


POSTERIORS = {"epsRon": posterior(0.002, 0.001, 0.004), "epsTsumo": posterior(0.002, 0.0008, 0.005),
              "rho": posterior(0.006, 0.004, 0.009)}


class ScenarioTests(unittest.TestCase):
    def test_d32b_08e_scenario_list_counts_and_chankan_rule(self) -> None:
        strata = [
            {"dimension": "dealer", "value": "dealer", "constant": "epsRon", "status": "difference_detected",
             "posterior": {"lower": 0.01, "upper": 0.02}},
            {"dimension": "turn", "value": "late", "constant": "rho", "status": "unsupported",
             "posterior": {"lower": 0.00001, "upper": 0.8}},
            {"dimension": "gap", "value": "top", "constant": "epsTsumo", "status": "consistent",
             "posterior": {"lower": 0.001, "upper": 0.004}},
        ]
        scenarios = build_scenarios(POSTERIORS, strata + strata[:1], theta_id="theta")
        self.assertEqual(scenario_count_summary(scenarios), {"K": 4, "D.3.3": 20, "D.3.4": 21})
        no_strata = build_scenarios(POSTERIORS, [], theta_id="theta")
        self.assertEqual(scenario_count_summary(no_strata), {"K": 0, "D.3.3": 16, "D.3.4": 17})
        by_id = {item["id"]: item for item in scenarios}
        self.assertEqual(by_id["oat_epsRon_low"]["constants"]["epsChankan"], 0.001)
        self.assertEqual(by_id["chankan_assumption"]["constants"]["epsChankan"], 0.5)
        self.assertEqual(by_id["chankan_assumption"]["constants"]["epsRon"], 0.002)
        for item in scenarios:
            self.assertEqual(set(item["constants"]), {"epsRon", "epsTsumo", "rho", "epsChankan"})
            self.assertEqual(item["thetaId"], "theta")
            self.assertIn("posteriorId", item)
        self.assertEqual(by_id["zero"]["posteriorId"], "base")
        with self.assertRaises(ValueError):
            require_scenario_for_stage(scenarios, "zero", "D.3.3")
        self.assertEqual(len(scenarios_for_stage(scenarios, "D.3.3")), 20)
        override = by_id["stratum_dealer-dealer_epsRon_high"]
        member = constants_for_seat(override, {"dealer": "dealer"})
        other = constants_for_seat(override, {"dealer": "nondealer"})
        self.assertEqual((member.eps_ron, member.eps_chankan), (0.02, 0.02))
        self.assertEqual(other.eps_ron, 0.002)

    def test_d32b_08e_wide_stratum_is_unsupported(self) -> None:
        overall = posterior(0.002, 0.001, 0.003)
        self.assertEqual(classify_stratum(overall, posterior(0.4, 0.00001, 0.8), 3), "unsupported")
        self.assertEqual(classify_stratum(overall, posterior(0.02, 0.01, 0.03), 50), "difference_detected")
        self.assertEqual(classify_stratum(overall, posterior(0.002, 0.0012, 0.0031), 500), "consistent")

    def test_d32b_08e_one_at_a_time_detects_flip_hidden_by_joint_corners(self) -> None:
        # ΔEV = 1 + 2x - 2y（第1回反証レビューR4の反例）。x=ε、y=ρ を0（下）か1（上）へ動かす。
        def delta(x: float, y: float) -> float:
            return 1 + 2 * x - 2 * y

        corners_only = {"base": delta(0, 0), "corner_ll": delta(0, 0), "corner_hh": delta(1, 1)}
        self.assertFalse(fixed_component_sensitivity(corners_only)["fixed_component_sensitive"])
        with_oat = {**corners_only, "oat_rho_high": delta(0, 1), "oat_epsRon_high": delta(1, 0)}
        result = fixed_component_sensitivity(with_oat)
        self.assertTrue(result["fixed_component_sensitive"])
        self.assertEqual(result["flippedScenarios"], ["oat_rho_high"])


if __name__ == "__main__":
    unittest.main()
