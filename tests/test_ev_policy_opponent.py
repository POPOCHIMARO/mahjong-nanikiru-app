from __future__ import annotations

import gzip
import json
import math
import sys
import tempfile
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
    fixed_components_declared,
    joint_resolution_likelihood_and_gradient,
    load_win_legality_report,
    opponent_adoption_holds,
    win_legality_satisfied,
    _public_resolution_kind_probabilities,
    _support_diagnostics,
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
        diagnostic = response_rate_diagnostic(metrics, "developmentConfirmation")
        self.assertEqual(diagnostic["status"], "hold")
        self.assertEqual(diagnostic["developmentConfirmationReuse"], "reused_non_independent")
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


def _self_window(split: str, *, learnable: bool) -> dict[str, object]:
    action = {"kind": "discard", "tile34": 3, "isRed": False, "origin": "drawn"}
    return {
        "windowId": f"{split}:self:{learnable}",
        "developmentSplit": split,
        "phase": "self_action_after_live",
        "actorSeat": 0,
        "legalActions": [action, {"kind": "tsumo"}],
        "winActionStatus": "known" if learnable else "hold",
        "observation": {"status": "exact", "mass": 1.0, "action": action},
        "learningMask": {"kind": learnable, "conditionalDetail": True},
    }


def _response_window(
    split: str,
    window_id: str,
    legal_by_seat: dict[str, list[dict[str, object]]],
    resolution: dict[str, object],
    labels: dict[str, dict[str, object] | None],
    *,
    joint_kind: bool,
) -> dict[str, object]:
    return {
        "windowId": window_id,
        "developmentSplit": split,
        "phase": "discard_response",
        "actorSeat": 3,
        "legalBySeat": {seat: {"actions": actions} for seat, actions in legal_by_seat.items()},
        "observation": {
            "mass": 1.0,
            "resolution": resolution,
            "perSeatLabels": [
                {"seat": seat, "status": "exact" if action is not None else "censored", "action": action}
                for seat, action in labels.items()
            ],
        },
        "learningMask": {"jointKind": joint_kind, "conditionalDetail": True},
    }


class SupportDiagnosticsTests(unittest.TestCase):
    def _write_dataset(self, root: Path, rows: list[dict[str, object]]) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        path = dataset / "teacher-windows.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return dataset

    def test_d32b_09_unlearnable_windows_are_excluded_from_observed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = [
                _self_window("train", learnable=True),
                # 打牌自体は合法だが、和了不合法とは無関係のwinActionStatus=holdで学習から外れる窓。
                _self_window("train", learnable=False),
            ]
            dataset = self._write_dataset(Path(temporary), rows)
            support = _support_diagnostics(dataset, None)
        train = support["bySplit"]["train"]
        self.assertEqual(train["observed"]["discard"], 1)
        self.assertEqual(train["opportunities"]["discard"], 2)
        self.assertEqual(train["heldOrCensored"]["heldSelfWindows"], 1)

    def test_d32b_09_ron_skip_confirmed_by_lower_priority_resolution_but_not_by_ron_headhane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ron = {"kind": "ron"}
            pon = {"kind": "pon", "consumed": []}
            chi = {"kind": "chi", "consumed": []}
            passing = {"kind": "pass"}
            rows = [
                # R3の反例：チーが成立（優先度2）したので、ロン合法だった席1のロン見送りが
                # exactラベルなしでも確定する。
                _response_window(
                    "calibration", "w1",
                    {"1": [ron, pon, passing], "2": [chi, passing]},
                    {"kind": "chi", "seat": 2, "action": chi},
                    {"1": None, "2": chi},
                    joint_kind=True,
                ),
                # 頭ハネ：resolutionはron（優先度0）で席1はロン合法だが勝者ではない。
                # 隠れた同時ロン希望がありうるため、見送りは確定しない。
                _response_window(
                    "calibration", "w2",
                    {"1": [ron, passing], "2": [ron, passing]},
                    {"kind": "ron", "winnerSeats": [2]},
                    {"1": None, "2": ron},
                    joint_kind=True,
                ),
                # 学習から外れた窓（jointKind=False、和了判定とは無関係のhold）でも、
                # 公開結果（pass）が確定させるロン見送りという論理的事実は変わらない。
                # ただしexactラベルはobservedへ混ぜない。
                _response_window(
                    "calibration", "w3",
                    {"1": [ron, passing]},
                    {"kind": "pass"},
                    {"1": passing},
                    joint_kind=False,
                ),
            ]
            dataset = self._write_dataset(Path(temporary), rows)
            support = _support_diagnostics(dataset, None)
        calibration = support["bySplit"]["calibration"]
        self.assertEqual(calibration["confirmedRonSkips"], 2)  # w1の席1、w3の席1
        self.assertEqual(calibration["legalWinSkips"]["ron"], 0)
        self.assertEqual(calibration["observed"].get("pass", 0), 0)
        self.assertEqual(calibration["opportunities"]["ron"], 4)

    def test_d32b_09_observed_never_exceeds_opportunities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            row = _self_window("train", learnable=True)
            row["legalActions"] = [{"kind": "tsumo"}]  # discardは非合法のはずが観測される矛盾データ。
            dataset = self._write_dataset(Path(temporary), [row])
            with self.assertRaisesRegex(ValueError, "観測が合法機会を上回る"):
                _support_diagnostics(dataset, None)


WELL_FORMED_FEATURE_MANIFEST = {"status": "complete", "implementedGroups": [
    "exact_candidate_ukeire", "danger_multi_riichi_v3", "yaku_shape_cues_v1",
]}
CLEAN_WIN_LEGALITY = {"unclassifiedCount": 0, "allObservedWinsInLegalSet": True, "residualExceptions": []}
CLEAN_SUPPORT = {"unidentifiedComponents": []}
CLEAN_RATES = {
    "calibration": {"status": "pass", "reasons": []},
    "developmentConfirmation": {"status": "pass", "reasons": []},
}


def _fixed_components(scenario_count: int = 17) -> dict[str, object]:
    constants = {"epsRon": 0.01, "epsTsumo": 0.01, "rho": 0.02, "epsChankan": 0.01}
    posteriors = {name: {"mean": value, "lower": value / 2, "upper": value * 2} for name, value in constants.items() if name != "epsChankan"}
    return {
        "roundTrips": {"final": {"constants": constants, "posteriors": posteriors}},
        "scenarios": [{"id": f"s{i}"} for i in range(scenario_count)],
    }


class WinLegalitySatisfiedTests(unittest.TestCase):
    def test_unclassified_remaining_blocks(self) -> None:
        self.assertFalse(win_legality_satisfied({**CLEAN_WIN_LEGALITY, "unclassifiedCount": 1}))

    def test_unverified_full_scan_blocks(self) -> None:
        self.assertFalse(win_legality_satisfied({**CLEAN_WIN_LEGALITY, "allObservedWinsInLegalSet": False}))

    def test_residual_exception_without_detector_blocks(self) -> None:
        report = {**CLEAN_WIN_LEGALITY, "residualExceptions": [
            {"cause": "x", "hasDetector": False, "hasThreeWayHandling": False}
        ]}
        self.assertFalse(win_legality_satisfied(report))

    def test_residual_exception_with_detector_but_no_three_way_handling_blocks(self) -> None:
        report = {**CLEAN_WIN_LEGALITY, "residualExceptions": [
            {"cause": "x", "hasDetector": True, "hasThreeWayHandling": False}
        ]}
        self.assertFalse(win_legality_satisfied(report))

    def test_fully_declared_residual_exception_passes(self) -> None:
        report = {**CLEAN_WIN_LEGALITY, "residualExceptions": [
            {"cause": "x", "hasDetector": True, "hasThreeWayHandling": True}
        ]}
        self.assertTrue(win_legality_satisfied(report))

    def test_zero_case_passes(self) -> None:
        self.assertTrue(win_legality_satisfied(CLEAN_WIN_LEGALITY))


class LoadWinLegalityReportTests(unittest.TestCase):
    def test_loads_probe_files_and_leaves_unrecognized_causes_as_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            probe_dir = Path(temporary)
            (probe_dir / "summary.json").write_text(json.dumps({
                "unclassified": 0,
                "byCauseAndExpected": [
                    {"cause": "chi_meld_order_adapter", "expected": {}, "count": 924},
                    {"cause": "rule_unresolved:some_form", "expected": {}, "count": 3},
                ],
            }), encoding="utf-8")
            (probe_dir / "verification.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
            report = load_win_legality_report(probe_dir)
        self.assertEqual(report["unclassifiedCount"], 0)
        self.assertTrue(report["allObservedWinsInLegalSet"])
        self.assertEqual([item["cause"] for item in report["residualExceptions"]], ["rule_unresolved:some_form"])
        self.assertFalse(win_legality_satisfied(report))  # 検出体制が未宣言のため解除されない


class FixedComponentsDeclaredTests(unittest.TestCase):
    def test_no_unidentified_components_is_trivially_satisfied(self) -> None:
        self.assertTrue(fixed_components_declared(CLEAN_SUPPORT, None))

    def test_missing_fixed_components_file_fails(self) -> None:
        support = {"unidentifiedComponents": [{"component": "ron_pass"}]}
        self.assertFalse(fixed_components_declared(support, None))

    def test_all_four_unidentified_components_covered(self) -> None:
        support = {"unidentifiedComponents": [
            {"component": "ron_pass"}, {"component": "tsumo_pass"},
            {"component": "daiminkan_policy"}, {"component": "chankan_response_policy"},
        ]}
        self.assertTrue(fixed_components_declared(support, _fixed_components()))

    def test_component_without_posterior_fails(self) -> None:
        support = {"unidentifiedComponents": [{"component": "ron_pass"}]}
        fixed = _fixed_components()
        del fixed["roundTrips"]["final"]["posteriors"]["epsRon"]
        self.assertFalse(fixed_components_declared(support, fixed))


class OpponentAdoptionHoldsTests(unittest.TestCase):
    def _holds(self, **overrides: object) -> dict[str, object]:
        args = dict(
            win_legality=CLEAN_WIN_LEGALITY,
            feature_manifest=WELL_FORMED_FEATURE_MANIFEST,
            support=CLEAN_SUPPORT,
            rate_diagnostics=CLEAN_RATES,
            fixed_components=None,
        )
        args.update(overrides)
        return opponent_adoption_holds(**args)

    def test_d32b_10_unclassified_win_legality_blocks_only_that_hold(self) -> None:
        result = self._holds(win_legality={**CLEAN_WIN_LEGALITY, "unclassifiedCount": 5})
        self.assertEqual(result["holds"], ["win_legality_unresolved"])

    def test_d32b_10_residual_exception_missing_detector_blocks(self) -> None:
        win_legality = {**CLEAN_WIN_LEGALITY, "residualExceptions": [
            {"cause": "x", "hasDetector": False, "hasThreeWayHandling": False}
        ]}
        result = self._holds(win_legality=win_legality)
        self.assertIn("win_legality_unresolved", result["holds"])

    def test_d32b_10_detector_without_three_way_handling_blocks(self) -> None:
        win_legality = {**CLEAN_WIN_LEGALITY, "residualExceptions": [
            {"cause": "x", "hasDetector": True, "hasThreeWayHandling": False}
        ]}
        result = self._holds(win_legality=win_legality)
        self.assertIn("win_legality_unresolved", result["holds"])

    def test_d32b_10_unresolved_known_win_windows_block(self) -> None:
        result = self._holds(win_legality={**CLEAN_WIN_LEGALITY, "allObservedWinsInLegalSet": False})
        self.assertIn("win_legality_unresolved", result["holds"])

    def test_d32b_10_missing_danger_feature_group(self) -> None:
        manifest = {"status": "complete", "implementedGroups": ["exact_candidate_ukeire", "yaku_shape_cues_v1"]}
        result = self._holds(feature_manifest=manifest)
        self.assertEqual(result["holds"], ["full_multi_riichi_danger_class"])

    def test_d32b_10_missing_yaku_feature_group(self) -> None:
        manifest = {"status": "complete", "implementedGroups": ["exact_candidate_ukeire", "danger_multi_riichi_v3"]}
        result = self._holds(feature_manifest=manifest)
        self.assertEqual(result["holds"], ["explicit_yaku_shape_features"])

    def test_d32b_10_unidentified_component_without_fixed_declaration(self) -> None:
        support = {"unidentifiedComponents": [{"component": "daiminkan_policy"}]}
        result = self._holds(support=support, fixed_components=None)
        self.assertEqual(result["holds"], ["opponent_component_unidentified"])
        result_declared = self._holds(support=support, fixed_components=_fixed_components())
        self.assertEqual(result_declared["holds"], [])

    def test_d32b_10_daiminkan_only_rate_miscalibration(self) -> None:
        rates = {
            "calibration": {"status": "pass", "reasons": []},
            "developmentConfirmation": {"status": "hold", "reasons": ["rare_rate_ratio_above_5:daiminkan"]},
        }
        result = self._holds(rate_diagnostics=rates)
        self.assertEqual(result["holds"], ["response_rate_miscalibration"])

    def test_d32b_10_correction_degradation_forces_rate_hold_even_when_diagnostics_pass(self) -> None:
        result = self._holds(rate_correction_degraded=True)
        self.assertEqual(result["holds"], ["response_rate_miscalibration"])

    def test_d32b_10_all_clear_yields_eligible_d33_with_conditions(self) -> None:
        result = self._holds(fixed_components=_fixed_components())
        self.assertEqual(result["holds"], [])
        self.assertTrue(result["eligibleForD33"])
        self.assertFalse(result["eligibleForAdoption"])
        self.assertEqual(result["d33Conditions"], ["fixed_component_sensitivity"])

    def test_d32b_10_declared_residual_exception_adds_d33_condition_even_when_satisfied(self) -> None:
        win_legality = {**CLEAN_WIN_LEGALITY, "residualExceptions": [
            {"cause": "x", "hasDetector": True, "hasThreeWayHandling": True}
        ]}
        result = self._holds(win_legality=win_legality, fixed_components=_fixed_components())
        self.assertEqual(result["holds"], [])
        self.assertEqual(
            set(result["d33Conditions"]), {"fixed_component_sensitivity", "residual_win_legality_detector"}
        )

    def test_d32b_10_eligible_for_adoption_is_always_false(self) -> None:
        self.assertFalse(self._holds(fixed_components=_fixed_components())["eligibleForAdoption"])
        self.assertFalse(self._holds(win_legality={**CLEAN_WIN_LEGALITY, "unclassifiedCount": 9})["eligibleForAdoption"])


if __name__ == "__main__":
    unittest.main()
