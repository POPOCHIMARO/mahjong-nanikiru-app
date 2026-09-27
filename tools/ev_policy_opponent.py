#!/usr/bin/env python3
"""D.3.2: 合法手だけに確率を置く相手行動モデル。

数学部分は局面復元から独立させてある。これにより、合法集合上の正規化と
打ち切られた共同応答尤度を小さなfixtureで厳密に検査できる。
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import inspect
import itertools
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:
    from .ev_calibration_state import shanten
    from .ev_policy_features import (
        CALCULATION_VERSION,
        DANGER_FEATURE_GROUP,
        FEATURE_CACHE_SCHEMA,
        IMPLEMENTED_FEATURE_GROUPS,
        YAKU_SHAPE_FEATURE_GROUP,
        MeldView,
        RiichiOpponentView,
        clear_all_shape_caches,
        danger_features,
        melds_after_action,
        yaku_shape_features,
        iter_feature_shard,
        make_action_view,
        project_action,
        sha256_file,
        write_feature_shard,
        write_json_atomic,
    )
    from .ev_calibration_model import DANGER_RATES
    from .ev_policy_fixed import (
        CHANKAN_ASSUMPTION,
        ESTIMATION_SPLITS,
        FixedConstants,
        build_scenarios,
        collect_estimation_inputs,
        compose_probabilities,
        constants_from_posteriors,
        estimate_base_posteriors,
        estimate_strata,
        fixed_kind_masses,
        initial_constants,
        scenario_count_summary,
        stratum_attributes,
    )
    from .ev_policy_observation import resolve_joint_response, response_priority
except ImportError:
    from ev_calibration_state import shanten
    from ev_policy_features import (
        CALCULATION_VERSION,
        DANGER_FEATURE_GROUP,
        FEATURE_CACHE_SCHEMA,
        IMPLEMENTED_FEATURE_GROUPS,
        YAKU_SHAPE_FEATURE_GROUP,
        MeldView,
        RiichiOpponentView,
        clear_all_shape_caches,
        danger_features,
        melds_after_action,
        yaku_shape_features,
        iter_feature_shard,
        make_action_view,
        project_action,
        sha256_file,
        write_feature_shard,
        write_json_atomic,
    )
    from ev_calibration_model import DANGER_RATES
    from ev_policy_fixed import (
        CHANKAN_ASSUMPTION,
        ESTIMATION_SPLITS,
        FixedConstants,
        build_scenarios,
        collect_estimation_inputs,
        compose_probabilities,
        constants_from_posteriors,
        estimate_base_posteriors,
        estimate_strata,
        fixed_kind_masses,
        initial_constants,
        scenario_count_summary,
        stratum_attributes,
    )
    from ev_policy_observation import resolve_joint_response, response_priority


MODEL_SCHEMA = "ev-policy-opponent-model/v3"
FIT_SCHEMA = "ev-policy-opponent-fit/v3"
EVALUATION_SCHEMA = "ev-policy-opponent-evaluation/v3"
FEATURE_SCHEMA = "ev-policy-opponent-features/v3"
MODEL_VERSION = "shared-hierarchical-softmax-v3-fixed-constants"
LABEL_DEFINITION_VERSION = "joint-public-resolution-v1"

KINDS = (
    "pass",
    "discard",
    "riichi_discard",
    "ankan",
    "kakan",
    "tsumo",
    "ron",
    "chi",
    "pon",
    "daiminkan",
)
KIND_INDEX = {kind: index for index, kind in enumerate(KINDS)}
PHASES = (
    "self_action_after_live",
    "self_action_after_call",
    "self_action_after_rinshan",
    "discard_response",
    "chankan_response",
)

# 候補種別スコア。局面特徴に、その種別内で最良の手牌形を加える。
KIND_FEATURE_NAMES = (
    "bias",
    "turn",
    "wall_remaining",
    "dealer",
    "score_delta",
    "leader_delta",
    "honba",
    "riichi_sticks",
    "kyoku",
    "fixed_melds",
    "own_riichi",
    "opponent_riichi_count",
    "hand_shanten",
    "dora_count",
    "honor_ratio",
    "terminal_honor_ratio",
    "suit_concentration",
    "best_action_shanten",
    "best_local_acceptance",
    "best_ukeire_count",
    "best_ukeire_kinds",
    "best_ukeire_applicable",
    # D.3.2b：安全に打てる候補の有無と、種別内で役の手掛かりが全くないか。
    "min_danger_max_rate",
    "min_open_no_listed_yaku_cue",
)

# 同じ種別内の具体行動スコア。tile34 one-hotは末尾に置く。
DETAIL_BASE_FEATURE_NAMES = (
    "bias",
    "action_shanten",
    "local_acceptance",
    "ukeire_count",
    "ukeire_kinds",
    "ukeire_applicable",
    "is_red",
    "is_dora",
    "origin_drawn",
    "honor",
    "terminal",
    "simple",
    "consumed_red_ratio",
    "turn_x_shanten",
    "late_x_dora",
    # D.3.2b 危険度（設計5.3節）。旧genbutsuはリーチ前の本人の捨牌を数えていなかった。
    "danger_applicable",
    "danger_max_rate",
    "danger_sum_rate",
    "danger_dealer_rate",
    "genbutsu_all",
    "genbutsu_any",
    "danger_group_safe",
    "danger_group_semi_safe",
    "danger_group_guarded",
    "danger_group_moderate_risk",
    "danger_group_high_risk",
    "riichi_x_genbutsu_all",
    # D.3.2b 役と形の手掛かり（設計6.3節）
    "menzen_after",
    "yakuhai_secured",
    "yakuhai_pairs",
    "tanyao_path",
    "tanyao_distance",
    "flush_path",
    "flush_distance",
    "toitoi_blocks",
    "toitoi_path",
    "pair_count",
    "chiitoi_applicable",
    "chiitoi_shanten",
    "open_no_listed_yaku_cue",
)
YAKU_SHAPE_DETAIL_NAMES = DETAIL_BASE_FEATURE_NAMES[DETAIL_BASE_FEATURE_NAMES.index("menzen_after") :]
DETAIL_FEATURE_NAMES = DETAIL_BASE_FEATURE_NAMES + tuple(f"tile34_{index}" for index in range(34))


def canonical_action(action: Mapping[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.ones(1, dtype=float)
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


@dataclass(frozen=True)
class EncodedCandidate:
    """一つの意味上の合法行動と、その二段の特徴。"""

    action: Mapping[str, Any]
    kind_features: np.ndarray
    detail_features: np.ndarray

    @property
    def kind(self) -> str:
        return str(self.action["kind"])


@dataclass
class ModelGradient:
    kind: np.ndarray
    detail: np.ndarray

    @classmethod
    def zeros(cls, model: "HierarchicalSoftmax") -> "ModelGradient":
        return cls(np.zeros_like(model.kind_weights), np.zeros_like(model.detail_weights))

    def add_scaled(self, other: "ModelGradient", scale: float) -> None:
        self.kind += other.kind * scale
        self.detail += other.detail * scale


@dataclass
class HierarchicalSoftmax:
    """行動種別と具体行動を分けた線形softmax。"""

    kind_weights: np.ndarray
    detail_weights: np.ndarray
    temperatures: dict[str, float]
    # D.3.2b：未識別成分の固定定数。Noneなら全種別を学習分布で扱う（v2までの動作）。
    fixed: FixedConstants | None = None

    @classmethod
    def zeros(cls, fixed: FixedConstants | None = None) -> "HierarchicalSoftmax":
        return cls(
            np.zeros((len(KINDS), len(KIND_FEATURE_NAMES)), dtype=float),
            np.zeros((len(KINDS), len(DETAIL_FEATURE_NAMES)), dtype=float),
            {phase: 1.0 for phase in PHASES},
            fixed,
        )

    def copy(self) -> "HierarchicalSoftmax":
        return HierarchicalSoftmax(
            self.kind_weights.copy(), self.detail_weights.copy(), dict(self.temperatures), self.fixed
        )

    def with_fixed(self, fixed: FixedConstants | None) -> "HierarchicalSoftmax":
        """同じ係数で固定定数だけを替えたモデル（感度シナリオ用）。"""
        return HierarchicalSoftmax(self.kind_weights, self.detail_weights, dict(self.temperatures), fixed)

    def probabilities(self, candidates: Sequence[EncodedCandidate], phase: str) -> np.ndarray:
        """合法集合上の行動確率。固定定数があれば、固定成分へ定数の質量を与える（設計7.2節）。"""
        if self.fixed is None:
            return self.base_probabilities(candidates, phase)
        kinds = [candidate.kind for candidate in candidates]
        return compose_probabilities(
            kinds,
            phase,
            self.fixed,
            lambda indices: self.base_probabilities([candidates[index] for index in indices], phase),
        )

    def base_probabilities(self, candidates: Sequence[EncodedCandidate], phase: str) -> np.ndarray:
        """固定成分を考えない学習分布（係数と温度だけで決まる階層softmax）。"""
        if not candidates:
            raise ValueError("合法手集合が空")
        temperature = float(self.temperatures.get(phase, 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"不正な温度: {phase}={temperature}")

        groups: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            if candidate.kind not in KIND_INDEX:
                raise ValueError(f"未知の行動種別: {candidate.kind}")
            groups.setdefault(candidate.kind, []).append(index)

        kinds = list(groups)
        kind_logits = np.asarray(
            [self.kind_weights[KIND_INDEX[kind]] @ candidates[groups[kind][0]].kind_features for kind in kinds]
        ) / temperature
        kind_probability = _softmax(kind_logits)
        result = np.zeros(len(candidates), dtype=float)
        for group_index, kind in enumerate(kinds):
            indices = groups[kind]
            detail_logits = np.asarray(
                [self.detail_weights[KIND_INDEX[kind]] @ candidates[index].detail_features for index in indices]
            ) / temperature
            detail_probability = _softmax(detail_logits)
            result[indices] = kind_probability[group_index] * detail_probability
        return result

    def action_probability(
        self, candidates: Sequence[EncodedCandidate], action: Mapping[str, Any], phase: str
    ) -> float:
        key = canonical_action(action)
        for index, candidate in enumerate(candidates):
            if canonical_action(candidate.action) == key:
                return float(self.probabilities(candidates, phase)[index])
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": MODEL_SCHEMA,
            "modelVersion": MODEL_VERSION,
            "featureSchemaVersion": FEATURE_SCHEMA,
            "exactCalculationVersion": CALCULATION_VERSION,
            "labelDefinitionVersion": LABEL_DEFINITION_VERSION,
            "kinds": list(KINDS),
            "kindFeatureNames": list(KIND_FEATURE_NAMES),
            "detailFeatureNames": list(DETAIL_FEATURE_NAMES),
            "kindWeights": self.kind_weights.tolist(),
            "detailWeights": self.detail_weights.tolist(),
            "temperatures": dict(sorted(self.temperatures.items())),
            "fixedConstants": None if self.fixed is None else self.fixed.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HierarchicalSoftmax":
        if value.get("schemaVersion") != MODEL_SCHEMA or tuple(value.get("kinds", ())) != KINDS:
            raise ValueError("相手モデルのschemaまたは行動種別が不一致")
        if value.get("labelDefinitionVersion") != LABEL_DEFINITION_VERSION:
            raise ValueError("相手モデルのラベル定義版が不一致")
        if value.get("featureSchemaVersion") != FEATURE_SCHEMA:
            raise ValueError("相手モデルの特徴版が不一致")
        if value.get("modelVersion") != MODEL_VERSION:
            raise ValueError("相手モデルの計算方式が不一致")
        if value.get("exactCalculationVersion") != CALCULATION_VERSION:
            raise ValueError("厳密受け入れ計算版が不一致")
        if tuple(value.get("kindFeatureNames", ())) != KIND_FEATURE_NAMES:
            raise ValueError("種別特徴schemaが不一致")
        if tuple(value.get("detailFeatureNames", ())) != DETAIL_FEATURE_NAMES:
            raise ValueError("具体行動特徴schemaが不一致")
        model = cls(
            np.asarray(value["kindWeights"], dtype=float),
            np.asarray(value["detailWeights"], dtype=float),
            {str(key): float(item) for key, item in value["temperatures"].items()},
            None if value.get("fixedConstants") is None else FixedConstants.from_dict(value["fixedConstants"]),
        )
        if "fixedConstants" not in value:
            raise ValueError("相手モデルに固定定数の欄がない")
        if model.kind_weights.shape != (len(KINDS), len(KIND_FEATURE_NAMES)):
            raise ValueError("種別係数shapeが不一致")
        if model.detail_weights.shape != (len(KINDS), len(DETAIL_FEATURE_NAMES)):
            raise ValueError("具体行動係数shapeが不一致")
        return model


def action_nll_and_gradient(
    model: HierarchicalSoftmax,
    candidates: Sequence[EncodedCandidate],
    observed_action: Mapping[str, Any],
    phase: str,
) -> tuple[float, ModelGradient]:
    """正確に観測した一行動のNLLと解析勾配。"""
    probabilities = model.probabilities(candidates, phase)
    observed_key = canonical_action(observed_action)
    observed_index = next(
        (index for index, candidate in enumerate(candidates) if canonical_action(candidate.action) == observed_key), None
    )
    if observed_index is None:
        raise ValueError("観測行動が合法集合にない")
    probability = max(float(probabilities[observed_index]), np.finfo(float).tiny)
    gradient = _log_probability_gradient(model, candidates, observed_index, phase)
    gradient.kind *= -1.0
    gradient.detail *= -1.0
    return -math.log(probability), gradient


def _log_probability_gradient(
    model: HierarchicalSoftmax,
    candidates: Sequence[EncodedCandidate],
    selected_index: int,
    phase: str,
) -> ModelGradient:
    """log M(a|v) の勾配。固定成分の確率は係数に依存しないため勾配0とする。"""
    if model.fixed is None:
        return _base_log_probability_gradient(model, candidates, selected_index, phase)
    masses = fixed_kind_masses([candidate.kind for candidate in candidates], phase, model.fixed)
    if not masses:
        return _base_log_probability_gradient(model, candidates, selected_index, phase)
    if candidates[selected_index].kind in masses:
        return ModelGradient.zeros(model)
    residual = [index for index, candidate in enumerate(candidates) if candidate.kind not in masses]
    # 残余候補の確率は (1 - 固定質量) × 残余上の学習分布。定数倍は勾配に効かない。
    return _base_log_probability_gradient(
        model, [candidates[index] for index in residual], residual.index(selected_index), phase
    )


def _base_log_probability_gradient(
    model: HierarchicalSoftmax,
    candidates: Sequence[EncodedCandidate],
    selected_index: int,
    phase: str,
) -> ModelGradient:
    """固定成分を考えない学習分布での log M(a|v) の勾配。"""
    temperature = float(model.temperatures.get(phase, 1.0))
    groups: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(candidate.kind, []).append(index)
    kinds = list(groups)
    kind_logits = np.asarray(
        [model.kind_weights[KIND_INDEX[kind]] @ candidates[groups[kind][0]].kind_features for kind in kinds]
    ) / temperature
    kind_probability = _softmax(kind_logits)
    selected_kind = candidates[selected_index].kind
    gradient = ModelGradient.zeros(model)
    for group_index, kind in enumerate(kinds):
        coefficient = (1.0 if kind == selected_kind else 0.0) - float(kind_probability[group_index])
        gradient.kind[KIND_INDEX[kind]] += coefficient * candidates[groups[kind][0]].kind_features / temperature

    indices = groups[selected_kind]
    detail_logits = np.asarray(
        [model.detail_weights[KIND_INDEX[selected_kind]] @ candidates[index].detail_features for index in indices]
    ) / temperature
    detail_probability = _softmax(detail_logits)
    for local_index, candidate_index in enumerate(indices):
        coefficient = (1.0 if candidate_index == selected_index else 0.0) - float(detail_probability[local_index])
        gradient.detail[KIND_INDEX[selected_kind]] += (
            coefficient * candidates[candidate_index].detail_features / temperature
        )
    return gradient


def enumerate_joint_indices(per_seat: Mapping[int, Sequence[EncodedCandidate]]) -> Iterator[dict[int, int]]:
    seats = sorted(per_seat)
    for values in itertools.product(*(range(len(per_seat[seat])) for seat in seats)):
        yield dict(zip(seats, values))


def joint_resolution_likelihood_and_gradient(
    model: HierarchicalSoftmax,
    discarder: int,
    per_seat: Mapping[int, Sequence[EncodedCandidate]],
    observed_resolution: Mapping[str, Any],
    phase: str,
) -> tuple[float, ModelGradient, int]:
    """公開結果と両立する全希望を足し、-log尤度と勾配を返す。"""
    if not per_seat:
        raise ValueError("応答者がいない")
    probabilities = {seat: model.probabilities(candidates, phase) for seat, candidates in per_seat.items()}
    compatible: list[tuple[float, dict[int, int]]] = []
    for indices in enumerate_joint_indices(per_seat):
        choices = {seat: dict(per_seat[seat][index].action) for seat, index in indices.items()}
        if resolve_joint_response(discarder, choices) != dict(observed_resolution):
            continue
        mass = math.prod(float(probabilities[seat][index]) for seat, index in indices.items())
        compatible.append((mass, indices))
    likelihood = math.fsum(mass for mass, _ in compatible)
    if likelihood <= 0 or not math.isfinite(likelihood):
        raise ValueError("公開結果と両立する共同応答の確率質量がない")

    # -log Σ_c Π_j M(c_j) の勾配は、両立する組合せの事後確率で
    # 各 log M の勾配を平均して符号を反転したものになる。
    gradient = ModelGradient.zeros(model)
    for mass, indices in compatible:
        posterior = mass / likelihood
        for seat, index in indices.items():
            gradient.add_scaled(_log_probability_gradient(model, per_seat[seat], index, phase), -posterior)
    return -math.log(likelihood), gradient, len(compatible)


def exhaustive_joint_likelihood(
    model: HierarchicalSoftmax,
    discarder: int,
    per_seat: Mapping[int, Sequence[EncodedCandidate]],
    observed_resolution: Mapping[str, Any],
    phase: str,
) -> float:
    """D32-02用の素朴な全列挙。集約実装との独立照合に使う。"""
    total = 0.0
    for indices in enumerate_joint_indices(per_seat):
        choices = {seat: dict(per_seat[seat][index].action) for seat, index in indices.items()}
        if resolve_joint_response(discarder, choices) == dict(observed_resolution):
            total += math.prod(
                model.action_probability(per_seat[seat], per_seat[seat][index].action, phase)
                for seat, index in indices.items()
            )
    return total


def joint_resolution_likelihood(
    model: HierarchicalSoftmax,
    discarder: int,
    per_seat: Mapping[int, Sequence[EncodedCandidate]],
    observed_resolution: Mapping[str, Any],
    phase: str,
) -> tuple[float, int]:
    """評価用。勾配を作らずに同じ全列挙尤度を計算する。"""
    probabilities = {seat: model.probabilities(candidates, phase) for seat, candidates in per_seat.items()}
    masses = []
    for indices in enumerate_joint_indices(per_seat):
        choices = {seat: dict(per_seat[seat][index].action) for seat, index in indices.items()}
        if resolve_joint_response(discarder, choices) == dict(observed_resolution):
            masses.append(math.prod(float(probabilities[seat][index]) for seat, index in indices.items()))
    likelihood = math.fsum(masses)
    if likelihood <= 0 or not math.isfinite(likelihood):
        raise ValueError("公開結果と両立する共同応答の確率質量がない")
    return likelihood, len(masses)


def _tile34(action: Mapping[str, Any]) -> int | None:
    if "tile34" in action:
        return int(action["tile34"])
    consumed = action.get("consumed") or []
    return int(consumed[0]["tile34"]) if consumed else None


def _dora_from_indicator(indicator: int) -> int:
    if indicator < 27:
        base = indicator - indicator % 9
        return base + (indicator % 9 + 1) % 9
    if indicator <= 30:
        return 27 + (indicator - 27 + 1) % 4
    return 31 + (indicator - 31 + 1) % 3


class RoundFeatureState:
    """公開列と各家の私有自摸を一度だけ進め、行動前の手牌特徴を作る。"""

    def __init__(self, public: Mapping[str, Any], private: Mapping[int, Mapping[str, Any]]):
        self.initial = dict(public["initial"])
        self.current_scores = [int(value) for value in self.initial["scores"]]
        self.events = list(public["events"])
        self.position = 0
        self.hands = {
            seat: [dict(tile) for tile in private[seat]["initialHand"]]
            for seat in range(4)
        }
        self.draws = {
            (seat, int(event["rawEventIndex"])): dict(event["tile"])
            for seat in range(4)
            for event in private[seat]["events"]
            if event["type"] == "draw_observation"
        }
        self.melds: dict[int, list[list[dict[str, Any]]]] = {seat: [] for seat in range(4)}
        # 役の手掛かりに使う面子の種類。melds と同じ順に並べる。
        self.meld_kinds: dict[int, list[str]] = {seat: [] for seat in range(4)}
        # リーチ後に他家から出て、そのリーチ者が和了しなかった牌（見逃しでフリテンが確定する）。
        self.passed_after_riichi: dict[int, set[int]] = {seat: set() for seat in range(4)}
        # 捨牌・加槓の応答が解決するまで保留する（打った家、牌種、その時点のリーチ者）。
        self.pending_pass: tuple[int, int, frozenset[int]] | None = None
        self.rivers: dict[int, list[dict[str, Any]]] = {seat: [] for seat in range(4)}
        self.riichi: set[int] = set()
        self.riichi_river_start: dict[int, int] = {}
        self.pending_kakan_dora: int | None = None
        self.draw_count = 0
        self.discard_count = 0
        self.visible = Counter({int(self.initial["doraIndicator"]["tile34"]): 1})
        self.dora_indicators = [int(self.initial["doraIndicator"]["tile34"])]

    @staticmethod
    def _same_tile(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return int(left["tile34"]) == int(right["tile34"]) and bool(left["isRed"]) == bool(right["isRed"])

    def _remove(self, seat: int, tile: Mapping[str, Any]) -> None:
        for index, value in enumerate(self.hands[seat]):
            if self._same_tile(value, tile):
                self.hands[seat].pop(index)
                return
        # 赤区別の曖昧な加槓でも牌種保存を優先する。
        for index, value in enumerate(self.hands[seat]):
            if int(value["tile34"]) == int(tile["tile34"]):
                self.hands[seat].pop(index)
                return
        raise ValueError(f"手牌に公開牌がない: seat={seat}, tile={tile}")

    def advance(self, public_event_count: int) -> None:
        if public_event_count < self.position:
            raise ValueError("teacher windowが公開時系列順でない")
        while self.position < public_event_count:
            event = self.events[self.position]
            self.position += 1
            kind = str(event["type"])
            if kind == "draw":
                seat = int(event["seat"])
                tile = self.draws[(seat, int(event["rawEventIndex"]))]
                self.hands[seat].append(tile)
                self.draw_count += int(event.get("source") == "live")
            elif kind == "discard":
                seat = int(event["seat"])
                tile = dict(event["tile"])
                self.pending_pass = (seat, int(tile["tile34"]), frozenset(self.riichi - {seat}))
                self._remove(seat, tile)
                self.rivers[seat].append({**tile, "riichiDeclaration": bool(event.get("riichiDeclaration"))})
                self.visible[int(tile["tile34"])] += 1
                self.discard_count += 1
                if event.get("riichiDeclaration"):
                    self.riichi.add(seat)
                    self.riichi_river_start[seat] = len(self.rivers[seat]) - 1
                    self.current_scores[seat] -= 1000
            elif kind in {"chi", "pon", "daiminkan"}:
                # 鳴きが成立したならロンはなかった。捨牌はリーチ者に見逃されている。
                self._commit_pass()
                seat = int(event["seat"])
                tiles = [dict(tile) for tile in event["tiles"]]
                previous = next(
                    river[-1] for other, river in self.rivers.items() if other == int(event["fromSeat"]) and river
                )
                removed_called = False
                consumed = []
                for tile in tiles:
                    if not removed_called and self._same_tile(tile, previous):
                        removed_called = True
                        continue
                    self._remove(seat, tile)
                    self.visible[int(tile["tile34"])] += 1
                    consumed.append(tile)
                self.melds[seat].append(tiles)
                self.meld_kinds[seat].append(kind)
                self._add_dora(event)
            elif kind == "ankan":
                seat = int(event["seat"])
                tiles = [dict(tile) for tile in event["tiles"]]
                for tile in tiles:
                    self._remove(seat, tile)
                    self.visible[int(tile["tile34"])] += 1
                self.melds[seat].append(tiles)
                self.meld_kinds[seat].append("ankan")
                self._add_dora(event)
            elif kind == "kakan":
                seat = int(event["seat"])
                tiles = [dict(tile) for tile in event["tiles"]]
                tile34 = int(tiles[0]["tile34"])
                meld_index = next(
                    index for index, meld in enumerate(self.melds[seat])
                    if len(meld) == 3 and int(meld[0]["tile34"]) == tile34
                )
                old = self.melds[seat][meld_index]
                remaining = list(tiles)
                for old_tile in old:
                    for index, candidate in enumerate(remaining):
                        if self._same_tile(old_tile, candidate):
                            remaining.pop(index)
                            break
                added = remaining[0] if remaining else {"tile34": tile34, "isRed": False}
                self._remove(seat, added)
                self.visible[tile34] += 1
                self.melds[seat][meld_index] = tiles
                self.meld_kinds[seat][meld_index] = "kakan"
                self.pending_pass = (seat, tile34, frozenset(self.riichi - {seat}))
                # 搶槓の応答時点では新ドラはまだ見えない。projected eventに
                # 指示牌が同居していても、公開解決を通過するまで特徴へ入れない。
                if "revealedDoraIndicator" in event:
                    self.pending_kakan_dora = int(event["revealedDoraIndicator"]["tile34"])
            elif kind == "response_resolution" and event.get("resolution", {}).get("kind") == "pass":
                self._commit_pass()
            elif kind == "chankan_resolution" and event.get("resolution", {}).get("kind") == "pass":
                self._commit_pass()
                if self.pending_kakan_dora is not None:
                    self.dora_indicators.append(self.pending_kakan_dora)
                    self.visible[self.pending_kakan_dora] += 1
                    self.pending_kakan_dora = None

    def _commit_pass(self) -> None:
        if self.pending_pass is None:
            return
        _, tile34, riichi_seats = self.pending_pass
        for seat in riichi_seats:
            self.passed_after_riichi[seat].add(tile34)
        self.pending_pass = None

    def safe_tiles(self, riichi_seat: int) -> frozenset[int]:
        """リーチ者に対する安全牌集合G_r：本人の河の全牌とリーチ後に見逃された牌。"""

        river = {int(tile["tile34"]) for tile in self.rivers[riichi_seat]}
        return frozenset(river | self.passed_after_riichi[riichi_seat])

    def _add_dora(self, event: Mapping[str, Any]) -> None:
        if "revealedDoraIndicator" in event:
            tile34 = int(event["revealedDoraIndicator"]["tile34"])
            self.dora_indicators.append(tile34)
            self.visible[tile34] += 1

    def context(self, seat: int) -> dict[str, float]:
        counts = Counter(int(tile["tile34"]) for tile in self.hands[seat])
        total = max(1, sum(counts.values()))
        scores = self.current_scores
        # 同じ表示牌が複数ある場合、同じ通常ドラも表示枚数ぶん数える。
        dora_multiplicity = Counter(_dora_from_indicator(value) for value in self.dora_indicators)
        dora_count = sum(counts[tile] * multiplier for tile, multiplier in dora_multiplicity.items())
        dora_count += sum(bool(tile["isRed"]) for tile in self.hands[seat])
        suits = [sum(counts[index] for index in range(start, start + 9)) for start in (0, 9, 18)]
        honors = sum(counts[index] for index in range(27, 34))
        terminal_honors = honors + sum(counts[index] for index in (0, 8, 9, 17, 18, 26))
        open_melds = len(self.melds[seat])
        hand_counts = tuple(counts[index] for index in range(34))
        return {
            "bias": 1.0,
            "turn": min(1.0, self.discard_count / 72.0),
            "wall_remaining": max(0.0, (70 - self.draw_count) / 70.0),
            "dealer": float(seat == int(self.initial["dealerSeat"])),
            "score_delta": (scores[seat] - sum(scores) / 4.0) / 10_000.0,
            "leader_delta": (scores[seat] - max(scores)) / 10_000.0,
            "honba": min(1.0, int(self.initial["honba"]) / 5.0),
            "riichi_sticks": min(1.0, (int(self.initial["riichiSticks"]) + len(self.riichi)) / 5.0),
            "kyoku": min(1.0, int(self.initial["kyoku"]) / 7.0),
            "fixed_melds": open_melds / 4.0,
            "own_riichi": float(seat in self.riichi),
            "opponent_riichi_count": len(self.riichi - {seat}) / 3.0,
            "hand_shanten": shanten(hand_counts, open_melds) / 6.0,
            "dora_count": min(1.0, dora_count / 5.0),
            "honor_ratio": honors / total,
            "terminal_honor_ratio": terminal_honors / total,
            "suit_concentration": max(suits) / total,
        }

    def cache_identity(self, seat: int, actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """同じ特徴入力だけが同じcache行を共有できるように境界をhash化する。"""

        hand = sorted(
            ({"tile34": int(tile["tile34"]), "isRed": bool(tile["isRed"])} for tile in self.hands[seat]),
            key=canonical_action,
        )
        return {
            "actorSeat": seat,
            "publicPrefixHash": _json_hash({"initial": self.initial, "events": self.events[: self.position]}),
            "privatePrefixHash": _json_hash({"hand": hand, "fixedMelds": len(self.melds[seat])}),
            "legalCandidateSetHash": _json_hash(list(actions)),
        }

    def encode(self, seat: int, actions: Sequence[Mapping[str, Any]]) -> list[EncodedCandidate]:
        context = self.context(seat)
        metrics = [self._action_metrics(seat, action, context) for action in actions]
        best_by_kind: dict[str, tuple[tuple[float, float, float, str], Mapping[str, float]]] = {}
        for action, metric in zip(actions, metrics):
            key = str(action["kind"])
            value = (
                metric["action_shanten"],
                -metric["ukeire_count"],
                -metric["ukeire_kinds"],
                canonical_action(action),
            )
            current = best_by_kind.get(key)
            if current is None or value < current[0]:
                best_by_kind[key] = (value, metric)
        details = [
            self._detail_values(seat, action, context, metric) for action, metric in zip(actions, metrics)
        ]
        # 種別headへ渡す集約：安全に打てる候補があるか、種別内の全候補で役の手掛かりがないか。
        min_danger: dict[str, float] = {}
        min_no_cue: dict[str, float] = {}
        for action, detail in zip(actions, details):
            key = str(action["kind"])
            min_danger[key] = min(min_danger.get(key, 1.0e9), detail["danger_max_rate"])
            min_no_cue[key] = min(min_no_cue.get(key, 1.0), detail["open_no_listed_yaku_cue"])
        encoded = []
        for action, metric, detail_values in zip(actions, metrics, details):
            key = str(action["kind"])
            best = best_by_kind[key][1]
            kind_values = {
                **context,
                "best_action_shanten": best["action_shanten"],
                "best_local_acceptance": best["local_acceptance"],
                "best_ukeire_count": best["ukeire_count"],
                "best_ukeire_kinds": best["ukeire_kinds"],
                "best_ukeire_applicable": best["ukeire_applicable"],
                "min_danger_max_rate": min_danger[key],
                "min_open_no_listed_yaku_cue": min_no_cue[key],
            }
            encoded.append(
                EncodedCandidate(
                    dict(action),
                    np.asarray([kind_values[name] for name in KIND_FEATURE_NAMES], dtype=float),
                    np.asarray([detail_values[name] for name in DETAIL_FEATURE_NAMES], dtype=float),
                )
            )
        return encoded

    def _action_metrics(
        self, seat: int, action: Mapping[str, Any], context: Mapping[str, float]
    ) -> dict[str, float]:
        counts = Counter(int(tile["tile34"]) for tile in self.hands[seat])
        called_tile34 = None
        if self.position:
            previous = self.events[self.position - 1]
            if previous.get("type") == "discard":
                called_tile34 = int(previous["tile"]["tile34"])
            elif previous.get("type") == "kakan":
                called_tile34 = int(previous["tiles"][0]["tile34"])
        view = make_action_view(
            tuple(counts[index] for index in range(34)),
            len(self.melds[seat]),
            tuple(self.visible[index] for index in range(34)),
            called_tile34,
        )
        projected = project_action(view, action)
        projected_counts = Counter(
            {index: amount for index, amount in enumerate(projected["counts34"]) if amount}
        )
        melds = tuple(
            MeldView(meld_kind, tuple(sorted(int(tile["tile34"]) for tile in tiles)))
            for meld_kind, tiles in zip(self.meld_kinds[seat], self.melds[seat])
        )
        yaku = yaku_shape_features(
            projected["counts34"],
            melds_after_action(melds, action, called_tile34),
            27 + (seat - int(self.initial["dealerSeat"])) % 4,
            27 + int(self.initial["kyoku"]) // 4,
        )
        return {
            "action_shanten": float(projected["shanten"]) / 6.0,
            "local_acceptance": self._local_acceptance(projected_counts),
            "ukeire_count": float(projected["ukeireCount"]) / 136.0,
            "ukeire_kinds": float(projected["ukeireKinds"]) / 34.0,
            "ukeire_applicable": float(projected["applicable"]),
            **yaku,
        }

    def _local_acceptance(self, counts: Counter[int]) -> float:
        # 厳密な受け入れは候補ごとに34回のシャンテン計算を要する。
        # 初版は牌効率に単調な局所連結度を使い、未実装項目をmanifestへ明記する。
        score = 0.0
        for tile34, amount in counts.items():
            if amount <= 0:
                continue
            if amount >= 2:
                score += 1.0
            if tile34 < 27:
                rank = tile34 % 9
                for delta in (-2, -1, 1, 2):
                    other = tile34 + delta
                    if 0 <= rank + delta < 9 and counts[other] > 0:
                        score += 0.25
        return min(1.0, score / 10.0)

    def _detail_values(
        self,
        seat: int,
        action: Mapping[str, Any],
        context: Mapping[str, float],
        metric: Mapping[str, float],
    ) -> dict[str, float]:
        tile34 = _tile34(action)
        red = bool(action.get("isRed", action.get("addedIsRed", False)))
        consumed = action.get("consumed") or []
        red_ratio = sum(bool(tile["isRed"]) for tile in consumed) / max(1, len(consumed))
        dora_tiles = {_dora_from_indicator(value) for value in self.dora_indicators}
        danger_tile = tile34 if action.get("kind") in {"discard", "riichi_discard"} else None
        dealer = int(self.initial["dealerSeat"])
        opponents = [
            RiichiOpponentView(other, other == dealer, self.safe_tiles(other))
            for other in sorted(self.riichi - {seat})
        ]
        # 見えている枚数は、公開済みの牌に行動する家自身の手牌を加えたもの。
        seen = [self.visible[index] for index in range(34)]
        for tile in self.hands[seat]:
            seen[int(tile["tile34"])] += 1
        danger = danger_features(danger_tile, opponents, seen)
        honor = float(tile34 is not None and tile34 >= 27)
        terminal = float(tile34 is not None and tile34 < 27 and tile34 % 9 in {0, 8})
        simple = float(tile34 is not None and tile34 < 27 and tile34 % 9 not in {0, 8})
        values = {
            "bias": 1.0,
            "action_shanten": float(metric["action_shanten"]),
            "local_acceptance": float(metric["local_acceptance"]),
            "ukeire_count": float(metric["ukeire_count"]),
            "ukeire_kinds": float(metric["ukeire_kinds"]),
            "ukeire_applicable": float(metric["ukeire_applicable"]),
            "is_red": float(red),
            "is_dora": float(tile34 in dora_tiles if tile34 is not None else False),
            "origin_drawn": float(action.get("origin") == "drawn"),
            "honor": honor,
            "terminal": terminal,
            "simple": simple,
            "consumed_red_ratio": red_ratio,
            "turn_x_shanten": context["turn"] * float(metric["action_shanten"]),
            "late_x_dora": context["turn"] * float(tile34 in dora_tiles if tile34 is not None else False),
            **danger,
            "riichi_x_genbutsu_all": context["opponent_riichi_count"] * danger["genbutsu_all"],
        }
        values.update({name: float(metric[name]) for name in YAKU_SHAPE_DETAIL_NAMES})
        values.update({f"tile34_{index}": float(tile34 == index) for index in range(34)})
        return values


def _iter_jsonl_gzip(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _deterministic_candidate(action: Mapping[str, Any]) -> EncodedCandidate:
    """一択集合では特徴によらず確率1なので、ゼロ特徴で同じ意味になる。"""
    return EncodedCandidate(
        dict(action),
        np.zeros(len(KIND_FEATURE_NAMES), dtype=float),
        np.zeros(len(DETAIL_FEATURE_NAMES), dtype=float),
    )


def _group_by_round(rows: Iterable[dict[str, Any]]) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    for round_id, group in itertools.groupby(rows, key=lambda row: str(row["roundId"])):
        yield round_id, list(group)


def iter_encoded_windows(
    dataset_dir: Path,
    maximum_windows: int | None = None,
    selected_splits: set[str] | None = None,
    selected_window_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """三つのD.3.1列をroundIdでlockstep結合する。"""
    public_rows = iter(_iter_jsonl_gzip(dataset_dir / "public-events.jsonl.gz"))
    private_groups = iter(_group_by_round(_iter_jsonl_gzip(dataset_dir / "private-events.jsonl.gz")))
    produced = Counter()
    public: dict[str, Any] | None = None
    private_round_id: str | None = None
    private_values: list[dict[str, Any]] | None = None
    remaining_window_ids = None if selected_window_ids is None else set(selected_window_ids)
    for round_id, windows in _group_by_round(_iter_jsonl_gzip(dataset_dir / "teacher-windows.jsonl.gz")):
        # 公開・私有列は、合法な教師窓が一つもない局も保持する上位集合。
        # 同じ抽出順を保ったまま、教師局に一致するまで両列を一緒に進める。
        while public is None or str(public["roundId"]) != round_id:
            try:
                public = next(public_rows)
                private_round_id, private_values = next(private_groups)
            except StopIteration as error:
                raise ValueError(f"教師局に対応する公開・私有列がない: {round_id}") from error
            if private_round_id != str(public["roundId"]):
                raise ValueError(f"公開・私有列のroundId順が不一致: {public['roundId']}")
        assert private_values is not None
        split = str(windows[0]["developmentSplit"])
        if selected_splits is not None and split not in selected_splits:
            continue
        if maximum_windows is not None and produced[split] >= maximum_windows:
            continue
        private = {int(row["seat"]): row for row in private_values}
        if set(private) != {0, 1, 2, 3}:
            raise ValueError(f"私有列が4家分でない: {round_id}")
        state = RoundFeatureState(public, private)
        for window in windows:
            state.advance(int(window["publicEventCount"]))
            window_id = str(window["windowId"])
            if remaining_window_ids is not None and window_id not in remaining_window_ids:
                continue
            if maximum_windows is not None and produced[split] >= maximum_windows:
                continue
            if "legalActions" in window:
                candidates = state.encode(int(window["actorSeat"]), window["legalActions"])
                yield {
                    "window": window,
                    "candidates": candidates,
                    "identities": [state.cache_identity(int(window["actorSeat"]), window["legalActions"])],
                }
            else:
                per_seat = {
                    int(seat): (
                        [_deterministic_candidate(value["actions"][0])]
                        if len(value["actions"]) == 1
                        else state.encode(int(seat), value["actions"])
                    )
                    for seat, value in window["legalBySeat"].items()
                }
                yield {
                    "window": window,
                    "perSeat": per_seat,
                    "identities": [
                        state.cache_identity(int(seat), value["actions"])
                        for seat, value in sorted(window["legalBySeat"].items(), key=lambda item: int(item[0]))
                    ],
                }
            produced[split] += 1
            if remaining_window_ids is not None:
                remaining_window_ids.remove(window_id)
                if not remaining_window_ids:
                    return
    if remaining_window_ids:
        raise ValueError(f"選定した性能標本が教師窓にない: {sorted(remaining_window_ids)[:3]}")


def _feature_code_hash() -> str:
    feature_module = Path(__file__).with_name("ev_policy_features.py").resolve()
    return _json_hash(
        {
            "exactFeatureModuleSha256": sha256_file(feature_module),
            "roundFeatureAdapter": inspect.getsource(RoundFeatureState),
            "adapterHelpers": inspect.getsource(_tile34) + inspect.getsource(_dora_from_indicator),
            "kindFeatureNames": KIND_FEATURE_NAMES,
            "detailFeatureNames": DETAIL_FEATURE_NAMES,
            # 危険率の表は別モジュールにあるため、値そのものを依存hashへ含める。
            "dangerRates": DANGER_RATES,
        }
    )


def _validate_feature_dataset(dataset_dir: Path) -> dict[str, Any]:
    verification = json.loads((dataset_dir / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") not in {"pass", "debug_pass"}:
        raise ValueError("検証済みD.3.1データが必要")
    summary = json.loads((dataset_dir / "extraction-summary.json").read_text(encoding="utf-8"))
    if summary.get("labelDefinition", {}).get("version") != LABEL_DEFINITION_VERSION:
        raise ValueError("D.3.1ラベル定義が相手モデルの固定版と不一致")
    input_manifest = summary.get("input", {})
    selected = set(input_manifest.get("selectedSeasons", ()))
    allowed = set(input_manifest.get("allowedSeasons", ()))
    forbidden = set(input_manifest.get("forbiddenSeasons", ()))
    if selected & forbidden:
        raise ValueError(f"将来評価期間を特徴入力へ使用できない: {sorted(selected & forbidden)}")
    if allowed and not selected <= allowed:
        raise ValueError(f"未指定期間を特徴入力へ使用できない: {sorted(selected - allowed)}")
    for item in summary.get("generatedFiles", {}).get("files", ()):
        path = dataset_dir / str(item["path"])
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"D.3.1入力ファイルhashが不一致: {item.get('path')}")
    return summary


def _candidate_payload(candidate: EncodedCandidate) -> dict[str, Any]:
    return {
        "action": dict(candidate.action),
        "kindFeatures": [float(value) for value in candidate.kind_features],
        "detailFeatures": [float(value) for value in candidate.detail_features],
    }


def _candidate_count(encoded: Mapping[str, Any]) -> int:
    if "candidates" in encoded:
        return len(encoded["candidates"])
    return sum(len(candidates) for candidates in encoded["perSeat"].values())


def _action_structure(value: Mapping[str, Any]) -> Any:
    if "candidates" in value:
        return [dict(candidate.action) for candidate in value["candidates"]]
    return {
        str(seat): [dict(candidate.action) for candidate in candidates]
        for seat, candidates in sorted(value["perSeat"].items())
    }


def _teacher_action_structure(window: Mapping[str, Any]) -> Any:
    if "legalActions" in window:
        return window["legalActions"]
    return {
        str(seat): value["actions"]
        for seat, value in sorted(window["legalBySeat"].items(), key=lambda item: int(item[0]))
    }


def _serialize_encoded(encoded: Mapping[str, Any]) -> dict[str, Any]:
    window = encoded["window"]
    base = {
        "schemaVersion": FEATURE_SCHEMA,
        "windowId": window["windowId"],
        "split": window["developmentSplit"],
        "phase": window["phase"],
        "candidateCount": _candidate_count(encoded),
        "candidateActionsHash": _json_hash(_action_structure(encoded)),
        "identities": encoded.get("identities", []),
        # 特徴計算が終わった後に教師行を結合して保存する。encodeへは渡さない。
        "teacherWindow": window,
    }
    if "candidates" in encoded:
        base.update({"mode": "self", "candidates": [_candidate_payload(value) for value in encoded["candidates"]]})
    else:
        base.update(
            {
                "mode": "response",
                "perSeat": {
                    str(seat): [_candidate_payload(value) for value in candidates]
                    for seat, candidates in sorted(encoded["perSeat"].items())
                },
            }
        )
    return base


def _deserialize_candidate(value: Mapping[str, Any]) -> EncodedCandidate:
    kind = np.asarray(value["kindFeatures"], dtype=float)
    detail = np.asarray(value["detailFeatures"], dtype=float)
    if kind.shape != (len(KIND_FEATURE_NAMES),) or not np.all(np.isfinite(kind)):
        raise ValueError("cacheの種別特徴shapeまたは数値が不正")
    if detail.shape != (len(DETAIL_FEATURE_NAMES),) or not np.all(np.isfinite(detail)):
        raise ValueError("cacheの具体行動特徴shapeまたは数値が不正")
    return EncodedCandidate(dict(value["action"]), kind, detail)


def _deserialize_encoded(record: Mapping[str, Any], window: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("schemaVersion") != FEATURE_SCHEMA:
        raise ValueError("cache行の特徴schemaが不一致")
    if record.get("windowId") != window.get("windowId"):
        raise ValueError("cacheと教師のwindowId順が不一致")
    expected_action_hash = _json_hash(_teacher_action_structure(window))
    if record.get("candidateActionsHash") != expected_action_hash:
        raise ValueError(f"cacheの合法候補集合が不一致: {window.get('windowId')}")
    if record.get("mode") == "self" and "legalActions" in window:
        result = {"window": window, "candidates": [_deserialize_candidate(value) for value in record["candidates"]]}
    elif record.get("mode") == "response" and "legalBySeat" in window:
        result = {
            "window": window,
            "perSeat": {
                int(seat): [_deserialize_candidate(value) for value in candidates]
                for seat, candidates in record["perSeat"].items()
            },
        }
    else:
        raise ValueError(f"cache行の窓種別が教師と不一致: {window.get('windowId')}")
    if _candidate_count(result) != int(record.get("candidateCount", -1)):
        raise ValueError(f"cache行の候補数が不一致: {window.get('windowId')}")
    return result


def _iter_teacher_windows(dataset_dir: Path, maximum_windows: int | None) -> Iterator[dict[str, Any]]:
    produced = Counter()
    for row in _iter_jsonl_gzip(dataset_dir / "teacher-windows.jsonl.gz"):
        split = str(row["developmentSplit"])
        if maximum_windows is not None and produced[split] >= maximum_windows:
            continue
        produced[split] += 1
        yield row


def _base_feature_manifest(dataset_dir: Path, maximum_windows: int | None, windows_per_shard: int) -> dict[str, Any]:
    return {
        "schemaVersion": FEATURE_CACHE_SCHEMA,
        "status": "building",
        "featureSchemaVersion": FEATURE_SCHEMA,
        "calculationVersion": CALCULATION_VERSION,
        "implementedGroups": list(IMPLEMENTED_FEATURE_GROUPS),
        "datasetManifestHash": _dataset_hash(dataset_dir),
        "codeDependencyHash": _feature_code_hash(),
        "labelDefinitionVersion": LABEL_DEFINITION_VERSION,
        "kindFeatureNames": list(KIND_FEATURE_NAMES),
        "detailFeatureNames": list(DETAIL_FEATURE_NAMES),
        "normalization": {"shanten": 6.0, "ukeireCount": 136.0, "ukeireKinds": 34.0},
        "maximumWindows": maximum_windows,
        "maximumWindowsPerShard": windows_per_shard,
        "shards": [],
        "countsBySplit": {},
        "totalWindows": 0,
        "totalCandidates": 0,
    }


def _keep_smallest(heap: list[tuple[int, int, str, str]], limit: int, rank: int, order: int, window_id: str, phase: str) -> None:
    item = (-rank, -order, window_id, phase)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _heap_values(heap: Sequence[tuple[int, int, str, str]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {"rank": -rank, "originalOrder": -order, "windowId": window_id, "phase": phase}
            for rank, order, window_id, phase in heap
        ),
        key=lambda item: (item["rank"], item["originalOrder"]),
    )


def _peak_working_set_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            get_process_memory_info.restype = wintypes.BOOL
            handle = get_current_process()
            if get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError):
        return None


def _performance_sample(
    dataset_dir: Path,
    *,
    seed: int,
    benchmark_windows: int,
    minimum_per_phase: int,
    edge_per_kind: int,
) -> dict[str, Any]:
    phase_heaps = {phase: [] for phase in PHASES}
    overall: list[tuple[int, int, str, str]] = []
    edge_heaps = {kind: [] for kind in KINDS}
    overall_limit = benchmark_windows + minimum_per_phase * len(PHASES)
    train_windows = 0
    for order, row in enumerate(_iter_jsonl_gzip(dataset_dir / "teacher-windows.jsonl.gz")):
        if row.get("developmentSplit") != "train":
            continue
        train_windows += 1
        window_id = str(row["windowId"])
        phase = str(row["phase"])
        rank = int(hashlib.sha256(f"{seed}:speed:{window_id}".encode()).hexdigest(), 16)
        _keep_smallest(overall, overall_limit, rank, order, window_id, phase)
        if phase in phase_heaps:
            _keep_smallest(phase_heaps[phase], minimum_per_phase, rank, order, window_id, phase)
        if "legalActions" in row:
            kinds = {str(action["kind"]) for action in row["legalActions"]}
        else:
            kinds = {
                str(action["kind"])
                for value in row["legalBySeat"].values()
                for action in value["actions"]
            }
        for kind in kinds:
            edge_rank = int(hashlib.sha256(f"{seed}:edge:{kind}:{window_id}".encode()).hexdigest(), 16)
            _keep_smallest(edge_heaps[kind], edge_per_kind, edge_rank, order, window_id, phase)

    mandatory = {
        item["windowId"]: item
        for heap in phase_heaps.values()
        for item in _heap_values(heap)
    }
    selected = dict(mandatory)
    for item in _heap_values(overall):
        if len(selected) >= benchmark_windows:
            break
        selected.setdefault(item["windowId"], item)
    performance = sorted(selected.values(), key=lambda item: item["originalOrder"])
    edges = {
        kind: sorted(_heap_values(heap), key=lambda item: item["originalOrder"])
        for kind, heap in edge_heaps.items()
    }
    return {
        "seed": seed,
        "trainWindows": train_windows,
        "performance": performance,
        "minimumPerPhaseRequested": minimum_per_phase,
        "performanceByPhase": dict(sorted(Counter(item["phase"] for item in performance).items())),
        "edges": edges,
        "edgeCounts": {kind: len(items) for kind, items in edges.items()},
    }


def probe_opponent_features(
    dataset_dir: Path,
    output_dir: Path,
    *,
    benchmark_windows: int = 20_000,
    seed: int = 20260909,
) -> dict[str, Any]:
    """固定標本でcold、warm、永続読込を測り、全件生成予算を判定する。"""

    _validate_feature_dataset(dataset_dir)
    if benchmark_windows < 1 or benchmark_windows > 20_000:
        raise ValueError("性能標本数は1〜20000が必要")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = _performance_sample(
        dataset_dir,
        seed=seed,
        benchmark_windows=benchmark_windows,
        minimum_per_phase=min(256, max(1, benchmark_windows // len(PHASES))),
        edge_per_kind=64,
    )
    debug = benchmark_windows != 20_000
    manifest = {
        "schemaVersion": "ev-policy-opponent-feature-probe-manifest/v1",
        "status": "debug" if debug else "frozen",
        "datasetManifestHash": _dataset_hash(dataset_dir),
        "featureSchemaVersion": FEATURE_SCHEMA,
        "calculationVersion": CALCULATION_VERSION,
        "codeDependencyHash": _feature_code_hash(),
        "sample": sample,
        "budgets": {
            "probeSeconds": 3600,
            "fullBuildAndVerifySeconds": 28_800,
            "peakWorkingSetBytes": 4 * 1024**3,
            "cacheBytes": 8 * 1024**3,
        },
    }
    write_json_atomic(output_dir / "probe-manifest.json", manifest)
    selected_ids = {item["windowId"] for item in sample["performance"]}

    def measure(clear: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if clear:
            clear_all_shape_caches()
        started = time.perf_counter()
        previous = started
        phase_seconds = Counter()
        rows = []
        candidates = 0
        for encoded in iter_encoded_windows(dataset_dir, selected_splits={"train"}, selected_window_ids=selected_ids):
            now = time.perf_counter()
            phase_seconds[str(encoded["window"]["phase"])] += now - previous
            previous = now
            candidates += _candidate_count(encoded)
            rows.append(_serialize_encoded(encoded))
        elapsed = time.perf_counter() - started
        return {
            "seconds": elapsed,
            "windows": len(rows),
            "candidates": candidates,
            "phaseSeconds": dict(sorted(phase_seconds.items())),
            "peakWorkingSetBytes": _peak_working_set_bytes(),
        }, rows

    cold, rows = measure(True)
    warm, warm_rows = measure(False)
    if [_json_hash(row) for row in rows] != [_json_hash(row) for row in warm_rows]:
        raise ValueError("coldとwarmで特徴内容が一致しない")
    shard = write_feature_shard(output_dir / "debug-features.jsonl.gz", rows)
    read_started = time.perf_counter()
    loaded = list(iter_feature_shard(output_dir / shard["path"]))
    persistent_seconds = time.perf_counter() - read_started
    if [_json_hash(row) for row in rows] != [_json_hash(row) for row in loaded]:
        raise ValueError("永続cache再読込で特徴内容が一致しない")
    total_dataset_windows = int(
        json.loads((dataset_dir / "extraction-summary.json").read_text(encoding="utf-8"))
        .get("totals", {})
        .get("teacherWindows", 0)
    )
    projected_seconds = (
        None
        if debug
        else cold["seconds"] * total_dataset_windows / max(1, cold["windows"])
    )
    peak = cold["peakWorkingSetBytes"] or warm["peakWorkingSetBytes"]
    within_budget = (
        (debug or cold["seconds"] <= 3600)
        and (peak is None or peak <= 4 * 1024**3)
    )
    report = {
        "schemaVersion": "ev-policy-opponent-feature-probe/v1",
        "status": "debug_complete" if debug else ("pass" if within_budget else "feature_budget_exceeded"),
        "cold": cold,
        "warm": warm,
        "persistentRead": {"seconds": persistent_seconds, "windows": len(loaded), **shard},
        "projection": {
            "datasetWindows": total_dataset_windows,
            "fullBuildSecondsFromColdMean": projected_seconds,
            "fullBuildProjectionIsGate": False,
        },
        "budgetGate": {"withinBudget": within_budget, "fullBuildRequiresActualMeasurement": True},
        "runtimeEnvironment": {
            "pythonVersion": platform.python_version(),
            "numpyVersion": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    write_json_atomic(output_dir / "probe.json", report)
    return report


def _validate_manifest_identity(manifest: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for name in (
        "schemaVersion", "featureSchemaVersion", "calculationVersion", "datasetManifestHash",
        "codeDependencyHash", "labelDefinitionVersion", "kindFeatureNames", "detailFeatureNames",
        "maximumWindows", "maximumWindowsPerShard",
    ):
        if manifest.get(name) != expected.get(name):
            raise ValueError(f"特徴cacheの再利用条件が不一致: {name}")


def build_opponent_feature_cache(
    dataset_dir: Path,
    output_dir: Path,
    *,
    maximum_windows: int | None = None,
    windows_per_shard: int = 10_000,
    resume: bool = False,
    maximum_seconds: float = 28_800.0,
    maximum_bytes: int = 8 * 1024**3,
    stop_after_shards: int | None = None,
) -> dict[str, Any]:
    """v2特徴を元の教師順でshard化する。完了manifestは全件照合後だけ作る。"""

    started = time.perf_counter()
    _validate_feature_dataset(dataset_dir)
    if windows_per_shard < 1 or windows_per_shard > 10_000:
        raise ValueError("一つの特徴shardは1〜10000窓が必要")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "build-state.json"
    manifest_path = output_dir / "manifest.json"
    expected = _base_feature_manifest(dataset_dir, maximum_windows, windows_per_shard)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest_identity(manifest, expected)
        if manifest.get("status") in {"complete", "debug"}:
            verify_opponent_feature_cache(dataset_dir, output_dir)
            return manifest
        raise ValueError("完了していないmanifestを再利用できない")
    if resume:
        if not state_path.exists():
            raise ValueError("再開対象のbuild-state.jsonがない")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_manifest_identity(state, expected)
    else:
        if state_path.exists() or any(output_dir.glob("features-*.jsonl.gz")):
            raise ValueError("既存の未完了cacheには--resumeが必要")
        state = expected
        write_json_atomic(state_path, state)

    for shard in state["shards"]:
        path = output_dir / shard["path"]
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"再開済みshardのhashが不一致: {shard['path']}")
    completed = int(state.get("totalWindows", 0))
    generated = iter_encoded_windows(dataset_dir, maximum_windows, None)
    for _ in range(completed):
        try:
            next(generated)
        except StopIteration as error:
            raise ValueError("再開位置が教師窓数を超える") from error
    previous_elapsed = float(state.get("elapsedSeconds", 0.0))
    shard_rows: list[dict[str, Any]] = []
    counts = Counter({str(key): int(value) for key, value in state.get("countsBySplit", {}).items()})
    shard_index = len(state["shards"])

    def flush() -> None:
        nonlocal shard_rows, shard_index
        if not shard_rows:
            return
        path = output_dir / f"features-{shard_index:05d}.jsonl.gz"
        shard = write_feature_shard(path, shard_rows)
        state["shards"].append(shard)
        state["totalWindows"] = int(state["totalWindows"]) + shard["windows"]
        state["totalCandidates"] = int(state["totalCandidates"]) + shard["candidates"]
        state["countsBySplit"] = dict(sorted(counts.items()))
        state["elapsedSeconds"] = previous_elapsed + time.perf_counter() - started
        write_json_atomic(state_path, state)
        shard_rows = []
        shard_index += 1
        print(json.dumps({"progress": "opponent_feature_shard_complete", **shard}, ensure_ascii=False), flush=True)

    for encoded in generated:
        row = _serialize_encoded(encoded)
        shard_rows.append(row)
        counts[str(row["split"])] += 1
        if len(shard_rows) >= windows_per_shard:
            flush()
            if stop_after_shards is not None and len(state["shards"]) >= stop_after_shards:
                raise RuntimeError("debug_feature_build_interrupted")
        elapsed = previous_elapsed + time.perf_counter() - started
        cache_bytes = sum(int(item["bytes"]) for item in state["shards"])
        if elapsed > maximum_seconds or cache_bytes > maximum_bytes:
            state["status"] = "feature_budget_exceeded"
            state["budget"] = {"elapsedSeconds": elapsed, "bytes": cache_bytes}
            write_json_atomic(state_path, state)
            raise ValueError("feature_budget_exceeded")
    flush()
    state["status"] = "debug" if maximum_windows is not None else "complete"
    state["countsBySplit"] = dict(sorted(counts.items()))
    state["elapsedSeconds"] = previous_elapsed + time.perf_counter() - started
    state["cacheBytes"] = sum(int(item["bytes"]) for item in state["shards"])
    write_json_atomic(manifest_path, state)
    return verify_opponent_feature_cache(dataset_dir, output_dir)


def verify_opponent_feature_cache(dataset_dir: Path, feature_dir: Path) -> dict[str, Any]:
    _validate_feature_dataset(dataset_dir)
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("特徴cacheの完了manifestがない")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _base_feature_manifest(
        dataset_dir,
        manifest.get("maximumWindows"),
        int(manifest.get("maximumWindowsPerShard", 0)),
    )
    _validate_manifest_identity(manifest, expected)
    if manifest.get("status") not in {"complete", "debug"}:
        raise ValueError(f"未完了の特徴cacheは利用できない: {manifest.get('status')}")
    seen: set[str] = set()
    rows = 0
    candidates = 0
    splits = Counter()
    for shard in manifest.get("shards", ()):
        path = feature_dir / str(shard["path"])
        if not path.is_file() or sha256_file(path) != shard.get("sha256"):
            raise ValueError(f"特徴shardのhashが不一致: {shard.get('path')}")
        shard_rows = 0
        shard_candidates = 0
        for record in iter_feature_shard(path):
            window_id = str(record.get("windowId"))
            if window_id in seen:
                raise ValueError(f"特徴cacheにwindowId重複: {window_id}")
            seen.add(window_id)
            shard_rows += 1
            shard_candidates += int(record.get("candidateCount", -1))
            splits[str(record.get("split"))] += 1
            # 配列shapeと有限値は教師との結合前にも検査する。
            values = record.get("candidates", ()) if record.get("mode") == "self" else itertools.chain.from_iterable(record.get("perSeat", {}).values())
            for value in values:
                _deserialize_candidate(value)
        if shard_rows != int(shard.get("windows", -1)) or shard_candidates != int(shard.get("candidates", -1)):
            raise ValueError(f"特徴shardの件数がmanifestと不一致: {shard.get('path')}")
        rows += shard_rows
        candidates += shard_candidates
    if rows != int(manifest.get("totalWindows", -1)) or candidates != int(manifest.get("totalCandidates", -1)):
        raise ValueError("特徴cache総件数がmanifestと不一致")
    if dict(sorted(splits.items())) != manifest.get("countsBySplit"):
        raise ValueError("特徴cacheのsplit件数がmanifestと不一致")
    expected_windows = sum(1 for _ in _iter_teacher_windows(dataset_dir, manifest.get("maximumWindows")))
    if rows != expected_windows:
        raise ValueError(f"特徴cacheに教師窓の欠落がある: expected={expected_windows}, actual={rows}")
    return {**manifest, "verification": {"status": "pass", "windows": rows, "candidates": candidates}}


def iter_cached_windows(
    dataset_dir: Path,
    feature_dir: Path,
    maximum_windows: int | None = None,
    selected_splits: set[str] | None = None,
    *,
    verified_manifest: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    manifest = dict(verified_manifest) if verified_manifest is not None else verify_opponent_feature_cache(dataset_dir, feature_dir)
    cache_limit = manifest.get("maximumWindows")
    if cache_limit is not None and (maximum_windows is None or maximum_windows > int(cache_limit)):
        raise ValueError("debug特徴cacheの範囲を超えてfit/evaluateできない")
    produced = Counter()
    for shard in manifest["shards"]:
        for record in iter_feature_shard(feature_dir / shard["path"]):
            window = record.get("teacherWindow")
            if not isinstance(window, Mapping):
                raise ValueError(f"特徴cacheに教師結合行がない: {record.get('windowId')}")
            encoded = _deserialize_encoded(record, window)
            split = str(window["developmentSplit"])
            if selected_splits is not None and split not in selected_splits:
                continue
            if maximum_windows is not None and produced[split] >= maximum_windows:
                continue
            produced[split] += 1
            yield encoded


def window_loss_and_gradient(
    model: HierarchicalSoftmax, encoded: Mapping[str, Any]
) -> tuple[float, ModelGradient, dict[str, Any]] | None:
    window = encoded["window"]
    phase = str(window["phase"])
    if "candidates" in encoded:
        observation = window["observation"]
        if observation["status"] != "exact" or not window["learningMask"].get("kind", False):
            return None
        loss, gradient = action_nll_and_gradient(model, encoded["candidates"], observation["action"], phase)
        return loss, gradient, {"compatibleJointClaims": None}
    if not window["learningMask"].get("jointKind", False):
        return None
    # 全員が一択なら尤度1であり、計算しても勾配は厳密に0。
    if all(len(actions) == 1 for actions in encoded["perSeat"].values()):
        return 0.0, ModelGradient.zeros(model), {"compatibleJointClaims": 1, "structural": True}
    loss, gradient, compatible = joint_resolution_likelihood_and_gradient(
        model,
        int(window["actorSeat"]),
        encoded["perSeat"],
        window["observation"]["resolution"],
        phase,
    )
    return loss, gradient, {"compatibleJointClaims": compatible, "structural": False}


def window_nll(model: HierarchicalSoftmax, encoded: Mapping[str, Any]) -> tuple[float, dict[str, Any]] | None:
    """評価専用のNLL。解析勾配を作らない。"""
    window = encoded["window"]
    phase = str(window["phase"])
    if "candidates" in encoded:
        observation = window["observation"]
        if observation["status"] != "exact" or not window["learningMask"].get("kind", False):
            return None
        probability = model.action_probability(encoded["candidates"], observation["action"], phase)
        if probability <= 0:
            raise ValueError("観測行動が合法集合にない")
        return -math.log(probability), {"structural": len(encoded["candidates"]) == 1}
    if not window["learningMask"].get("jointKind", False):
        return None
    if all(len(actions) == 1 for actions in encoded["perSeat"].values()):
        return 0.0, {"compatibleJointClaims": 1, "structural": True}
    likelihood, compatible = joint_resolution_likelihood(
        model,
        int(window["actorSeat"]),
        encoded["perSeat"],
        window["observation"]["resolution"],
        phase,
    )
    return -math.log(likelihood), {"compatibleJointClaims": compatible, "structural": False}


def _dataset_hash(dataset_dir: Path) -> str:
    summary = json.loads((dataset_dir / "extraction-summary.json").read_text(encoding="utf-8"))
    payload = json.dumps(summary.get("generatedFiles", {}), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase_metrics_template() -> dict[str, dict[str, float]]:
    return {phase: {"windows": 0, "nll": 0.0} for phase in PHASES}


def _metric_accumulator() -> dict[str, Any]:
    return {
        "phases": _phase_metrics_template(),
        "kinds": Counter(),
        "windows": 0,
        "nll": 0.0,
        "skipped": Counter(),
        "brier": 0.0,
        "brierWindows": 0,
        "selfNll": 0.0,
        "selfWindows": 0,
        "responseNll": 0.0,
        "responseWindows": 0,
        "predictedSelfKinds": Counter(),
        "observedResponseKinds": Counter(),
        "predictedResponseKinds": Counter(),
        "calibrationCount": np.zeros(10, dtype=int),
        "calibrationConfidence": np.zeros(10, dtype=float),
        "calibrationCorrect": np.zeros(10, dtype=float),
    }


def _add_metric(accumulator: dict[str, Any], model: HierarchicalSoftmax, encoded: Mapping[str, Any]) -> None:
    window = encoded["window"]
    result = window_nll(model, encoded)
    if result is None:
        accumulator["skipped"]["masked"] += 1
        return
    loss, detail = result
    phase = str(window["phase"])
    accumulator["phases"][phase]["windows"] += 1
    accumulator["phases"][phase]["nll"] += loss
    accumulator["windows"] += 1
    accumulator["nll"] += loss
    if "candidates" in encoded:
        probabilities = model.probabilities(encoded["candidates"], phase)
        kind_probabilities = Counter()
        for candidate, probability in zip(encoded["candidates"], probabilities):
            kind_probabilities[candidate.kind] += float(probability)
        observed_kind = str(window["observation"]["action"]["kind"])
        accumulator["kinds"][observed_kind] += 1
        accumulator["selfNll"] += loss
        accumulator["selfWindows"] += 1
        for kind, probability in kind_probabilities.items():
            accumulator["predictedSelfKinds"][kind] += probability
        accumulator["brier"] += sum(
            (probability - float(kind == observed_kind)) ** 2
            for kind, probability in kind_probabilities.items()
        )
        accumulator["brierWindows"] += 1
        predicted_kind, confidence = max(kind_probabilities.items(), key=lambda item: (item[1], item[0]))
        bucket = min(9, int(confidence * 10))
        accumulator["calibrationCount"][bucket] += 1
        accumulator["calibrationConfidence"][bucket] += confidence
        accumulator["calibrationCorrect"][bucket] += float(predicted_kind == observed_kind)
    else:
        accumulator["responseNll"] += loss
        accumulator["responseWindows"] += 1
        resolution_probabilities = _public_resolution_kind_probabilities(model, encoded)
        observed_kind = str(window["observation"]["resolution"]["kind"])
        accumulator["observedResponseKinds"][observed_kind] += 1
        for kind, probability in resolution_probabilities.items():
            accumulator["predictedResponseKinds"][kind] += probability
        if detail.get("structural"):
            accumulator["skipped"]["structural_probability_one"] += 1


def _public_resolution_kind_probabilities(
    model: HierarchicalSoftmax, encoded: Mapping[str, Any]
) -> Counter[str]:
    """三家の希望分布を公開解決の行動種別率へ写像する。"""
    window = encoded["window"]
    phase = str(window["phase"])
    per_seat = encoded["perSeat"]
    probabilities = {seat: model.probabilities(candidates, phase) for seat, candidates in per_seat.items()}
    result: Counter[str] = Counter()
    for indices in enumerate_joint_indices(per_seat):
        choices = {seat: dict(per_seat[seat][index].action) for seat, index in indices.items()}
        resolution = resolve_joint_response(int(window["actorSeat"]), choices)
        mass = math.prod(float(probabilities[seat][index]) for seat, index in indices.items())
        result[str(resolution["kind"])] += mass
    if not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("公開応答結果の予測確率和が1でない")
    return result


def _finish_metric(accumulator: dict[str, Any]) -> dict[str, Any]:
    for value in accumulator["phases"].values():
        value["meanNll"] = value["nll"] / value["windows"] if value["windows"] else None
    calibration = []
    ece = 0.0
    calibration_total = int(np.sum(accumulator["calibrationCount"]))
    for index in range(10):
        count = int(accumulator["calibrationCount"][index])
        mean_confidence = accumulator["calibrationConfidence"][index] / count if count else None
        accuracy = accumulator["calibrationCorrect"][index] / count if count else None
        if count and calibration_total:
            ece += count / calibration_total * abs(float(accuracy) - float(mean_confidence))
        calibration.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "windows": count,
                "meanConfidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    def rates(counter: Mapping[str, float], denominator: int) -> dict[str, float]:
        return {kind: float(counter.get(kind, 0.0)) / denominator for kind in KINDS if counter.get(kind, 0.0)} if denominator else {}

    return {
        "windows": accumulator["windows"],
        "meanNll": accumulator["nll"] / accumulator["windows"] if accumulator["windows"] else None,
        "selfMeanNll": (
            accumulator["selfNll"] / accumulator["selfWindows"] if accumulator["selfWindows"] else None
        ),
        "responseMeanNll": (
            accumulator["responseNll"] / accumulator["responseWindows"]
            if accumulator["responseWindows"]
            else None
        ),
        "selfKindBrier": (
            accumulator["brier"] / accumulator["brierWindows"]
            if accumulator["brierWindows"]
            else None
        ),
        "phases": accumulator["phases"],
        "observedKinds": dict(sorted(accumulator["kinds"].items())),
        "rates": {
            "selfAction": {
                "windows": accumulator["selfWindows"],
                "observed": rates(accumulator["kinds"], accumulator["selfWindows"]),
                "predicted": rates(accumulator["predictedSelfKinds"], accumulator["selfWindows"]),
            },
            "publicResponseResolution": {
                "windows": accumulator["responseWindows"],
                "observed": rates(accumulator["observedResponseKinds"], accumulator["responseWindows"]),
                "predicted": rates(accumulator["predictedResponseKinds"], accumulator["responseWindows"]),
            },
        },
        "selfTopKindCalibration": {"ece": ece if calibration_total else None, "bins": calibration},
        "skipped": dict(sorted(accumulator["skipped"].items())),
    }


def _evaluate_stream(
    model: HierarchicalSoftmax,
    dataset_dir: Path,
    feature_dir: Path,
    splits: set[str],
    maximum_windows: int | None,
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    accumulator = _metric_accumulator()
    for encoded in iter_cached_windows(
        dataset_dir, feature_dir, maximum_windows, splits, verified_manifest=feature_manifest
    ):
        _add_metric(accumulator, model, encoded)
    return _finish_metric(accumulator)


def _evaluate_models_one_split(
    models: Sequence[HierarchicalSoftmax],
    dataset_dir: Path,
    feature_dir: Path,
    split: str,
    maximum_windows: int | None,
    feature_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    accumulators = [_metric_accumulator() for _ in models]
    for encoded in iter_cached_windows(
        dataset_dir, feature_dir, maximum_windows, {split}, verified_manifest=feature_manifest
    ):
        for model, accumulator in zip(models, accumulators):
            _add_metric(accumulator, model, encoded)
    return [_finish_metric(value) for value in accumulators]


def _evaluate_one_model_all_splits(
    model: HierarchicalSoftmax,
    dataset_dir: Path,
    feature_dir: Path,
    maximum_windows: int | None,
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    splits = ("train", "selection", "calibration", "developmentConfirmation")
    accumulators = {split: _metric_accumulator() for split in splits}
    for encoded in iter_cached_windows(
        dataset_dir, feature_dir, maximum_windows, set(splits), verified_manifest=feature_manifest
    ):
        split = str(encoded["window"]["developmentSplit"])
        _add_metric(accumulators[split], model, encoded)
    return {split: _finish_metric(accumulators[split]) for split in splits}


def _support_diagnostics(dataset_dir: Path, maximum_windows: int | None) -> dict[str, Any]:
    """支持件数の集計（設計7.1節、D.3.2b工程4）。

    D.3.2b工程1の和了判定修正後は、和了headのholdは0件になった。
    それでも学習に使わない窓（learningMaskが偽）はここでも「observed」から除く。
    exactラベルの見送りに加え、公開結果の優先順位から論理的に確定するロン見送り
    （`confirmedSkips`）を別欄で数える。ロンは最優先（priority 0）なので、
    公開結果が別の種別なら、その種別より優先度で劣後する候補のうちロンが合法な
    家は全員ロンを選ばなかったと確定できる（頭ハネされた同順位の希望は確定しない）。
    """

    opportunities = Counter()
    observed = Counter()
    legal_win_skips = Counter()
    confirmed_ron_skips = Counter()
    held_or_censored = Counter()
    response_windows = Counter()
    informative_response_windows = Counter()
    split_processed = Counter()
    for row in _iter_jsonl_gzip(dataset_dir / "teacher-windows.jsonl.gz"):
        split = str(row["developmentSplit"])
        if maximum_windows is not None and split_processed[split] >= maximum_windows:
            continue
        split_processed[split] += 1
        if "legalActions" in row:
            legal_kinds = {str(action["kind"]) for action in row["legalActions"]}
            for kind in legal_kinds:
                opportunities[(split, kind)] += 1
            learnable = bool(row.get("learningMask", {}).get("kind", False))
            action = row["observation"].get("action")
            if row["observation"]["status"] == "exact" and action and learnable:
                observed[(split, str(action["kind"]))] += 1
                if "tsumo" in legal_kinds and action["kind"] != "tsumo":
                    legal_win_skips[(split, "tsumo")] += 1
            else:
                held_or_censored[(split, "heldSelfWindows")] += 1
        else:
            phase = str(row["phase"])
            response_windows[(split, phase)] += 1
            if any(len(value["actions"]) > 1 for value in row["legalBySeat"].values()):
                informative_response_windows[(split, phase)] += 1
            learnable = bool(row.get("learningMask", {}).get("jointKind", False))
            resolution = row["observation"]["resolution"]
            resolution_priority = response_priority(str(resolution["kind"]))
            for seat, value in row["legalBySeat"].items():
                legal_kinds = {str(action["kind"]) for action in value["actions"]}
                for kind in legal_kinds:
                    opportunities[(split, kind)] += 1
                label = next(item for item in row["observation"]["perSeatLabels"] if str(item["seat"]) == seat)
                if label["status"] == "exact" and learnable:
                    observed[(split, str(label["action"]["kind"]))] += 1
                    if "ron" in legal_kinds and label["action"]["kind"] != "ron":
                        legal_win_skips[(split, "ron")] += 1
                elif "ron" in legal_kinds and resolution_priority > response_priority("ron"):
                    # 公開結果の優先順位がロンより低い（数が大きい）ため、その家がロンを
                    # 選んでいれば必ずロンが公開結果になっていたはず。実際はそうならなかった
                    # ので、exactラベルの有無に関わらずロン見送りが確定する（R3の反例）。
                    confirmed_ron_skips[split] += 1
                    held_or_censored[(split, "censoredResponseLabels")] += 1
                else:
                    held_or_censored[(split, "censoredResponseLabels")] += 1
    values = {}
    for split in ("train", "selection", "calibration", "developmentConfirmation"):
        values[split] = {
            "opportunities": {kind: opportunities[(split, kind)] for kind in KINDS},
            "observed": {kind: observed[(split, kind)] for kind in KINDS},
            "legalWinSkips": {kind: legal_win_skips[(split, kind)] for kind in ("tsumo", "ron")},
            "confirmedRonSkips": confirmed_ron_skips[split],
            "heldOrCensored": {
                kind: held_or_censored[(split, kind)]
                for kind in ("heldSelfWindows", "censoredResponseLabels")
            },
            "responseWindows": {
                phase: response_windows[(split, phase)]
                for phase in ("discard_response", "chankan_response")
            },
            "informativeResponseWindows": {
                phase: informative_response_windows[(split, phase)]
                for phase in ("discard_response", "chankan_response")
            },
        }
        for kind in KINDS:
            if values[split]["observed"][kind] > values[split]["opportunities"][kind]:
                raise ValueError(
                    f"観測が合法機会を上回る: split={split}, kind={kind}, "
                    f"observed={values[split]['observed'][kind]}, opportunities={values[split]['opportunities'][kind]}"
                )
    total_ron_skips = sum(legal_win_skips[(split, "ron")] for split in values)
    total_confirmed_ron_skips = sum(confirmed_ron_skips[split] for split in values)
    total_tsumo_skips = sum(legal_win_skips[(split, "tsumo")] for split in values)
    unidentified = []
    ron_evidence = total_ron_skips + total_confirmed_ron_skips
    if ron_evidence < 30:
        unidentified.append({
            "component": "ron_pass",
            "status": "opponent_component_unidentified",
            "observedSkips": total_ron_skips,
            "confirmedSkips": total_confirmed_ron_skips,
            "reason": "observed_skip_below_30",
        })
    if total_tsumo_skips < 30:
        unidentified.append({
            "component": "tsumo_pass",
            "status": "opponent_component_unidentified",
            "observedSkips": total_tsumo_skips,
            "reason": "observed_skip_below_30",
        })
    chankan_informative = sum(
        informative_response_windows[(split, "chankan_response")] for split in values
    )
    if chankan_informative == 0:
        unidentified.append(
            {
                "component": "chankan_response_policy",
                "status": "opponent_component_unidentified",
                "informativeWindows": 0,
                "reason": "all_legal_sets_are_structural_pass_only",
            }
        )
    total_daiminkan = sum(observed[(split, "daiminkan")] for split in values)
    if total_daiminkan < 100:
        unidentified.append(
            {
                "component": "daiminkan_policy",
                "status": "opponent_component_unidentified",
                "observedActions": total_daiminkan,
                "reason": "observed_action_below_100_and_rate_unstable",
            }
        )
    return {
        "bySplit": values,
        "unidentifiedComponents": unidentified,
        "thresholdRole": "diagnostic_only_D4_must_freeze_adoption_thresholds",
    }


def response_rate_diagnostic(metrics: Mapping[str, Any], period: str) -> dict[str, Any]:
    """指定期間の公開応答結果の率ずれを採用holdとして再現可能に判定する（設計8.3節）。

    固定成分（ロン・大明槓の定数）を含む合成方策の確率（model.probabilitiesの出力）を
    _add_metricが使うため、この診断は固定成分を除外しない。設計9節の要件を、較正期間と
    開発確認期間の両方で呼び出すことで満たす（opponent_adoption_holdsを参照）。
    """
    rates = metrics[period]["rates"]["publicResponseResolution"]
    observed = rates["observed"]
    predicted = rates["predicted"]
    differences = {
        kind: {
            "observed": float(observed.get(kind, 0.0)),
            "predicted": float(predicted.get(kind, 0.0)),
            "absoluteError": abs(float(predicted.get(kind, 0.0)) - float(observed.get(kind, 0.0))),
        }
        for kind in ("pass", "chi", "pon", "daiminkan", "ron")
    }
    reasons = []
    if max(value["absoluteError"] for value in differences.values()) > 0.005:
        reasons.append("maximum_absolute_rate_error_above_0.005")
    for kind, value in differences.items():
        if 0 < value["observed"] <= 0.001 and value["predicted"] / value["observed"] > 5:
            reasons.append(f"rare_rate_ratio_above_5:{kind}")
    return {
        "status": "hold" if reasons else "pass",
        "period": period,
        "developmentConfirmationReuse": "reused_non_independent" if period == "developmentConfirmation" else None,
        "diagnosticThresholdsNotAdoptionThresholds": {
            "maximumAbsoluteRateError": 0.005,
            "maximumRareRateRatio": 5.0,
            "rareObservedRateMaximum": 0.001,
        },
        "differences": differences,
        "reasons": reasons,
    }


WIN_LEGALITY_DECLARED_REASONS = frozenset({
    "observed_skip_below_30", "observed_action_below_100_and_rate_unstable", "all_legal_sets_are_structural_pass_only",
})
FIXED_COMPONENT_BY_UNIDENTIFIED = {"ron_pass": "epsRon", "tsumo_pass": "epsTsumo",
                                    "daiminkan_policy": "rho", "chankan_response_policy": "epsChankan"}


def win_legality_satisfied(win_legality: Mapping[str, Any]) -> bool:
    """4.4節の解除条件。未分類0件、全教師窓の和了照合合格、残存例外の検出体制の3点を検査する。"""

    if int(win_legality.get("unclassifiedCount", 1)) != 0:
        return False
    if not bool(win_legality.get("allObservedWinsInLegalSet", False)):
        return False
    for item in win_legality.get("residualExceptions", []):
        if not (item.get("hasDetector") and item.get("hasThreeWayHandling")):
            return False
    return True


def load_win_legality_report(probe_dir: Path) -> dict[str, Any]:
    """工程1の分類記録（calibration/probes/win-legality-d32b）からwin_legality入力を作る。"""

    summary = json.loads((probe_dir / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((probe_dir / "verification.json").read_text(encoding="utf-8"))
    residual = [
        {"cause": entry["cause"], "hasDetector": False, "hasThreeWayHandling": False}
        for entry in summary.get("byCauseAndExpected", [])
        if entry["cause"] not in {"chi_meld_order_adapter", "truncated_prefix_terminal_label"}
    ]
    return {
        "unclassifiedCount": int(summary["unclassified"]),
        "allObservedWinsInLegalSet": verification.get("status") == "pass",
        "residualExceptions": residual,
        "evidence": {"summaryPath": "summary.json", "verificationPath": "verification.json"},
    }


def fixed_components_declared(support: Mapping[str, Any], fixed_components: Mapping[str, Any] | None) -> bool:
    """未識別成分すべてに、対応する固定成分（定数・推定方法・事後区間・シナリオ一覧）の宣言があるかを検査する。"""

    unidentified = support.get("unidentifiedComponents", [])
    if not unidentified:
        return True
    if fixed_components is None:
        return False
    constants = fixed_components.get("roundTrips", {}).get("final", {}).get("constants")
    posteriors = fixed_components.get("roundTrips", {}).get("final", {}).get("posteriors", {})
    scenarios = fixed_components.get("scenarios", [])
    if not constants or not scenarios:
        return False
    for item in unidentified:
        name = FIXED_COMPONENT_BY_UNIDENTIFIED.get(str(item.get("component")))
        if name is None or name not in constants:
            return False
        # epsChankanはモデル仮定であり事後分布を推定しない（設計7.3節）。他3件は事後分布が必要。
        if name != "epsChankan" and name not in posteriors:
            return False
    return True


def opponent_adoption_holds(
    win_legality: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    support: Mapping[str, Any],
    rate_diagnostics: Mapping[str, Mapping[str, Any]],
    fixed_components: Mapping[str, Any] | None,
    *,
    rate_correction_degraded: bool = False,
) -> dict[str, Any]:
    """fitと再読込evaluateが同じ根拠から同じhold集合・条件を作る（設計9節）。"""

    holds: list[str] = []
    if not win_legality_satisfied(win_legality):
        holds.append("win_legality_unresolved")
    implemented = set(feature_manifest.get("implementedGroups", []))
    if feature_manifest.get("status") != "complete":
        holds.append("exact_candidate_ukeire_not_full_verified")
    if DANGER_FEATURE_GROUP not in implemented:
        holds.append("full_multi_riichi_danger_class")
    if YAKU_SHAPE_FEATURE_GROUP not in implemented:
        holds.append("explicit_yaku_shape_features")
    if not fixed_components_declared(support, fixed_components):
        holds.append("opponent_component_unidentified")
    rate_hold = any(diagnostic.get("status") == "hold" for diagnostic in rate_diagnostics.values())
    if rate_hold or rate_correction_degraded:
        holds.append("response_rate_miscalibration")

    d33_conditions: list[str] = []
    if fixed_components is not None:
        d33_conditions.append("fixed_component_sensitivity")
    if win_legality.get("residualExceptions"):
        d33_conditions.append("residual_win_legality_detector")

    return {
        "holds": holds,
        "eligibleForD33": not holds,
        "eligibleForAdoption": False,
        "d33Conditions": d33_conditions,
    }


def _select_phase_temperatures(
    model: HierarchicalSoftmax,
    dataset_dir: Path,
    feature_dir: Path,
    grid: Sequence[float],
    maximum_windows: int | None,
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """較正期間を一度だけ復元し、phase別の温度を独立に選ぶ。"""
    totals = {phase: {float(value): [0, 0.0] for value in grid} for phase in PHASES}
    trials = {float(value): model.copy() for value in grid}
    for temperature, trial in trials.items():
        trial.temperatures = {phase: temperature for phase in PHASES}
    for encoded in iter_cached_windows(
        dataset_dir, feature_dir, maximum_windows, {"calibration"}, verified_manifest=feature_manifest
    ):
        phase = str(encoded["window"]["phase"])
        for temperature, trial in trials.items():
            result = window_nll(trial, encoded)
            if result is None:
                continue
            loss, _ = result
            totals[phase][temperature][0] += 1
            totals[phase][temperature][1] += loss
    selection = {}
    for phase in PHASES:
        candidates = [
            {
                "temperature": temperature,
                "windows": totals[phase][temperature][0],
                "meanNll": (
                    totals[phase][temperature][1] / totals[phase][temperature][0]
                    if totals[phase][temperature][0]
                    else None
                ),
            }
            for temperature in map(float, grid)
        ]
        available = [item for item in candidates if item["meanNll"] is not None]
        chosen = (
            min(available, key=lambda item: (item["meanNll"], abs(item["temperature"] - 1.0)))
            if available
            else {"temperature": 1.0, "windows": 0, "meanNll": None}
        )
        model.temperatures[phase] = float(chosen["temperature"])
        selection[phase] = {"selected": chosen, "candidates": candidates}
    return selection


def fit_opponent_model(
    dataset_dir: Path,
    feature_dir: Path,
    output_dir: Path,
    win_legality_dir: Path,
    *,
    maximum_windows: int | None = None,
) -> dict[str, Any]:
    """固定manifestどおりに正則化選択、温度較正、確認評価を行う。"""
    _validate_feature_dataset(dataset_dir)
    win_legality = load_win_legality_report(win_legality_dir)
    feature_manifest = verify_opponent_feature_cache(dataset_dir, feature_dir)
    debug = maximum_windows is not None
    if feature_manifest.get("status") == "debug" and maximum_windows is None:
        raise ValueError("debug特徴cacheを全件fitへ使用できない")
    manifest = {
        "schemaVersion": "ev-policy-opponent-fit-manifest/v3",
        "seed": 20260909,
        "optimizer": "deterministic_minibatch_sgd",
        "batchSize": 256,
        "epochs": 1 if debug else 2,
        "learningRate": 0.03,
        "regularizationGrid": [0.01, 0.1, 1.0, 10.0, 100.0],
        "temperatureGrid": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "trainSplit": "train",
        "selectionSplit": "selection",
        "calibrationSplit": "calibration",
        "confirmationSplit": "developmentConfirmation",
        "maximumWindows": maximum_windows,
        "datasetManifestHash": _dataset_hash(dataset_dir),
        "featureCacheManifestHash": _json_hash(feature_manifest),
        "featureSchemaVersion": FEATURE_SCHEMA,
        "exactCalculationVersion": CALCULATION_VERSION,
        "labelDefinitionVersion": LABEL_DEFINITION_VERSION,
        # D.3.2b 固定成分（設計7.3節）。往復回数と格子の設定を学習前に固定する。
        "fixedComponents": {
            "estimationSplits": sorted(ESTIMATION_SPLITS),
            "thetaConstantRoundTrips": 1,
            "gridStartCells": 400,
            "gridMaxCells": 3200,
            "gridRelativeTolerance": 0.01,
            "prior": "jeffreys_beta_0.5_0.5",
            "chankanRule": CHANKAN_ASSUMPTION,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "fit-manifest.json", manifest)
    print(json.dumps({"progress": "opponent_fit_manifest_frozen", "epochs": manifest["epochs"]}), flush=True)

    fixed_options = manifest["fixedComponents"]
    grid_options = {
        "start_cells": int(fixed_options["gridStartCells"]),
        "max_cells": int(fixed_options["gridMaxCells"]),
        "tolerance": float(fixed_options["gridRelativeTolerance"]),
    }

    def estimation_inputs(current: HierarchicalSoftmax):
        windows = iter_cached_windows(
            dataset_dir, feature_dir, maximum_windows, set(ESTIMATION_SPLITS), verified_manifest=feature_manifest
        )
        return collect_estimation_inputs(current, windows, _stratum_attributes_of)

    # 1. 初期定数：exactに分かる件数比（Jeffreys補正）。θに依存しない。
    initial = initial_constants(estimation_inputs(HierarchicalSoftmax.zeros()))
    print(json.dumps({"progress": "opponent_initial_fixed_constants", **initial.to_dict()}), flush=True)

    # 2. 初期定数の下でθを学習し、正則化を選び、温度を較正する。
    lambdas = manifest["regularizationGrid"]
    models, train_counts = _train_models(
        dataset_dir, feature_dir, feature_manifest, manifest, lambdas, initial, maximum_windows
    )
    selection_metrics = _evaluate_models_one_split(
        models, dataset_dir, feature_dir, "selection", maximum_windows, feature_manifest
    )
    print(json.dumps({"progress": "opponent_regularization_selection_complete"}), flush=True)
    selection = [
        {"lambda": regularization, "metrics": metrics}
        for regularization, metrics in zip(lambdas, selection_metrics)
    ]
    eligible = [item for item in selection if item["metrics"]["meanNll"] is not None]
    if not eligible:
        raise ValueError("selection期間の評価窓がない")
    selected = min(eligible, key=lambda item: (item["metrics"]["meanNll"], item["lambda"]))
    first_model = models[lambdas.index(selected["lambda"])]
    first_temperatures = _select_phase_temperatures(
        first_model, dataset_dir, feature_dir, manifest["temperatureGrid"], maximum_windows, feature_manifest
    )

    # 3. 学習したθを固定して、定数の事後分布を求める。
    first_posteriors = estimate_base_posteriors(estimation_inputs(first_model), initial, **grid_options)
    first_constants = constants_from_posteriors(first_posteriors)
    print(json.dumps({"progress": "opponent_fixed_constants_round1", **first_constants.to_dict()}), flush=True)

    # 4. 推定した定数の下でθを学習し直す（選んだ正則化だけ）。温度も較正し直す。
    refit_models, refit_counts = _train_models(
        dataset_dir, feature_dir, feature_manifest, manifest, [selected["lambda"]], first_constants, maximum_windows
    )
    model = refit_models[0]
    temperature_selection = _select_phase_temperatures(
        model, dataset_dir, feature_dir, manifest["temperatureGrid"], maximum_windows, feature_manifest
    )
    print(json.dumps({"progress": "opponent_temperature_calibration_complete"}), flush=True)

    # 5. 再学習したθで定数を推定し直し、最終の定数・層別結果・シナリオ一覧を固定する。
    final_inputs = estimation_inputs(model)
    final_posteriors = estimate_base_posteriors(final_inputs, first_constants, **grid_options)
    final_constants = constants_from_posteriors(final_posteriors)
    model.fixed = final_constants
    strata = estimate_strata(final_inputs, final_constants, final_posteriors, **grid_options)
    theta_id = _json_hash(
        {"kind": model.kind_weights.tolist(), "detail": model.detail_weights.tolist(), "temperatures": model.temperatures}
    )
    scenarios = build_scenarios(final_posteriors, strata, theta_id=theta_id)
    fixed_components = {
        "schemaVersion": "ev-policy-opponent-fixed-components/v1",
        "modelVersion": MODEL_VERSION,
        "estimationSplits": sorted(ESTIMATION_SPLITS),
        "roundTrips": {
            "initial": initial.to_dict(),
            "afterFirstFit": {
                "constants": first_constants.to_dict(),
                "posteriors": {key: value.to_dict() for key, value in first_posteriors.items()},
                "temperatures": first_temperatures,
            },
            "final": {
                "constants": final_constants.to_dict(),
                "posteriors": {key: value.to_dict() for key, value in final_posteriors.items()},
            },
        },
        "excludedWindows": final_inputs.excluded,
        "observations": {"tsumoWindows": len(final_inputs.tsumo), "responseWindows": len(final_inputs.response)},
        "assumptions": [CHANKAN_ASSUMPTION, "constants_conditioned_only_on_legality"],
        "strata": strata,
        "thetaId": theta_id,
        "scenarios": scenarios,
        "scenarioCounts": scenario_count_summary(scenarios),
    }
    _write_json(output_dir / "fixed-components.json", fixed_components)
    print(json.dumps({"progress": "opponent_fixed_constants_final", **final_constants.to_dict()}), flush=True)

    metrics = _evaluate_one_model_all_splits(
        model, dataset_dir, feature_dir, maximum_windows, feature_manifest
    )
    print(json.dumps({"progress": "opponent_period_evaluation_complete"}), flush=True)
    support = _support_diagnostics(dataset_dir, maximum_windows)
    rate_diagnostics = {
        period: response_rate_diagnostic(metrics, period) for period in ("calibration", "developmentConfirmation")
    }
    adoption = opponent_adoption_holds(win_legality, feature_manifest, support, rate_diagnostics, fixed_components)
    holds = adoption["holds"]
    print(json.dumps({"progress": "opponent_identifiability_audit_complete"}), flush=True)
    train_counts["refitVisited"] = refit_counts.get("visited", 0)
    model_payload = model.to_dict()
    model_payload.update(
        {
            "labelDefinitionVersion": LABEL_DEFINITION_VERSION,
            "selectedLambda": selected["lambda"],
            "datasetManifestHash": manifest["datasetManifestHash"],
            "featureCacheManifestHash": manifest["featureCacheManifestHash"],
            "trainingSplits": ["train", "selection", "calibration"],
            "fixedComponentsPath": "fixed-components.json",
            "featureCoverage": {
                "implemented": list(feature_manifest.get("implementedGroups", [])),
                "held": [value for value in holds if value in {
                    "exact_candidate_ukeire_not_full_verified",
                    "full_multi_riichi_danger_class",
                    "explicit_yaku_shape_features",
                }],
                "reason": "D.3.2b_feature_gate_and_remaining_D.3_holds",
            },
        }
    )
    _write_json(output_dir / "model.json", model_payload)
    summary = {
        "schemaVersion": FIT_SCHEMA,
        "status": "debug_complete" if debug else "complete_with_fixed_components",
        "eligibleForD33": adoption["eligibleForD33"],
        "eligibleForAdoption": adoption["eligibleForAdoption"],
        "d33Conditions": adoption["d33Conditions"],
        "modelPath": "model.json",
        "manifestPath": "fit-manifest.json",
        "fixedComponentsPath": "fixed-components.json",
        "selection": selection,
        "selectedLambda": selected["lambda"],
        "temperatureSelection": temperature_selection,
        "fixedConstants": final_constants.to_dict(),
        "metrics": metrics,
        "support": support,
        "training": dict(train_counts),
        "winLegality": win_legality,
        "rateDiagnostics": rate_diagnostics,
        "holds": holds,
    }
    _write_json(output_dir / "fit-summary.json", summary)
    return summary


def _stratum_attributes_of(candidate: EncodedCandidate) -> dict[str, str]:
    """種別特徴から、定数を使う家の層を決める（設計7.4節）。巡目は捨牌総数から近似する。"""
    values = dict(zip(KIND_FEATURE_NAMES, candidate.kind_features))
    discards = round(float(values["turn"]) * 72.0)
    return stratum_attributes(
        dealer=bool(values["dealer"] >= 0.5),
        own_riichi=bool(values["own_riichi"] >= 0.5),
        gap_to_top=-float(values["leader_delta"]) * 10_000.0,
        junme=discards // 4 + 1,
    )


def _train_models(
    dataset_dir: Path,
    feature_dir: Path,
    feature_manifest: Mapping[str, Any],
    manifest: Mapping[str, Any],
    lambdas: Sequence[float],
    fixed: FixedConstants,
    maximum_windows: int | None,
) -> tuple[list[HierarchicalSoftmax], Counter]:
    """固定定数を合成した方策で、正則化ごとのモデルをdeterministic minibatch SGDで学習する。"""
    models = [HierarchicalSoftmax.zeros(fixed) for _ in lambdas]
    batch_gradients = [ModelGradient.zeros(model) for model in models]
    batch_counts = [0 for _ in models]
    train_counts = Counter()
    for epoch in range(int(manifest["epochs"])):
        for encoded in iter_cached_windows(
            dataset_dir, feature_dir, maximum_windows, {"train"}, verified_manifest=feature_manifest
        ):
            window = encoded["window"]
            if window["developmentSplit"] != "train":
                continue
            for model_index, model in enumerate(models):
                result = window_loss_and_gradient(model, encoded)
                if result is None:
                    continue
                _, gradient, detail = result
                if detail.get("structural"):
                    train_counts["structural"] += int(model_index == 0)
                    continue
                batch_gradients[model_index].add_scaled(gradient, 1.0)
                batch_counts[model_index] += 1
                if batch_counts[model_index] >= int(manifest["batchSize"]):
                    _sgd_update(model, batch_gradients[model_index], batch_counts[model_index], float(lambdas[model_index]), float(manifest["learningRate"]))
                    batch_gradients[model_index] = ModelGradient.zeros(model)
                    batch_counts[model_index] = 0
            train_counts["visited"] += 1
        train_counts["epochs"] += 1
        print(json.dumps({"progress": "opponent_train_epoch_complete", "epoch": epoch + 1}), flush=True)
    for model_index, model in enumerate(models):
        if batch_counts[model_index]:
            _sgd_update(model, batch_gradients[model_index], batch_counts[model_index], float(lambdas[model_index]), float(manifest["learningRate"]))
    return models, train_counts


def _sgd_update(
    model: HierarchicalSoftmax,
    gradient: ModelGradient,
    count: int,
    regularization: float,
    learning_rate: float,
) -> None:
    scale = 1.0 / count
    model.kind_weights -= learning_rate * (gradient.kind * scale + regularization * model.kind_weights / count)
    model.detail_weights -= learning_rate * (gradient.detail * scale + regularization * model.detail_weights / count)


def evaluate_opponent_model(
    dataset_dir: Path,
    feature_dir: Path,
    model_dir: Path,
    win_legality_dir: Path,
    *,
    maximum_windows: int | None = None,
) -> dict[str, Any]:
    feature_manifest = verify_opponent_feature_cache(dataset_dir, feature_dir)
    if feature_manifest.get("status") == "debug" and maximum_windows is None:
        raise ValueError("debug特徴cacheを全件evaluateへ使用できない")
    win_legality = load_win_legality_report(win_legality_dir)
    payload = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if payload.get("datasetManifestHash") != _dataset_hash(dataset_dir):
        raise ValueError("モデルとD.3.1データのmanifest hashが不一致")
    if payload.get("featureCacheManifestHash") != _json_hash(feature_manifest):
        raise ValueError("モデルと特徴cacheのmanifest hashが不一致")
    fixed_components = None
    fixed_components_path = payload.get("fixedComponentsPath")
    if fixed_components_path:
        fixed_components = json.loads((model_dir / fixed_components_path).read_text(encoding="utf-8"))
    model = HierarchicalSoftmax.from_dict(payload)
    metrics = _evaluate_one_model_all_splits(
        model, dataset_dir, feature_dir, maximum_windows, feature_manifest
    )
    support = _support_diagnostics(dataset_dir, maximum_windows)
    rate_diagnostics = {
        period: response_rate_diagnostic(metrics, period) for period in ("calibration", "developmentConfirmation")
    }
    adoption = opponent_adoption_holds(win_legality, feature_manifest, support, rate_diagnostics, fixed_components)
    holds = adoption["holds"]
    report = {
        "schemaVersion": EVALUATION_SCHEMA,
        "status": "debug_complete" if maximum_windows is not None else "confirmation_only_not_independent_test",
        "finalTest": False,
        "developmentConfirmationIsIndependent": False,
        "metrics": metrics,
        "support": support,
        "winLegality": win_legality,
        "rateDiagnostics": rate_diagnostics,
        "holds": holds,
        "eligibleForD33": adoption["eligibleForD33"],
        "eligibleForAdoption": adoption["eligibleForAdoption"],
        "d33Conditions": adoption["d33Conditions"],
    }
    _write_json(model_dir / "evaluation.json", report)
    return report
