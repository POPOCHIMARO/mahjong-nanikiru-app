#!/usr/bin/env python3
"""D.3.2b: 未識別成分の固定定数、事後分布、層別推定、感度シナリオ（設計7節）。

相手モデルの学習では識別できないロン見送り、ツモ見送り、大明槓、搶槓応答を、
合法性だけに条件付けた定数として置く。これは実戦の事実ではなくモデル仮定である。

このモジュールは相手モデル（ev_policy_opponent）をimportしない。
モデルは引数で受け取り、`base_probabilities`（固定成分を除いた学習分布）だけを呼ぶ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    from .ev_policy_observation import resolve_joint_response
except ImportError:
    from ev_policy_observation import resolve_joint_response


CONSTANT_NAMES = ("epsRon", "epsTsumo", "rho", "epsChankan")
ESTIMATED_CONSTANTS = ("epsRon", "epsTsumo", "rho")
ESTIMATION_SPLITS = frozenset({"train", "selection", "calibration"})
SELF_PHASES = frozenset({"self_action_after_live", "self_action_after_call", "self_action_after_rinshan"})
CHANKAN_ASSUMPTION = "chankan_uses_discard_ron_constant"
CHANKAN_STRESS_VALUE = 0.5


# ---------------------------------------------------------------------------
# 定数と確率の合成（設計7.2節）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedConstants:
    """各家が合法時に固定成分を選ぶ確率を決める定数。

    eps_ron: ロン合法時の見送り率。p_ron = 1 - eps_ron。
    eps_tsumo: ツモ合法時の見送り率。
    rho: ロンしなかった条件の下での大明槓選択率。p_dmk = (1 - p_ron) * rho。
    eps_chankan: 搶槓ロン合法時の見送り率。観測がないため推定しない。
    """

    eps_ron: float
    eps_tsumo: float
    rho: float
    eps_chankan: float

    def validate(self, *, open_interval: bool) -> "FixedConstants":
        """open_interval=Trueは粒子重みに使う場合。0や1は真の世界を重み0にするため拒否する。"""

        for name, value in self.to_dict().items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"固定定数が有限の数でない: {name}={value!r}")
            if open_interval and not 0.0 < value < 1.0:
                raise ValueError(f"固定定数は0と1を含まない区間が必要: {name}={value}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"固定定数は0以上1以下が必要: {name}={value}")
        return self

    def to_dict(self) -> dict[str, float]:
        return {
            "epsRon": float(self.eps_ron),
            "epsTsumo": float(self.eps_tsumo),
            "rho": float(self.rho),
            "epsChankan": float(self.eps_chankan),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FixedConstants":
        missing = [name for name in CONSTANT_NAMES if name not in value]
        if missing:
            raise ValueError(f"固定定数が不足: {missing}")
        return cls(
            float(value["epsRon"]), float(value["epsTsumo"]), float(value["rho"]), float(value["epsChankan"])
        ).validate(open_interval=False)


def fixed_kind_masses(kinds: Sequence[str], phase: str, constants: FixedConstants) -> dict[str, float]:
    """合法候補の種別列から、固定成分の種別ごとの確率質量を返す（逐次の合成規則）。"""

    present = set(kinds)
    masses: dict[str, float] = {}
    if phase in SELF_PHASES:
        if "tsumo" in present:
            masses["tsumo"] = 1.0 - constants.eps_tsumo
    elif phase == "discard_response":
        p_ron = 1.0 - constants.eps_ron if "ron" in present else 0.0
        if "ron" in present:
            masses["ron"] = p_ron
        if "daiminkan" in present:
            masses["daiminkan"] = (1.0 - p_ron) * constants.rho
    elif phase == "chankan_response":
        if "ron" in present:
            masses["ron"] = 1.0 - constants.eps_chankan
    else:
        raise ValueError(f"未知のphase: {phase}")
    return masses


def compose_probabilities(
    kinds: Sequence[str],
    phase: str,
    constants: FixedConstants,
    residual_probabilities: Callable[[list[int]], np.ndarray],
) -> np.ndarray:
    """固定成分へ定数の質量を与え、残りを残余候補上の学習分布で配る。

    residual_probabilities(indices) は、指定した候補だけを合法集合とみなした学習分布を返す。
    固定種別に候補が複数ある場合（赤の消費の違い）は、その種別の質量を等分する。
    """

    if not kinds:
        raise ValueError("合法手集合が空")
    masses = fixed_kind_masses(kinds, phase, constants)
    result = np.zeros(len(kinds), dtype=float)
    residual = [index for index, kind in enumerate(kinds) if kind not in masses]
    if not residual:
        if len(kinds) == 1:
            result[0] = 1.0
            return result
        raise ValueError(f"残余候補のない合法集合はルール上生成されない: {sorted(set(kinds))}")
    for kind, mass in masses.items():
        members = [index for index, value in enumerate(kinds) if value == kind]
        for index in members:
            result[index] = mass / len(members)
    remaining = 1.0 - sum(masses.values())
    result[residual] = remaining * np.asarray(residual_probabilities(residual), dtype=float)
    return result


# ---------------------------------------------------------------------------
# Beta分布（ツモ見送り率の厳密な事後分布）
# ---------------------------------------------------------------------------


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """正則化不完全ベータ関数の連分数部（Lentz法）。"""

    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1.0e-15:
            return h
    raise ArithmeticError("不完全ベータ関数の連分数が収束しない")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not (a > 0 and b > 0):
        raise ValueError("ベータ分布の母数は正が必要")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def beta_quantile(a: float, b: float, probability: float) -> float:
    """二分法による分位点。0と1の端点は返さない。"""

    if not 0.0 < probability < 1.0:
        raise ValueError("分位点の確率は0と1の間が必要")
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if regularized_incomplete_beta(a, b, middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True)
class Posterior:
    """一つの定数の事後分布の要約。"""

    mean: float
    lower: float  # 2.5%点
    upper: float  # 97.5%点
    method: str
    opportunities: int
    detail: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
            "opportunities": self.opportunities,
            **dict(self.detail),
        }


def jeffreys_beta_posterior(skips: int, opportunities: int) -> Posterior:
    if skips < 0 or opportunities < skips:
        raise ValueError(f"件数が不正: skips={skips}, opportunities={opportunities}")
    a, b = skips + 0.5, opportunities - skips + 0.5
    return Posterior(
        mean=a / (a + b),
        lower=beta_quantile(a, b, 0.025),
        upper=beta_quantile(a, b, 0.975),
        method="jeffreys_beta_exact",
        opportunities=opportunities,
        detail={"skips": skips, "alpha": a, "beta": b},
    )


# ---------------------------------------------------------------------------
# 変数変換p = sin²(u)による中点則の格子（設計7.3節）
# ---------------------------------------------------------------------------


def u_grid(cells: int) -> tuple[np.ndarray, np.ndarray]:
    """(0, π/2)をcells等分した中点uと、対応するp=sin²(u)。端点p=0と1は評価しない。

    Jeffreys事前分布Beta(0.5, 0.5)はuの上で一様になるため、事後密度はuの上で尤度に比例する。
    """

    if cells < 2:
        raise ValueError("格子の区間数は2以上が必要")
    width = (math.pi / 2.0) / cells
    u = (np.arange(cells, dtype=float) + 0.5) * width
    return u, np.sin(u) ** 2


def summarize_u_posterior(log_weights: np.ndarray, cells: int) -> tuple[float, float, float]:
    """uの中点格子上の対数重みから、pの事後平均と2.5%点・97.5%点を返す。"""

    u, p = u_grid(cells)
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    weights /= float(np.sum(weights))
    mean = float(np.dot(weights, p))
    width = (math.pi / 2.0) / cells
    edges = np.concatenate([[0.0], np.cumsum(weights)])

    def quantile(probability: float) -> float:
        index = int(np.searchsorted(edges, probability, side="left")) - 1
        index = min(max(index, 0), cells - 1)
        inside = (probability - edges[index]) / max(weights[index], 1.0e-300)
        value_u = (index + min(max(inside, 0.0), 1.0)) * width
        return float(math.sin(value_u) ** 2)

    return mean, quantile(0.025), quantile(0.975)


# ---------------------------------------------------------------------------
# 応答窓の尤度を(ε_ron, ρ)の多項式として表す
# ---------------------------------------------------------------------------

# 係数配列 c[i, j] は ε^i ρ^j の係数。三家までなので各次数は3以下。
_DEGREE = 4


def _seat_polynomials(kinds: Sequence[str], residual: np.ndarray, eps_var: bool, rho_var: bool,
                      constants: FixedConstants) -> list[np.ndarray]:
    """一家の各候補の確率を(ε, ρ)の多項式で表す。

    eps_var/rho_var がFalseの家は、その定数を constants の値で固定した数として扱う（層別推定用）。
    residual は残余候補（ron、daiminkan以外）上の学習分布で、候補順に並べた全長配列。
    """

    has_ron = "ron" in kinds
    dmk_members = [index for index, kind in enumerate(kinds) if kind == "daiminkan"]
    polys = [np.zeros((_DEGREE, _DEGREE), dtype=float) for _ in kinds]

    def eps_poly(coefficient_one: float, coefficient_eps: float) -> np.ndarray:
        # coefficient_one + coefficient_eps * ε
        poly = np.zeros((_DEGREE, _DEGREE), dtype=float)
        if eps_var:
            poly[0, 0] += coefficient_one
            poly[1, 0] += coefficient_eps
        else:
            poly[0, 0] += coefficient_one + coefficient_eps * constants.eps_ron
        return poly

    def times_rho(poly: np.ndarray, rho_part: bool) -> np.ndarray:
        # rho_part=True なら ρ を、False なら (1-ρ) を掛ける
        result = np.zeros_like(poly)
        if rho_var:
            if rho_part:
                result[:, 1:] += poly[:, :-1]
            else:
                result += poly
                result[:, 1:] -= poly[:, :-1]
        else:
            result += poly * (constants.rho if rho_part else 1.0 - constants.rho)
        return result

    non_ron = eps_poly(0.0, 1.0) if has_ron else eps_poly(1.0, 0.0)  # 1 - p_ron
    for index, kind in enumerate(kinds):
        if kind == "ron":
            polys[index] = eps_poly(1.0, -1.0)
        elif kind == "daiminkan":
            polys[index] = times_rho(non_ron, True) / len(dmk_members)
        else:
            share = times_rho(non_ron, False) if dmk_members else non_ron
            polys[index] = share * float(residual[index])
    return polys


def _poly_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros((_DEGREE, _DEGREE), dtype=float)
    for i in range(_DEGREE):
        for j in range(_DEGREE):
            if left[i, j] == 0.0:
                continue
            result[i:, j:] += left[i, j] * right[: _DEGREE - i, : _DEGREE - j]
    return result


def response_window_polynomial(
    discarder: int,
    per_seat_kinds: Mapping[int, Sequence[str]],
    per_seat_actions: Mapping[int, Sequence[Mapping[str, Any]]],
    per_seat_residual: Mapping[int, np.ndarray],
    observed_resolution: Mapping[str, Any],
    *,
    variable_seats: Mapping[int, tuple[bool, bool]],
    constants: FixedConstants,
) -> np.ndarray:
    """捨牌応答窓の公開結果尤度を ε, ρ の多項式係数として返す。

    variable_seats[seat] = (εを変数にするか, ρを変数にするか)。
    観測と両立する全希望の組を足すため、優先順位で隠れた希望を負例にしない。
    """

    seats = sorted(per_seat_kinds)
    polys = {
        seat: _seat_polynomials(
            per_seat_kinds[seat], per_seat_residual[seat], *variable_seats.get(seat, (False, False)),
            constants=constants,
        )
        for seat in seats
    }
    total = np.zeros((_DEGREE, _DEGREE), dtype=float)
    observed = dict(observed_resolution)

    def walk(position: int, choices: dict[int, Mapping[str, Any]], poly: np.ndarray) -> None:
        nonlocal total
        if position == len(seats):
            if resolve_joint_response(discarder, {seat: dict(action) for seat, action in choices.items()}) == observed:
                total += poly
            return
        seat = seats[position]
        for index, action in enumerate(per_seat_actions[seat]):
            seat_poly = polys[seat][index]
            if not seat_poly.any():
                continue
            walk(position + 1, {**choices, seat: action}, _poly_multiply(poly, seat_poly))

    start = np.zeros((_DEGREE, _DEGREE), dtype=float)
    start[0, 0] = 1.0
    walk(0, {}, start)
    return total


def evaluate_log_polynomials(polynomials: Sequence[np.ndarray], eps: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """多項式の積の対数を格子上で足す。戻り値の形は (len(eps), len(rho))。"""

    eps_powers = np.vstack([eps**power for power in range(_DEGREE)]).T  # (Ne, 4)
    rho_powers = np.vstack([rho**power for power in range(_DEGREE)])  # (4, Nr)
    total = np.zeros((len(eps), len(rho)), dtype=float)
    for poly in polynomials:
        values = eps_powers @ poly @ rho_powers
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("格子の内点で尤度が正でない")
        total += np.log(values)
    return total


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(values - peak), axis=axis))


def _relative_change(before: tuple[float, ...], after: tuple[float, ...]) -> float:
    return max(abs(a - b) / max(abs(b), 1.0e-300) for a, b in zip(before, after))


def grid_posterior(
    polynomials: Sequence[np.ndarray],
    *,
    variables: tuple[str, ...],
    opportunities: Mapping[str, int],
    start_cells: int = 400,
    max_cells: int = 3200,
    tolerance: float = 0.01,
) -> dict[str, Posterior]:
    """ε_ronとρ（またはその一方）の事後分布を、u上の中点則で求める。

    区間数を倍にしても平均と2.5%点・97.5%点の相対差がtolerance以内になるまで細かくする。
    variablesに含めない軸は長さ1の格子（値は多項式に現れないので任意）で評価する。
    """

    unknown = set(variables) - {"epsRon", "rho"}
    if unknown or not variables:
        raise ValueError(f"格子推定できる定数はepsRonとrhoだけ: {variables}")

    def summarize(cells: int) -> dict[str, tuple[float, float, float]]:
        _, p = u_grid(cells)
        eps = p if "epsRon" in variables else np.asarray([0.5])
        rho = p if "rho" in variables else np.asarray([0.5])
        log_joint = evaluate_log_polynomials(polynomials, eps, rho)
        result = {}
        if "epsRon" in variables:
            result["epsRon"] = summarize_u_posterior(_logsumexp(log_joint, axis=1), cells)
        if "rho" in variables:
            result["rho"] = summarize_u_posterior(_logsumexp(log_joint, axis=0), cells)
        return result

    cells = start_cells
    previous = summarize(cells)
    history = []
    while True:
        refined = summarize(cells * 2)
        change = max(_relative_change(previous[name], refined[name]) for name in variables)
        history.append({"cells": cells, "refinedCells": cells * 2, "maximumRelativeChange": change})
        if change <= tolerance:
            break
        cells *= 2
        previous = refined
        if cells * 2 > max_cells:
            raise ArithmeticError(f"格子の区間数{max_cells}まで細かくしても収束しない: {history}")
    return {
        name: Posterior(
            mean=refined[name][0],
            lower=refined[name][1],
            upper=refined[name][2],
            method="jeffreys_u_midpoint_grid",
            opportunities=int(opportunities.get(name, 0)),
            detail={"cells": cells * 2, "convergence": history},
        )
        for name in variables
    }


# ---------------------------------------------------------------------------
# 推定入力の収集（設計7.3節の「有効な窓」）
# ---------------------------------------------------------------------------

STRATUM_DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dealer", ("dealer", "nondealer")),
    ("riichi", ("riichi", "no_riichi")),
    ("gap", ("top", "lt8000", "ge8000")),
    ("turn", ("early", "mid", "late")),
)


def stratum_attributes(*, dealer: bool, own_riichi: bool, gap_to_top: float, junme: int) -> dict[str, str]:
    """定数を使う家自身の属性から層を決める（設計7.4節）。"""

    if gap_to_top <= 0:
        gap = "top"
    elif gap_to_top < 8000:
        gap = "lt8000"
    else:
        gap = "ge8000"
    turn = "early" if junme <= 6 else "mid" if junme <= 12 else "late"
    return {
        "dealer": "dealer" if dealer else "nondealer",
        "riichi": "riichi" if own_riichi else "no_riichi",
        "gap": gap,
        "turn": turn,
    }


@dataclass(frozen=True)
class TsumoObservation:
    skipped: bool
    attributes: Mapping[str, str]


@dataclass(frozen=True)
class ResponseObservation:
    discarder: int
    kinds: Mapping[int, tuple[str, ...]]
    actions: Mapping[int, tuple[Mapping[str, Any], ...]]
    residual: Mapping[int, np.ndarray]
    resolution: Mapping[str, Any]
    attributes: Mapping[int, Mapping[str, str]]


@dataclass
class EstimationInputs:
    tsumo: list[TsumoObservation]
    response: list[ResponseObservation]
    excluded: dict[str, int]


def _residual_distribution(model: Any, candidates: Sequence[Any], phase: str) -> np.ndarray:
    kinds = [candidate.kind for candidate in candidates]
    fixed = {"ron", "daiminkan"} if phase == "discard_response" else {"tsumo"}
    residual = [index for index, kind in enumerate(kinds) if kind not in fixed]
    values = np.zeros(len(candidates), dtype=float)
    if residual:
        values[residual] = model.base_probabilities([candidates[index] for index in residual], phase)
    return values


def collect_estimation_inputs(
    model: Any,
    encoded_windows: Iterable[Mapping[str, Any]],
    attributes_of: Callable[[Any], Mapping[str, str]],
) -> EstimationInputs:
    """固定成分の推定に使う窓を集める。学習・選択・較正の3期間だけを受け付ける。

    attributes_of(candidate) は、その家の候補1件の種別特徴から層の属性を返す。
    """

    tsumo: list[TsumoObservation] = []
    response: list[ResponseObservation] = []
    excluded: dict[str, int] = {}

    def skip(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for encoded in encoded_windows:
        window = encoded["window"]
        split = str(window["developmentSplit"])
        if split not in ESTIMATION_SPLITS:
            raise ValueError(f"固定定数の推定に{split}期間を混ぜない")
        phase = str(window["phase"])
        if "candidates" in encoded:
            kinds = [candidate.kind for candidate in encoded["candidates"]]
            if "tsumo" not in kinds:
                continue
            observation = window["observation"]
            if observation["status"] != "exact" or not window["learningMask"].get("kind", False):
                skip("self_window_not_exact_or_masked")
                continue
            if window.get("winActionStatus") != "known":
                skip("self_win_action_hold")
                continue
            tsumo.append(
                TsumoObservation(observation["action"]["kind"] != "tsumo", attributes_of(encoded["candidates"][0]))
            )
            continue
        if phase != "discard_response":
            continue
        per_seat = encoded["perSeat"]
        kinds_by_seat = {seat: tuple(c.kind for c in candidates) for seat, candidates in per_seat.items()}
        if not any({"ron", "daiminkan"} & set(kinds) for kinds in kinds_by_seat.values()):
            continue
        if not window["learningMask"].get("jointKind", False):
            skip("response_window_masked")
            continue
        if any(value.get("winActionStatus") != "known" for value in window.get("legalBySeat", {}).values()):
            skip("response_win_action_hold")
            continue
        response.append(
            ResponseObservation(
                discarder=int(window["actorSeat"]),
                kinds=kinds_by_seat,
                actions={seat: tuple(dict(c.action) for c in candidates) for seat, candidates in per_seat.items()},
                residual={seat: _residual_distribution(model, candidates, phase) for seat, candidates in per_seat.items()},
                resolution=dict(window["observation"]["resolution"]),
                attributes={seat: attributes_of(candidates[0]) for seat, candidates in per_seat.items()},
            )
        )
    return EstimationInputs(tsumo, response, excluded)


def initial_constants(inputs: EstimationInputs) -> FixedConstants:
    """最初のθ学習に使う定数。exactに分かる件数比にJeffreys補正をした値。"""

    tsumo_skips = sum(item.skipped for item in inputs.tsumo)
    eps_tsumo = (tsumo_skips + 0.5) / (len(inputs.tsumo) + 1.0)
    ron_legal = ron_skip = dmk_legal = dmk_taken = 0
    for item in inputs.response:
        resolution = item.resolution
        for seat, kinds in item.kinds.items():
            ron_won = resolution.get("kind") == "ron" and seat in resolution.get("winnerSeats", [])
            if "ron" in kinds:
                ron_legal += 1
                ron_skip += int(resolution.get("kind") == "pass")
            if "daiminkan" in kinds and not ron_won:
                dmk_legal += 1
                dmk_taken += int(resolution.get("kind") == "daiminkan" and resolution.get("seat") == seat)
    eps_ron = (ron_skip + 0.5) / (ron_legal + 1.0)
    rho = (dmk_taken + 0.5) / (dmk_legal + 1.0)
    return FixedConstants(eps_ron, eps_tsumo, rho, eps_ron).validate(open_interval=True)


# ---------------------------------------------------------------------------
# 全体と層別の事後分布（設計7.3節、7.4節）
# ---------------------------------------------------------------------------


def _opportunities(inputs: EstimationInputs, seat_filter: Callable[[Mapping[str, str]], bool]) -> dict[str, int]:
    counts = {"epsRon": 0, "rho": 0, "epsTsumo": 0}
    for item in inputs.response:
        for seat, kinds in item.kinds.items():
            if not seat_filter(item.attributes[seat]):
                continue
            counts["epsRon"] += int("ron" in kinds)
            counts["rho"] += int("daiminkan" in kinds)
    counts["epsTsumo"] = sum(seat_filter(item.attributes) for item in inputs.tsumo)
    return counts


def _response_polynomials(
    inputs: EstimationInputs,
    constants: FixedConstants,
    variables: tuple[str, ...],
    seat_filter: Callable[[Mapping[str, str]], bool],
) -> list[np.ndarray]:
    polys = []
    for item in inputs.response:
        variable_seats = {
            seat: ("epsRon" in variables and seat_filter(item.attributes[seat]),
                   "rho" in variables and seat_filter(item.attributes[seat]))
            for seat in item.kinds
        }
        relevant = any(
            (eps and "ron" in item.kinds[seat]) or (rho and "daiminkan" in item.kinds[seat])
            for seat, (eps, rho) in variable_seats.items()
        )
        if not relevant:
            continue
        polys.append(
            response_window_polynomial(
                item.discarder, item.kinds, item.actions, item.residual, item.resolution,
                variable_seats=variable_seats, constants=constants,
            )
        )
    return polys


def estimate_base_posteriors(inputs: EstimationInputs, current: FixedConstants, **grid_options: Any) -> dict[str, Posterior]:
    """全体の事後分布。ε_tsumoはBeta厳密解、ε_ronとρは2次元の同時事後分布の周辺。"""

    def everyone(attributes: Mapping[str, str]) -> bool:
        return True

    opportunities = _opportunities(inputs, everyone)
    skips = sum(item.skipped for item in inputs.tsumo)
    result = {"epsTsumo": jeffreys_beta_posterior(skips, len(inputs.tsumo))}
    polys = _response_polynomials(inputs, current, ("epsRon", "rho"), everyone)
    result.update(grid_posterior(polys, variables=("epsRon", "rho"), opportunities=opportunities, **grid_options))
    return result


def constants_from_posteriors(posteriors: Mapping[str, Posterior]) -> FixedConstants:
    eps_ron = posteriors["epsRon"].mean
    # 搶槓ロンの見送り率は観測がないため、捨牌ロンの定数を流用する（モデル仮定）。
    return FixedConstants(eps_ron, posteriors["epsTsumo"].mean, posteriors["rho"].mean, eps_ron).validate(
        open_interval=True
    )


def _prior_summary() -> tuple[float, float, float]:
    # Jeffreys事前分布Beta(0.5, 0.5)はarcsin分布。p = sin²(πq/2) が q分位点になる。
    return 0.5, math.sin(math.pi * 0.025 / 2) ** 2, math.sin(math.pi * 0.975 / 2) ** 2


def classify_stratum(overall: Posterior, stratum: Posterior | None, opportunities: int) -> str:
    """層を difference_detected / unsupported / consistent に分ける。"""

    if opportunities == 0 or stratum is None:
        return "unsupported"
    if stratum.upper < overall.lower or stratum.lower > overall.upper:
        return "difference_detected"
    overall_width = overall.upper - overall.lower
    contains = stratum.lower <= overall.lower and stratum.upper >= overall.upper
    if contains and (stratum.upper - stratum.lower) >= 5.0 * overall_width:
        return "unsupported"
    return "consistent"


def estimate_strata(
    inputs: EstimationInputs,
    base: FixedConstants,
    overall: Mapping[str, Posterior],
    **grid_options: Any,
) -> list[dict[str, Any]]:
    """層ごと・定数ごとに、層に属する家の定数だけを変数とした事後分布を求める。"""

    results = []
    for dimension, values in STRATUM_DIMENSIONS:
        for value in values:
            def member(attributes: Mapping[str, str], d: str = dimension, v: str = value) -> bool:
                return attributes[d] == v

            opportunities = _opportunities(inputs, member)
            for constant in ESTIMATED_CONSTANTS:
                count = opportunities[constant]
                posterior: Posterior | None = None
                if count > 0:
                    if constant == "epsTsumo":
                        members = [item for item in inputs.tsumo if member(item.attributes)]
                        posterior = jeffreys_beta_posterior(sum(item.skipped for item in members), len(members))
                    else:
                        polys = _response_polynomials(inputs, base, (constant,), member)
                        posterior = grid_posterior(
                            polys, variables=(constant,), opportunities={constant: count}, **grid_options
                        )[constant]
                status = classify_stratum(overall[constant], posterior, count)
                if posterior is None:
                    mean, lower, upper = _prior_summary()
                    summary = {"mean": mean, "lower": lower, "upper": upper, "method": "jeffreys_prior_only"}
                else:
                    summary = posterior.to_dict()
                results.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "constant": constant,
                        "opportunities": count,
                        "status": status,
                        "posterior": summary,
                    }
                )
    return results


# ---------------------------------------------------------------------------
# 感度シナリオ（設計7.6節）
# ---------------------------------------------------------------------------

_ATTRIBUTE_BY_CONSTANT = {"epsRon": "eps_ron", "epsTsumo": "eps_tsumo", "rho": "rho", "epsChankan": "eps_chankan"}


def _with_constant(constants: FixedConstants, name: str, value: float) -> FixedConstants:
    changed = replace(constants, **{_ATTRIBUTE_BY_CONSTANT[name]: value})
    if name == "epsRon":
        # chankan_assumption以外では、搶槓の定数は捨牌ロンの定数に追従する。
        changed = replace(changed, eps_chankan=value)
    return changed


def build_scenarios(
    posteriors: Mapping[str, Posterior],
    strata: Sequence[Mapping[str, Any]],
    *,
    theta_id: str,
) -> list[dict[str, Any]]:
    """シナリオ一覧を作る。重複を除いた後の一覧を実行前にmanifestへ固定する。"""

    base = constants_from_posteriors(posteriors)
    bounds = {name: (posteriors[name].lower, posteriors[name].upper) for name in ESTIMATED_CONSTANTS}
    scenarios: list[dict[str, Any]] = []

    def add(identifier: str, kind: str, constants: FixedConstants, overrides: list[dict[str, Any]], posterior_id: str) -> None:
        scenarios.append(
            {
                "id": identifier,
                "type": kind,
                "constants": constants.to_dict(),
                "stratumOverrides": overrides,
                "thetaId": theta_id,
                "posteriorId": posterior_id,
                "usableInD33": kind == "history_and_future",
            }
        )

    add("base", "history_and_future", base, [], "base")
    for name in ESTIMATED_CONSTANTS:
        for side, value in zip(("low", "high"), bounds[name]):
            identifier = f"oat_{name}_{side}"
            add(identifier, "history_and_future", _with_constant(base, name, value), [], identifier)
    for sides in ((a, b, c) for a in "lh" for b in "lh" for c in "lh"):
        constants = base
        for name, side in zip(ESTIMATED_CONSTANTS, sides):
            constants = _with_constant(constants, name, bounds[name][0 if side == "l" else 1])
        identifier = "corner_" + "".join(sides)
        add(identifier, "history_and_future", constants, [], identifier)
    add("chankan_assumption", "history_and_future", replace(base, eps_chankan=CHANKAN_STRESS_VALUE), [], "chankan_assumption")
    for item in strata:
        if item["status"] not in {"difference_detected", "unsupported"}:
            continue
        for side, key in (("low", "lower"), ("high", "upper")):
            identifier = f"stratum_{item['dimension']}-{item['value']}_{item['constant']}_{side}"
            override = {
                "dimension": item["dimension"],
                "value": item["value"],
                "constant": item["constant"],
                "constantValue": float(item["posterior"][key]),
                "stratumStatus": item["status"],
            }
            add(identifier, "history_and_future", base, [override], identifier)
    add("zero", "future_stress", FixedConstants(0.0, 0.0, 0.0, 0.0), [], "base")

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scenario in scenarios:
        key = repr(
            (
                sorted(scenario["constants"].items()),
                [sorted(override.items()) for override in scenario["stratumOverrides"]],
                scenario["type"],
            )
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(scenario)
    return unique


def scenario_count_summary(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "K": sum(bool(item["stratumOverrides"]) for item in scenarios),
        "D.3.3": sum(bool(item["usableInD33"]) for item in scenarios),
        "D.3.4": len(scenarios),
    }


def scenarios_for_stage(scenarios: Sequence[Mapping[str, Any]], stage: str) -> list[Mapping[str, Any]]:
    """D.3.3では将来だけのストレス試験（zero）を使わず、全定数が0と1を含まないことを検査する。"""

    if stage == "D.3.4":
        return list(scenarios)
    if stage != "D.3.3":
        raise ValueError(f"未知の段階: {stage}")
    selected = [item for item in scenarios if item["usableInD33"]]
    for item in selected:
        FixedConstants.from_dict(item["constants"]).validate(open_interval=True)
        for override in item["stratumOverrides"]:
            if not 0.0 < float(override["constantValue"]) < 1.0:
                raise ValueError(f"D.3.3で0または1の定数は使えない: {item['id']}")
    return selected


def require_scenario_for_stage(scenarios: Sequence[Mapping[str, Any]], identifier: str, stage: str) -> Mapping[str, Any]:
    for item in scenarios_for_stage(scenarios, stage):
        if item["id"] == identifier:
            return item
    raise ValueError(f"{stage}で使えないシナリオ: {identifier}")


def constants_for_seat(scenario: Mapping[str, Any], attributes: Mapping[str, str]) -> FixedConstants:
    """層を適用するシナリオで、その家の属性に応じた定数を返す。"""

    constants = FixedConstants.from_dict(scenario["constants"])
    for override in scenario["stratumOverrides"]:
        if attributes.get(override["dimension"]) == override["value"]:
            constants = _with_constant(constants, override["constant"], float(override["constantValue"]))
    return constants


def fixed_component_sensitivity(deltas: Mapping[str, float], base_id: str = "base") -> dict[str, Any]:
    """押しとオリのEV差の符号がbaseと異なるシナリオを列挙する（D.3.4用）。"""

    if base_id not in deltas:
        raise ValueError("baseシナリオのEV差がない")

    def sign(value: float) -> float:
        return 0.0 if value == 0 else math.copysign(1.0, value)

    flipped = sorted(
        identifier for identifier, value in deltas.items()
        if identifier != base_id and sign(value) != sign(deltas[base_id])
    )
    return {"fixed_component_sensitive": bool(flipped), "flippedScenarios": flipped}
