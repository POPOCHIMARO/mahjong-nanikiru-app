#!/usr/bin/env python3
"""D.3.3 工程1：家ごとの履歴評価器（calibration/PHASE_D33_DESIGN.md 5節）。

他家1人の「配牌と自摸の物理牌」の仮説を受け取り、判断時点までの公開履歴を
その家の視点で順に評価して、次を返す。

- 自己行動窓・応答窓ごとの合法集合（RoundStateと同じ規則）
- フリテン3種とリーチ・一発の履歴
- 観測行動の確率（v3相手モデル、固定定数とシナリオを合成）と、その対数和
- 硬い制約（打牌整合、リーチ時テンパイ）の違反

局全体の状態機械は回さない。各家の尤度は、その家の手牌と公開履歴だけで決まる
（設計5.4節）。RoundStateで同じ割当を進めた参照経路（`reference_evaluations`）との
一致を受入試験D33-07で確かめる。

情報境界（D33-06）：`DecisionContext`は公開履歴と対象家自身の私有履歴だけから作る。
他家の実際の手牌、判断時点より後の公開イベント、局の結果は受け取らない。
"""

from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

from tools.ev_calibration_state import shanten
from tools.ev_policy_fixed import compose_probabilities, constants_for_seat
from tools.ev_policy_observation import (
    _deduplicate_actions,
    semantic_response_actions,
    semantic_self_actions,
)
from tools.ev_policy_opponent import (
    HierarchicalSoftmax,
    RoundFeatureState,
    _deterministic_candidate,
    _stratum_attributes_of,
    canonical_action,
)
from tools.ev_policy_round import (
    NON_WINNING_SCORING_CODES,
    IllegalAction,
    RoundState,
    kuikae_forbidden_tile34,
    riichi_ankan_legal,
    riichi_timing_legal,
    waiting_tile34,
)
from tools.ev_policy_scoring import (
    HandFlags,
    HandScoreRequest,
    HandScoringError,
    ScoringDependencyError,
    score_hand,
)
from tools.ev_policy_state import TileInstance, canonical_tile_set


BELIEF_EVALUATOR_VERSION = "ev-policy-belief-seat-evaluator/v1"
LIVE_WALL_SIZE = 70
TILES: tuple[TileInstance, ...] = canonical_tile_set()
TILE_BY_ID: dict[int, TileInstance] = {tile.tile_id: tile for tile in TILES}
# 入口条件（設計2.1節）で判断時点までに現れてよい公開イベント。
ALLOWED_PREFIX_EVENTS = frozenset({"draw", "discard", "response_resolution"})


class ContextError(ValueError):
    """推論入力が入口条件を満たさない、または内部で矛盾する。"""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tile_key(tile: Mapping[str, Any]) -> tuple[int, bool]:
    """公開記録の牌（tile34, isRed）を比較用の組にする。"""
    return int(tile["tile34"]), bool(tile["isRed"])


def id_key(tile_id: int) -> tuple[int, bool]:
    tile = TILE_BY_ID[tile_id]
    return tile.tile34, tile.is_red


def ids_of_key(key: tuple[int, bool]) -> list[int]:
    """その牌種・赤の物理IDをすべて返す（赤5は1枚、通常5は3枚、他は4枚）。"""
    return [tile.tile_id for tile in TILES if (tile.tile34, tile.is_red) == key]


def counts34(tile_ids: Sequence[int]) -> tuple[int, ...]:
    values = [0] * 34
    for tile_id in tile_ids:
        values[TILE_BY_ID[tile_id].tile34] += 1
    return tuple(values)


def _record(tile_id: int) -> dict[str, Any]:
    tile = TILE_BY_ID[tile_id]
    return {"tile34": tile.tile34, "isRed": tile.is_red}


# ---------------------------------------------------------------------------
# 判断文脈（情報境界の内側）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeatTurn:
    """他家の1回の自摸と、その直後の打牌。"""

    event_position: int  # 公開イベント列でのdrawの位置（0始まり）
    raw_event_index: int
    discard: tuple[int, bool]
    tsumogiri: bool
    riichi_declaration: bool


@dataclass(frozen=True)
class DecisionContext:
    """判断時点の推論入力。公開履歴と対象家の私有履歴だけから作る（D33-06）。"""

    decision_id: str
    round_id: str
    information_state_hash: str
    target_seat: int
    riichi_seat: int
    initial: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    target_initial: tuple[tuple[int, bool], ...]
    target_draws: Mapping[int, tuple[int, bool]]  # rawEventIndex -> 牌
    turns: Mapping[int, tuple[SeatTurn, ...]]  # 他家ごと

    @property
    def other_seats(self) -> tuple[int, ...]:
        return tuple(seat for seat in range(4) if seat != self.target_seat)

    @property
    def dealer_seat(self) -> int:
        return int(self.initial["dealerSeat"])

    @property
    def round_wind_tile34(self) -> int:
        return 27 + min(int(self.initial["kyoku"]) // 4, 3)

    @property
    def dora_indicator(self) -> tuple[int, bool]:
        return tile_key(self.initial["doraIndicator"])

    def public_row(self) -> dict[str, Any]:
        return {"roundId": self.round_id, "initial": dict(self.initial), "events": [dict(e) for e in self.events]}

    def target_private_row(self) -> dict[str, Any]:
        return {
            "seat": self.target_seat,
            "initialHand": [{"tile34": t, "isRed": r} for t, r in self.target_initial],
            "events": [
                {"type": "draw_observation", "rawEventIndex": raw, "tile": {"tile34": t, "isRed": r}}
                for raw, (t, r) in sorted(self.target_draws.items())
            ],
        }

    def input_hash(self) -> str:
        """推定器へ入る情報の全体。他家の教師情報を差し替えても変わらないこと（D33-06）。"""
        payload = {
            "decisionId": self.decision_id,
            "public": self.public_row(),
            "target": self.target_private_row(),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @classmethod
    def from_records(
        cls,
        prefix: Mapping[str, Any],
        public_row: Mapping[str, Any],
        target_private_row: Mapping[str, Any],
    ) -> "DecisionContext":
        """推論prefix、公開列、対象家の私有列から作る。他家の私有列は受け取らない。"""

        target = int(prefix["seat"])
        if int(target_private_row["seat"]) != target:
            raise ContextError("私有列が対象家のものでない")
        if str(public_row["roundId"]) != str(prefix["roundId"]):
            raise ContextError("公開列のroundIdが不一致")
        count = int(prefix["publicEventCount"])
        events = tuple(dict(event) for event in public_row["events"][:count])
        if len(events) != count:
            raise ContextError("公開イベントがprefixより短い")
        for event in events:
            if event["type"] not in ALLOWED_PREFIX_EVENTS:
                raise ContextError(f"入口条件外の公開イベント: {event['type']}")
            if event["type"] == "response_resolution" and event["resolution"]["kind"] != "pass":
                raise ContextError("判断時点までに全員パス以外の応答がある")
        # 対象家の自摸は、判断時点までに公開された（本人に見えた）ものだけを使う。
        draws = {
            int(item["rawEventIndex"]): tile_key(item["tile"])
            for item in target_private_row["events"]
            if item["type"] == "draw_observation" and int(item["availableAtPublicEventCount"]) <= count
        }
        turns: dict[int, list[SeatTurn]] = {seat: [] for seat in range(4) if seat != target}
        riichi = set()
        for position, event in enumerate(events):
            if event["type"] != "draw":
                continue
            seat = int(event["seat"])
            if event.get("source") != "live":
                raise ContextError("入口条件外の嶺上自摸")
            if seat == target:
                if int(event["rawEventIndex"]) not in draws:
                    raise ContextError("対象家の自摸が私有列にない")
                continue
            following = events[position + 1] if position + 1 < len(events) else None
            if following is None or following["type"] != "discard" or int(following["seat"]) != seat:
                raise ContextError("他家の自摸の直後に打牌がない")
            declaration = bool(following.get("riichiDeclaration"))
            if declaration:
                riichi.add(seat)
            turns[seat].append(
                SeatTurn(
                    position,
                    int(event["rawEventIndex"]),
                    tile_key(following["tile"]),
                    following["origin"] == "drawn",
                    declaration,
                )
            )
        if len(riichi) != 1:
            raise ContextError("成立済み他家リーチがちょうど1人でない")
        return cls(
            decision_id=str(prefix["decisionId"]),
            round_id=str(prefix["roundId"]),
            information_state_hash=str(prefix["informationStateHash"]),
            target_seat=target,
            riichi_seat=next(iter(riichi)),
            initial=dict(public_row["initial"]),
            events=events,
            target_initial=tuple(tile_key(tile) for tile in target_private_row["initialHand"]),
            target_draws=draws,
            turns={seat: tuple(value) for seat, value in turns.items()},
        )


# ---------------------------------------------------------------------------
# 家ごとの仮説と評価結果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeatHypothesis:
    """他家1人の配牌13枚と、各自摸（rawEventIndex）の物理ID。"""

    seat: int
    initial: tuple[int, ...]
    draws: Mapping[int, int]

    def private_row(self, context: DecisionContext) -> dict[str, Any]:
        """RoundFeatureStateへ渡す私有列の形。"""
        return {
            "seat": self.seat,
            "initialHand": [_record(tile_id) for tile_id in self.initial],
            "events": [
                {"type": "draw_observation", "rawEventIndex": turn.raw_event_index, "tile": _record(self.draws[turn.raw_event_index])}
                for turn in context.turns[self.seat]
            ],
        }


@dataclass
class WindowRecord:
    kind: str  # "self" または "response"
    event_position: int  # 窓の基準となる公開イベントの位置
    legal_actions: list[dict[str, Any]]
    win_action_status: str
    observed: dict[str, Any]
    probability: float | None = None
    furiten: tuple[str, ...] = ()


@dataclass
class SeatEvaluation:
    seat: int
    log_likelihood: float
    windows: list[WindowRecord] = field(default_factory=list)
    violation: str | None = None  # 硬い制約の違反（H=0）
    holds: list[str] = field(default_factory=list)  # rule_unresolved の原因
    riichi_furiten: bool = False
    temporary_furiten: bool = False
    ippatsu: bool = False  # 判断時点の一発の資格
    final_hand: tuple[int, ...] = ()  # 判断時点の手牌（物理ID）
    river34: tuple[int, ...] = ()  # その家の河（牌種）

    @property
    def consistent(self) -> bool:
        return self.violation is None


# ---------------------------------------------------------------------------
# 特徴状態：その家だけを追うRoundFeatureState
# ---------------------------------------------------------------------------


class _PlaceholderDraws(dict):
    """対象外の家の自摸は特徴に使わないので、仮の牌で埋める。"""

    def __missing__(self, key: tuple[int, int]) -> dict[str, Any]:
        return {"tile34": 0, "isRed": False}


class SeatFeatureState(RoundFeatureState):
    """RoundFeatureStateのうち、focus家の手牌だけを正しく追う版。

    特徴計算（context、encode）は、行動する家の手牌と公開情報しか読まない。
    そのため他家の手牌は空のまま扱い、打牌による除去も無視する。
    """

    def __init__(self, public: Mapping[str, Any], focus_seat: int, focus_private: Mapping[str, Any]):
        private = {
            seat: focus_private if seat == focus_seat else {"initialHand": [], "events": []}
            for seat in range(4)
        }
        super().__init__(public, private)
        self.focus_seat = focus_seat
        self.draws = _PlaceholderDraws(self.draws)

    def _remove(self, seat: int, tile: Mapping[str, Any]) -> None:
        if seat == self.focus_seat:
            super()._remove(seat, tile)


# ---------------------------------------------------------------------------
# 規則側：その家の視点の合法集合とフリテン（RoundStateと同じ判定）
# ---------------------------------------------------------------------------


@dataclass
class _SeatRuleState:
    context: DecisionContext
    seat: int
    hand: list[int]
    drawn: int | None = None
    live_draws: int = 0
    river34: list[int] = field(default_factory=list)
    active_riichi: bool = False
    double_riichi: bool = False
    riichi_base_counts: tuple[int, ...] | None = None
    pending_riichi: bool = False
    ippatsu: bool = False
    temporary_furiten: bool = False
    riichi_furiten: bool = False

    @property
    def live_remaining(self) -> int:
        return LIVE_WALL_SIZE - self.live_draws

    def waits(self, tile_ids: Sequence[int] | None = None) -> tuple[int, ...]:
        return waiting_tile34(counts34(self.hand if tile_ids is None else tile_ids), 0)

    def discard_furiten(self, waits: Sequence[int]) -> bool:
        return bool(set(waits) & set(self.river34))

    def is_furiten(self, waits: Sequence[int]) -> bool:
        return self.temporary_furiten or self.riichi_furiten or self.discard_furiten(waits)

    def furiten_reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.discard_furiten(self.waits()):
            reasons.append("own_discard")
        if self.temporary_furiten:
            reasons.append("temporary")
        if self.riichi_furiten:
            reasons.append("riichi_pass")
        return tuple(reasons)

    def _dora_indicators(self, tiles: Sequence[int]) -> tuple[TileInstance, ...]:
        # 得点器は表示牌の牌種だけを使うが、手牌と同じ物理IDは拒否する（indicator_tile_in_hand）。
        # 表示牌は手牌とは別の物理牌なので、手牌にない同じ牌種のIDが必ず残る。
        used = set(tiles)
        return (TILE_BY_ID[next(i for i in ids_of_key(self.context.dora_indicator) if i not in used)],)

    def _score(self, tiles: Sequence[int], win_tile_id: int, flags: HandFlags) -> None:
        score_hand(
            HandScoreRequest(
                tiles=tuple(TILE_BY_ID[tile_id] for tile_id in tiles),
                win_tile_id=win_tile_id,
                seat_wind_tile34=27 + ((self.seat - self.context.dealer_seat) % 4),
                round_wind_tile34=self.context.round_wind_tile34,
                melds=(),
                dora_indicators=self._dora_indicators(tiles),
                ura_dora_indicators=(),
                flags=flags,
            )
        )

    def _win_status(self, attempt: Callable[[], None]) -> tuple[bool, str]:
        """和了の可否。役なしは不可、その他の得点エラーはhold（設計10.2節 rule_unresolved）。"""
        try:
            attempt()
        except ScoringDependencyError:
            raise
        except HandScoringError as error:
            if error.code in NON_WINNING_SCORING_CODES:
                return False, "known"
            return False, f"hold:{error.code}"
        return True, "known"

    def self_actions(self) -> tuple[list[dict[str, Any]], str]:
        """通常自摸後の意味上の合法手（semantic_self_actionsと同じ形）。"""
        assert self.drawn is not None
        is_last = self.live_remaining == 0
        actions: list[dict[str, Any]] = []
        discard_ids = [self.drawn] if self.active_riichi else list(self.hand)
        riichi_by_type: dict[int, bool] = {}
        for tile_id in discard_ids:
            tile = TILE_BY_ID[tile_id]
            origin = "drawn" if tile_id == self.drawn else "concealed"
            actions.append({"kind": "discard", "tile34": tile.tile34, "isRed": tile.is_red, "origin": origin})
            if tile.tile34 not in riichi_by_type:
                rest = list(self.hand)
                rest.remove(tile_id)
                riichi_by_type[tile.tile34] = (
                    not self.active_riichi
                    and riichi_timing_legal("live", is_last)
                    and shanten(counts34(rest), 0) == 0
                )
            if riichi_by_type[tile.tile34]:
                actions.append({"kind": "riichi_discard", "tile34": tile.tile34, "isRed": tile.is_red, "origin": origin})
        # 暗槓（槓0回、嶺上牌4枚が残る入口条件では、生牌が残る限り開始できる）。
        if self.live_remaining > 0:
            by_type: dict[int, list[int]] = {}
            for tile_id in self.hand:
                by_type.setdefault(TILE_BY_ID[tile_id].tile34, []).append(tile_id)
            for tile_type, ids in by_type.items():
                if len(ids) == 4 and (
                    not self.active_riichi or riichi_ankan_legal(self.riichi_base_counts, tile_type, 0)
                ):
                    actions.append(
                        {"kind": "ankan", "tile34": tile_type, "redCount": sum(TILE_BY_ID[i].is_red for i in ids)}
                    )
        status = "known"
        rest = list(self.hand)
        rest.remove(self.drawn)
        if TILE_BY_ID[self.drawn].tile34 in self.waits(rest):
            flags = HandFlags(
                is_tsumo=True,
                is_riichi=self.active_riichi and not self.double_riichi,
                is_double_riichi=self.active_riichi and self.double_riichi,
                is_ippatsu=self.ippatsu,
                is_haitei=is_last,
            )
            legal, status = self._win_status(lambda: self._score(self.hand, self.drawn, flags))
            if legal:
                actions.append({"kind": "tsumo"})
        return _deduplicate_actions(actions), status

    def response_actions(self, discarder: int, tile: tuple[int, bool]) -> tuple[list[dict[str, Any]], str]:
        """他家の捨牌への意味上の合法応答（semantic_response_actionsと同じ形）。"""
        called34, called_red = tile
        actions: list[dict[str, Any]] = [{"kind": "pass"}]
        if not self.active_riichi and self.live_remaining > 0:
            matching = [tile_id for tile_id in self.hand if TILE_BY_ID[tile_id].tile34 == called34]
            for size, kind in ((2, "pon"), (3, "daiminkan")):
                for combo in combinations(matching, size):
                    actions.append({"kind": kind, "consumed": _consumed(combo)})
            if self.seat == (discarder + 1) % 4 and called34 < 27:
                actions.extend(self._chi_actions(called34))
        waits = self.waits()
        status = "known"
        if called34 in waits and not self.is_furiten(waits):
            win_id = next(i for i in ids_of_key(tile) if i not in self.hand)
            flags = HandFlags(
                is_tsumo=False,
                is_riichi=self.active_riichi and not self.double_riichi,
                is_double_riichi=self.active_riichi and self.double_riichi,
                is_ippatsu=self.ippatsu,
                is_houtei=self.live_remaining == 0,
            )
            legal, status = self._win_status(lambda: self._score([*self.hand, win_id], win_id, flags))
            if legal:
                actions.append({"kind": "ron"})
        return _deduplicate_actions(actions), status

    def _chi_actions(self, called34: int) -> list[dict[str, Any]]:
        suit_start = (called34 // 9) * 9
        by_type: dict[int, list[int]] = {}
        for tile_id in self.hand:
            by_type.setdefault(TILE_BY_ID[tile_id].tile34, []).append(tile_id)
        actions = []
        for start in range(max(suit_start, called34 - 2), min(suit_start + 6, called34) + 1):
            sequence = {start, start + 1, start + 2}
            if called34 not in sequence:
                continue
            needed = sorted(sequence - {called34})
            if not all(by_type.get(t) for t in needed):
                continue
            for left in by_type[needed[0]]:
                for right in by_type[needed[1]]:
                    forbidden = kuikae_forbidden_tile34("chi", called34, (needed[0], needed[1]))
                    remaining = [i for i in self.hand if i not in (left, right)]
                    if any(TILE_BY_ID[i].tile34 not in forbidden for i in remaining):
                        actions.append({"kind": "chi", "consumed": _consumed((left, right))})
        return actions

    def pass_resolved(self, discarder: int, tile34: int) -> None:
        """全員パスで解決した捨牌の見逃しフリテン（RoundState.resolve_discard_responsesと同じ）。"""
        if discarder == self.seat:
            if self.pending_riichi:
                self.active_riichi = True
                self.double_riichi = len(self.river34) == 1  # 入口条件で鳴きは起きていない
                self.riichi_base_counts = counts34(self.hand)
                self.ippatsu = True
                self.pending_riichi = False
            return
        waits = self.waits()
        if tile34 in waits and not self.is_furiten(waits):
            if self.active_riichi:
                self.riichi_furiten = True
            else:
                self.temporary_furiten = True


def _consumed(tile_ids: Sequence[int]) -> list[dict[str, Any]]:
    return sorted((_record(tile_id) for tile_id in tile_ids), key=_canonical_json)


# ---------------------------------------------------------------------------
# 評価器本体
# ---------------------------------------------------------------------------


ModelResolver = Callable[[Sequence[Any]], HierarchicalSoftmax]


def model_identity(model: HierarchicalSoftmax) -> str:
    """学習分布を決めるモデル識別子（係数と温度のハッシュ）。固定定数は含めない（8.2節）。"""
    payload = {
        "kindWeights": np.asarray(model.kind_weights).tolist(),
        "detailWeights": np.asarray(model.detail_weights).tolist(),
        "temperatures": dict(sorted(model.temperatures.items())),
    }
    bias = getattr(model, "riichi_response_bias", None)
    if bias:  # θ変種（9.3節）は別の識別子を持ち、キャッシュの学習分布を共有しない
        payload["riichiResponseBias"] = dict(sorted(bias.items()))
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class ScenarioResolver:
    """窓の候補から、その家に使うモデルを返す（呼び出すとHierarchicalSoftmax）。

    シナリオがあれば、その家の層（候補の種別特徴から決まる）の固定定数を合成する。
    キャッシュを使う評価では、保存した学習分布へ、このシナリオの定数を最後に合成する（8.2節）。
    """

    def __init__(self, model: HierarchicalSoftmax, scenario: Mapping[str, Any] | None = None):
        self.model = model
        self.scenario = scenario
        self.model_id = model_identity(model)

    def __call__(self, candidates: Sequence[Any]) -> HierarchicalSoftmax:
        if self.scenario is None:
            return self.model
        return self.model.with_fixed(constants_for_seat(self.scenario, _stratum_attributes_of(candidates[0])))

    def cached_probability(self, entry: "WindowEntry", index: int, phase: str, cache: "WindowCache") -> float:
        """probabilities(candidates, phase)[index] と同じ値を、保存した学習分布から計算する。"""
        if self.scenario is None:
            fixed = self.model.fixed
        else:
            fixed = constants_for_seat(self.scenario, entry.attributes)

        def base(indices: Sequence[int]) -> np.ndarray:
            key = (self.model_id, tuple(indices))
            value = entry.base.get(key)
            if value is None:
                cache.base_misses += 1
                value = self.model.base_probabilities([entry.candidates[i] for i in indices], phase)
                entry.base[key] = value
            else:
                cache.base_hits += 1
            return value

        if fixed is None:
            return float(base(list(range(len(entry.candidates))))[index])
        return float(compose_probabilities(entry.kinds, phase, fixed, base)[index])


def model_resolver(model: HierarchicalSoftmax, scenario: Mapping[str, Any] | None = None) -> ScenarioResolver:
    return ScenarioResolver(model, scenario)


@dataclass
class WindowEntry:
    """窓キャッシュの値：符号化した候補（モデルに依存しない）と、モデル識別子ごとの学習分布。"""

    candidates: list[Any]
    kinds: tuple[str, ...]
    attributes: dict[str, str]
    base: dict[tuple[str, tuple[int, ...]], np.ndarray] = field(default_factory=dict)


class WindowCache:
    """判断1件の窓キャッシュ（件数上限つきLRU）。同じ判断の全シナリオ・全鎖で共有する。

    鍵は判断、phase、窓の位置、家、その時点の手牌の牌種、自摸牌の牌種またはフリテン状態、
    合法候補の列。学習分布はさらにモデル識別子で分ける（θ変種と共有しない）。
    """

    def __init__(self, capacity: int = 200_000):
        if capacity < 1:
            raise ValueError("キャッシュの上限は1以上")
        from collections import OrderedDict

        self.capacity = capacity
        self.entries: "OrderedDict[tuple, WindowEntry]" = OrderedDict()
        self.hits = self.misses = self.evictions = self.base_hits = self.base_misses = 0

    def get(self, key: tuple) -> WindowEntry | None:
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return entry

    def put(self, key: tuple, entry: WindowEntry) -> None:
        self.entries[key] = entry
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
            self.evictions += 1

    def statistics(self) -> dict[str, Any]:
        lookups = self.hits + self.misses
        return {
            "capacity": self.capacity, "entries": len(self.entries), "hits": self.hits, "misses": self.misses,
            "evictions": self.evictions, "hitRate": self.hits / lookups if lookups else None,
            "baseHits": self.base_hits, "baseMisses": self.base_misses,
        }


def _window_probability(
    features: RoundFeatureState,
    seat: int,
    actions: Sequence[Mapping[str, Any]],
    observed: Mapping[str, Any],
    phase: str,
    resolver: ModelResolver,
) -> float:
    if len(actions) == 1:
        return 1.0 if canonical_action(actions[0]) == canonical_action(observed) else 0.0
    candidates = features.encode(seat, actions)
    return resolver(candidates).action_probability(candidates, observed, phase)


def window_cache_key(
    context: DecisionContext, phase: str, position: int, seat: int, hand: Sequence[int], state: tuple, names: Sequence[str]
) -> tuple:
    """窓キャッシュの鍵（8.2節）。stateは自己行動窓なら("drawn", 自摸牌の牌種)、応答窓ならフリテンとリーチ。

    同じ14枚でも自摸牌が違えば合法候補が変わる（反証レビューR3）ので、自摸牌と合法候補の列を鍵に入れる。
    """
    return (context.decision_id, phase, position, seat, tuple(sorted(id_key(tile_id) for tile_id in hand)), state, tuple(names))


def evaluate_seat(
    context: DecisionContext,
    hypothesis: SeatHypothesis,
    resolver: ModelResolver | None,
    cache: WindowCache | None = None,
) -> SeatEvaluation:
    """他家1人の履歴を判断時点まで評価する（設計5節）。

    resolverがNoneなら合法集合とフリテンだけを求め、確率は計算しない（規則だけの検査用）。
    硬い制約に違反したら、その時点で対数尤度を-infとして返す。
    cacheがあれば、窓の符号化と学習分布を再利用する（8.2節）。特徴状態は、キャッシュに
    ない窓に出会ったときだけ作って進める。キャッシュの有無で結果は一致する（D33-07）。
    """

    seat = hypothesis.seat
    if seat not in context.other_seats:
        raise ValueError("評価できるのは他家だけ")
    if len(hypothesis.initial) != 13 or len(set(hypothesis.initial)) != 13:
        raise ValueError("配牌は異なる物理ID13枚が必要")
    result = SeatEvaluation(seat, 0.0)
    rule = _SeatRuleState(context, seat, list(hypothesis.initial))
    features = None
    if resolver is not None and cache is None:
        features = SeatFeatureState(context.public_row(), seat, hypothesis.private_row(context))

    def window_probability(
        position: int, actions: list[dict[str, Any]], observed: Mapping[str, Any], phase: str, state: tuple
    ) -> float:
        nonlocal features
        if cache is None:
            features.advance(position + 1)  # type: ignore[union-attr]
            return _window_probability(features, seat, actions, observed, phase, resolver)  # type: ignore[arg-type]
        if len(actions) == 1:
            return 1.0 if canonical_action(actions[0]) == canonical_action(observed) else 0.0
        names = tuple(canonical_action(action) for action in actions)
        observed_name = canonical_action(observed)
        if observed_name not in names:
            return 0.0
        key = window_cache_key(context, phase, position, seat, rule.hand, state, names)
        entry = cache.get(key)
        if entry is None:
            if features is None:
                features = SeatFeatureState(context.public_row(), seat, hypothesis.private_row(context))
            features.advance(position + 1)
            candidates = features.encode(seat, actions)
            entry = WindowEntry(candidates, tuple(c.kind for c in candidates), _stratum_attributes_of(candidates[0]))
            cache.put(key, entry)
        return resolver.cached_probability(entry, names.index(observed_name), phase, cache)  # type: ignore[union-attr]
    pending: tuple[int, tuple[int, bool]] | None = None
    events = context.events

    def fail(reason: str) -> SeatEvaluation:
        result.violation = reason
        result.log_likelihood = -math.inf
        return result

    for position, event in enumerate(events):
        kind = event["type"]
        if kind == "draw":
            rule.live_draws += 1
            if int(event["seat"]) != seat:
                continue
            raw = int(event["rawEventIndex"])
            tile_id = hypothesis.draws.get(raw)
            if tile_id is None or tile_id in rule.hand:
                raise ValueError("仮説の自摸が欠けているか配牌と重複")
            rule.hand.append(tile_id)
            rule.drawn = tile_id
            discard = events[position + 1]
            observed = {
                "kind": "riichi_discard" if discard.get("riichiDeclaration") else "discard",
                "tile34": int(discard["tile"]["tile34"]),
                "isRed": bool(discard["tile"]["isRed"]),
                "origin": discard["origin"],
            }
            actions, status = rule.self_actions()
            window = WindowRecord("self", position, actions, status, observed)
            result.windows.append(window)
            if status != "known":
                result.holds.append(status)
            if canonical_action(observed) not in {canonical_action(a) for a in actions}:
                return fail(f"observed_self_action_not_legal:{position}")
            if resolver is not None:
                window.probability = window_probability(
                    position, actions, observed, "self_action_after_live", ("drawn", id_key(tile_id))
                )
                result.log_likelihood += _log(window.probability)
        elif kind == "discard":
            discarder = int(event["seat"])
            key = tile_key(event["tile"])
            if discarder == seat:
                # 観測した打牌は合法集合に含まれることを確認済み。どの物理コピーかは尤度に影響しない。
                if event["origin"] == "drawn":
                    removed = rule.drawn
                else:
                    removed = next(i for i in rule.hand if id_key(i) == key and i != rule.drawn)
                rule.hand.remove(removed)
                rule.river34.append(key[0])
                if rule.temporary_furiten and not rule.active_riichi:
                    rule.temporary_furiten = False
                if rule.active_riichi:
                    rule.ippatsu = False
                rule.pending_riichi = bool(event.get("riichiDeclaration"))
                rule.drawn = None
            else:
                actions, status = rule.response_actions(discarder, key)
                window = WindowRecord(
                    "response", position, actions, status, {"kind": "pass"}, furiten=rule.furiten_reasons()
                )
                result.windows.append(window)
                if status != "known":
                    result.holds.append(status)
                if resolver is not None:
                    window.probability = window_probability(
                        position, actions, {"kind": "pass"}, "discard_response",
                        ("response", rule.furiten_reasons(), rule.active_riichi),
                    )
                    result.log_likelihood += _log(window.probability)
            pending = (discarder, key)
        elif kind == "response_resolution":
            assert pending is not None
            rule.pass_resolved(pending[0], pending[1][0])
            pending = None
    if seat == context.riichi_seat and not rule.active_riichi:
        return fail("riichi_not_committed")
    result.riichi_furiten = rule.riichi_furiten
    result.temporary_furiten = rule.temporary_furiten
    result.ippatsu = rule.ippatsu
    result.final_hand = tuple(rule.hand)
    result.river34 = tuple(rule.river34)
    return result


def _log(probability: float) -> float:
    return math.log(probability) if probability > 0 else -math.inf


# ---------------------------------------------------------------------------
# 参照経路：同じ割当をRoundStateと全家のRoundFeatureStateで進める（D33-07）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldAssignment:
    """判断時点までの136枚の配置。参照経路と検査だけで使う。"""

    hypotheses: Mapping[int, SeatHypothesis]
    target_initial: tuple[int, ...]
    target_draws: Mapping[int, int]
    dora_indicator: int
    pool: tuple[int, ...]  # 残りの生牌と王牌（順不同）

    def validate(self) -> None:
        used = [*self.target_initial, *self.target_draws.values(), self.dora_indicator, *self.pool]
        for hypothesis in self.hypotheses.values():
            used.extend(hypothesis.initial)
            used.extend(hypothesis.draws.values())
        if sorted(used) != list(range(136)):
            raise ValueError("割当が136枚の物理牌をちょうど1回ずつ使っていない")


def build_reference_state(context: DecisionContext, world: WorldAssignment) -> RoundState:
    """割当から開始時の136牌状態を作る。生牌は公開された自摸の順、残りをその後ろへ置く。"""
    world.validate()
    hands: list[list[int]] = [[] for _ in range(4)]
    hands[context.target_seat] = list(world.target_initial)
    for seat, hypothesis in world.hypotheses.items():
        hands[seat] = list(hypothesis.initial)
    live_front: list[int] = []
    for event in context.events:
        if event["type"] != "draw":
            continue
        seat, raw = int(event["seat"]), int(event["rawEventIndex"])
        live_front.append(world.target_draws[raw] if seat == context.target_seat else world.hypotheses[seat].draws[raw])
    pool = list(world.pool)
    live_count = LIVE_WALL_SIZE - len(live_front)
    live_wall = [*live_front, *pool[:live_count]]
    rest = pool[live_count:]
    dead_wall = [*rest[:4], world.dora_indicator, *rest[4:13]]
    scores = context.initial["scores"]
    state = RoundState.from_parts(
        hands,
        live_wall,
        dead_wall,
        scores=scores,
        dealer_seat=context.dealer_seat,
        round_wind_tile34=context.round_wind_tile34,
        kyoku=int(context.initial["kyoku"]),
        honba=int(context.initial["honba"]),
        kyoutaku=int(context.initial["riichiSticks"]),
    )
    # 裏ドラは判断時点で未知。和了可否には影響しないので参照させない（replayと同じ扱い）。
    state.ura_indicator_slots = ()
    state.validate()
    return state


def reference_evaluations(
    context: DecisionContext,
    world: WorldAssignment,
    resolver: ModelResolver | None,
) -> dict[int, SeatEvaluation]:
    """RoundStateと全家のRoundFeatureStateで同じ割当を進め、家ごとの評価を作る。"""

    state = build_reference_state(context, world)
    private = {context.target_seat: {**context.target_private_row()}}
    for seat, hypothesis in world.hypotheses.items():
        private[seat] = hypothesis.private_row(context)
    features = RoundFeatureState(context.public_row(), private) if resolver is not None else None
    results = {seat: SeatEvaluation(seat, 0.0) for seat in context.other_seats}
    events = context.events
    for position, event in enumerate(events):
        kind = event["type"]
        try:
            if kind == "draw":
                state.draw()
                seat = int(event["seat"])
                if seat == context.target_seat or results[seat].violation is not None:
                    continue
                discard = events[position + 1]
                observed = {
                    "kind": "riichi_discard" if discard.get("riichiDeclaration") else "discard",
                    "tile34": int(discard["tile"]["tile34"]),
                    "isRed": bool(discard["tile"]["isRed"]),
                    "origin": discard["origin"],
                }
                actions, status, reason = semantic_self_actions(state)
                status = "known" if status == "known" else f"hold:{reason}"
                window = WindowRecord("self", position, actions, status, observed)
                results[seat].windows.append(window)
                if status != "known":
                    results[seat].holds.append(status)
                if canonical_action(observed) not in {canonical_action(a) for a in actions}:
                    results[seat].violation = f"observed_self_action_not_legal:{position}"
                    results[seat].log_likelihood = -math.inf
                elif features is not None:
                    features.advance(position + 1)
                    window.probability = _window_probability(
                        features, seat, actions, observed, "self_action_after_live", resolver  # type: ignore[arg-type]
                    )
                    results[seat].log_likelihood += _log(window.probability)
            elif kind == "discard":
                seat = int(event["seat"])
                key = tile_key(event["tile"])
                if event["origin"] == "drawn":
                    tile_id = state.drawn_tile_id
                else:
                    tile_id = next(
                        (i for i in state.hands[seat] if id_key(i) == key and i != state.drawn_tile_id), None
                    )
                if tile_id is None or id_key(tile_id) != key:
                    # 違反した家はそれ以降比較しない。局の進行だけを続けるため同じ牌種の牌を探す。
                    raise _ReferenceViolation(seat, f"observed_self_action_not_legal:{position - 1}")
                state.discard(tile_id, declare_riichi=bool(event.get("riichiDeclaration")))
                for other in context.other_seats:
                    if other == seat or results[other].violation is not None:
                        continue
                    actions, status, reason = semantic_response_actions(state, other)
                    status = "known" if status == "known" else f"hold:{reason}"
                    window = WindowRecord(
                        "response", position, actions, status, {"kind": "pass"}, furiten=state.furiten_reasons(other)
                    )
                    results[other].windows.append(window)
                    if status != "known":
                        results[other].holds.append(status)
                    if features is not None:
                        features.advance(position + 1)
                        window.probability = _window_probability(
                            features, other, actions, {"kind": "pass"}, "discard_response", resolver  # type: ignore[arg-type]
                        )
                        results[other].log_likelihood += _log(window.probability)
            elif kind == "response_resolution":
                state.resolve_discard_responses([])
        except _ReferenceViolation as violation:
            results[violation.seat].violation = violation.reason
            results[violation.seat].log_likelihood = -math.inf
            return _stop_reference(results)
        except IllegalAction:
            seat = int(event["seat"])
            results[seat].violation = f"observed_self_action_not_legal:{position - 1}"
            results[seat].log_likelihood = -math.inf
            return _stop_reference(results)
    for seat in context.other_seats:
        if results[seat].violation is None:
            results[seat].riichi_furiten = seat in state.riichi_furiten
            results[seat].temporary_furiten = seat in state.temporary_furiten
            if seat == context.riichi_seat and seat not in state.active_riichi:
                results[seat].violation = "riichi_not_committed"
                results[seat].log_likelihood = -math.inf
    return results


class _ReferenceViolation(Exception):
    def __init__(self, seat: int, reason: str):
        super().__init__(reason)
        self.seat = seat
        self.reason = reason


def _stop_reference(results: dict[int, SeatEvaluation]) -> dict[int, SeatEvaluation]:
    """局の進行が止まった後の家は比較できないので、その旨を記録する。"""
    for evaluation in results.values():
        if evaluation.violation is None:
            evaluation.violation = "reference_stopped_by_other_seat"
    return results


# ---------------------------------------------------------------------------
# 検査用の割当生成（設計6節の逆算。受入試験と照合probeだけで使う）
# ---------------------------------------------------------------------------


def _backward_history(
    context: DecisionContext,
    seat: int,
    final_hand: Sequence[int],
    discard_ids: Mapping[int, int],
    rng: random.Random,
) -> SeatHypothesis:
    """判断時点の手から時間を遡り、自摸と配牌を作る（手出し窓の自摸は後の手から一様に選ぶ）。"""
    hand = list(final_hand)
    draws: dict[int, int] = {}
    for turn in reversed(context.turns[seat]):
        discard_id = discard_ids[turn.raw_event_index]
        if turn.tsumogiri:
            draws[turn.raw_event_index] = discard_id
            continue
        drawn = rng.choice(hand)
        hand.remove(drawn)
        hand.append(discard_id)
        draws[turn.raw_event_index] = drawn
    return SeatHypothesis(seat, tuple(sorted(hand)), draws)


def _take(available: list[int], key: tuple[int, bool], rng: random.Random) -> int:
    choices = [tile_id for tile_id in available if id_key(tile_id) == key]
    if not choices:
        raise ValueError(f"牌在庫が足りない: {key}")
    chosen = rng.choice(choices)
    available.remove(chosen)
    return chosen


def _random_tenpai_hand(available: list[int], rng: random.Random, attempts: int = 20_000) -> list[int]:
    """在庫から面子手または七対子のテンパイ13枚を無作為に作る（検査用。分布は問わない）。"""
    for _ in range(attempts):
        counts = Counter(TILE_BY_ID[i].tile34 for i in available)
        picked: list[int] = []
        if rng.random() < 0.9:
            pair = rng.randrange(34)
            groups = [(pair, pair)]
            for _ in range(4):
                if rng.random() < 0.3:
                    t = rng.randrange(34)
                    groups.append((t, t, t))
                else:
                    suit, start = rng.randrange(3), rng.randrange(7)
                    groups.append(tuple(suit * 9 + start + k for k in range(3)))
            types = [t for group in groups for t in group]
        else:
            types = [t for t in rng.sample(range(34), 7) for _ in range(2)]
        types.remove(rng.choice(types))
        need = Counter(types)
        if any(counts[t] < n for t, n in need.items()):
            continue
        pool = list(available)
        for t in types:
            chosen = rng.choice([i for i in pool if TILE_BY_ID[i].tile34 == t])
            pool.remove(chosen)
            picked.append(chosen)
        if shanten(counts34(picked), 0) == 0:
            for i in picked:
                available.remove(i)
            return picked
    raise ValueError("テンパイ形を作れない")


def random_world(context: DecisionContext, rng: random.Random, *, riichi_tenpai: bool = True) -> WorldAssignment:
    """公開履歴と整合する割当を無作為に作る（打牌整合は常に満たす）。

    riichi_tenpai=Falseなら、リーチ者の宣言時手牌をテンパイに限らず一様に選ぶ（H=0の例を作る）。
    """
    available = list(range(136))
    target_initial = [_take(available, key, rng) for key in context.target_initial]
    target_draws = {raw: _take(available, key, rng) for raw, key in sorted(context.target_draws.items())}
    dora = _take(available, context.dora_indicator, rng)
    discard_ids: dict[int, dict[int, int]] = {}
    for seat in context.other_seats:
        discard_ids[seat] = {turn.raw_event_index: _take(available, turn.discard, rng) for turn in context.turns[seat]}
    hypotheses = {}
    order = [context.riichi_seat, *[s for s in context.other_seats if s != context.riichi_seat]]
    for seat in order:
        if seat == context.riichi_seat and riichi_tenpai:
            final = _random_tenpai_hand(available, rng)
        else:
            final = rng.sample(available, 13)
            # H=0の例では、リーチ者の宣言時手牌が偶然テンパイになった割当を引き直す。
            while seat == context.riichi_seat and shanten(counts34(final), 0) == 0:
                final = rng.sample(available, 13)
            for i in final:
                available.remove(i)
        hypotheses[seat] = _backward_history(context, seat, final, discard_ids[seat], rng)
    rng.shuffle(available)
    return WorldAssignment(hypotheses, tuple(target_initial), target_draws, dora, tuple(available))


def world_from_teacher(
    context: DecisionContext, private_rows: Mapping[int, Mapping[str, Any]], rng: random.Random
) -> WorldAssignment:
    """実際の手牌（教師情報）から割当を作る。推定には使わず、教師窓との照合だけに使う。"""
    available = list(range(136))
    target_initial = [_take(available, key, rng) for key in context.target_initial]
    target_draws = {raw: _take(available, key, rng) for raw, key in sorted(context.target_draws.items())}
    dora = _take(available, context.dora_indicator, rng)
    hypotheses = {}
    for seat in context.other_seats:
        row = private_rows[seat]
        initial = tuple(_take(available, tile_key(tile), rng) for tile in row["initialHand"])
        observed = {int(item["rawEventIndex"]): tile_key(item["tile"]) for item in row["events"] if item["type"] == "draw_observation"}
        draws = {turn.raw_event_index: _take(available, observed[turn.raw_event_index], rng) for turn in context.turns[seat]}
        hypotheses[seat] = SeatHypothesis(seat, initial, draws)
    rng.shuffle(available)
    return WorldAssignment(hypotheses, tuple(target_initial), target_draws, dora, tuple(available))


# ---------------------------------------------------------------------------
# 照合
# ---------------------------------------------------------------------------


def compare_evaluations(fast: SeatEvaluation, reference: SeatEvaluation, *, tolerance: float = 1e-9) -> list[str]:
    """評価器と参照経路の差分を文字列で返す（空なら一致）。"""
    if reference.violation == "reference_stopped_by_other_seat":
        return []
    differences = []
    if (fast.violation is None) != (reference.violation is None):
        differences.append(f"violation {fast.violation!r} != {reference.violation!r}")
        return differences
    if fast.violation is not None:
        return differences
    if len(fast.windows) != len(reference.windows):
        return [f"window count {len(fast.windows)} != {len(reference.windows)}"]
    for mine, theirs in zip(fast.windows, reference.windows):
        where = f"{mine.kind}@{mine.event_position}"
        if (mine.kind, mine.event_position) != (theirs.kind, theirs.event_position):
            differences.append(f"window order {where} != {theirs.kind}@{theirs.event_position}")
            continue
        if [canonical_action(a) for a in mine.legal_actions] != [canonical_action(a) for a in theirs.legal_actions]:
            differences.append(f"legal {where}: {mine.legal_actions} != {theirs.legal_actions}")
        if mine.win_action_status != theirs.win_action_status:
            differences.append(f"win status {where}: {mine.win_action_status} != {theirs.win_action_status}")
        if mine.furiten != theirs.furiten:
            differences.append(f"furiten {where}: {mine.furiten} != {theirs.furiten}")
        if (mine.probability is None) != (theirs.probability is None) or (
            mine.probability is not None and abs(mine.probability - theirs.probability) > tolerance
        ):
            differences.append(f"probability {where}: {mine.probability} != {theirs.probability}")
    if (fast.riichi_furiten, fast.temporary_furiten) != (reference.riichi_furiten, reference.temporary_furiten):
        differences.append("final furiten state differs")
    if math.isfinite(fast.log_likelihood) != math.isfinite(reference.log_likelihood) or (
        math.isfinite(fast.log_likelihood) and abs(fast.log_likelihood - reference.log_likelihood) > tolerance
    ):
        differences.append(f"log likelihood {fast.log_likelihood} != {reference.log_likelihood}")
    return differences


def teacher_window_differences(
    evaluation: SeatEvaluation, teacher_windows: Sequence[Mapping[str, Any]], decision_count: int
) -> list[str]:
    """実際の手牌での評価と、D.3.1の教師窓（学習データ）の合法集合と位置を比べる。"""
    mine = {(w.kind, w.event_position + 1): [canonical_action(a) for a in w.legal_actions] for w in evaluation.windows}
    theirs: dict[tuple[str, int], list[str]] = {}
    seat = evaluation.seat
    for window in teacher_windows:
        count = int(window["publicEventCount"])
        if count > decision_count:
            continue
        if "legalActions" in window:
            if int(window["actorSeat"]) == seat and window["phase"] == "self_action_after_live":
                theirs[("self", count)] = [canonical_action(a) for a in window["legalActions"]]
        elif str(seat) in window.get("legalBySeat", {}) and window["phase"] == "discard_response":
            theirs[("response", count)] = [canonical_action(a) for a in window["legalBySeat"][str(seat)]["actions"]]
    differences = [f"window only in one side: {key}" for key in sorted(set(mine) ^ set(theirs))]
    differences.extend(f"legal differs at {key}" for key in sorted(set(mine) & set(theirs)) if mine[key] != theirs[key])
    return differences


# ---------------------------------------------------------------------------
# 照合probe（受入試験D33-07の1,000割当。calibrate_ev probe-policy-belief から呼ぶ）
# ---------------------------------------------------------------------------


def _iter_gzip_jsonl(path: Any) -> Any:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_seat_evaluator(
    dataset_dir: Any,
    model_dir: Any,
    output_dir: Any,
    *,
    decisions: int = 100,
    assignments_per_decision: int = 10,
    violations_per_decision: int = 1,
    seed: int = 20260929,
) -> dict[str, Any]:
    """評価器と参照経路（RoundState）を、固定seedの無作為な割当で照合する（D33-07）。

    学習期間の判断から無作為に選び、各判断で整合する割当（H=1）とリーチ者を非テンパイにした
    割当（H=0の例）を作る。実際の手牌では、評価器の窓がD.3.1の教師窓と一致することも確かめる。
    """
    import time
    from pathlib import Path

    dataset_dir, model_dir, output_dir = Path(dataset_dir), Path(model_dir), Path(output_dir)
    rng = random.Random(seed)
    train = [row for row in _iter_gzip_jsonl(dataset_dir / "inference-prefixes.jsonl.gz") if row["developmentSplit"] == "train"]
    selected = rng.sample(train, decisions)
    rounds = {row["roundId"] for row in selected}
    public = {row["roundId"]: row for row in _iter_gzip_jsonl(dataset_dir / "public-events.jsonl.gz") if row["roundId"] in rounds}
    private: dict[str, dict[int, Any]] = {}
    for row in _iter_gzip_jsonl(dataset_dir / "private-events.jsonl.gz"):
        if row["roundId"] in rounds:
            private.setdefault(row["roundId"], {})[int(row["seat"])] = row
    teacher: dict[str, list[Any]] = {}
    for row in _iter_gzip_jsonl(dataset_dir / "teacher-windows.jsonl.gz"):
        if row["roundId"] in rounds:
            teacher.setdefault(row["roundId"], []).append(row)

    model = HierarchicalSoftmax.from_dict(json.loads((model_dir / "model.json").read_text(encoding="utf-8")))
    components = json.loads((model_dir / "fixed-components.json").read_text(encoding="utf-8"))
    scenarios = [s for s in components["scenarios"] if s.get("usableInD33")]
    base = next(s for s in scenarios if s["id"] == "base")
    stratified = next(s for s in scenarios if s["stratumOverrides"])
    resolvers = [(base["id"], model_resolver(model, base)), (stratified["id"], model_resolver(model, stratified))]

    coverage: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    fast_seconds = reference_seconds = 0.0
    compared = violation_cases = violation_agreements = teacher_seats = teacher_mismatch = 0
    for index, prefix in enumerate(selected):
        round_id = prefix["roundId"]
        seat = int(prefix["seat"])
        context = DecisionContext.from_records(prefix, public[round_id], private[round_id][seat])
        count = int(prefix["publicEventCount"])
        truth = world_from_teacher(context, private[round_id], rng)
        for other, hypothesis in truth.hypotheses.items():
            evaluation = evaluate_seat(context, hypothesis, None)
            teacher_seats += 1
            differences = ["teacher_hand_violation:" + str(evaluation.violation)] if evaluation.violation else []
            differences += teacher_window_differences(evaluation, teacher[round_id], count)
            if differences:
                teacher_mismatch += 1
                mismatches.append({"decisionId": prefix["decisionId"], "seat": other, "check": "teacher", "differences": differences[:5]})
        for number in range(assignments_per_decision + violations_per_decision):
            is_violation_case = number >= assignments_per_decision
            world = random_world(context, rng, riichi_tenpai=not is_violation_case)
            scenario_id, resolver = resolvers[number % len(resolvers)]
            start = time.perf_counter()
            reference = reference_evaluations(context, world, resolver)
            reference_seconds += time.perf_counter() - start
            for other, hypothesis in world.hypotheses.items():
                start = time.perf_counter()
                fast = evaluate_seat(context, hypothesis, resolver)
                fast_seconds += time.perf_counter() - start
                differences = compare_evaluations(fast, reference[other])
                if is_violation_case and other == context.riichi_seat:
                    violation_cases += 1
                    violation_agreements += int(fast.violation is not None and reference[other].violation is not None)
                if not is_violation_case:
                    compared += 1
                    if not math.isfinite(fast.log_likelihood):
                        differences.append("non_finite_log_likelihood_for_consistent_assignment")
                    for window in fast.windows:
                        for kind in {a["kind"] for a in window.legal_actions}:
                            coverage[f"legal_{kind}"] += 1
                        for reason in window.furiten:
                            coverage[f"furiten_{reason}"] += 1
                        if window.win_action_status != "known":
                            coverage["win_action_hold"] += 1
                if differences:
                    mismatches.append(
                        {"decisionId": prefix["decisionId"], "seat": other, "assignment": number, "scenario": scenario_id, "differences": differences[:5]}
                    )
    assignments = decisions * assignments_per_decision
    status = "pass" if not mismatches and violation_agreements == violation_cases else "fail"
    report = {
        "schemaVersion": "ev-policy-belief-evaluator-check/v1",
        "evaluatorVersion": BELIEF_EVALUATOR_VERSION,
        "status": status,
        "seed": seed,
        "split": "train",
        "decisions": decisions,
        "consistentAssignments": assignments,
        "seatComparisons": compared,
        "violationCases": violation_cases,
        "violationAgreements": violation_agreements,
        "teacherSeatsChecked": teacher_seats,
        "teacherMismatches": teacher_mismatch,
        "mismatchCount": len(mismatches),
        "mismatches": mismatches[:50],
        "scenarios": [scenario_id for scenario_id, _ in resolvers],
        "coverage": dict(sorted(coverage.items())),
        "seconds": {"fastEvaluator": round(fast_seconds, 3), "reference": round(reference_seconds, 3)},
        "inputs": {
            "datasetSha256": {
                name: _sha256_file(dataset_dir / name)
                for name in ("inference-prefixes.jsonl.gz", "public-events.jsonl.gz", "private-events.jsonl.gz", "teacher-windows.jsonl.gz")
            },
            "modelSha256": _sha256_file(model_dir / "model.json"),
            "fixedComponentsSha256": _sha256_file(model_dir / "fixed-components.json"),
        },
        "decisionIds": [row["decisionId"] for row in selected],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluator-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


# ===========================================================================
# 工程2：基準SMC（上位設計7.1〜7.3節、PHASE_D33_DESIGN.md 3.2節）
# ===========================================================================
#
# 一段先の提案：各イベントeで、粒子ごとに取り得る選択を全列挙し、
#   K_e(選択) = 抽出確率 × 整合指示子 × 行動確率
#   G_e = Σ K_e、  提案 q_e = K_e / G_e、  重み w_e = w_(e-1) × G_e
# とする。G_e=0の粒子は重み0、全粒子0なら posterior_zero_mass で数値を返さない。
# SMC本体はイベントの中身を知らず、「粒子から選択肢と対数項を列挙する」関数だけを使う。
# これにより、全列挙できる小例（D33-09）と麻雀の局で同じ本体を検査できる。


def _logsumexp(values: Sequence[float]) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    top = max(finite)
    return top + math.log(math.fsum(math.exp(value - top) for value in finite))


class SmcEvent:
    """SMCの1イベント。expandは(対数項, 選択)の列、applyは選んだ選択を粒子へ反映する。"""

    name = "event"

    def expand(self, particle: Any) -> list[tuple[float, Any]]:
        raise NotImplementedError

    def apply(self, particle: Any, choice: Any) -> None:
        raise NotImplementedError


@dataclass
class SmcResult:
    status: str  # "ok" または "posterior_zero_mass"
    log_normalizer: float | None
    particles: list[Any]
    log_weights: list[float]
    initial_ancestors: list[int]
    event_diagnostics: list[dict[str, Any]]
    zero_mass_event: dict[str, Any] | None = None
    resampling_count: int = 0

    def normalized_weights(self) -> list[float]:
        return _normalize(self.log_weights)


def _normalize(log_weights: Sequence[float]) -> list[float]:
    total = _logsumexp(log_weights)
    if total == -math.inf:
        return [0.0] * len(log_weights)
    return [math.exp(value - total) if value != -math.inf else 0.0 for value in log_weights]


def _ess(normalized: Sequence[float]) -> float:
    denominator = math.fsum(value * value for value in normalized)
    return 1.0 / denominator if denominator > 0 else 0.0


def _systematic_resample(normalized: Sequence[float], rng: random.Random) -> list[int]:
    count = len(normalized)
    offset = rng.random() / count
    indices = []
    cumulative = normalized[0]
    index = 0
    for step in range(count):
        position = offset + step / count
        while cumulative < position and index < count - 1:
            index += 1
            cumulative += normalized[index]
        indices.append(index)
    return indices


def _sample_index(log_terms: Sequence[float], log_total: float, rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    last_positive = 0
    for index, value in enumerate(log_terms):
        if value == -math.inf:
            continue
        cumulative += math.exp(value - log_total)
        last_positive = index
        if threshold < cumulative:
            return index
    return last_positive


def run_smc(
    particles: list[Any],
    events: Sequence[SmcEvent],
    rng: random.Random,
    *,
    ess_fraction: float = 0.5,
    copy_particle: Callable[[Any], Any] = lambda particle: particle.copy(),
    on_event: Callable[[int, SmcEvent, list[Any], list[float]], None] | None = None,
) -> SmcResult:
    """一段先の提案による基準SMC（上位設計7.1節）。初期粒子はp0から引く（初期重みは一様）。

    各イベントの前にESSがN×ess_fractionを下回っていればsystematic resamplingを行い、
    その直前のESSを記録する。正規化定数は各段の Σ W_(e-1) G_e の積の対数和で推定する。
    """

    count = len(particles)
    log_weights = [0.0] * count
    initial = list(range(count))
    log_normalizer = 0.0
    diagnostics: list[dict[str, Any]] = []
    resamplings = 0
    for number, event in enumerate(events):
        normalized_before = _normalize(log_weights)
        ess_before = _ess(normalized_before)
        resampled = False
        previous = list(range(count))
        if ess_before < ess_fraction * count:
            chosen = _systematic_resample(normalized_before, rng)
            particles = [copy_particle(particles[i]) for i in chosen]
            initial = [initial[i] for i in chosen]
            previous = list(chosen)
            log_weights = [0.0] * count
            normalized_before = [1.0 / count] * count
            resamplings += 1
            resampled = True
        expansions = [
            event.expand(particle) if weight > 0 else [] for particle, weight in zip(particles, normalized_before)
        ]
        log_g = [_logsumexp([term for term, _ in terms]) for terms in expansions]
        increment = _logsumexp([
            math.log(w) + g for w, g in zip(normalized_before, log_g) if w > 0 and g != -math.inf
        ])
        record: dict[str, Any] = {
            "event": number,
            "name": event.name,
            "essBefore": ess_before,
            "resampled": resampled,
            "aliveBefore": sum(w > 0 for w in normalized_before),
            "positiveG": sum(g != -math.inf for g, w in zip(log_g, normalized_before) if w > 0),
        }
        if increment == -math.inf:
            record["zeroMass"] = True
            diagnostics.append(record)
            return SmcResult(
                "posterior_zero_mass", None, particles, [-math.inf] * count, initial, diagnostics,
                zero_mass_event={"event": number, "name": event.name}, resampling_count=resamplings,
            )
        log_normalizer += increment
        for index, (terms, g) in enumerate(zip(expansions, log_g)):
            log_weights[index] = log_weights[index] + g if normalized_before[index] > 0 else -math.inf
            if g == -math.inf:
                continue
            choice = terms[_sample_index([term for term, _ in terms], g, rng)][1]
            event.apply(particles[index], choice)
        normalized = _normalize(log_weights)
        alive = [i for i, w in enumerate(normalized) if w > 0]
        record.update({
            "essAfter": _ess(normalized),
            "maxWeight": max(normalized),
            "initialAncestors": len({initial[i] for i in alive}),
            "previousAncestors": len({previous[i] for i in alive}),
        })
        diagnostics.append(record)
        if on_event is not None:
            on_event(number, event, particles, log_weights)
    return SmcResult("ok", log_normalizer, particles, log_weights, initial, diagnostics, resampling_count=resamplings)


# ---------------------------------------------------------------------------
# 有限小例（D33-09）：全列挙で正確な事後分布と正規化定数を作れる縮小規則
# ---------------------------------------------------------------------------
#
# 牌種3種×3枚の9枚。相手の手牌は2枚。対象家は既知の1枚を持つ。
# 観測列：相手の自摸＋打牌（牌種、手出し/ツモ切り）、対象家の既知の自摸、相手の自摸＋「リーチ」打牌。
# リーチ打牌は、打牌後の2枚が対子であることを硬い制約とする。
# 相手の打牌の確率は、合法な（牌種、origin）の組の上の固定スコアのsoftmax。


TOY_SCORES = {(0, "drawn"): 0.3, (0, "concealed"): -0.2, (1, "drawn"): 0.5, (1, "concealed"): 0.1,
              (2, "drawn"): -0.4, (2, "concealed"): 0.6}


def toy_discard_probability(hand: Sequence[int], observed: tuple[int, str], riichi: bool) -> float:
    """手牌（牌種の列。最後が今回の自摸）で観測打牌を選ぶ確率。非合法なら0。"""
    options = set()
    for index, tile in enumerate(hand):
        origin = "drawn" if index == len(hand) - 1 else "concealed"
        rest = [t for j, t in enumerate(hand) if j != index]
        if riichi and not (len(rest) == 2 and rest[0] == rest[1]):
            continue
        options.add((tile, origin))
    if observed not in options:
        return 0.0
    weights = {option: math.exp(TOY_SCORES[option]) for option in options}
    return weights[observed] / math.fsum(weights.values())


@dataclass
class ToyParticle:
    pool: Counter
    hand: list[int]

    def copy(self) -> "ToyParticle":
        return ToyParticle(Counter(self.pool), list(self.hand))


class ToyDrawDiscard(SmcEvent):
    """相手の未知の自摸と、直後の観測打牌をまとめた一イベント。"""

    def __init__(self, discard: int, origin: str, riichi: bool):
        self.name = f"toy_draw_discard:{discard}:{origin}:{riichi}"
        self.discard, self.origin, self.riichi = discard, origin, riichi

    def expand(self, particle: ToyParticle) -> list[tuple[float, Any]]:
        total = sum(particle.pool.values())
        terms = []
        for tile, count in sorted(particle.pool.items()):
            if count <= 0:
                continue
            probability = toy_discard_probability([*particle.hand, tile], (self.discard, self.origin), self.riichi)
            if probability > 0:
                terms.append((math.log(count / total) + math.log(probability), tile))
        return terms

    def apply(self, particle: ToyParticle, tile: int) -> None:
        particle.pool[tile] -= 1
        if self.origin == "drawn":
            return
        particle.hand.remove(self.discard)  # 手出し：自摸牌とは別の、手中の同じ牌種を切る
        particle.hand.append(tile)


class ToyKnownDraw(SmcEvent):
    """対象家の既知の自摸。残数比n/Nを重みに残す（上位設計7.1節）。"""

    name = "toy_target_known_draw"

    def __init__(self, tile: int):
        self.tile = tile

    def expand(self, particle: ToyParticle) -> list[tuple[float, Any]]:
        total = sum(particle.pool.values())
        count = particle.pool[self.tile]
        return [(math.log(count / total), None)] if count > 0 else []

    def apply(self, particle: ToyParticle, choice: Any) -> None:
        particle.pool[self.tile] -= 1


def toy_problem(target_initial: int, observations: Sequence[tuple[str, Any]]) -> tuple[Counter, list[SmcEvent]]:
    """対象家の既知の1枚を除いた初期プールとイベント列を作る。"""
    pool = Counter({0: 3, 1: 3, 2: 3})
    pool[target_initial] -= 1
    events: list[SmcEvent] = []
    for kind, value in observations:
        events.append(ToyKnownDraw(value) if kind == "target_draw" else ToyDrawDiscard(*value))
    return pool, events


def toy_initial_particles(pool: Counter, count: int, rng: random.Random) -> list[ToyParticle]:
    """p0：残りの物理牌から相手の2枚を一様に非復元抽出する。"""
    tiles = [tile for tile, n in sorted(pool.items()) for _ in range(n)]
    particles = []
    for _ in range(count):
        hand = rng.sample(tiles, 2)
        rest = Counter(pool)
        rest.subtract(hand)
        particles.append(ToyParticle(rest, list(hand)))
    return particles


def toy_exact(pool: Counter, events: Sequence[SmcEvent]) -> tuple[float, dict[tuple[int, ...], float]]:
    """全列挙：正規化定数（観測の確率）と、最終手牌（牌種の昇順）の事後分布。

    配牌は物理牌の順序付き2枚、自摸は各段の残数比で分岐する。SMCとは独立に、
    全経路の確率を直接足し合わせる。
    """
    tiles = [tile for tile, n in sorted(pool.items()) for _ in range(n)]
    total = 0.0
    posterior: dict[tuple[int, ...], float] = {}
    for first in range(len(tiles)):
        for second in range(len(tiles)):
            if first == second:
                continue
            start = ToyParticle(Counter(pool), [tiles[first], tiles[second]])
            start.pool.subtract(start.hand)
            stack = [(start, 1.0 / (len(tiles) * (len(tiles) - 1)))]
            for event in events:
                following = []
                for current, mass in stack:
                    for log_term, choice in event.expand(current):
                        nxt = current.copy()
                        event.apply(nxt, choice)
                        following.append((nxt, mass * math.exp(log_term)))
                stack = following
            for current, mass in stack:
                total += mass
                key = tuple(sorted(current.hand))
                posterior[key] = posterior.get(key, 0.0) + mass
    return total, ({key: value / total for key, value in posterior.items()} if total > 0 else {})


# ---------------------------------------------------------------------------
# 麻雀の局への適用（遅延割当。未割当の物理牌を牌種ごとに持ち、必要な位置へ残数比で割り当てる）
# ---------------------------------------------------------------------------


class PublicFeatureState(RoundFeatureState):
    """公開情報だけを追う特徴状態。encode_forでその家の手牌を差し込んでから特徴を作る。

    特徴は行動する家の手牌と公開情報しか読まないので、全粒子で1つの状態を共有できる。
    """

    def __init__(self, public: Mapping[str, Any]):
        super().__init__(public, {seat: {"initialHand": [], "events": []} for seat in range(4)})
        self.draws = _PlaceholderDraws(self.draws)

    def _remove(self, seat: int, tile: Mapping[str, Any]) -> None:
        return

    def encode_for(self, seat: int, hand_ids: Sequence[int], actions: Sequence[Mapping[str, Any]]) -> list[Any]:
        self.hands[seat] = [_record(tile_id) for tile_id in hand_ids]
        return self.encode(seat, actions)


@dataclass
class MahjongParticle:
    """他家3人の手牌と規則の状態、未割当の物理牌、経路の記録。"""

    rules: dict[int, _SeatRuleState]
    pool: dict[tuple[int, bool], list[int]]
    pool_size: int
    initial: dict[int, tuple[int, ...]]
    draws: dict[int, dict[int, int]]
    log_model: float = 0.0  # 選んだ経路の行動確率の対数和（評価器との照合用）

    def copy(self) -> "MahjongParticle":
        rules = {
            seat: _SeatRuleState(
                rule.context, rule.seat, list(rule.hand), rule.drawn, rule.live_draws, list(rule.river34),
                rule.active_riichi, rule.double_riichi, rule.riichi_base_counts, rule.pending_riichi,
                rule.ippatsu, rule.temporary_furiten, rule.riichi_furiten,
            )
            for seat, rule in self.rules.items()
        }
        return MahjongParticle(
            rules, {key: list(ids) for key, ids in self.pool.items()}, self.pool_size,
            dict(self.initial), {seat: dict(values) for seat, values in self.draws.items()}, self.log_model,
        )

    def take(self, key: tuple[int, bool]) -> int:
        self.pool_size -= 1
        return self.pool[key].pop()

    def hypothesis(self, seat: int) -> SeatHypothesis:
        return SeatHypothesis(seat, self.initial[seat], dict(self.draws[seat]))


def _rule_signature(rule: _SeatRuleState) -> tuple:
    return (
        tuple(sorted(id_key(i) for i in rule.hand)), id_key(rule.drawn) if rule.drawn is not None else None,
        tuple(rule.river34), rule.active_riichi, rule.double_riichi, rule.riichi_base_counts, rule.ippatsu,
        rule.temporary_furiten, rule.riichi_furiten,
    )


@dataclass
class _EventStatistics:
    win_action_holds: int = 0
    probability_cache_hits: int = 0
    probability_cache_misses: int = 0


class _TargetDrawEvent(SmcEvent):
    """対象家の既知の自摸。残数比n/Nを重みに残す（上位設計7.1節）。"""

    def __init__(self, key: tuple[int, bool], position: int):
        self.key = key
        self.name = f"target_draw@{position}"

    def expand(self, particle: MahjongParticle) -> list[tuple[float, Any]]:
        count = len(particle.pool.get(self.key, ()))
        return [(math.log(count / particle.pool_size), None)] if count else []

    def apply(self, particle: MahjongParticle, choice: Any) -> None:
        particle.take(self.key)


class _SeatDrawDiscardEvent(SmcEvent):
    """他家の未知の自摸と直後の観測打牌。自摸の牌種を全列挙する（上位設計7.1節の一イベント）。"""

    def __init__(self, context: DecisionContext, turn: SeatTurn, seat: int, live_draws: int, discard: Mapping[str, Any],
                 features: PublicFeatureState, resolver: ModelResolver, statistics: _EventStatistics):
        self.context, self.turn, self.seat, self.live_draws = context, turn, seat, live_draws
        self.observed = {
            "kind": "riichi_discard" if discard.get("riichiDeclaration") else "discard",
            "tile34": int(discard["tile"]["tile34"]),
            "isRed": bool(discard["tile"]["isRed"]),
            "origin": discard["origin"],
        }
        kind = "riichi_declaration" if turn.riichi_declaration else "seat_draw_discard"
        self.name = f"{kind}:seat{seat}@{turn.event_position}"
        self.features, self.resolver, self.statistics = features, resolver, statistics
        self.cache: dict[tuple, float] = {}

    def _probability(self, rule: _SeatRuleState) -> float:
        signature = _rule_signature(rule)
        cached = self.cache.get(signature)
        if cached is not None:
            self.statistics.probability_cache_hits += 1
            return cached
        self.statistics.probability_cache_misses += 1
        actions, status = rule.self_actions()
        if status != "known":
            self.statistics.win_action_holds += 1
        if canonical_action(self.observed) not in {canonical_action(a) for a in actions}:
            probability = 0.0
        elif len(actions) == 1:
            probability = 1.0
        else:
            self.features.advance(self.turn.event_position + 1)
            candidates = self.features.encode_for(self.seat, rule.hand, actions)
            probability = self.resolver(candidates).action_probability(candidates, self.observed, "self_action_after_live")
        self.cache[signature] = probability
        return probability

    def expand(self, particle: MahjongParticle) -> list[tuple[float, Any]]:
        rule = particle.rules[self.seat]
        rule.live_draws = self.live_draws
        terms = []
        for key, ids in particle.pool.items():
            if not ids or (self.turn.tsumogiri and key != self.turn.discard):
                continue
            rule.hand.append(ids[-1])
            rule.drawn = ids[-1]
            probability = self._probability(rule)
            rule.hand.pop()
            rule.drawn = None
            if probability > 0:
                terms.append((math.log(len(ids) / particle.pool_size) + math.log(probability), (key, math.log(probability))))
        return terms

    def apply(self, particle: MahjongParticle, choice: tuple[tuple[int, bool], float]) -> None:
        key, log_probability = choice
        rule = particle.rules[self.seat]
        tile_id = particle.take(key)
        particle.draws[self.seat][self.turn.raw_event_index] = tile_id
        particle.log_model += log_probability
        rule.hand.append(tile_id)
        # 打牌（evaluate_seatの打牌処理と同じ）。どの物理コピーを切ったかは尤度に影響しない。
        if self.turn.tsumogiri:
            removed = tile_id
        else:
            removed = next(i for i in rule.hand if id_key(i) == self.turn.discard and i != tile_id)
        rule.hand.remove(removed)
        rule.river34.append(self.turn.discard[0])
        if rule.temporary_furiten and not rule.active_riichi:
            rule.temporary_furiten = False
        if rule.active_riichi:
            rule.ippatsu = False
        rule.pending_riichi = self.turn.riichi_declaration
        rule.drawn = None


class _ResponseEvent(SmcEvent):
    """捨牌への他家の応答。公開結果は全員パスなので、各家のpass確率の積を重みに掛ける。"""

    def __init__(self, context: DecisionContext, discarder: int, key: tuple[int, bool], position: int, live_draws: int,
                 features: PublicFeatureState, resolver: ModelResolver, statistics: _EventStatistics):
        self.context, self.discarder, self.key, self.position, self.live_draws = context, discarder, key, position, live_draws
        self.features, self.resolver, self.statistics = features, resolver, statistics
        self.name = f"response:discarder{discarder}@{position}"
        self.cache: dict[tuple, float] = {}

    def _pass_probability(self, rule: _SeatRuleState) -> float:
        signature = (rule.seat, _rule_signature(rule))
        cached = self.cache.get(signature)
        if cached is not None:
            self.statistics.probability_cache_hits += 1
            return cached
        self.statistics.probability_cache_misses += 1
        actions, status = rule.response_actions(self.discarder, self.key)
        if status != "known":
            self.statistics.win_action_holds += 1
        if len(actions) == 1:
            probability = 1.0
        else:
            self.features.advance(self.position + 1)
            candidates = self.features.encode_for(rule.seat, rule.hand, actions)
            probability = self.resolver(candidates).action_probability(candidates, {"kind": "pass"}, "discard_response")
        self.cache[signature] = probability
        return probability

    def expand(self, particle: MahjongParticle) -> list[tuple[float, Any]]:
        total = 0.0
        for seat, rule in particle.rules.items():
            if seat == self.discarder:
                continue
            rule.live_draws = self.live_draws
            probability = self._pass_probability(rule)
            if probability <= 0:
                return []
            total += math.log(probability)
        return [(total, total)]

    def apply(self, particle: MahjongParticle, choice: float) -> None:
        particle.log_model += choice


class _ResolutionEvent(SmcEvent):
    """全員パスの解決。見逃しフリテンと、打牌者のリーチ成立を更新する（重みは変えない）。"""

    def __init__(self, discarder: int, tile34: int, position: int):
        self.discarder, self.tile34 = discarder, tile34
        self.name = f"resolution@{position}"

    def expand(self, particle: MahjongParticle) -> list[tuple[float, Any]]:
        return [(0.0, None)]

    def apply(self, particle: MahjongParticle, choice: Any) -> None:
        for rule in particle.rules.values():
            rule.pass_resolved(self.discarder, self.tile34)


def build_smc_events(
    context: DecisionContext, resolver: ModelResolver
) -> tuple[list[SmcEvent], _EventStatistics]:
    """判断時点までの公開イベントを、SMCのイベント列へ変換する。"""
    features = PublicFeatureState(context.public_row())
    statistics = _EventStatistics()
    turns = {turn.event_position: (seat, turn) for seat, values in context.turns.items() for turn in values}
    events: list[SmcEvent] = []
    live_draws = 0
    pending: tuple[int, tuple[int, bool]] | None = None
    for position, event in enumerate(context.events):
        kind = event["type"]
        if kind == "draw":
            live_draws += 1
            seat = int(event["seat"])
            if seat == context.target_seat:
                events.append(_TargetDrawEvent(context.target_draws[int(event["rawEventIndex"])], position))
            else:
                _, turn = turns[position]
                events.append(_SeatDrawDiscardEvent(
                    context, turn, seat, live_draws, context.events[position + 1], features, resolver, statistics
                ))
        elif kind == "discard":
            pending = (int(event["seat"]), tile_key(event["tile"]))
            events.append(_ResponseEvent(context, pending[0], pending[1], position, live_draws, features, resolver, statistics))
        elif kind == "response_resolution":
            assert pending is not None
            events.append(_ResolutionEvent(pending[0], pending[1][0], position))
            pending = None
    return events, statistics


def initial_mahjong_particles(context: DecisionContext, count: int, rng: random.Random) -> list[MahjongParticle]:
    """p0：対象家の配牌と最初のドラ表示牌を除いた物理牌から、他家3人の13枚を一様に非復元抽出する。"""
    available = list(range(136))
    for key in (*context.target_initial, context.dora_indicator):
        available.remove(next(i for i in available if id_key(i) == key))
    particles = []
    for _ in range(count):
        shuffled = list(available)
        rng.shuffle(shuffled)
        rules, initial = {}, {}
        for offset, seat in enumerate(context.other_seats):
            hand = tuple(shuffled[offset * 13:(offset + 1) * 13])
            initial[seat] = hand
            rules[seat] = _SeatRuleState(context, seat, list(hand))
        pool: dict[tuple[int, bool], list[int]] = {}
        for tile_id in shuffled[39:]:
            pool.setdefault(id_key(tile_id), []).append(tile_id)
        particles.append(MahjongParticle(rules, pool, len(shuffled) - 39, initial, {seat: {} for seat in context.other_seats}))
    return particles


def run_reference_smc(
    context: DecisionContext, resolver: ModelResolver, particle_count: int, seed: int
) -> tuple[SmcResult, dict[str, Any]]:
    """基準SMCを1回走らせ、結果と要約（3.2節の記録項目）を返す。"""
    import time

    rng = random.Random(seed)
    start = time.perf_counter()
    events, statistics = build_smc_events(context, resolver)
    particles = initial_mahjong_particles(context, particle_count, rng)
    result = run_smc(particles, events, rng)
    seconds = time.perf_counter() - start
    riichi_events = [record for record in result.event_diagnostics if record["name"].startswith("riichi_declaration")]
    summary: dict[str, Any] = {
        "decisionId": context.decision_id,
        "particleCount": particle_count,
        "seed": seed,
        "status": result.status,
        "logNormalizer": result.log_normalizer,
        "events": len(events),
        "eventsProcessed": len(result.event_diagnostics),
        "resamplingCount": result.resampling_count,
        "zeroMassEvent": result.zero_mass_event,
        "riichiDeclarationPositiveG": riichi_events[0]["positiveG"] if riichi_events else None,
        "riichiDeclarationAliveBefore": riichi_events[0]["aliveBefore"] if riichi_events else None,
        "minimumEssAfter": min((r["essAfter"] for r in result.event_diagnostics if "essAfter" in r), default=None),
        "winActionHolds": statistics.win_action_holds,
        "probabilityCacheHits": statistics.probability_cache_hits,
        "probabilityCacheMisses": statistics.probability_cache_misses,
        "seconds": round(seconds, 3),
    }
    if result.status == "ok":
        weights = result.normalized_weights()
        alive = [i for i, w in enumerate(weights) if w > 0]
        riichi = context.riichi_seat
        signatures = {tuple(sorted(id_key(t) for t in result.particles[i].rules[riichi].hand)) for i in alive}
        waits: dict[int, float] = {}
        furiten = 0.0
        for i in alive:
            rule = result.particles[i].rules[riichi]
            for tile in rule.waits():
                waits[tile] = waits.get(tile, 0.0) + weights[i]
            furiten += weights[i] * bool(rule.furiten_reasons())
        summary.update({
            "finalEss": _ess(weights),
            "finalMaxWeight": max(weights),
            "finalInitialAncestors": len({result.initial_ancestors[i] for i in alive}),
            "uniqueRiichiHandSignatures": len(signatures),
            "riichiWaitMarginals": {str(k): v for k, v in sorted(waits.items())},
            "riichiFuritenRate": furiten,
        })
    else:
        riichi_index = next((i for i, e in enumerate(events) if e.name.startswith("riichi_declaration")), None)
        zero_index = result.zero_mass_event["event"] if result.zero_mass_event else None
        summary["zeroMassBeforeRiichiDeclaration"] = (
            None if riichi_index is None or zero_index is None else zero_index < riichi_index
        )
        summary["riichiDeclarationEventIndex"] = riichi_index
    return result, summary


# ---------------------------------------------------------------------------
# pilotの判断選択（PHASE_D33_DESIGN.md 12.1節）と基準SMCの測定（3.2〜3.3節）
# ---------------------------------------------------------------------------

PILOT_SCHEMA = "ev-policy-belief-pilot-manifest/v1"
RIICHI_TIMING_BINS = (("early", 1, 6), ("mid", 7, 11), ("late", 12, 99))


def _pilot_stratum(prefix: Mapping[str, Any], public: Mapping[str, Any]) -> dict[str, str]:
    """公開情報だけから層を決める：rのリーチ巡目（宣言までのrの打牌数）、rと対象家の親子。"""
    count = int(prefix["publicEventCount"])
    target = int(prefix["seat"])
    dealer = int(public["initial"]["dealerSeat"])
    discards: Counter[int] = Counter()
    riichi_seat, riichi_turn = None, None
    for event in public["events"][:count]:
        if event["type"] != "discard":
            continue
        seat = int(event["seat"])
        discards[seat] += 1
        if event.get("riichiDeclaration") and seat != target:
            riichi_seat, riichi_turn = seat, discards[seat]
    if riichi_seat is None:
        raise ContextError("他家リーチがない")
    timing = next(name for name, low, high in RIICHI_TIMING_BINS if low <= riichi_turn <= high)
    return {
        "riichiTiming": timing,
        "riichiSeatDealer": "dealer" if riichi_seat == dealer else "nondealer",
        "targetDealer": "dealer" if target == dealer else "nondealer",
    }


def select_pilot_decisions(dataset_dir: Any, *, count: int = 16, seed: int = 20260930) -> dict[str, Any]:
    """学習期間の推論入力から、公開情報だけで層別して判断を選ぶ。

    規則：12層を固定順に並べ、空でない層から1件ずつ無作為に選ぶ。残りの枠は、未選択の判断全体から
    一様に選ぶ。同じ局から2件は選ばない。失敗した判断は後で補充しない。
    """
    from pathlib import Path

    dataset_dir = Path(dataset_dir)
    rng = random.Random(seed)
    train = [row for row in _iter_gzip_jsonl(dataset_dir / "inference-prefixes.jsonl.gz") if row["developmentSplit"] == "train"]
    rounds = {row["roundId"] for row in train}
    public = {row["roundId"]: row for row in _iter_gzip_jsonl(dataset_dir / "public-events.jsonl.gz") if row["roundId"] in rounds}
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in train:
        stratum = _pilot_stratum(row, public[row["roundId"]])
        strata.setdefault((stratum["riichiTiming"], stratum["riichiSeatDealer"], stratum["targetDealer"]), []).append(
            {"decisionId": row["decisionId"], "roundId": row["roundId"], "stratum": stratum}
        )
    order = [(t, r, d) for t, _, _ in RIICHI_TIMING_BINS for r in ("dealer", "nondealer") for d in ("dealer", "nondealer")]
    chosen: list[dict[str, Any]] = []
    used_rounds: set[str] = set()
    for key in order:
        candidates = [c for c in sorted(strata.get(key, []), key=lambda c: c["decisionId"]) if c["roundId"] not in used_rounds]
        if candidates and len(chosen) < count:
            pick = rng.choice(candidates)
            chosen.append(pick)
            used_rounds.add(pick["roundId"])
    remaining = sorted(
        (c for values in strata.values() for c in values if c["roundId"] not in used_rounds), key=lambda c: c["decisionId"]
    )
    rng.shuffle(remaining)
    for candidate in remaining:
        if len(chosen) >= count:
            break
        if candidate["roundId"] in used_rounds:
            continue
        chosen.append(candidate)
        used_rounds.add(candidate["roundId"])
    return {
        "schemaVersion": PILOT_SCHEMA,
        "seed": seed,
        "split": "train",
        "decisionCount": len(chosen),
        "selectionRule": "one_per_nonempty_stratum_in_fixed_order_then_uniform_fill_distinct_rounds",
        "riichiTimingBins": {name: [low, high] for name, low, high in RIICHI_TIMING_BINS},
        "stratumSizes": {"/".join(key): len(values) for key, values in sorted(strata.items())},
        "replaceFailedDecisions": False,
        "decisions": chosen,
        "inputSha256": {
            name: _sha256_file(dataset_dir / name) for name in ("inference-prefixes.jsonl.gz", "public-events.jsonl.gz")
        },
    }


def _run_seed(base: int, decision_id: str, particles: int, replicate: int) -> int:
    digest = hashlib.sha256(f"{base}:{decision_id}:{particles}:{replicate}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


_WORKER: dict[str, Any] = {}


def _worker_init(dataset_dir: str, model_dir: str, decision_ids: Sequence[str]) -> None:
    from pathlib import Path

    dataset, models = Path(dataset_dir), Path(model_dir)
    wanted = set(decision_ids)
    prefixes = {row["decisionId"]: row for row in _iter_gzip_jsonl(dataset / "inference-prefixes.jsonl.gz") if row["decisionId"] in wanted}
    rounds = {row["roundId"] for row in prefixes.values()}
    public = {row["roundId"]: row for row in _iter_gzip_jsonl(dataset / "public-events.jsonl.gz") if row["roundId"] in rounds}
    targets = {(row["roundId"], int(row["seat"])) for row in prefixes.values()}
    private = {
        (row["roundId"], int(row["seat"])): row
        for row in _iter_gzip_jsonl(dataset / "private-events.jsonl.gz")
        if (row["roundId"], int(row["seat"])) in targets
    }
    model = HierarchicalSoftmax.from_dict(json.loads((models / "model.json").read_text(encoding="utf-8")))
    base = next(s for s in json.loads((models / "fixed-components.json").read_text(encoding="utf-8"))["scenarios"] if s["id"] == "base")
    _WORKER["contexts"] = {
        decision: DecisionContext.from_records(row, public[row["roundId"]], private[(row["roundId"], int(row["seat"]))])
        for decision, row in prefixes.items()
    }
    _WORKER["resolver"] = model_resolver(model, base)


def _worker_run(task: tuple[str, int, int, int]) -> dict[str, Any]:
    decision_id, particles, replicate, seed = task
    _, summary = run_reference_smc(_WORKER["contexts"][decision_id], _WORKER["resolver"], particles, seed)
    summary["replicate"] = replicate
    return summary


def reference_smc_gate(runs: Sequence[Mapping[str, Any]], *, particles: int = 4096, expected: int = 64) -> dict[str, Any]:
    """3.3節の関門。判定に使うのはN=4,096の64反復だけ。"""
    decisive = [run for run in runs if int(run["particleCount"]) == particles]
    zero = [run for run in decisive if run["status"] == "posterior_zero_mass"]
    if zero:
        verdict = "proceed_to_mcmc_implementation"
    elif len(decisive) == expected:
        verdict = "return_to_design_as_undetermined"
    else:
        verdict = "incomplete"
    return {
        "particleCountForDecision": particles,
        "expectedRuns": expected,
        "completedRuns": len(decisive),
        "zeroMassRuns": len(zero),
        "verdict": verdict,
    }


def measure_reference_smc(
    dataset_dir: Any,
    model_dir: Any,
    manifest_path: Any,
    output_dir: Any,
    *,
    particle_grid: Sequence[int] = (256, 1024, 4096),
    replicates: int = 4,
    processes: int = 3,
    wall_clock_seconds: float = 86_400.0,
) -> dict[str, Any]:
    """manifestの全判断で基準SMCを走らせ、結果を1行ずつ保存する（途中から再開できる）。"""
    import multiprocessing
    import time
    from pathlib import Path

    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision_ids = [item["decisionId"] for item in manifest["decisions"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "reference-smc-runs.jsonl"
    done: dict[tuple[str, int, int], dict[str, Any]] = {}
    if runs_path.is_file():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["decisionId"], int(row["particleCount"]), int(row["replicate"]))] = row
    tasks = [
        (decision, particles, replicate, _run_seed(int(manifest["seed"]), decision, particles, replicate))
        for particles in particle_grid
        for decision in decision_ids
        for replicate in range(replicates)
        if (decision, particles, replicate) not in done
    ]
    start = time.perf_counter()
    status = "complete"
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes, initializer=_worker_init, initargs=(str(dataset_dir), str(model_dir), decision_ids)) as pool:
        with runs_path.open("a", encoding="utf-8") as handle:
            for summary in pool.imap_unordered(_worker_run, tasks):
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                handle.flush()
                done[(summary["decisionId"], int(summary["particleCount"]), int(summary["replicate"]))] = summary
                if time.perf_counter() - start > wall_clock_seconds:
                    status = "resource_budget_exceeded"
                    pool.terminate()
                    break
    runs = list(done.values())
    report = {
        "schemaVersion": "ev-policy-belief-reference-smc/v1",
        "status": status,
        "manifestSha256": _sha256_file(manifest_path),
        "particleGrid": list(particle_grid),
        "replicates": replicates,
        "processes": processes,
        "scenario": "base",
        "runs": len(runs),
        "zeroMassBySetting": {
            str(n): sum(r["status"] == "posterior_zero_mass" for r in runs if int(r["particleCount"]) == n) for n in particle_grid
        },
        "gate": reference_smc_gate(runs, expected=len(decision_ids) * replicates),
        "wallClockSeconds": round(time.perf_counter() - start, 1),
    }
    (output_dir / "reference-smc-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


# ===========================================================================
# 工程3：構成的初期化（PHASE_D33_DESIGN.md 6節、7.3節の生成器g_r）
# ===========================================================================

# テンパイ形の系統の確率（7.3節）。すべて正でなければならない（0だと国士などの分断が残る）。
# 値は効率だけに影響し、正しさには影響しない。pilot前にmanifestで固定する。
TENPAI_FAMILY_PROBABILITIES = (("regular", 0.90), ("chiitoi", 0.05), ("kokushi", 0.05))
YAOCHUU = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
MENTSU_KINDS: tuple[tuple[int, int, int], ...] = tuple(
    [(t, t, t) for t in range(34)] + [(s * 9 + i, s * 9 + i + 1, s * 9 + i + 2) for s in range(3) for i in range(7)]
)


def validate_family_probabilities(probabilities: Sequence[tuple[str, float]]) -> None:
    names = [name for name, _ in probabilities]
    if sorted(names) != ["chiitoi", "kokushi", "regular"]:
        raise ValueError("系統は面子手・七対子・国士の3つが必要")
    if any(not value > 0 for _, value in probabilities) or abs(math.fsum(v for _, v in probabilities) - 1.0) > 1e-12:
        raise ValueError("系統の確率はすべて正で、和が1である必要がある（7.3節）")


def sample_tenpai_shape(
    rng: random.Random, probabilities: Sequence[tuple[str, float]] = TENPAI_FAMILY_PROBABILITIES
) -> tuple[str, list[int]]:
    """生成器g_rの牌種（34種）の段：系統を選び、14枚の完成形から1枚を一様に除いた13枚を返す。

    面子手は雀頭と面子を独立に選ぶので、同じ牌種が5枚以上の形も作る。在庫の判定は呼び出し側で行う。
    """
    validate_family_probabilities(probabilities)
    threshold, cumulative, family = rng.random(), 0.0, probabilities[-1][0]
    for name, value in probabilities:
        cumulative += value
        if threshold < cumulative:
            family = name
            break
    if family == "regular":
        tiles = [rng.randrange(34)] * 2
        for _ in range(4):
            tiles.extend(MENTSU_KINDS[rng.randrange(len(MENTSU_KINDS))])
    elif family == "chiitoi":
        tiles = [t for t in rng.sample(range(34), 7) for _ in range(2)]
    else:
        tiles = [*YAOCHUU, rng.choice(YAOCHUU)]
    tiles.pop(rng.randrange(14))
    return family, sorted(tiles)


@dataclass
class InitialWorld:
    """構成的初期化の結果（6節）。"""

    status: str  # "ok" または "init_failed"
    world: WorldAssignment | None
    family: str | None
    attempts: int
    seconds: float
    failure_reasons: dict[str, int]
    log_likelihood: dict[int, float] | None = None


def _discard_ids(context: DecisionContext, available: list[int], rng: random.Random) -> dict[int, dict[int, int]]:
    """他家の打牌（手出し・ツモ切り）の物理IDを先に確保する。打牌はその家の位置にあった牌（4.2節）。"""
    return {
        seat: {turn.raw_event_index: _take(available, turn.discard, rng) for turn in context.turns[seat]}
        for seat in context.other_seats
    }


def _take_shape(available: list[int], shape: Sequence[int], rng: random.Random) -> list[int] | None:
    """34種の牌姿を、利用可能な物理IDから一様に割り当てる（赤は同じ牌種の物理牌の中で一様＝超幾何）。"""
    by_type: dict[int, list[int]] = {}
    for tile_id in available:
        by_type.setdefault(TILE_BY_ID[tile_id].tile34, []).append(tile_id)
    need = Counter(shape)
    if any(len(by_type.get(t, ())) < n for t, n in need.items()):
        return None
    picked = [tile_id for t, n in sorted(need.items()) for tile_id in rng.sample(by_type[t], n)]
    for tile_id in picked:
        available.remove(tile_id)
    return picked


def construct_initial_world(
    context: DecisionContext,
    rng: random.Random,
    *,
    resolver: ModelResolver | None = None,
    probabilities: Sequence[tuple[str, float]] = TENPAI_FAMILY_PROBABILITIES,
    max_attempts: int = 10_000,
    max_seconds: float = 60.0,
    cache: WindowCache | None = None,
) -> InitialWorld:
    """H=1（かつresolverがあれば尤度が正）の出発点を構成的に作る（6節）。

    入力は判断文脈（公開履歴と対象家の私有履歴）だけで、他家の教師手牌は読まない。
    分布は事後分布と一致しなくてよい。出発点の影響は複数の鎖とburn-inで診断する。
    """
    import time

    validate_family_probabilities(probabilities)
    start = time.perf_counter()
    failures: Counter[str] = Counter()
    for attempt in range(1, max_attempts + 1):
        if time.perf_counter() - start > max_seconds:
            break
        available = list(range(136))
        target_initial = [_take(available, key, rng) for key in context.target_initial]
        target_draws = {raw: _take(available, key, rng) for raw, key in sorted(context.target_draws.items())}
        dora = _take(available, context.dora_indicator, rng)
        discard_ids = _discard_ids(context, available, rng)
        # 1. リーチ者の宣言時手牌（判断時点の手と同じ）をテンパイ形の生成器で作る。
        family, shape = sample_tenpai_shape(rng, probabilities)
        riichi_hand = _take_shape(available, shape, rng)
        if riichi_hand is None:
            failures["riichi_shape_inventory"] += 1
            continue
        # 2〜3. 自摸の逆算。リーチしていない2家は残りから一様に13枚。
        hypotheses = {context.riichi_seat: _backward_history(context, context.riichi_seat, riichi_hand, discard_ids[context.riichi_seat], rng)}
        for seat in context.other_seats:
            if seat == context.riichi_seat:
                continue
            final = rng.sample(available, 13)
            for tile_id in final:
                available.remove(tile_id)
            hypotheses[seat] = _backward_history(context, seat, final, discard_ids[seat], rng)
        rng.shuffle(available)
        world = WorldAssignment(hypotheses, tuple(target_initial), target_draws, dora, tuple(available))
        # 4. 検査：Hと（resolverがあれば）尤度が正。得点器のholdは出発点に使わない。
        log_likelihood: dict[int, float] = {}
        reason = None
        for seat, hypothesis in hypotheses.items():
            evaluation = evaluate_seat(context, hypothesis, resolver, cache)
            if evaluation.violation is not None:
                reason = "hard_constraint_violation"
            elif evaluation.holds:
                reason = "win_action_hold"
            elif resolver is not None and not math.isfinite(evaluation.log_likelihood):
                reason = "zero_likelihood"
            if reason:
                break
            log_likelihood[seat] = evaluation.log_likelihood
        if reason:
            failures[reason] += 1
            continue
        world.validate()
        return InitialWorld("ok", world, family, attempt, time.perf_counter() - start, dict(failures),
                            log_likelihood if resolver is not None else None)
    return InitialWorld("init_failed", None, None, sum(failures.values()), time.perf_counter() - start, dict(failures))


def record_pilot_initialization(
    dataset_dir: Any, model_dir: Any, manifest_path: Any, *, chains: int = 4
) -> dict[str, Any]:
    """pilot 16判断×4鎖で構成的初期化を行い、成否・試行回数・時間・系統を記録する（12.4節）。"""
    from pathlib import Path

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision_ids = [item["decisionId"] for item in manifest["decisions"]]
    _worker_init(str(dataset_dir), str(model_dir), decision_ids)
    rows = []
    for decision in decision_ids:
        context = _WORKER["contexts"][decision]
        for chain in range(chains):
            seed = _run_seed(int(manifest["seed"]), decision, 0, 1000 + chain)
            result = construct_initial_world(context, random.Random(seed), resolver=_WORKER["resolver"])
            rows.append({
                "decisionId": decision, "chain": chain, "seed": seed, "status": result.status,
                "family": result.family, "attempts": result.attempts, "seconds": round(result.seconds, 4),
                "failureReasons": result.failure_reasons,
            })
    report = {
        "schemaVersion": "ev-policy-belief-initialization/v1",
        "manifestSha256": _sha256_file(manifest_path),
        "familyProbabilities": dict(TENPAI_FAMILY_PROBABILITIES),
        "scenario": "base",
        "chains": chains,
        "failed": sum(row["status"] != "ok" for row in rows),
        "families": dict(Counter(row["family"] for row in rows if row["family"])),
        "maxAttempts": max(row["attempts"] for row in rows),
        "maxSeconds": max(row["seconds"] for row in rows),
        "rows": rows,
    }
    (manifest_path.parent / "initialization.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report



# ===========================================================================
# 工程4：MCMCの移動M1〜M4と受理計算（PHASE_D33_DESIGN.md 4.2節、7節）
# ===========================================================================
#
# 鎖の状態は「位置→物理ID」の割当（物理水準）で持つ。位置は、他家ごとに配牌（hand_size個）と
# 手出し窓の自摸を並べ、最後に未割当プールを置く。ツモ切り窓の自摸位置は物理IDを固定した
# 条件付きの空間（4.2節）とし、状態に含めない（牌種は打牌と同じなので型だけ持つ）。
# 牌種は整数で表す。麻雀では tile34 + 34*赤、類（生成器が扱う牌の種類）は tile34。
# 小例（受入試験D33-01・D33-02）は同じ実装に、任意の牌種・類・生成器を渡して使う。
#
# 乱数は「選択器」を通して引く。RandomChooserは通常の実行、_ReplayChooserは全分岐を列挙して
# 小例の正確な遷移行列を作る。同じ移動の実装を両方で使うので、行列の検査は実装そのものを検査する。


class RuleUnresolvedError(RuntimeError):
    """履歴評価器が得点器の未知エラーに遭遇した（10.2節 rule_unresolved）。"""


class RandomChooser:
    """乱数で1つの分岐を選ぶ。"""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def index(self, n: int) -> int:
        return self.rng.randrange(n)

    def weighted(self, weights: Sequence[float]) -> int:
        threshold = self.rng.random() * math.fsum(weights)
        cumulative, last = 0.0, 0
        for position, weight in enumerate(weights):
            if weight <= 0:
                continue
            cumulative += weight
            last = position
            if threshold < cumulative:
                return position
        return last

    def accept(self, log_ratio: float) -> bool:
        """MHの受理。log_ratio = log(受理比)。"""
        if log_ratio >= 0.0:
            return True
        if log_ratio == -math.inf:
            return False
        return self.rng.random() < math.exp(log_ratio)


class _ReplayChooser:
    """決めた分岐の列をなぞり、その先は確率が正の最初の分岐を選んで、通った分岐を記録する。"""

    def __init__(self, prefix: Sequence[int]):
        self.prefix = prefix
        self.trace: list[tuple[int, tuple[float, ...]]] = []

    def _choose(self, probabilities: tuple[float, ...]) -> int:
        depth = len(self.trace)
        if depth < len(self.prefix):
            choice = self.prefix[depth]
        else:
            choice = next(i for i, p in enumerate(probabilities) if p > 0)
        self.trace.append((choice, probabilities))
        return choice

    def index(self, n: int) -> int:
        return self._choose((1.0 / n,) * n)

    def weighted(self, weights: Sequence[float]) -> int:
        total = math.fsum(weights)
        return self._choose(tuple(weight / total for weight in weights))

    def accept(self, log_ratio: float) -> bool:
        if log_ratio >= 0.0:
            return True
        if log_ratio == -math.inf:
            return False
        probability = math.exp(log_ratio)
        return self._choose((probability, 1.0 - probability)) == 0


def enumerate_outcomes(step: Callable[[Any], Any]) -> dict[Any, float]:
    """step(chooser)の全分岐を深さ優先でたどり、結果ごとの確率を返す（小例の遷移行列用）。"""
    outcomes: dict[Any, float] = {}
    stack: list[tuple[int, ...]] = [()]
    while stack:
        prefix = stack.pop()
        chooser = _ReplayChooser(prefix)
        result = step(chooser)
        probability = 1.0
        for choice, probabilities in chooser.trace:
            probability *= probabilities[choice]
        outcomes[result] = outcomes.get(result, 0.0) + probability
        # prefixより先で初めて通った分岐点について、選ばなかった兄弟を積む。
        for depth in range(len(prefix), len(chooser.trace)):
            choice, probabilities = chooser.trace[depth]
            head = tuple(c for c, _ in chooser.trace[:depth])
            for alternative in range(choice + 1, len(probabilities)):
                if probabilities[alternative] > 0:
                    stack.append(head + (alternative,))
    return outcomes


# ---------------------------------------------------------------------------
# 位置の構造と、牌種水準の見方
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnLayout:
    """他家の1回の自摸と打牌。ツモ切りなら自摸の牌種は打牌と同じで、位置は動かさない。"""

    tsumogiri: bool
    discard_type: int


@dataclass(frozen=True)
class SeatLayout:
    seat: int
    riichi: bool
    turns: tuple[TurnLayout, ...]


class BeliefLayout:
    """判断ごとに固定する位置の構造。

    type_of：物理ID→牌種（添字で引ける列または辞書）。class_of：牌種→類。
    """

    def __init__(
        self,
        hand_size: int,
        seats: Sequence[SeatLayout],
        pool_size: int,
        type_of: Sequence[int] | Mapping[int, int],
        class_of: Mapping[int, int],
    ):
        self.hand_size = hand_size
        self.seats = tuple(seats)
        self.pool_size = pool_size
        self.type_of = type_of
        self.class_of = class_of
        riichi = [i for i, seat in enumerate(self.seats) if seat.riichi]
        if len(riichi) > 1:
            raise ValueError("リーチ者は高々1人")
        self.riichi_index: int | None = riichi[0] if riichi else None
        self.offsets: list[int] = []
        self.tedashi: list[tuple[int, ...]] = []  # 家ごとの手出し窓の添字（turnsの中の位置）
        position = 0
        for seat in self.seats:
            self.offsets.append(position)
            tedashi = tuple(k for k, turn in enumerate(seat.turns) if not turn.tsumogiri)
            self.tedashi.append(tedashi)
            position += hand_size + len(tedashi)
        self.seat_size = position  # |S|：牌種が未知の位置（配牌と手出し窓の自摸）
        self.size = position + pool_size  # |U| = |S| + プール
        self.position_seat = [i for i, seat in enumerate(self.seats) for _ in range(hand_size + len(self.tedashi[i]))]
        self.position_seat += [-1] * pool_size
        # M3の順序：リーチ者、リーチしていない家の順（7.5節）。M4の生成順も同じ（7.4節）。
        self.m3_order = tuple(([self.riichi_index] if self.riichi_index is not None else [])
                              + [i for i in range(len(self.seats)) if i != self.riichi_index])

    def initial_positions(self, i: int) -> range:
        return range(self.offsets[i], self.offsets[i] + self.hand_size)

    def draw_positions(self, i: int) -> range:
        start = self.offsets[i] + self.hand_size
        return range(start, start + len(self.tedashi[i]))

    def seat_positions(self, i: int) -> range:
        return range(self.offsets[i], self.offsets[i] + self.hand_size + len(self.tedashi[i]))

    @property
    def pool_positions(self) -> range:
        return range(self.seat_size, self.size)


# 家の牌種水準の状態：（配牌の牌種の昇順タプル, 各窓の自摸の牌種のタプル（ツモ切り窓も含む））
SeatTypes = tuple[tuple[int, ...], tuple[int, ...]]


def seat_type_state(layout: BeliefLayout, ids: Sequence[int], i: int, override: Mapping[int, int] | None = None) -> SeatTypes:
    """位置の物理IDから家iの牌種水準の状態を作る。overrideは位置→牌種の差し替え（交換の試算用）。"""

    def type_at(position: int) -> int:
        if override is not None and position in override:
            return override[position]
        return layout.type_of[ids[position]]

    initial = tuple(sorted(type_at(p) for p in layout.initial_positions(i)))
    draws: list[int] = []
    position = layout.offsets[i] + layout.hand_size
    for turn in layout.seats[i].turns:
        if turn.tsumogiri:
            draws.append(turn.discard_type)
        else:
            draws.append(type_at(position))
            position += 1
    return initial, tuple(draws)


def seat_final_hand(layout: BeliefLayout, i: int, initial: Sequence[int], draws: Sequence[int]) -> Counter | None:
    """判断時点の手（牌種の計数）。打牌整合（4.4節の2）に反すればNone。

    手出しの打牌は、その時点の手中にあり、直前の自摸とは別の物理牌（同じ牌種でもよい）。
    """
    hand = Counter(initial)
    for turn, drawn in zip(layout.seats[i].turns, draws):
        if turn.tsumogiri:
            continue  # 自摸と同じ牌を切るので手は変わらない
        hand[drawn] += 1
        if hand[turn.discard_type] - (1 if drawn == turn.discard_type else 0) < 1:
            return None
        hand[turn.discard_type] -= 1
    return +hand


def tedashi_discards(layout: BeliefLayout, i: int) -> Counter:
    """家iの手出しの打牌の牌種の計数（D_j）。"""
    return Counter(layout.seats[i].turns[k].discard_type for k in layout.tedashi[i])


def tedashi_draws(layout: BeliefLayout, i: int, draws: Sequence[int]) -> list[int]:
    return [draws[k] for k in layout.tedashi[i]]


def class_counts(problem: "BeliefProblem", hand: Mapping[int, int]) -> tuple[int, ...]:
    values = [0] * problem.rules.class_count
    for tile_type, count in hand.items():
        values[problem.layout.class_of[tile_type]] += count
    return tuple(values)


# ---------------------------------------------------------------------------
# 問題の定義：位置の構造、テンパイ形の規則（生成器g_r）、家ごとの尤度因子
# ---------------------------------------------------------------------------


@dataclass
class BeliefProblem:
    """MCMCの対象。model.log_factor(i, initial, draws)は家iの対数尤度（H=1を前提に呼ぶ）。"""

    layout: BeliefLayout
    rules: Any
    model: Any
    max_generator_attempts: int = 1000

    def __post_init__(self) -> None:
        # 系統の確率はすべて正（7.3節）。0を設定した実行は開始前に拒否する（D33-01(e)）。
        values = [value for _, value in self.rules.families]
        if any(not value > 0 for value in values) or abs(math.fsum(values) - 1.0) > 1e-12:
            raise ValueError("系統の確率はすべて正で、和が1である必要がある（7.3節）")
        if self.max_generator_attempts < 1:
            raise ValueError("生成器の引き直しの上限は1以上")


def seat_hard_violation(problem: BeliefProblem, i: int, initial: Sequence[int], draws: Sequence[int]) -> str | None:
    """家iの硬い制約（4.4節）の違反理由。牌在庫は位置の割当が保証する。"""
    final = seat_final_hand(problem.layout, i, initial, draws)
    if final is None:
        return "discard_consistency"
    if problem.layout.seats[i].riichi and not problem.rules.is_tenpai(class_counts(problem, final)):
        return "tenpai"
    return None


def seat_log_factor(problem: BeliefProblem, i: int, state: SeatTypes) -> float:
    """家iの因子 log(H×L)。H=0なら-inf。"""
    if seat_hard_violation(problem, i, *state) is not None:
        return -math.inf
    return float(problem.model.log_factor(i, *state))


# ---------------------------------------------------------------------------
# テンパイ形の生成器g_rと、その密度（7.3節）
# ---------------------------------------------------------------------------


def tenpai_removal_classes(rules: Any, counts: Sequence[int]) -> Sequence[int]:
    """経路の和で除いた牌として数える類。合法な待ちに限らず、すべての類（7.3節、再レビュー指摘1）。"""
    return range(rules.class_count)


def _family_densities(rules: Any, counts: tuple[int, ...]) -> list[float]:
    size = rules.hand_size + 1
    values = []
    for family, _ in rules.families:
        total = 0.0
        for removed in tenpai_removal_classes(rules, counts):
            complete = counts[:removed] + (counts[removed] + 1,) + counts[removed + 1:]
            probability = rules.complete_probability(family, complete)
            if probability > 0:
                total += probability * complete[removed] / size
        values.append(total)
    return values


def tenpai_class_density(rules: Any, counts: tuple[int, ...]) -> float:
    """g_r(h)の類の段：hに至るすべての生成経路の確率の和（系統ごとの確率を掛けて足す）。"""
    return math.fsum(probability * value for (_, probability), value in zip(rules.families, _family_densities(rules, counts)))


def tenpai_families(rules: Any, counts: tuple[int, ...]) -> tuple[str, ...]:
    """hを生成できる系統の名前（診断の「系統」。面子手と七対子の両方になる手もある）。"""
    return tuple(name for (name, _), value in zip(rules.families, _family_densities(rules, counts)) if value > 0)


def sample_tenpai_classes(rules: Any, chooser: Any) -> list[int]:
    """g_rの類の段の抽出：系統を選び、完成形から1枚を一様に除く。"""
    family = rules.families[chooser.weighted([value for _, value in rules.families])][0]
    complete = list(rules.sample_complete(family, chooser))
    complete.pop(chooser.index(len(complete)))
    return sorted(complete)


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _draw_without_replacement(items: Sequence[int], count: int, chooser: Any) -> list[int]:
    remaining = list(items)
    return [remaining.pop(chooser.index(len(remaining))) for _ in range(count)]


def _variants_log_density(problem: BeliefProblem, available: Counter, hand: Counter) -> float:
    """類を決めた後、各類の中で物理牌を一様に選ぶ段（赤の超幾何分布）の確率。"""
    class_of = problem.layout.class_of
    total = 0.0
    need: Counter = Counter()
    supply: Counter = Counter()
    for tile_type, count in hand.items():
        total += _log_comb(available[tile_type], count)
        need[class_of[tile_type]] += count
    for tile_type, count in available.items():
        supply[class_of[tile_type]] += count
    for tile_class, count in need.items():
        total -= _log_comb(supply[tile_class], count)
    return total


def _uniform_log_density(available: Counter, hand: Counter) -> float:
    """リーチしていない家のg_j：availableから手の枚数を一様に引く（超幾何分布）。"""
    total = -_log_comb(sum(available.values()), sum(hand.values()))
    for tile_type, count in hand.items():
        total += _log_comb(available[tile_type], count)
    return total


def propose_final_hands(problem: BeliefProblem, order: Sequence[int], available: Counter, chooser: Any) -> list[Counter] | None:
    """判断時点の手を家の順に作る（7.3節の段1、7.4節の段1〜2）。

    リーチ者はg_rを、類の在庫に収まるまで引き直す（上限で失敗ならNone＝現状維持）。
    引き直しで失われる確率はavailableだけで決まり、移動の前後で相殺する。
    """
    layout, rules = problem.layout, problem.rules
    remaining = Counter(available)
    hands = []
    for i in order:
        if layout.seats[i].riichi:
            supply = class_counts(problem, remaining)
            classes = None
            for _ in range(problem.max_generator_attempts):
                candidate = Counter(sample_tenpai_classes(rules, chooser))
                if all(supply[c] >= n for c, n in candidate.items()):
                    classes = candidate
                    break
            if classes is None:
                return None
            hand: Counter = Counter()
            for tile_class in sorted(classes):
                items = sorted(t for t in remaining.elements() if layout.class_of[t] == tile_class)
                hand.update(_draw_without_replacement(items, classes[tile_class], chooser))
        else:
            hand = Counter(_draw_without_replacement(sorted(remaining.elements()), layout.hand_size, chooser))
        remaining -= hand
        hands.append(hand)
    return hands


def final_hands_log_density(problem: BeliefProblem, order: Sequence[int], available: Counter, hands: Sequence[Counter]) -> float:
    """propose_final_handsが手の列を作る確率（引き直しの正規化を除く）。"""
    remaining = Counter(available)
    total = 0.0
    for i, hand in zip(order, hands):
        if problem.layout.seats[i].riichi:
            density = tenpai_class_density(problem.rules, class_counts(problem, hand))
            if density <= 0:
                return -math.inf
            total += math.log(density) + _variants_log_density(problem, remaining, hand)
        else:
            total += _uniform_log_density(remaining, hand)
        remaining -= hand
    return total


def backward_history(layout: BeliefLayout, i: int, final: Counter, chooser: Any) -> SeatTypes:
    """自摸の逆算（7.3節の段2）。手出し窓では自摸を打牌後の手の物理牌から一様に選ぶ。"""
    hand = Counter(final)
    draws: list[int] = [0] * len(layout.seats[i].turns)
    for k in reversed(range(len(layout.seats[i].turns))):
        turn = layout.seats[i].turns[k]
        if turn.tsumogiri:
            draws[k] = turn.discard_type
            continue
        drawn = _draw_without_replacement(sorted(hand.elements()), 1, chooser)[0]
        draws[k] = drawn
        hand[turn.discard_type] += 1
        hand[drawn] -= 1
    return tuple(sorted((+hand).elements())), tuple(draws)


def backward_log_density(layout: BeliefLayout, i: int, initial: Sequence[int], draws: Sequence[int]) -> float:
    """逆算が自摸の列を作る確率 Π n_after(d)/手の枚数。"""
    hand = Counter(initial)
    total = 0.0
    for turn, drawn in zip(layout.seats[i].turns, draws):
        if turn.tsumogiri:
            continue
        hand[drawn] += 1
        hand[turn.discard_type] -= 1
        total += math.log(hand[drawn] / layout.hand_size)
    return total


def initial_log_weight(initial: Sequence[int]) -> float:
    """牌種水準の目標の重みのうち配牌の分 Π 1/h0_t!（4.2節）。"""
    return -math.fsum(math.lgamma(count + 1) for count in Counter(initial).values())


def pool_log_weight(pool: Counter) -> float:
    """牌種水準の目標の重みのうちプールの分 Π 1/pool_t!（4.2節）。"""
    return -math.fsum(math.lgamma(count + 1) for count in pool.values())


# ---------------------------------------------------------------------------
# 鎖の状態と移動の統計
# ---------------------------------------------------------------------------


@dataclass
class ChainState:
    ids: list[int]  # 位置→物理ID
    log_factors: list[float]  # 家ごとの log(H×L)

    def copy(self) -> "ChainState":
        return ChainState(list(self.ids), list(self.log_factors))


def initial_chain_state(problem: BeliefProblem, ids: Sequence[int]) -> ChainState:
    layout = problem.layout
    if len(ids) != layout.size or len(set(ids)) != len(ids):
        raise ValueError("位置の数と物理IDが一致しない")
    factors = [seat_log_factor(problem, i, seat_type_state(layout, ids, i)) for i in range(len(layout.seats))]
    if not all(math.isfinite(value) for value in factors):
        raise ValueError("出発点がH=1かつ尤度が正でない")
    return ChainState(list(ids), factors)


@dataclass
class MoveStatistics:
    """移動の種類ごとの提案・受理・棄却理由・系統をまたいだ受理（10.1節）。"""

    proposed: Counter = field(default_factory=Counter)
    accepted: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)  # "移動:理由"
    null_moves: Counter = field(default_factory=Counter)  # 同じ牌種どうしの交換（受理比1の空移動）
    cross_family: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {name: dict(sorted(getattr(self, name).items())) for name in
                ("proposed", "accepted", "rejected", "null_moves", "cross_family")}


# ---------------------------------------------------------------------------
# M1：物理牌の交換（7.1節）
# ---------------------------------------------------------------------------


def m1_partner(layout: BeliefLayout, ids: Sequence[int], a: int, chooser: Any) -> int:
    """U = S ∪ プールのうちa以外から一様に選ぶ。同じ牌種どうしも候補から除かない（7.1節）。"""
    k = chooser.index(layout.size - 1)
    return k if k < a else k + 1


def _swap_factors(problem: BeliefProblem, state: ChainState, a: int, b: int) -> tuple[dict[int, float], str | None]:
    """a、bの物理牌を交換したときに変わる家の新しい因子。硬い制約の違反なら理由を返す。"""
    layout, ids = problem.layout, state.ids
    type_a, type_b = layout.type_of[ids[a]], layout.type_of[ids[b]]
    override = {a: type_b, b: type_a}
    factors: dict[int, float] = {}
    changed: dict[int, SeatTypes] = {}
    for i in sorted({layout.position_seat[a], layout.position_seat[b]} - {-1}):
        after = seat_type_state(layout, ids, i, override)
        if after == seat_type_state(layout, ids, i):
            factors[i] = state.log_factors[i]
        else:
            changed[i] = after
    # 変わる家すべての硬い制約を先に確かめる。1家だけ整合でも、もう1家が公開の打牌を持てない
    # 提案では世界全体の牌在庫が崩れ、尤度の特徴計算が成り立たない。
    for i, after in changed.items():
        violation = seat_hard_violation(problem, i, *after)
        if violation is not None:
            return factors, violation
    for i, after in changed.items():
        factors[i] = float(problem.model.log_factor(i, *after))
    return factors, None


def m1_step(problem: BeliefProblem, state: ChainState, chooser: Any, stats: MoveStatistics) -> None:
    layout = problem.layout
    stats.proposed["m1"] += 1
    a = chooser.index(layout.seat_size)
    b = m1_partner(layout, state.ids, a, chooser)
    if layout.type_of[state.ids[a]] == layout.type_of[state.ids[b]]:
        stats.null_moves["m1"] += 1
    factors, violation = _swap_factors(problem, state, a, b)
    if violation is not None:
        stats.rejected[f"m1:{violation}"] += 1
        return
    log_ratio = math.fsum(value - state.log_factors[i] for i, value in factors.items())
    if not chooser.accept(log_ratio):
        stats.rejected["m1:mh"] += 1
        return
    state.ids[a], state.ids[b] = state.ids[b], state.ids[a]
    for i, value in factors.items():
        state.log_factors[i] = value
    stats.accepted["m1"] += 1


# ---------------------------------------------------------------------------
# M2：リーチ者のテンパイを保つ交換（7.2節）
# ---------------------------------------------------------------------------


def m2_positions(problem: BeliefProblem, ids: Sequence[int]) -> list[int]:
    """S_r：リーチ者の牌種が未知の位置（配牌と手出し窓の自摸）。交換で変わらない固定の集合。"""
    return list(problem.layout.seat_positions(problem.layout.riichi_index))  # type: ignore[arg-type]


def m2_candidates(problem: BeliefProblem, ids: Sequence[int], a: int) -> list[int]:
    """V(z, a)：aと交換してもすべての硬い制約を満たす、プールとリーチしていない家の位置。"""
    layout = problem.layout
    r = layout.riichi_index
    type_a = layout.type_of[ids[a]]
    riichi_ok: dict[int, bool] = {}
    result = []
    for b in range(layout.size):
        i = layout.position_seat[b]
        if i == r:
            continue
        type_b = layout.type_of[ids[b]]
        if type_b not in riichi_ok:
            riichi_ok[type_b] = seat_hard_violation(problem, r, *seat_type_state(layout, ids, r, {a: type_b})) is None  # type: ignore[arg-type]
        if not riichi_ok[type_b]:
            continue
        if i != -1 and seat_hard_violation(problem, i, *seat_type_state(layout, ids, i, {b: type_a})) is not None:
            continue
        result.append(b)
    return result


def m2_log_correction(size_before: int, size_after: int) -> float:
    """提案の非対称の補正 log(|V(z,a)| / |V(z',a)|)。"""
    return math.log(size_before) - math.log(size_after)


def m2_step(problem: BeliefProblem, state: ChainState, chooser: Any, stats: MoveStatistics, a: int | None = None) -> None:
    """aを固定した核は可逆。a=Noneなら状態によらず一様に選ぶ（その混合も可逆）。"""
    stats.proposed["m2"] += 1
    if a is None:
        positions = m2_positions(problem, state.ids)
        a = positions[chooser.index(len(positions))]
    before = m2_candidates(problem, state.ids, a)
    if not before:
        stats.rejected["m2:no_candidate"] += 1
        return
    b = before[chooser.index(len(before))]
    layout = problem.layout
    if layout.type_of[state.ids[a]] == layout.type_of[state.ids[b]]:
        stats.null_moves["m2"] += 1
    factors, violation = _swap_factors(problem, state, a, b)
    if violation is not None:  # V(z,a)の定義から起きない
        raise AssertionError(f"M2の候補が硬い制約に反する: {violation}")
    swapped = list(state.ids)
    swapped[a], swapped[b] = swapped[b], swapped[a]
    after = m2_candidates(problem, swapped, a)
    log_ratio = math.fsum(value - state.log_factors[i] for i, value in factors.items())
    log_ratio += m2_log_correction(len(before), len(after))
    if not chooser.accept(log_ratio):
        stats.rejected["m2:mh"] += 1
        return
    state.ids = swapped
    for i, value in factors.items():
        state.log_factors[i] = value
    stats.accepted["m2"] += 1


# ---------------------------------------------------------------------------
# M3・M4：家ごと／他家全員の履歴の再生成（7.3〜7.4節）
# ---------------------------------------------------------------------------


def _shuffled(items: Sequence[int], chooser: Any) -> list[int]:
    """一様な順列（Fisher–Yates）。"""
    values = list(items)
    for k in range(len(values) - 1, 0, -1):
        j = chooser.index(k + 1)
        values[k], values[j] = values[j], values[k]
    return values


def arrange_pool_types(pool_types: Sequence[int], chooser: Any) -> list[int]:
    """復元の段3：プールの牌種をプール位置へ一様な順列で並べる（改訂3で追加。省くと目標を保たない）。"""
    return _shuffled(pool_types, chooser)


def restore_block(
    problem: BeliefProblem, state: ChainState, order: Sequence[int], proposed: Mapping[int, SeatTypes], pool: Counter, chooser: Any
) -> None:
    """条件付き復元（4.2節）：牌種状態に一致するブロック内の物理配置全体から一様に1つ選ぶ。"""
    layout = problem.layout
    labels: dict[int, int] = {}
    for i in order:
        initial, draws = proposed[i]
        for position, tile_type in zip(layout.initial_positions(i), _shuffled(initial, chooser)):
            labels[position] = tile_type  # 段1：配牌の多重集合を配牌位置へ一様に並べる
        for position, tile_type in zip(layout.draw_positions(i), tedashi_draws(layout, i, draws)):
            labels[position] = tile_type
    for position, tile_type in zip(layout.pool_positions, arrange_pool_types(sorted(pool.elements()), chooser)):
        labels[position] = tile_type
    by_type: dict[int, list[int]] = {}
    for position in labels:
        tile_id = state.ids[position]
        by_type.setdefault(layout.type_of[tile_id], []).append(tile_id)
    for tile_type in sorted(by_type):
        # 段2：その牌種の物理IDを、その牌種の位置へ一様に割り当てる（非復元）。
        positions = [p for p in sorted(labels) if labels[p] == tile_type]
        tile_ids = _shuffled(sorted(by_type[tile_type]), chooser)
        if len(positions) != len(tile_ids):
            raise AssertionError("復元の牌種の計数がブロックと一致しない")
        for position, tile_id in zip(positions, tile_ids):
            state.ids[position] = tile_id


@dataclass(frozen=True)
class RegenerationBlock:
    """M3・M4のブロック（移動する家の未知の位置とプール）の現在の牌種状態。"""

    order: tuple[int, ...]  # 生成順（リーチ者が先）
    current: Mapping[int, SeatTypes]
    pool: Counter  # 現在のプールの牌種
    block: Counter  # ブロック内の牌種の合計（移動で変わらない）
    available: Counter  # block − 手出しの打牌（判断時点の手を引く元）


def regeneration_block(problem: BeliefProblem, ids: Sequence[int], seats: Sequence[int]) -> RegenerationBlock:
    layout = problem.layout
    chosen = set(seats)
    order = tuple(i for i in layout.m3_order if i in chosen)
    current = {i: seat_type_state(layout, ids, i) for i in order}
    pool = Counter(layout.type_of[ids[p]] for p in layout.pool_positions)
    block = Counter(pool)
    discards: Counter = Counter()
    for i in order:
        block.update(current[i][0])
        block.update(tedashi_draws(layout, i, current[i][1]))
        discards += tedashi_discards(layout, i)
    available = block - discards
    if sum(available.values()) + sum(discards.values()) != sum(block.values()):
        raise AssertionError("手出しの打牌がブロックにない")
    return RegenerationBlock(order, current, pool, block, available)


def propose_regeneration(problem: BeliefProblem, block: RegenerationBlock, chooser: Any) -> dict[int, SeatTypes] | None:
    """新しい牌種状態x'を作る（判断時点の手の生成と自摸の逆算）。生成器が上限に達したらNone。

    結果はブロックの牌種の合計と手出しの打牌だけに依存し、物理配置には依存しない。
    """
    hands = propose_final_hands(problem, block.order, block.available, chooser)
    if hands is None:
        return None
    return {i: backward_history(problem.layout, i, hand, chooser) for i, hand in zip(block.order, hands)}


def regeneration_pool(problem: BeliefProblem, block: RegenerationBlock, proposed: Mapping[int, SeatTypes]) -> Counter:
    pool = Counter(block.block)
    for i in block.order:
        pool.subtract(proposed[i][0])
        pool.subtract(tedashi_draws(problem.layout, i, proposed[i][1]))
    if any(count < 0 for count in pool.values()):
        raise AssertionError("提案がブロックの牌種を超える")
    return +pool


def regeneration_log_ratio(
    problem: BeliefProblem, block: RegenerationBlock, log_factors: Sequence[float], proposed: Mapping[int, SeatTypes]
) -> tuple[float, dict[int, float]]:
    """log([w(x')L(x')q(x)] / [w(x)L(x)q(x')]) と、新しい家ごとの因子。"""
    layout = problem.layout
    order = block.order
    hands = [seat_final_hand(layout, i, *proposed[i]) for i in order]
    current_hands = [seat_final_hand(layout, i, *block.current[i]) for i in order]
    log_q_new = final_hands_log_density(problem, order, block.available, hands)  # type: ignore[arg-type]
    log_q_old = final_hands_log_density(problem, order, block.available, current_hands)  # type: ignore[arg-type]
    log_w_new = pool_log_weight(regeneration_pool(problem, block, proposed))
    log_w_old = pool_log_weight(block.pool)
    for i in order:
        log_q_new += backward_log_density(layout, i, *proposed[i])
        log_q_old += backward_log_density(layout, i, *block.current[i])
        log_w_new += initial_log_weight(proposed[i][0])
        log_w_old += initial_log_weight(block.current[i][0])
    factors = {i: seat_log_factor(problem, i, proposed[i]) for i in order}
    if not all(math.isfinite(value) for value in factors.values()):
        # 構成でH=1を保証する（打牌整合は逆算、テンパイは生成器）。起きれば実装の誤り。
        raise AssertionError(f"再生成の提案がH=0: {factors}")
    log_ratio = (log_w_new + math.fsum(factors.values()) + log_q_old) - (
        log_w_old + math.fsum(log_factors[i] for i in order) + log_q_new
    )
    return log_ratio, factors


def regenerate_step(
    problem: BeliefProblem, state: ChainState, seats: Sequence[int], chooser: Any, stats: MoveStatistics, move: str
) -> None:
    """seatsの未知の位置とプールをブロックとする独立型MH（M3は1家、M4は他家全員）。

    α = min(1, [w(x')L(x')q(x)] / [w(x)L(x)q(x')])。提案が同じ牌種状態でもα=1で復元する。
    棄却したときは物理配置も変えない（牌種水準のMHの後の一様な復元と同じ遷移になる）。
    """
    stats.proposed[move] += 1
    block = regeneration_block(problem, state.ids, seats)
    proposed = propose_regeneration(problem, block, chooser)
    if proposed is None:
        stats.rejected[f"{move}:generator_cap"] += 1
        return
    log_ratio, factors = regeneration_log_ratio(problem, block, state.log_factors, proposed)
    if not chooser.accept(log_ratio):
        stats.rejected[f"{move}:mh"] += 1
        return
    restore_block(problem, state, block.order, proposed, regeneration_pool(problem, block, proposed), chooser)
    for i in block.order:
        state.log_factors[i] = factors[i]
    stats.accepted[move] += 1
    r = problem.layout.riichi_index
    if r is not None and r in block.order:
        before = tenpai_families(problem.rules, class_counts(problem, seat_final_hand(problem.layout, r, *block.current[r])))  # type: ignore[arg-type]
        after = tenpai_families(problem.rules, class_counts(problem, seat_final_hand(problem.layout, r, *proposed[r])))  # type: ignore[arg-type]
        if before != after:
            stats.cross_family[move] += 1



def m3_step(problem: BeliefProblem, state: ChainState, i: int, chooser: Any, stats: MoveStatistics) -> None:
    regenerate_step(problem, state, (i,), chooser, stats, f"m3:{problem.layout.seats[i].seat}")


def m4_step(problem: BeliefProblem, state: ChainState, chooser: Any, stats: MoveStatistics) -> None:
    regenerate_step(problem, state, range(len(problem.layout.seats)), chooser, stats, "m4")


ALL_MOVES = ("m1", "m2", "m3", "m4")


def mcmc_iteration(
    problem: BeliefProblem, state: ChainState, chooser: Any, stats: MoveStatistics, moves: Sequence[str] = ALL_MOVES
) -> None:
    """1反復＝順序付き合成（7.5節）：M1を|S|回、M2を|S_r|回、M3を家ごとに1回、最後にM4を1回。

    movesで移動を外せるのは診断の検査（D33-04）だけ。本番の実行は全移動を使う。
    """
    layout = problem.layout
    if "m1" in moves:
        for _ in range(layout.seat_size):
            m1_step(problem, state, chooser, stats)
    if "m2" in moves and layout.riichi_index is not None:
        for _ in range(len(layout.seat_positions(layout.riichi_index))):
            m2_step(problem, state, chooser, stats)
    if "m3" in moves:
        for i in layout.m3_order:
            m3_step(problem, state, i, chooser, stats)
    if "m4" in moves:
        m4_step(problem, state, chooser, stats)  # 到達可能性の論証のため必ず最後（7.4節）


# ---------------------------------------------------------------------------
# 規則：小例用（完成形の明示的な一覧）と麻雀用（面子手・七対子・国士）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplicitTenpaiRules:
    """小例用のテンパイ形の規則。系統ごとに完成形（類の列）の一覧から一様に選ぶ。"""

    families: tuple[tuple[str, float], ...]
    shapes: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]
    class_count: int
    hand_size: int

    def _shapes(self, family: str) -> tuple[tuple[int, ...], ...]:
        return dict(self.shapes)[family]

    def sample_complete(self, family: str, chooser: Any) -> list[int]:
        options = self._shapes(family)
        return list(options[chooser.index(len(options))])

    def complete_probability(self, family: str, counts: tuple[int, ...]) -> float:
        options = self._shapes(family)
        hits = sum(1 for shape in options if tuple(shape.count(c) for c in range(self.class_count)) == counts)
        return hits / len(options)

    def is_tenpai(self, counts: tuple[int, ...]) -> bool:
        for family, _ in self.families:
            for removed in range(self.class_count):
                complete = counts[:removed] + (counts[removed] + 1,) + counts[removed + 1:]
                if self.complete_probability(family, complete) > 0:
                    return True
        return False


MENTSU_SHAPE_COUNT = len(MENTSU_KINDS)  # 55（刻子34、順子21）


def _ordered_mentsu_count(counts: tuple[int, ...], start: int, remaining: int) -> int:
    """countsを面子remaining個へ分ける、面子の順序付きの列の数（枚数の上限なし）。

    最小の牌種iは、iの刻子かiから始まる順子に必ず入る。その個数(t, s)の組で分けると、
    面子の多重集合を重複なく数えられ、順序付きの数は多項係数 Π C(k, t)C(k−t, s) になる。
    """
    i = start
    while i < 34 and counts[i] == 0:
        i += 1
    if i == 34:
        return 1 if remaining == 0 else 0
    n = counts[i]
    total = 0
    for triplets in range(n // 3 + 1):
        runs = n - 3 * triplets
        if triplets + runs > remaining:
            continue
        following = list(counts)
        following[i] = 0
        if runs:
            if i >= 27 or i % 9 > 6 or counts[i + 1] < runs or counts[i + 2] < runs:
                continue
            following[i + 1] -= runs
            following[i + 2] -= runs
        total += math.comb(remaining, triplets) * math.comb(remaining - triplets, runs) * _ordered_mentsu_count(
            tuple(following), i + 1, remaining - triplets - runs
        )
    return total


@lru_cache(maxsize=500_000)
def regular_complete_paths(counts: tuple[int, ...]) -> int:
    """14枚の完成形Cに至る面子手の生成経路（雀頭、順序付きの面子4つ）の数。"""
    total = 0
    for pair in range(34):
        if counts[pair] >= 2:
            rest = list(counts)
            rest[pair] -= 2
            total += _ordered_mentsu_count(tuple(rest), 0, 4)
    return total


def regular_path_numerator(counts: tuple[int, ...]) -> int:
    """面子手の項の分子（共通分母34×55⁴×14）：34種すべての除去牌xについてΣ 経路数×c_{h+x}(x)。"""
    total = 0
    for removed in range(34):
        complete = counts[:removed] + (counts[removed] + 1,) + counts[removed + 1:]
        total += regular_complete_paths(complete) * complete[removed]
    return total


@dataclass(frozen=True)
class MahjongTenpaiRules:
    """麻雀のテンパイ形の生成器g_r（7.3節）。sample_tenpai_shapeと同じ構成を選択器で引く。"""

    families: tuple[tuple[str, float], ...] = TENPAI_FAMILY_PROBABILITIES
    class_count: int = 34
    hand_size: int = 13

    def sample_complete(self, family: str, chooser: Any) -> list[int]:
        if family == "regular":
            tiles = [chooser.index(34)] * 2
            for _ in range(4):
                tiles.extend(MENTSU_KINDS[chooser.index(MENTSU_SHAPE_COUNT)])
            return tiles
        if family == "chiitoi":
            return [t for t in _draw_without_replacement(range(34), 7, chooser) for _ in range(2)]
        return [*YAOCHUU, YAOCHUU[chooser.index(len(YAOCHUU))]]

    def complete_probability(self, family: str, counts: tuple[int, ...]) -> float:
        if family == "regular":
            paths = regular_complete_paths(counts)
            return paths / (34 * MENTSU_SHAPE_COUNT**4) if paths else 0.0
        if family == "chiitoi":
            values = [c for c in counts if c]
            return 1.0 / math.comb(34, 7) if values == [2] * 7 else 0.0
        orphans = sum(counts[t] for t in YAOCHUU)
        if orphans == 14 and all(counts[t] >= 1 for t in YAOCHUU):
            return 1.0 / len(YAOCHUU)
        return 0.0

    def is_tenpai(self, counts: tuple[int, ...]) -> bool:
        return shanten(counts, 0) == 0


# ---------------------------------------------------------------------------
# 実局面への接続：判断文脈と初期化の割当から問題と鎖の状態を作る
# ---------------------------------------------------------------------------


def type_of_key(key: tuple[int, bool]) -> int:
    return key[0] + 34 * int(key[1])


def key_of_type(tile_type: int) -> tuple[int, bool]:
    return tile_type % 34, tile_type >= 34


TYPE_OF_ID: tuple[int, ...] = tuple(type_of_key(id_key(tile_id)) for tile_id in range(136))
MAHJONG_CLASS_OF: dict[int, int] = {tile_type: tile_type % 34 for tile_type in set(TYPE_OF_ID)}


class MahjongSeatModel:
    """家ごとの因子を履歴評価器（5節）で計算する。resolver=Noneなら尤度1（規則だけ）。

    評価器の結果は牌種だけで決まるので、牌種ごとに物理IDを順に当てて評価する。
    """

    def __init__(
        self, context: DecisionContext, layout: BeliefLayout, resolver: ModelResolver | None, cache: WindowCache | None = None
    ):
        self.context = context
        self.layout = layout
        self.resolver = resolver
        self.cache = cache
        self.evaluations = 0

    def hypothesis(self, i: int, initial: Sequence[int], draws: Sequence[int]) -> SeatHypothesis:
        seat = self.layout.seats[i].seat
        supply: dict[int, list[int]] = {}

        def take(tile_type: int) -> int:
            return supply.setdefault(tile_type, ids_of_key(key_of_type(tile_type))).pop(0)

        initial_ids = tuple(take(t) for t in initial)
        draw_ids = {turn.raw_event_index: take(t) for turn, t in zip(self.context.turns[seat], draws)}
        return SeatHypothesis(seat, initial_ids, draw_ids)

    def log_factor(self, i: int, initial: Sequence[int], draws: Sequence[int]) -> float:
        self.evaluations += 1
        evaluation = evaluate_seat(self.context, self.hypothesis(i, initial, draws), self.resolver, self.cache)
        if evaluation.holds:
            raise RuleUnresolvedError(f"{self.context.decision_id}:{self.layout.seats[i].seat}:{evaluation.holds}")
        return evaluation.log_likelihood


@dataclass
class MahjongBelief:
    """判断1件のMCMCの問題と、位置⇔割当の変換に要る固定部分（対象家の牌、表示牌、ツモ切りの自摸）。"""

    context: DecisionContext
    problem: BeliefProblem
    target_initial: tuple[int, ...]
    target_draws: Mapping[int, int]
    dora_indicator: int
    fixed_draws: Mapping[int, Mapping[int, int]]  # 家 → ツモ切り窓のrawEventIndex → 物理ID

    def ids_from_world(self, world: WorldAssignment) -> list[int]:
        ids: list[int] = []
        for seat_layout in self.problem.layout.seats:
            hypothesis = world.hypotheses[seat_layout.seat]
            ids.extend(hypothesis.initial)
            turns = self.context.turns[seat_layout.seat]
            ids.extend(hypothesis.draws[turn.raw_event_index] for turn in turns if not turn.tsumogiri)
        ids.extend(world.pool)
        return ids

    def world_from_state(self, state: ChainState) -> WorldAssignment:
        layout = self.problem.layout
        hypotheses = {}
        for i, seat_layout in enumerate(layout.seats):
            seat = seat_layout.seat
            draws = dict(self.fixed_draws[seat])
            tedashi = [turn for turn in self.context.turns[seat] if not turn.tsumogiri]
            for turn, position in zip(tedashi, layout.draw_positions(i)):
                draws[turn.raw_event_index] = state.ids[position]
            initial = tuple(state.ids[p] for p in layout.initial_positions(i))
            hypotheses[seat] = SeatHypothesis(seat, initial, draws)
        pool = tuple(state.ids[p] for p in layout.pool_positions)
        return WorldAssignment(hypotheses, self.target_initial, self.target_draws, self.dora_indicator, pool)


def mahjong_belief(
    context: DecisionContext,
    world: WorldAssignment,
    resolver: ModelResolver | None,
    *,
    probabilities: Sequence[tuple[str, float]] = TENPAI_FAMILY_PROBABILITIES,
    max_generator_attempts: int = 1000,
    cache: WindowCache | None = None,
) -> tuple[MahjongBelief, ChainState]:
    """初期化の割当（6節）から、判断1件の問題と出発点の状態を作る。"""
    validate_family_probabilities(probabilities)
    world.validate()
    seats = tuple(
        SeatLayout(seat, seat == context.riichi_seat,
                   tuple(TurnLayout(turn.tsumogiri, type_of_key(turn.discard)) for turn in context.turns[seat]))
        for seat in context.other_seats
    )
    layout = BeliefLayout(13, seats, len(world.pool), TYPE_OF_ID, MAHJONG_CLASS_OF)
    problem = BeliefProblem(layout, MahjongTenpaiRules(tuple(probabilities)),
                            MahjongSeatModel(context, layout, resolver, cache), max_generator_attempts)
    fixed = {
        seat: {turn.raw_event_index: world.hypotheses[seat].draws[turn.raw_event_index]
               for turn in context.turns[seat] if turn.tsumogiri}
        for seat in context.other_seats
    }
    belief = MahjongBelief(context, problem, world.target_initial, world.target_draws, world.dora_indicator, fixed)
    return belief, initial_chain_state(problem, belief.ids_from_world(world))


# ===========================================================================
# 工程5：鎖の実行、標本の要約、診断、hold、判断単位の実行（PHASE_D33_DESIGN.md 9〜12節）
# ===========================================================================

BELIEF_VERSION = "ev-policy-belief-mcmc/v1"
POSTERIOR_SCHEMA = "ev-policy-belief-posterior/v1"
HOLD_REASONS = (
    "init_failed",
    "chain_disagreement",
    "resource_budget_exceeded",
    "fixed_component_sensitivity_missing",
    "rule_unresolved",
    "state_reconstruction_mismatch",
)


@dataclass(frozen=True)
class ChainSettings:
    """鎖の長さ（7.5節）。checkpointsは同じ鎖の途中経過として診断する反復数（例：10,000と40,000）。"""

    iterations: int
    burn_in: int
    thin: int
    checkpoints: tuple[int, ...] = ()
    moves: tuple[str, ...] = ALL_MOVES

    def __post_init__(self) -> None:
        if self.iterations < 1 or not 0 <= self.burn_in < self.iterations or self.thin < 1:
            raise ValueError("反復数・burn-in・保存間隔が不正")
        if any(not self.burn_in < c <= self.iterations for c in self.checkpoints):
            raise ValueError("途中経過はburn-inより後、反復数以下")
        if not set(self.moves) <= set(ALL_MOVES):
            raise ValueError("未知の移動")

    def to_dict(self) -> dict[str, Any]:
        return {"iterations": self.iterations, "burnIn": self.burn_in, "thin": self.thin,
                "checkpoints": list(self.checkpoints), "moves": list(self.moves)}


@dataclass(frozen=True)
class DisagreementThresholds:
    """chain_disagreementの閾値。pilot後に固定する（10.2節）。Noneの間は数値を報告するだけ。"""

    max_rhat: float | None = None
    min_shape_overlap: float | None = None
    require_supported_families_visited: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"maxRhat": self.max_rhat, "minShapeOverlap": self.min_shape_overlap,
                "requireSupportedFamiliesVisited": self.require_supported_families_visited}


class BudgetExceeded(RuntimeError):
    """実行中に時間の上限を超えた（resource_budget_exceeded）。鎖や反復を減らして続けない。"""


@dataclass
class ChainRun:
    samples: list[dict[str, Any]]
    checkpoint_sizes: dict[int, int]  # 途中経過の反復数 → その時点の標本数
    stats: MoveStatistics
    seconds: float


def run_chain(
    problem: BeliefProblem,
    state: ChainState,
    chooser: Any,
    settings: ChainSettings,
    summarize: Callable[[ChainState], dict[str, Any]],
    deadline: float | None = None,
) -> ChainRun:
    """1本の鎖を走らせ、burn-in後に保存間隔ごとの要約を集める。deadline（time.time()）を超えたら止める。"""
    import time

    start = time.perf_counter()
    stats = MoveStatistics()
    samples: list[dict[str, Any]] = []
    checkpoints: dict[int, int] = {}
    for iteration in range(1, settings.iterations + 1):
        mcmc_iteration(problem, state, chooser, stats, settings.moves)
        if iteration > settings.burn_in and (iteration - settings.burn_in) % settings.thin == 0:
            samples.append(summarize(state))
        if iteration in settings.checkpoints:
            checkpoints[iteration] = len(samples)
        if deadline is not None and time.time() > deadline:
            raise BudgetExceeded(f"iteration {iteration}")
    return ChainRun(samples, checkpoints, stats, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# 標本の要約
# ---------------------------------------------------------------------------


def riichi_waits(rules: Any, counts: tuple[int, ...]) -> tuple[int, ...]:
    """小例の待ち：hに1枚足すといずれかの系統の完成形になる類。"""
    waits = []
    for removed in range(rules.class_count):
        complete = counts[:removed] + (counts[removed] + 1,) + counts[removed + 1:]
        if any(rules.complete_probability(family, complete) > 0 for family, _ in rules.families):
            waits.append(removed)
    return tuple(waits)


def generic_summary(problem: BeliefProblem, state: ChainState) -> dict[str, Any]:
    """家ごとの判断時点の手（牌種）、プール、リーチ者の形・系統・待ち（小例の標本）。"""
    layout = problem.layout
    hands = {}
    for i, seat_layout in enumerate(layout.seats):
        final = seat_final_hand(layout, i, *seat_type_state(layout, state.ids, i))
        hands[str(seat_layout.seat)] = sorted(final.elements())  # type: ignore[union-attr]
    pool = Counter(layout.type_of[state.ids[p]] for p in layout.pool_positions)
    sample: dict[str, Any] = {"hands": hands, "pool": {str(t): n for t, n in sorted(pool.items())}}
    r = layout.riichi_index
    if r is not None:
        counts = class_counts(problem, Counter(hands[str(layout.seats[r].seat)]))
        sample["riichiShape"] = list(counts)
        sample["riichiFamilies"] = list(tenpai_families(problem.rules, counts))
        sample["riichiWaits"] = list(riichi_waits(problem.rules, counts))
    return sample


def generic_scalars(sample: Mapping[str, Any]) -> dict[str, float]:
    """R̂と実効標本数を計算する要約統計：待ちの周辺、系統、（あれば）フリテンとドラ枚数。"""
    values: dict[str, float] = {}
    for tile in sample.get("riichiWaits", ()):
        values[f"wait:{tile}"] = 1.0
    for family in sample.get("riichiFamilies", ()):
        values[f"family:{family}"] = 1.0
    if "riichiFuriten" in sample:
        values["furiten"] = float(any(sample["riichiFuriten"].values()))
    for seat, count in sample.get("doraCounts", {}).items():
        values[f"dora:{seat}"] = float(count)
    return values


def sample_shape(sample: Mapping[str, Any]) -> tuple:
    return tuple(sample.get("riichiShape", ()))


def sample_family(sample: Mapping[str, Any]) -> str:
    return "+".join(sample.get("riichiFamilies", ())) or "none"


# ---------------------------------------------------------------------------
# 系統の支持（10.1節：他家の配置を固定しない、牌在庫とrの打牌整合の下での判定）
# ---------------------------------------------------------------------------


def _regular_supported(supply: Sequence[int]) -> bool:
    """雀頭と面子4つの完成形Cで、Cから1枚除いた13枚がsupplyに収まるものがあるか。"""

    def excess(values: Sequence[int]) -> int:
        return sum(max(0, v - c) for v, c in zip(values, supply))

    def walk(values: list[int], start: int, depth: int) -> bool:
        if depth == 4:
            return True
        for kind in range(start, len(MENTSU_KINDS)):
            for t in MENTSU_KINDS[kind]:
                values[t] += 1
            if excess(values) <= 1 and walk(values, kind, depth + 1):
                return True
            for t in MENTSU_KINDS[kind]:
                values[t] -= 1
        return False

    for pair in range(34):
        values = [0] * 34
        values[pair] = 2
        if excess(values) <= 1 and walk(values, 0, 0):
            return True
    return False


def family_supported(rules: Any, family: str, supply: Sequence[int]) -> bool:
    """類ごとの在庫supplyの下で、その系統のテンパイ形を作れるか。"""
    if isinstance(rules, MahjongTenpaiRules):
        if family == "regular":
            return _regular_supported(supply)
        if family == "chiitoi":
            pairs = sum(value >= 2 for value in supply)
            singles = sum(value >= 1 for value in supply)
            return pairs >= 7 or (pairs >= 6 and singles >= 7)
        for duplicate in YAOCHUU:
            for removed in YAOCHUU:
                counts = Counter(YAOCHUU)
                counts[duplicate] += 1
                counts[removed] -= 1
                if all(supply[t] >= n for t, n in counts.items()):
                    return True
        return False
    for shape in rules._shapes(family):
        for removed in set(shape):
            hand = Counter(shape)
            hand[removed] -= 1
            if all(supply[c] >= n for c, n in hand.items()):
                return True
    return False


def family_support(problem: BeliefProblem, ids: Sequence[int]) -> dict[str, bool]:
    """ブロック全体（他家の未知の位置とプール）から全員の手出しの打牌を除いた在庫で判定する。"""
    layout = problem.layout
    if layout.riichi_index is None:
        return {}
    supply = Counter(layout.type_of[ids[p]] for p in range(layout.size))
    for i in range(len(layout.seats)):
        supply -= tedashi_discards(layout, i)
    counts = class_counts(problem, supply)
    return {family: family_supported(problem.rules, family, counts) for family, _ in problem.rules.families}


# ---------------------------------------------------------------------------
# 診断（10.1節）
# ---------------------------------------------------------------------------


def split_rhat(chains: Sequence[Sequence[float]]) -> float | None:
    """鎖を前後半に分けたR̂。分散0の鎖どうしで平均が異なればinf（別の状態に閉じ込められている）。"""
    if len(chains) < 2 or min(len(c) for c in chains) < 4:
        return None
    half = min(len(c) for c in chains) // 2
    pieces = [np.asarray(c[:half], dtype=float) for c in chains] + [np.asarray(c[half:2 * half], dtype=float) for c in chains]
    means = np.asarray([piece.mean() for piece in pieces])
    within = float(np.mean([piece.var(ddof=1) for piece in pieces]))
    between = half * float(means.var(ddof=1))
    if within == 0.0:
        return 1.0 if between == 0.0 else math.inf
    pooled = (half - 1) / half * within + between / half
    return math.sqrt(pooled / within)


def batch_means_ess(series: Sequence[float]) -> float | None:
    """batch meansによる実効標本数（バッチの大きさ≒√n）。値が一定ならNone（定義できない）。"""
    n = len(series)
    size = int(math.sqrt(n))
    batches = n // size if size else 0
    if batches < 2:
        return None
    values = np.asarray(series[: batches * size], dtype=float)
    variance = float(values.var(ddof=1))
    if variance == 0.0:
        return None
    batch_variance = size * float(values.reshape(batches, size).mean(axis=1).var(ddof=1))
    if batch_variance == 0.0:
        return float(n)
    return n * variance / batch_variance


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def chain_diagnostics(
    chains: Sequence[Sequence[Mapping[str, Any]]],
    stats: Sequence[MoveStatistics],
    support: Mapping[str, bool],
    thresholds: DisagreementThresholds,
    scalars: Callable[[Mapping[str, Any]], dict[str, float]] = generic_scalars,
    shape: Callable[[Mapping[str, Any]], Any] = sample_shape,
    family: Callable[[Mapping[str, Any]], str] = sample_family,
) -> dict[str, Any]:
    """鎖ごとの標本と移動の統計から、10.1節の診断とchain_disagreementの判定を作る。"""
    names = sorted({name for chain in chains for sample in chain for name in scalars(sample)})
    series = {name: [[scalars(sample).get(name, 0.0) for sample in chain] for chain in chains] for name in names}
    summary: dict[str, Any] = {}
    rhats = []
    for name in names:
        rhat = split_rhat(series[name])
        esses = [batch_means_ess(values) for values in series[name]]
        ess = sum(value for value in esses if value is not None) if any(v is not None for v in esses) else None
        mean = float(np.mean([v for values in series[name] for v in values])) if any(series[name]) else None
        summary[name] = {"rhat": rhat, "ess": ess, "mean": mean}
        if rhat is not None:
            rhats.append(rhat)
    shape_sets = [{shape(sample) for sample in chain} for chain in chains]
    pairs = [_jaccard(a, b) for k, a in enumerate(shape_sets) for b in shape_sets[k + 1:]]
    union = set().union(*shape_sets) if shape_sets else set()
    visits = []
    for chain in chains:
        labels = Counter(family(sample) for sample in chain)
        visits.append({label: count / len(chain) for label, count in sorted(labels.items())} if chain else {})
    visited = {part for chain in chains for sample in chain for part in family(sample).split("+")}
    unvisited = sorted(name for name, ok in support.items() if ok and name not in visited)
    total = MoveStatistics()
    for item in stats:
        for field_name in ("proposed", "accepted", "rejected", "null_moves", "cross_family"):
            getattr(total, field_name).update(getattr(item, field_name))
    moves = total.as_dict()
    moves["acceptanceRate"] = {move: total.accepted[move] / count for move, count in sorted(total.proposed.items()) if count}
    result = {
        "samplesPerChain": [len(chain) for chain in chains],
        "scalars": summary,
        "maxRhat": max(rhats) if rhats else None,
        "shapes": {
            "perChain": [len(item) for item in shape_sets],
            "distinct": len(union),
            "sharedByAllChains": len(set.intersection(*shape_sets)) if shape_sets else 0,
            "minPairwiseOverlap": min(pairs) if pairs else None,
            "meanPairwiseOverlap": float(np.mean(pairs)) if pairs else None,
        },
        "families": {"support": dict(support), "visitsPerChain": visits, "unvisitedSupported": unvisited,
                     "crossFamilyAccepted": moves["cross_family"]},
        "moves": moves,
    }
    violations = []
    if thresholds.max_rhat is not None and (result["maxRhat"] is None or result["maxRhat"] > thresholds.max_rhat):
        violations.append("rhat")
    overlap = result["shapes"]["minPairwiseOverlap"]
    if thresholds.min_shape_overlap is not None and (overlap is None or overlap < thresholds.min_shape_overlap):
        violations.append("shape_overlap")
    if thresholds.require_supported_families_visited and unvisited:
        violations.append("unvisited_supported_family")
    result["chainDisagreement"] = {"thresholds": thresholds.to_dict(), "violations": violations,
                                   "judged": any(v is not None for v in thresholds.to_dict().values())}
    return result


# ---------------------------------------------------------------------------
# 鎖の組の実行とhold（10.2節）：一様配布への退避、成功した鎖だけの集計、判断の差替えをしない
# ---------------------------------------------------------------------------


@dataclass
class ChainStart:
    """鎖1本の出発点。problemは鎖ごと（ツモ切り窓の物理IDなど、出発点で固定する部分がある）。"""

    problem: BeliefProblem | None
    state: ChainState | None
    summarize: Callable[[ChainState], dict[str, Any]] | None
    info: dict[str, Any]


def run_chain_group(
    starts: Callable[[int, int], ChainStart],
    seeds: Sequence[int],
    settings: ChainSettings,
    thresholds: DisagreementThresholds,
    *,
    deadline: float | None = None,
    scalars: Callable[[Mapping[str, Any]], dict[str, float]] = generic_scalars,
) -> dict[str, Any]:
    """同じ目標の独立な鎖を走らせ、診断とholdをまとめる。heldなら標本を返さない。"""
    import time

    begin = time.perf_counter()
    record: dict[str, Any] = {"status": "ok", "holdReasons": [], "seeds": list(seeds), "chains": len(seeds),
                              "settings": settings.to_dict(), "initialization": []}

    def held(reason: str, detail: Any = None) -> dict[str, Any]:
        record["status"] = "held"
        record["holdReasons"].append(reason)
        if detail is not None:
            record.setdefault("holdDetails", {})[reason] = detail
        record["samples"] = None
        record["runtime"] = {"seconds": round(time.perf_counter() - begin, 4)}
        return record

    chain_starts = []
    try:
        for chain, seed in enumerate(seeds):
            start = starts(chain, seed)
            record["initialization"].append(start.info)
            chain_starts.append(start)
    except RuleUnresolvedError as error:
        return held("rule_unresolved", str(error))
    if any(start.state is None for start in chain_starts):
        return held("init_failed", [start.info for start in chain_starts if start.state is None])
    runs = []
    try:
        for chain, (start, seed) in enumerate(zip(chain_starts, seeds)):
            runs.append(run_chain(start.problem, start.state, RandomChooser(random.Random(seed + 1)),  # type: ignore[arg-type]
                                  settings, start.summarize, deadline))  # type: ignore[arg-type]
    except RuleUnresolvedError as error:
        return held("rule_unresolved", str(error))
    except BudgetExceeded as error:
        return held("resource_budget_exceeded", {"completedChains": len(runs), "at": str(error)})
    first = chain_starts[0]
    support = family_support(first.problem, first.state.ids)  # type: ignore[arg-type,union-attr]
    samples = [run.samples for run in runs]
    stats = [run.stats for run in runs]
    record["diagnostics"] = chain_diagnostics(samples, stats, support, thresholds, scalars)
    record["checkpoints"] = {
        str(point): chain_diagnostics([run.samples[: run.checkpoint_sizes[point]] for run in runs], stats, support,
                                      thresholds, scalars)
        for point in settings.checkpoints
    }
    record["runtime"] = {"seconds": round(time.perf_counter() - begin, 4), "chainSeconds": [round(r.seconds, 4) for r in runs]}
    if record["diagnostics"]["chainDisagreement"]["violations"]:
        return held("chain_disagreement", record["diagnostics"]["chainDisagreement"]["violations"])
    record["samples"] = [{"chain": chain, **sample} for chain, run in enumerate(runs) for sample in run.samples]
    return record


# ---------------------------------------------------------------------------
# 実局面：判断1件の全シナリオ
# ---------------------------------------------------------------------------


def reconstruction_mismatches(context: DecisionContext) -> list[str]:
    """状態復元の検査（4.4節の4）：リーチ者の宣言が1回で、宣言後の打牌はすべてツモ切り。"""
    problems = []
    turns = context.turns[context.riichi_seat]
    declarations = [k for k, turn in enumerate(turns) if turn.riichi_declaration]
    if len(declarations) != 1:
        problems.append("riichi_declaration_count")
    elif any(not turn.tsumogiri for turn in turns[declarations[0] + 1:]):
        problems.append("tedashi_after_riichi")
    return problems


def mahjong_summary(belief: MahjongBelief, state: ChainState) -> dict[str, Any]:
    """11節の標本：他家3人の手牌（牌種・赤）、rの待ちとフリテン、一発の資格、未割当プールの多重集合。"""
    from tools.ev_policy_opponent import _dora_from_indicator

    problem, context = belief.problem, belief.context
    sample = generic_summary(problem, state)
    world = belief.world_from_state(state)
    evaluation = evaluate_seat(context, world.hypotheses[context.riichi_seat], None)
    waits = waiting_tile34(counts34(evaluation.final_hand), 0)
    sample["riichiWaits"] = list(waits)
    sample["riichiFuriten"] = {
        "ownDiscard": bool(set(waits) & set(evaluation.river34)),
        "temporary": evaluation.temporary_furiten,
        "riichiPass": evaluation.riichi_furiten,
    }
    sample["ippatsu"] = evaluation.ippatsu
    dora = _dora_from_indicator(context.dora_indicator[0])
    sample["doraCounts"] = {seat: sum(1 for t in hand if t % 34 == dora) for seat, hand in sample["hands"].items()}
    sample["redCounts"] = {seat: sum(1 for t in hand if t >= 34) for seat, hand in sample["hands"].items()}
    return sample


def _chain_seed(base: int, decision_id: str, scenario_id: str, chain: int) -> int:
    digest = hashlib.sha256(f"{base}:{decision_id}:{scenario_id}:{chain}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def run_decision_scenarios(
    context: DecisionContext,
    model: HierarchicalSoftmax,
    scenarios: Sequence[Mapping[str, Any]],
    settings: ChainSettings,
    *,
    required_scenarios: Sequence[str],
    chains: int = 4,
    seed: int = 20261001,
    cache_capacity: int = 200_000,
    thresholds: DisagreementThresholds = DisagreementThresholds(),
    deadline: float | None = None,
    probabilities: Sequence[tuple[str, float]] = TENPAI_FAMILY_PROBABILITIES,
    init_attempts: int = 10_000,
    init_seconds: float = 60.0,
    metadata: Mapping[str, Any] | None = None,
    scenario_models: Mapping[str, HierarchicalSoftmax] | None = None,
) -> dict[str, Any]:
    """判断1件について、シナリオごとに独立な鎖を走らせる（9.1節）。窓キャッシュは全シナリオ・全鎖で共有する。

    必要なシナリオのどれかの鎖が欠ければ、判断全体をfixed_component_sensitivity_missingでheldにする。
    scenario_modelsはシナリオごとに別のθを使う場合（θ変種、9.3節）。モデル識別子が違うので学習分布は共有しない。
    """
    import time

    begin = time.perf_counter()
    cache = WindowCache(cache_capacity)
    target = context.target_private_row()
    base_record = {
        "schemaVersion": POSTERIOR_SCHEMA,
        "beliefVersion": BELIEF_VERSION,
        "decisionId": context.decision_id,
        "informationStateHash": context.information_state_hash,
        "privatePrefixHash": hashlib.sha256(_canonical_json(target).encode("utf-8")).hexdigest(),
        "inputHash": context.input_hash(),
        "familyProbabilities": dict(probabilities),
        **dict(metadata or {}),
    }
    mismatches = reconstruction_mismatches(context)
    outputs = []
    for scenario in scenarios:
        resolver = model_resolver((scenario_models or {}).get(str(scenario["id"]), model), scenario)
        record = {**base_record, "scenarioId": scenario["id"], "thetaId": resolver.model_id}
        if mismatches:
            record.update({"status": "held", "holdReasons": ["state_reconstruction_mismatch"],
                           "holdDetails": {"state_reconstruction_mismatch": mismatches}, "samples": None})
            outputs.append(record)
            continue

        def start(chain: int, chain_seed: int, resolver: ScenarioResolver = resolver) -> ChainStart:
            initial = construct_initial_world(context, random.Random(chain_seed), resolver=resolver,
                                              probabilities=probabilities, max_attempts=init_attempts,
                                              max_seconds=init_seconds, cache=cache)
            info = {"chain": chain, "seed": chain_seed, "status": initial.status, "family": initial.family,
                    "attempts": initial.attempts, "seconds": round(initial.seconds, 4),
                    "failureReasons": initial.failure_reasons}
            if initial.world is None:
                return ChainStart(None, None, None, info)
            belief, state = mahjong_belief(context, initial.world, resolver, probabilities=probabilities, cache=cache)
            return ChainStart(belief.problem, state, lambda s, belief=belief: mahjong_summary(belief, s), info)

        seeds = [_chain_seed(seed, context.decision_id, str(scenario["id"]), chain) for chain in range(chains)]
        record.update(run_chain_group(start, seeds, settings, thresholds, deadline=deadline))
        outputs.append(record)
        if "resource_budget_exceeded" in record["holdReasons"]:
            break  # 予算を超えたら残りのシナリオを走らせない（減らして続けない）
    present = {record["scenarioId"] for record in outputs if record["status"] == "ok"}
    missing = sorted(set(required_scenarios) - present)
    holds = sorted({reason for record in outputs for reason in record["holdReasons"]})
    if missing:
        holds = sorted(set(holds) | {"fixed_component_sensitivity_missing"})
    return {
        "decisionId": context.decision_id,
        "status": "held" if holds else "ok",
        "holdReasons": holds,
        "missingScenarios": missing,
        "scenarios": outputs,
        "cacheStatistics": cache.statistics(),
        "runtime": {"seconds": round(time.perf_counter() - begin, 4)},
    }


def belief_scenarios(fixed_components: Mapping[str, Any]) -> list[dict[str, Any]]:
    """D.3.3で事後分布を作るシナリオ（`zero`を除く22件、9.1節）。"""
    return [dict(s) for s in fixed_components["scenarios"] if s.get("usableInD33", True) and s["id"] != "zero"]


def _json_safe(value: Any) -> Any:
    """infやnanをJSONで表せる文字列に置き換える（R̂がinfになる例がある）。"""
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


_BELIEF_WORKER: dict[str, Any] = {}


def _belief_worker_init(dataset_dir: str, model_dir: str, decision_ids: Sequence[str], options: Mapping[str, Any]) -> None:
    from pathlib import Path

    _worker_init(dataset_dir, model_dir, decision_ids)
    install_fast_shape()  # 推定器のプロセスでだけ、形の計算を同じ値の速い実装へ差し替える（段階1のA1）
    models = Path(model_dir)
    fixed_path = models / "fixed-components.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    _BELIEF_WORKER.update({
        "model": HierarchicalSoftmax.from_dict(json.loads((models / "model.json").read_text(encoding="utf-8"))),
        "fixed": fixed,
        "fixedHash": _sha256_file(fixed_path),
        "options": dict(options),
    })


def decision_file_name(decision_id: str) -> str:
    """判断IDの出力ファイル名。IDの`:`はWindowsで使えないので置き換え、衝突を避けるハッシュを付ける。"""
    import re

    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", decision_id)
    return f"{safe}-{hashlib.sha256(decision_id.encode('utf-8')).hexdigest()[:8]}.json.gz"


def _belief_worker_run(decision_id: str) -> dict[str, Any]:
    import gzip
    from pathlib import Path

    options = _BELIEF_WORKER["options"]
    scenarios = belief_scenarios(_BELIEF_WORKER["fixed"])
    if options.get("scenarioIds"):
        scenarios = [s for s in scenarios if s["id"] in set(options["scenarioIds"])]
    settings = ChainSettings(**options["settings"])
    model = _BELIEF_WORKER["model"]
    required = [s["id"] for s in belief_scenarios(_BELIEF_WORKER["fixed"])]
    scenario_models = {}
    if options.get("thetaVariant") is not None:
        variant = load_theta_variant(model, options["thetaVariant"])
        scenario_models[THETA_VARIANT_ID] = variant
        required.append(THETA_VARIANT_ID)
        if not options.get("scenarioIds") or THETA_VARIANT_ID in options["scenarioIds"]:
            scenarios.append(theta_variant_scenario(belief_scenarios(_BELIEF_WORKER["fixed"]), variant))
    result = run_decision_scenarios(
        _WORKER["contexts"][decision_id], model, scenarios, settings,
        required_scenarios=required, scenario_models=scenario_models,
        chains=int(options["chains"]), seed=int(options["seed"]), cache_capacity=int(options["cacheCapacity"]),
        thresholds=DisagreementThresholds(**options.get("thresholds", {})), deadline=options.get("deadline"),
        metadata={"modelVersion": model.to_dict()["modelVersion"], "fixedComponentsHash": _BELIEF_WORKER["fixedHash"]},
    )
    path = Path(options["outputDir"]) / decision_file_name(decision_id)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, ensure_ascii=False)
    return {
        "decisionId": decision_id, "status": result["status"], "holdReasons": result["holdReasons"],
        "missingScenarios": result["missingScenarios"], "cacheStatistics": result["cacheStatistics"],
        "seconds": result["runtime"]["seconds"], "path": path.name,
    }


def run_belief_decisions(
    dataset_dir: Any,
    model_dir: Any,
    decision_ids: Sequence[str],
    output_dir: Any,
    settings: ChainSettings,
    *,
    chains: int = 4,
    seed: int = 20261001,
    processes: int = 3,
    cache_capacity: int = 200_000,
    wall_clock_seconds: float = 86_400.0,
    scenario_ids: Sequence[str] | None = None,
    thresholds: DisagreementThresholds = DisagreementThresholds(),
    theta_variant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """判断単位で並列に実行する（P≤3、12.3節）。同じ判断の全シナリオは同じプロセスでキャッシュを共有する。"""
    import multiprocessing
    import time
    from pathlib import Path

    if not 1 <= processes <= 3:
        raise ValueError("並列数は1〜3（設計12.3節）")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    options: dict[str, Any] = {
        "settings": {"iterations": settings.iterations, "burn_in": settings.burn_in, "thin": settings.thin,
                     "checkpoints": tuple(settings.checkpoints), "moves": tuple(settings.moves)},
        "chains": chains, "seed": seed, "cacheCapacity": cache_capacity, "outputDir": str(output_dir),
        "deadline": start + wall_clock_seconds, "scenarioIds": list(scenario_ids or []),
        "thetaVariant": dict(theta_variant) if theta_variant is not None else None,
        "thresholds": {"max_rhat": thresholds.max_rhat, "min_shape_overlap": thresholds.min_shape_overlap,
                       "require_supported_families_visited": thresholds.require_supported_families_visited},
    }
    rows = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes, initializer=_belief_worker_init,
                      initargs=(str(dataset_dir), str(model_dir), list(decision_ids), options)) as pool:
        for row in pool.imap_unordered(_belief_worker_run, list(decision_ids)):
            rows.append(row)
    rows.sort(key=lambda row: decision_ids.index(row["decisionId"]))
    report = {
        "schemaVersion": "ev-policy-belief-run/v1",
        "beliefVersion": BELIEF_VERSION,
        "settings": settings.to_dict(),
        "chains": chains, "seed": seed, "processes": processes, "cacheCapacity": cache_capacity,
        "scenarioIds": list(scenario_ids) if scenario_ids else "all",
        "thresholds": thresholds.to_dict(),
        "thetaVariant": None if theta_variant is None else {k: theta_variant.get(k) for k in ("thetaId", "bias", "manifestSha256")},
        "decisions": rows,
        "held": sum(row["status"] != "ok" for row in rows),
        "wallClockSeconds": round(time.time() - start, 1),
    }
    (output_dir / "run-summary.json").write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    return report


# ===========================================================================
# 工程6：θ変種 theta_riichi_response_recalibrated の較正（PHASE_D33_DESIGN.md 9.3節）
# ===========================================================================
#
# v3モデルはリーチ中の応答窓でポンを2〜4倍に過大予測する（D.3.2bレポート8節）。その偏りが
# 事後分布へ与える影響を測るための診断専用の変種。採用はしない。較正期間は温度の選択と共用する
# ので、較正期間での適合は独立な検証にならず、補正後の適合を性能改善の証拠と呼ばない。

THETA_VARIANT_ID = "theta_riichi_response_recalibrated"
THETA_VARIANT_MANIFEST_SCHEMA = "ev-policy-belief-theta-variant-manifest/v1"
THETA_VARIANT_RESULT_SCHEMA = "ev-policy-belief-theta-variant/v1"
THETA_VARIANT_KINDS = ("chi", "pon")
# 探索の設定（manifestへ写して固定する）。推定値を見る前に決めた値で、変えるときはmanifestを作り直す。
THETA_VARIANT_SEARCH = {
    "min": -6.0,
    "max": 3.0,
    "gridSteps": [0.5, 0.1, 0.02],  # 粗い格子の最良点の周り±1段の範囲を、次の細かさで探し直す
    "regularization": None,
}
THETA_VARIANT_MINIMUM_SUPPORT = {"legalResponderWindows": 200, "observedCalls": 20}  # 種別ごと
THETA_VARIANT_HOLD_CONDITIONS = (
    "insufficient_support",  # どちらかの種別で合法な応答者窓または観測した鳴きが最小件数未満
    "boundary_optimum",  # 最良点が探索範囲の境界
    "non_finite_likelihood",  # 尤度が0または非有限になる対象窓がある
    "support_mismatch",  # 推定時に数え直した支持件数がmanifestと一致しない
)


def _opponent_riichi_index() -> int:
    from tools.ev_policy_opponent import KIND_FEATURE_NAMES

    return KIND_FEATURE_NAMES.index("opponent_riichi_count")


@dataclass
class RiichiResponseBiasSoftmax(HierarchicalSoftmax):
    """discard_responseで、応答者から見たリーチ人数が1以上の窓に限り、種別スコアにβを加えるθ変種。

    βは温度で割る前のスコアに加える：種別ロジット = (w_k·f + β_k) / T。
    """

    riichi_response_bias: dict[str, float] = field(default_factory=dict)

    def copy(self) -> "RiichiResponseBiasSoftmax":
        return RiichiResponseBiasSoftmax(self.kind_weights.copy(), self.detail_weights.copy(), dict(self.temperatures),
                                         self.fixed, dict(self.riichi_response_bias))

    def with_fixed(self, fixed: Any) -> "RiichiResponseBiasSoftmax":
        return RiichiResponseBiasSoftmax(self.kind_weights, self.detail_weights, dict(self.temperatures), fixed,
                                         dict(self.riichi_response_bias))

    def applies(self, candidates: Sequence[Any], phase: str) -> bool:
        return (
            phase == "discard_response"
            and bool(self.riichi_response_bias)
            and float(candidates[0].kind_features[_opponent_riichi_index()]) > 0.0
        )

    def base_probabilities(self, candidates: Sequence[Any], phase: str) -> np.ndarray:
        if not candidates or not self.applies(candidates, phase):
            return super().base_probabilities(candidates, phase)
        from tools.ev_policy_opponent import KIND_INDEX, _softmax

        temperature = float(self.temperatures.get(phase, 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"不正な温度: {phase}={temperature}")
        groups: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            groups.setdefault(candidate.kind, []).append(index)
        kinds = list(groups)
        kind_logits = np.asarray([
            float(self.kind_weights[KIND_INDEX[kind]] @ candidates[groups[kind][0]].kind_features)
            + float(self.riichi_response_bias.get(kind, 0.0))
            for kind in kinds
        ]) / temperature
        kind_probability = _softmax(kind_logits)
        result = np.zeros(len(candidates), dtype=float)
        for group_index, kind in enumerate(kinds):
            indices = groups[kind]
            detail_logits = np.asarray(
                [self.detail_weights[KIND_INDEX[kind]] @ candidates[index].detail_features for index in indices]
            ) / temperature
            result[indices] = kind_probability[group_index] * _softmax(detail_logits)
        return result


def theta_variant_model(model: HierarchicalSoftmax, bias: Mapping[str, float]) -> RiichiResponseBiasSoftmax:
    return RiichiResponseBiasSoftmax(model.kind_weights, model.detail_weights, dict(model.temperatures), model.fixed,
                                     {kind: float(bias[kind]) for kind in THETA_VARIANT_KINDS})


def _responder_targeted(candidates: Sequence[Any]) -> bool:
    """βが効く応答者窓：リーチ人数≥1で、チーかポンが合法。"""
    return float(candidates[0].kind_features[_opponent_riichi_index()]) > 0.0 and any(
        c.kind in THETA_VARIANT_KINDS for c in candidates)


def theta_variant_targets(windows: Any) -> list[dict[str, Any]]:
    """共同応答の窓のうち、βで尤度が変わるもの（少なくとも1人の応答者がβの対象）だけを残す。"""
    targets = []
    for encoded in windows:
        window = encoded.get("window", {})
        if "perSeat" not in encoded or window.get("phase") != "discard_response":
            continue
        if not window.get("learningMask", {}).get("jointKind", False):
            continue
        if any(_responder_targeted(candidates) for candidates in encoded["perSeat"].values()):
            targets.append(encoded)
    return targets


def theta_variant_support(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """対象窓の支持件数：窓数、種別ごとの合法な応答者窓と、対象の応答者が鳴いた観測数。"""
    legal: Counter = Counter()
    observed: Counter = Counter()
    for encoded in targets:
        resolution = encoded["window"]["observation"]["resolution"]
        for seat, candidates in encoded["perSeat"].items():
            if not _responder_targeted(candidates):
                continue
            kinds = {c.kind for c in candidates}
            for kind in THETA_VARIANT_KINDS:
                legal[kind] += int(kind in kinds)
            if resolution.get("kind") in THETA_VARIANT_KINDS and int(resolution.get("seat", -1)) == int(seat):
                observed[resolution["kind"]] += 1
    return {
        "targetWindows": len(targets),
        "legalResponderWindows": {kind: legal[kind] for kind in THETA_VARIANT_KINDS},
        "observedCalls": {kind: observed[kind] for kind in THETA_VARIANT_KINDS},
    }


def theta_variant_nll(model: HierarchicalSoftmax, targets: Sequence[Mapping[str, Any]], bias: Mapping[str, float]) -> float:
    """対象窓の共同応答の負の対数尤度（公開結果と両立する全希望の和、D.3.2と同じ尤度）。"""
    from tools.ev_policy_opponent import joint_resolution_likelihood

    variant = theta_variant_model(model, bias)
    total = []
    for encoded in targets:
        window = encoded["window"]
        try:
            likelihood, _ = joint_resolution_likelihood(
                variant, int(window["actorSeat"]), encoded["perSeat"], window["observation"]["resolution"], str(window["phase"]))
        except ValueError:
            return math.inf
        total.append(-math.log(likelihood))
    value = math.fsum(total)
    return value if math.isfinite(value) else math.inf


def _grid(low: float, high: float, step: float) -> list[float]:
    count = int(round((high - low) / step))
    return [round(low + k * step, 10) for k in range(count + 1)]


def estimate_theta_variant(
    model: HierarchicalSoftmax, targets: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """manifestの探索設定で、βを決定的な格子探索で推定する。失敗はholdとして返し、値を返さない。"""
    support = theta_variant_support(targets)
    holds = []
    if support != manifest["support"]:
        holds.append("support_mismatch")
    minimum = manifest["minimumSupport"]
    for kind in THETA_VARIANT_KINDS:
        if (support["legalResponderWindows"][kind] < minimum["legalResponderWindows"]
                or support["observedCalls"][kind] < minimum["observedCalls"]):
            holds.append("insufficient_support")
            break
    result: dict[str, Any] = {"support": support, "holdReasons": holds}
    if holds:
        return {**result, "status": "held", "bias": None}
    search = manifest["search"]
    low, high = float(search["min"]), float(search["max"])
    baseline = theta_variant_nll(model, targets, {"chi": 0.0, "pon": 0.0})
    best: tuple[float, float, float] | None = None
    path = []
    window_low = {kind: low for kind in THETA_VARIANT_KINDS}
    window_high = {kind: high for kind in THETA_VARIANT_KINDS}
    for step in search["gridSteps"]:
        evaluated = []
        for chi in _grid(window_low["chi"], window_high["chi"], step):
            for pon in _grid(window_low["pon"], window_high["pon"], step):
                evaluated.append((theta_variant_nll(model, targets, {"chi": chi, "pon": pon}), chi, pon))
        # 同値は小さいβ（chi、ponの順）を選ぶ：並べ方によらない決定的な選択
        value, chi, pon = min(evaluated)
        if not math.isfinite(value):
            return {**result, "status": "held", "bias": None, "holdReasons": ["non_finite_likelihood"]}
        best = (value, chi, pon)
        path.append({"step": step, "bias": {"chi": chi, "pon": pon}, "nll": value, "points": len(evaluated)})
        for kind, center in (("chi", chi), ("pon", pon)):
            window_low[kind] = max(low, round(center - step, 10))
            window_high[kind] = min(high, round(center + step, 10))
    assert best is not None
    value, chi, pon = best
    bias = {"chi": chi, "pon": pon}
    if any(abs(v - low) < 1e-9 or abs(v - high) < 1e-9 for v in bias.values()):
        return {**result, "status": "held", "bias": None, "holdReasons": ["boundary_optimum"], "searchPath": path}
    return {**result, "status": "estimated", "bias": bias, "nll": value, "baselineNll": baseline, "searchPath": path}


def theta_variant_rates(model: HierarchicalSoftmax, targets: Sequence[Mapping[str, Any]], bias: Mapping[str, float]) -> dict[str, Any]:
    """対象窓の公開結果の種別率：観測と予測（記録用。性能改善の証拠とは呼ばない）。"""
    from tools.ev_policy_opponent import _public_resolution_kind_probabilities

    variant = theta_variant_model(model, bias)
    predicted: Counter = Counter()
    observed: Counter = Counter()
    for encoded in targets:
        predicted.update(_public_resolution_kind_probabilities(variant, encoded))
        observed[str(encoded["window"]["observation"]["resolution"]["kind"])] += 1
    count = len(targets)
    kinds = sorted(set(predicted) | set(observed))
    return {kind: {"observed": observed[kind] / count, "predicted": predicted[kind] / count} for kind in kinds}


def load_theta_variant_targets(dataset_dir: Any, feature_dir: Any, seasons: Sequence[str] = ("2024-25",)) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """特徴cacheから較正期間の応答窓だけを読み、βの対象窓を返す（cacheの検証を通す）。"""
    from pathlib import Path

    from tools.ev_policy_opponent import _deserialize_encoded, iter_feature_shard, verify_opponent_feature_cache

    dataset_dir, feature_dir = Path(dataset_dir), Path(feature_dir)
    manifest = verify_opponent_feature_cache(dataset_dir, feature_dir)
    selected = []
    for shard in manifest["shards"]:
        for record in iter_feature_shard(feature_dir / shard["path"]):
            window = record.get("teacherWindow") or {}
            if window.get("developmentSplit") != "calibration" or window.get("phase") != "discard_response":
                continue
            if str(window["roundId"]).split(":")[1] not in seasons:
                raise ValueError("較正期間の窓が想定のシーズンでない")
            selected.append(_deserialize_encoded(record, window))
    identity = {"featureManifestHash": hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest(),
                "calibrationResponseWindows": len(selected)}
    return theta_variant_targets(selected), identity


def build_theta_variant_manifest(
    model_path: Any, targets: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]
) -> dict[str, Any]:
    """推定の前に固定するmanifest（9.3節）。"""
    return {
        "schemaVersion": THETA_VARIANT_MANIFEST_SCHEMA,
        "variantId": THETA_VARIANT_ID,
        "baseModelSha256": _sha256_file(model_path),
        "period": {"split": "calibration", "seasons": ["2024-25"],
                   "note": "温度の選択と共用するため独立な検証ではない。開発確認期間を後で独立な採用判定に使い直さない"},
        "phase": "discard_response",
        "condition": "responder_opponent_riichi_count_at_least_1",
        "kinds": list(THETA_VARIANT_KINDS),
        "applicationOrder": "added_to_kind_score_before_temperature",
        "fixedConstants": "base_model_fixed_constants",
        "objective": "joint_public_resolution_negative_log_likelihood_over_target_windows",
        "search": dict(THETA_VARIANT_SEARCH),
        "minimumSupport": dict(THETA_VARIANT_MINIMUM_SUPPORT),
        "holdConditions": list(THETA_VARIANT_HOLD_CONDITIONS),
        "support": theta_variant_support(targets),
        "data": dict(identity),
        "adoption": "diagnostic_only_not_adopted",
    }


def calibrate_theta_variant(
    model: HierarchicalSoftmax, targets: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    estimate = estimate_theta_variant(model, targets, manifest)
    record = {
        "schemaVersion": THETA_VARIANT_RESULT_SCHEMA,
        "variantId": THETA_VARIANT_ID,
        "manifestSha256": manifest_sha256,
        "baseThetaId": model_identity(model),
        **estimate,
    }
    if estimate["status"] == "estimated":
        variant = theta_variant_model(model, estimate["bias"])
        record["thetaId"] = model_identity(variant)
        record["rates"] = {"base": theta_variant_rates(model, targets, {"chi": 0.0, "pon": 0.0}),
                           "variant": theta_variant_rates(model, targets, estimate["bias"])}
        record["note"] = "較正期間での適合。温度の選択と共用するため、性能改善の証拠ではない（9.3節）"
    return record


def load_theta_variant(model: HierarchicalSoftmax, result: Mapping[str, Any]) -> RiichiResponseBiasSoftmax:
    """推定済みの結果からθ変種を作る。heldの結果や、別の基準モデルの結果は使わない。"""
    if result.get("status") != "estimated":
        raise ValueError("θ変種の較正がheld（変種のシナリオは実行できない）")
    if result["baseThetaId"] != model_identity(model):
        raise ValueError("θ変種の基準モデルが一致しない")
    variant = theta_variant_model(model, result["bias"])
    if model_identity(variant) != result["thetaId"]:
        raise ValueError("θ変種の識別子が一致しない")
    return variant


def theta_variant_scenario(scenarios: Sequence[Mapping[str, Any]], variant: HierarchicalSoftmax) -> dict[str, Any]:
    """変種のシナリオ：baseの固定定数で1シナリオ（9.3節）。thetaIdは変種のもの。"""
    base = next(s for s in scenarios if s["id"] == "base")
    return {**dict(base), "id": THETA_VARIANT_ID, "posteriorId": THETA_VARIANT_ID, "thetaId": model_identity(variant)}


# ===========================================================================
# 工程7：マイクロベンチマークと予算の関門（PHASE_D33_DESIGN.md 12.3節）
# ===========================================================================
#
# 12.3節：1判断・1シナリオ・1鎖・10,000反復を測り、全格子（基準SMC、初期化、MCMC）の見積もりが
# 予算（24時間、P≤3）を超えればpilotを始めずにresource_budget_exceededとして設計へ戻る。
# 10,000反復の経過時間が「1反復あたりの予算×10,000」を超えた時点で判定は確定する（その後どれだけ
# 速くなっても覆らない）。そこで観測の上限時間をmanifestで固定し、判定の確定と見積もりの材料を記録する。

BENCHMARK_MANIFEST_SCHEMA = "ev-policy-belief-benchmark-manifest/v1"
BENCHMARK_SCHEMA = "ev-policy-belief-benchmark/v1"
PILOT_FULL_GRID_ITERATIONS = 16 * 23 * 4 * 160_000  # 16判断×23シナリオ×4鎖×160,000反復（12.3節）


def benchmark_budget(*, wall_clock_hours: float, processes: int, spent_seconds: float, iterations: int) -> dict[str, Any]:
    """予算から1反復あたりの上限と、ベンチマークの判定が確定する時間を求める。"""
    remaining = wall_clock_hours * 3600.0 - spent_seconds
    per_iteration = remaining * processes / PILOT_FULL_GRID_ITERATIONS
    return {
        "wallClockHours": wall_clock_hours,
        "processes": processes,
        "spentSeconds": spent_seconds,
        "remainingSeconds": remaining,
        "fullGridIterations": PILOT_FULL_GRID_ITERATIONS,
        "perIterationBudgetSeconds": per_iteration,
        "benchmarkIterations": iterations,
        "decisiveSeconds": per_iteration * iterations,
    }


def benchmark_chain(
    context: DecisionContext,
    model: HierarchicalSoftmax,
    scenario: Mapping[str, Any],
    *,
    iterations: int,
    max_seconds: float,
    decisive_seconds: float,
    seed: int,
    cache_capacity: int,
    probabilities: Sequence[tuple[str, float]] = TENPAI_FAMILY_PROBABILITIES,
) -> dict[str, Any]:
    """1判断・1シナリオ・1鎖を、iterationsかmax_secondsの早い方まで走らせ、反復ごとの時間を記録する。"""
    import time

    cache = WindowCache(cache_capacity)
    resolver = model_resolver(model, scenario)
    start = time.perf_counter()
    initial = construct_initial_world(context, random.Random(seed), resolver=resolver, probabilities=probabilities, cache=cache)
    init_seconds = time.perf_counter() - start
    record: dict[str, Any] = {"decisionId": context.decision_id, "scenarioId": scenario["id"], "seed": seed,
                              "initialization": {"status": initial.status, "family": initial.family,
                                                 "attempts": initial.attempts, "seconds": round(init_seconds, 4)}}
    if initial.world is None:
        return {**record, "status": "init_failed"}
    belief, state = mahjong_belief(context, initial.world, resolver, probabilities=probabilities, cache=cache)
    model_counter = belief.problem.model
    chooser = RandomChooser(random.Random(seed + 1))
    stats = MoveStatistics()
    seconds: list[float] = []
    trace = []
    decisive_at = None
    begin = time.perf_counter()
    for iteration in range(1, iterations + 1):
        before = time.perf_counter()
        mcmc_iteration(belief.problem, state, chooser, stats)
        seconds.append(time.perf_counter() - before)
        elapsed = time.perf_counter() - begin
        if decisive_at is None and elapsed > decisive_seconds:
            decisive_at = iteration
        if iteration % 10 == 0 or iteration == 1:
            trace.append({"iteration": iteration, "elapsed": round(elapsed, 3), **{
                key: cache.statistics()[key] for key in ("entries", "hits", "misses", "hitRate")},
                "evaluatorCalls": model_counter.evaluations})
        if elapsed > max_seconds:
            break
    completed = len(seconds)
    half = seconds[completed // 2:] or seconds
    return {
        **record,
        "status": "complete" if completed == iterations else "stopped_at_observation_cap",
        "iterationsCompleted": completed,
        "elapsedSeconds": round(sum(seconds), 4),
        "decisiveReachedAtIteration": decisive_at,
        "secondsPerIteration": {
            "mean": sum(seconds) / completed,
            "meanSecondHalf": sum(half) / len(half),
            "minimum": min(seconds),
            "median": float(np.median(seconds)),
        },
        "iterationSeconds": [round(value, 4) for value in seconds],
        "trace": trace,
        "evaluatorCalls": model_counter.evaluations,
        "cacheStatistics": cache.statistics(),
        "moves": stats.as_dict(),
    }


def _benchmark_worker_run(task: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    decision_id, options = task
    scenario = next(s for s in belief_scenarios(_BELIEF_WORKER["fixed"]) if s["id"] == options["scenario"])
    return benchmark_chain(
        _WORKER["contexts"][decision_id], _BELIEF_WORKER["model"], scenario,
        iterations=int(options["iterations"]), max_seconds=float(options["maxSeconds"]),
        decisive_seconds=float(options["decisiveSeconds"]),
        seed=_chain_seed(int(options["seed"]), decision_id, options["scenario"], 0),
        cache_capacity=int(options["cacheCapacity"]),
    )


def run_benchmark(dataset_dir: Any, model_dir: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """固定したmanifestどおりにベンチマークを並列で測り、12.3節の関門を機械的に判定する。"""
    import multiprocessing
    import time

    decisions = list(manifest["decisions"])
    options = {key: manifest[key] for key in ("scenario", "iterations", "maxSeconds", "seed", "cacheCapacity")}
    options["decisiveSeconds"] = manifest["budget"]["decisiveSeconds"]
    start = time.time()
    context = multiprocessing.get_context("spawn")
    with context.Pool(len(decisions), initializer=_belief_worker_init,
                      initargs=(str(dataset_dir), str(model_dir), decisions, {})) as pool:
        runs = pool.map(_benchmark_worker_run, [(decision, options) for decision in decisions])
    return {
        "schemaVersion": BENCHMARK_SCHEMA,
        "runs": runs,
        "gate": benchmark_gate(runs, manifest["budget"]),
        "wallClockSeconds": round(time.time() - start, 1),
    }


def benchmark_gate(runs: Sequence[Mapping[str, Any]], budget: Mapping[str, Any]) -> dict[str, Any]:
    """12.3節の関門。どれかの判断で10,000反復が判定の確定時間を超えれば、全格子は予算に収まらない。

    見積もりは、各判断の後半の1反復あたり平均（キャッシュが温まった側）を全格子に掛け、P並列で割る。
    見積もりは参考値で、判定には「確定時間を超えたか」と「見積もりが残り予算を超えるか」を使う。
    """
    if any(run.get("status") == "init_failed" for run in runs):
        return {"verdict": "init_failed_in_benchmark", "runs": len(runs)}
    per_iteration = max(run["secondsPerIteration"]["meanSecondHalf"] for run in runs)
    optimistic = min(run["secondsPerIteration"]["minimum"] for run in runs)
    processes = int(budget["processes"])
    projected = per_iteration * budget["fullGridIterations"] / processes
    lower_bound = optimistic * budget["fullGridIterations"] / processes
    decisive = [run["decisionId"] for run in runs if run["decisiveReachedAtIteration"] is not None]
    exceeded = bool(decisive) or projected > budget["remainingSeconds"]
    return {
        "verdict": "resource_budget_exceeded" if exceeded else "within_budget",
        "decisiveReachedIn": decisive,
        "perIterationBudgetSeconds": budget["perIterationBudgetSeconds"],
        "observedSecondsPerIteration": per_iteration,
        "ratioToBudget": per_iteration / budget["perIterationBudgetSeconds"],
        "projectedMcmcSeconds": projected,
        "projectedMcmcDays": projected / 86_400.0,
        "lowerBoundMcmcSeconds": lower_bound,
        "lowerBoundRatioToRemainingBudget": lower_bound / budget["remainingSeconds"],
        "remainingBudgetSeconds": budget["remainingSeconds"],
    }


# ===========================================================================
# 再判断の段階1（A1）：手牌の形の計算を、同じ値のまま速くする
# ===========================================================================
#
# 相手モデルの特徴（シャンテン数と改善牌mask）は ev_policy_features._exact_shape_cached が計算する。
# 1枚足した34通りの手を調べるとき、変わるのは足した牌の色（萬子・筒子・索子・字牌）だけなのに、元の実装は
# 毎回4色の組み合わせを最初から結合する。組み合わせの結合は和なので順序によらず、面子数の上限による
# 絞り込みも途中で行っても最後に行っても同じ集合になる。そこで、残り3色の結合を先に作って使い回す。
#
# 特徴コードのファイルは特徴cacheの検証ハッシュに含まれる（D.3.2b）ため、書き換えない。推定器を走らせる
# プロセスの中でだけ、同じ署名の関数へ差し替える（install_fast_shape）。値の一致はテストで確かめる。

FAST_SHAPE_CACHE_ENTRIES = 1_000_000


@lru_cache(maxsize=200_000)
def _combine_shape_groups(groups: tuple[tuple[tuple[int, int, int], ...], ...], target_melds: int) -> tuple[tuple[int, int, int], ...]:
    """色ごとの(面子, 塔子, 対子)の候補を足し合わせた状態の集合（面子数が上限以下のものだけ）。"""
    states = {(0, 0, 0)}
    for group in groups:
        combined = set()
        for left in states:
            for right in group:
                melds = left[0] + right[0]
                if melds <= target_melds:
                    combined.add((melds, left[1] + right[1], left[2] + right[2]))
        states = combined
    return tuple(sorted(states))


def _best_standard_shanten(
    states: Sequence[tuple[int, int, int]], group: Sequence[tuple[int, int, int]], fixed_melds: int,
    stop_below: int | None = None,
) -> int:
    """残り3色の状態と1色の候補を結合したときの、面子手のシャンテン数（元の実装の最後の段と同じ式）。

    stop_belowを渡すと、それより小さい値が見つかった時点で返す（改善牌の判定は「下がるか」だけで足りる）。
    """
    target_melds = 4 - fixed_melds
    best = 8
    for left_melds, left_taatsu, left_pairs in states:
        for right_melds, right_taatsu, right_pairs in group:
            melds = left_melds + right_melds
            if melds > target_melds:
                continue
            pairs = left_pairs + right_pairs
            taatsu = left_taatsu + right_taatsu + max(0, pairs - 1)
            value = 8 - 2 * (fixed_melds + melds) - min(taatsu, max(0, target_melds - melds)) - int(pairs > 0)
            if value < best:
                best = value
                if stop_below is not None and best < stop_below:
                    return best
    return best


def _other_form_shanten(counts: Sequence[int]) -> int:
    """七対子と国士のシャンテン数（元の実装と同じ式）。"""
    from tools.ev_policy_features import TERMINAL_HONORS

    pairs = sum(count >= 2 for count in counts)
    unique = sum(count > 0 for count in counts)
    chiitoi = 6 - pairs + max(0, 7 - unique)
    terminal_unique = sum(counts[index] > 0 for index in TERMINAL_HONORS)
    terminal_pair = any(counts[index] >= 2 for index in TERMINAL_HONORS)
    return min(chiitoi, 13 - terminal_unique - int(terminal_pair))


_GROUP_BOUNDS = ((0, 9, True), (9, 18, True), (18, 27, True), (27, 34, False))


@lru_cache(maxsize=FAST_SHAPE_CACHE_ENTRIES)
def fast_exact_shape_cached(counts: tuple[int, ...], fixed_melds: int) -> Any:
    """ev_policy_features._exact_shape_cachedと同じShapeResultを返す。"""
    from tools.ev_calibration_state import _group_shapes
    from tools.ev_policy_features import ShapeResult

    from tools.ev_policy_features import TERMINAL_HONORS

    target = 4 - fixed_melds
    groups = tuple(_group_shapes(counts[start:end], suited) for start, end, suited in _GROUP_BOUNDS)
    rests = tuple(_combine_shape_groups(groups[:k] + groups[k + 1:], target) for k in range(4))
    # 七対子・国士の数え上げ。1枚足した手では、対子・種類・么九の数が高々1つ増えるだけなので差分で求める。
    pairs = sum(count >= 2 for count in counts)
    unique = sum(count > 0 for count in counts)
    terminal_unique = sum(counts[index] > 0 for index in TERMINAL_HONORS)
    terminal_pair = any(counts[index] >= 2 for index in TERMINAL_HONORS)
    honors = set(TERMINAL_HONORS)

    def other_forms(p: int, u: int, tu: int, tp: bool) -> int:
        return min(6 - p + max(0, 7 - u), 13 - tu - int(tp))

    standard = _best_standard_shanten(rests[0], groups[0], fixed_melds)
    current = min(standard, other_forms(pairs, unique, terminal_unique, terminal_pair)) if fixed_melds == 0 else standard
    mask = 0
    work = list(counts)
    for tile34 in range(34):
        before = work[tile34]
        if before >= 4:
            continue
        work[tile34] += 1
        improved = False
        if fixed_melds == 0:
            terminal = tile34 in honors
            improved = other_forms(
                pairs + int(before == 1), unique + int(before == 0),
                terminal_unique + int(terminal and before == 0), terminal_pair or (terminal and before == 1)) < current
        if not improved:
            index = min(3, tile34 // 9)
            start, end, suited = _GROUP_BOUNDS[index]
            group = _group_shapes(tuple(work[start:end]), suited)
            improved = _best_standard_shanten(rests[index], group, fixed_melds, stop_below=current) < current
        if improved:
            mask |= 1 << tile34
        work[tile34] -= 1
    return ShapeResult(current, mask)


def _fast_counts34(values: Sequence[Any], label: str) -> tuple[int, ...]:
    """ev_policy_features._counts34と同じ検査の速い版。0〜4のint（boolを除く）34個なら同じタプルを返し、
    それ以外は元の関数へ渡して同じ例外を出させる。"""
    if len(values) == 34:
        counts = tuple(values)
        if all(type(value) is int and 0 <= value <= 4 for value in counts):
            return counts
    return _ORIGINAL_FEATURE_FUNCTIONS["_counts34"](values, label)


_ORIGINAL_FEATURE_FUNCTIONS: dict[str, Any] = {}


def install_fast_shape() -> None:
    """このプロセスの特徴計算で、形の計算と入力検査を速い実装へ差し替える（値と例外は同じ）。何度呼んでもよい。"""
    import tools.ev_policy_features as features

    if features._exact_shape_cached is fast_exact_shape_cached:
        return
    _ORIGINAL_FEATURE_FUNCTIONS["_exact_shape_cached"] = features._exact_shape_cached
    _ORIGINAL_FEATURE_FUNCTIONS["_counts34"] = features._counts34
    features._exact_shape_cached = fast_exact_shape_cached
    features._counts34 = _fast_counts34


def uninstall_fast_shape() -> None:
    import tools.ev_policy_features as features

    for name, function in _ORIGINAL_FEATURE_FUNCTIONS.items():
        setattr(features, name, function)
    _ORIGINAL_FEATURE_FUNCTIONS.clear()
