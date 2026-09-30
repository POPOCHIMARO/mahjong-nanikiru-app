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
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

from tools.ev_calibration_state import shanten
from tools.ev_policy_fixed import constants_for_seat
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


def model_resolver(model: HierarchicalSoftmax, scenario: Mapping[str, Any] | None = None) -> ModelResolver:
    """窓の候補から、その家に使うモデルを返す関数を作る。

    シナリオがあれば、その家の層（候補の種別特徴から決まる）の固定定数を合成する。
    """

    if scenario is None:
        return lambda candidates: model

    def resolve(candidates: Sequence[Any]) -> HierarchicalSoftmax:
        return model.with_fixed(constants_for_seat(scenario, _stratum_attributes_of(candidates[0])))

    return resolve


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


def evaluate_seat(
    context: DecisionContext,
    hypothesis: SeatHypothesis,
    resolver: ModelResolver | None,
) -> SeatEvaluation:
    """他家1人の履歴を判断時点まで評価する（設計5節）。

    resolverがNoneなら合法集合とフリテンだけを求め、確率は計算しない（規則だけの検査用）。
    硬い制約に違反したら、その時点で対数尤度を-infとして返す。
    """

    seat = hypothesis.seat
    if seat not in context.other_seats:
        raise ValueError("評価できるのは他家だけ")
    if len(hypothesis.initial) != 13 or len(set(hypothesis.initial)) != 13:
        raise ValueError("配牌は異なる物理ID13枚が必要")
    result = SeatEvaluation(seat, 0.0)
    rule = _SeatRuleState(context, seat, list(hypothesis.initial))
    features = None
    if resolver is not None:
        features = SeatFeatureState(context.public_row(), seat, hypothesis.private_row(context))
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
            if features is not None:
                features.advance(position + 1)
                window.probability = _window_probability(
                    features, seat, actions, observed, "self_action_after_live", resolver  # type: ignore[arg-type]
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
                if features is not None:
                    features.advance(position + 1)
                    window.probability = _window_probability(
                        features, seat, actions, {"kind": "pass"}, "discard_response", resolver  # type: ignore[arg-type]
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
