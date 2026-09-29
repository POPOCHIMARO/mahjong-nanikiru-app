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
