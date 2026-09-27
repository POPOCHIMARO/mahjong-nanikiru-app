"""D.3.1の観測イベント、教師窓、推論prefixを生成する。"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version as package_version
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__:
    from .ev_calibration_state import call_parts, is_red, tile34
    from .ev_policy_replay import (
        ParsedRound,
        RecordedChronologyError,
        RecordedEvent,
        RuntimeReplayError,
        apply_recorded_event,
        build_runtime_world,
        parse_recorded_round,
    )
    from .ev_policy_round import (
        IllegalAction,
        PHASE_CHANKAN_RESPONSES,
        PHASE_SELF_ACTION,
        ResponseClaim,
        RoundState,
        SelfAction,
    )
    from .ev_policy_scoring import HandScoringError, ScoringDependencyError
    from .ev_policy_state import select_development_inputs, verify_policy_dataset
else:
    from ev_calibration_state import call_parts, is_red, tile34
    from ev_policy_replay import (
        ParsedRound,
        RecordedChronologyError,
        RecordedEvent,
        RuntimeReplayError,
        apply_recorded_event,
        build_runtime_world,
        parse_recorded_round,
    )
    from ev_policy_round import (
        IllegalAction,
        PHASE_CHANKAN_RESPONSES,
        PHASE_SELF_ACTION,
        ResponseClaim,
        RoundState,
        SelfAction,
    )
    from ev_policy_scoring import HandScoringError, ScoringDependencyError
    from ev_policy_state import select_development_inputs, verify_policy_dataset


OBSERVATION_SCHEMA = "ev-policy-observation-dataset/v1"
PUBLIC_ROUND_SCHEMA = "ev-policy-public-events/v1"
PRIVATE_ROUND_SCHEMA = "ev-policy-private-events/v1"
TEACHER_WINDOW_SCHEMA = "ev-policy-teacher-window/v1"
INFERENCE_PREFIX_SCHEMA = "ev-policy-inference-prefix/v1"
OUTCOME_SCHEMA = "ev-policy-observed-outcome/v1"
REJECTION_SCHEMA = "ev-policy-observation-rejection/v1"
APP_ROOT = Path(__file__).resolve().parents[1]
CODE_DEPENDENCIES = (
    "tools/calibrate_ev.py",
    "tools/ev_policy_observation.py",
    "tools/ev_policy_replay.py",
    "tools/ev_policy_round.py",
    "tools/ev_policy_scoring.py",
    "tools/ev_policy_state.py",
    "tools/ev_calibration_state.py",
    "calibration/PHASE_D3_DESIGN.md",
)


@dataclass(frozen=True)
class RoundProjection:
    public_record: dict[str, Any]
    private_records: tuple[dict[str, Any], ...]
    raw_cutoffs: Mapping[int, int]
    response_boundaries: Mapping[int, int]
    response_resolutions: Mapping[int, dict[str, Any]]


def _pair(raw: int) -> tuple[int, bool]:
    index = tile34(raw)
    if index is None:
        raise RecordedChronologyError(f"牌tokenが不正: {raw!r}")
    return index, is_red(raw)


def _tile_record(raw: int) -> dict[str, Any]:
    index, red = _pair(raw)
    return {"tile34": index, "isRed": red}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _result_details(result: Any) -> list[dict[str, int]]:
    if not isinstance(result, list):
        return []
    values = []
    for item in result[2:]:
        if isinstance(item, list) and len(item) >= 3 and isinstance(item[0], int) and isinstance(item[1], int):
            values.append({"winnerSeat": int(item[0]), "loserSeat": int(item[1])})
    return values


def _result_delta(result: Any) -> list[int] | None:
    if not isinstance(result, list):
        return None
    vectors = [
        item
        for item in result[1:]
        if isinstance(item, list)
        and len(item) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in item)
    ]
    if not vectors:
        return None
    return [sum(int(vector[seat]) for vector in vectors) for seat in range(4)]


def _terminal_kind(parsed: ParsedRound) -> str:
    result = parsed.result
    if not isinstance(result, list) or not result:
        return "unknown"
    if result[0] == "流局":
        return "exhaustive_draw"
    if result[0] != "和了":
        return "unsupported"
    details = _result_details(result)
    return "tsumo" if details and all(item["winnerSeat"] == item["loserSeat"] for item in details) else "ron"


def _call_action(event: RecordedEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "consumed": sorted((_tile_record(raw) for raw in event.consumed_raw), key=_canonical),
    }


def response_resolution(parsed: ParsedRound, position: int) -> dict[str, Any]:
    """捨牌または加槓に対する公開解決だけを返す。隠れた希望は推定しない。"""
    event = parsed.events[position]
    next_event = parsed.events[position + 1] if position + 1 < len(parsed.events) else None
    if event.kind == "discard" and next_event is not None and next_event.kind in {"chi", "pon", "daiminkan"}:
        return {"kind": next_event.kind, "seat": next_event.seat, "action": _call_action(next_event)}
    if event.kind == "kakan" and next_event is not None and next_event.kind == "draw" and next_event.draw_source == "rinshan":
        return {"kind": "pass"}
    if event.kind in {"discard", "kakan"} and position == len(parsed.events) - 1 and _terminal_kind(parsed) == "ron":
        return {
            "kind": "ron",
            "winnerSeats": sorted({item["winnerSeat"] for item in _result_details(parsed.result)}),
        }
    if event.kind in {"discard", "kakan"}:
        return {"kind": "pass"}
    raise ValueError("応答窓を持たないイベント")


def _project_public_event(event: RecordedEvent, dora_raw: int | None) -> dict[str, Any]:
    base: dict[str, Any] = {"type": event.kind, "rawEventIndex": event.event_index, "seat": event.seat}
    if event.kind == "draw":
        base["source"] = event.draw_source
    elif event.kind == "discard" and event.raw is not None:
        base.update(
            {
                "tile": _tile_record(event.raw),
                "origin": "drawn" if event.is_tsumogiri else "concealed",
                "riichiDeclaration": event.is_riichi,
            }
        )
    elif event.kind in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
        _, all_tiles, _, _ = call_parts(event.token)
        base["tiles"] = [_tile_record(raw) for raw in all_tiles]
        if event.from_seat is not None:
            base["fromSeat"] = event.from_seat
        if dora_raw is not None:
            base["revealedDoraIndicator"] = _tile_record(dora_raw)
    return base


def project_round(parsed: ParsedRound, round_id: str) -> RoundProjection:
    """公開列と各家の私有列を作り、応答解決を行動前prefixから分離する。"""
    if not parsed.dora_raw:
        raise RecordedChronologyError("initial_dora_missing")
    public_events: list[dict[str, Any]] = []
    raw_cutoffs: dict[int, int] = {0: 0}
    response_boundaries: dict[int, int] = {}
    response_resolutions: dict[int, dict[str, Any]] = {}
    response_available: dict[int, int] = {}
    kan_count = 0

    for position, event in enumerate(parsed.events):
        dora = None
        if event.kind in {"ankan", "kakan", "daiminkan"}:
            kan_count += 1
            if kan_count < len(parsed.dora_raw):
                dora = parsed.dora_raw[kan_count]
        public_events.append(_project_public_event(event, dora))
        raw_cutoffs[event.event_index + 1] = len(public_events)
        if event.kind not in {"discard", "kakan"}:
            continue
        response_boundaries[event.event_index] = len(public_events)
        resolution = response_resolution(parsed, position)
        response_resolutions[event.event_index] = resolution
        next_event = parsed.events[position + 1] if position + 1 < len(parsed.events) else None
        if resolution["kind"] in {"chi", "pon", "daiminkan"}:
            response_available[event.event_index] = len(public_events) + 1
        else:
            public_events.append(
                {
                    "type": "response_resolution" if event.kind == "discard" else "chankan_resolution",
                    "afterRawEventIndex": event.event_index,
                    "resolution": resolution,
                }
            )
            response_available[event.event_index] = len(public_events)
        if next_event is not None:
            # 次の記録イベントの直前では、直前の捨牌・加槓に対する
            # 公開解決も既知である。最初に置いた暫定境界をここで更新する。
            raw_cutoffs[next_event.event_index] = len(public_events)

    initial_public = {
        "dealerSeat": parsed.dealer_seat,
        "kyoku": parsed.kyoku,
        "honba": parsed.honba,
        "riichiSticks": parsed.kyoutaku,
        "scores": list(parsed.start_scores),
        "doraIndicator": _tile_record(parsed.dora_raw[0]),
    }
    public_record = {
        "schemaVersion": PUBLIC_ROUND_SCHEMA,
        "roundId": round_id,
        "initial": initial_public,
        "events": public_events,
    }

    private_records = []
    for seat in range(4):
        private_events: list[dict[str, Any]] = []
        for event in parsed.events:
            if event.kind == "draw" and event.seat == seat and event.raw is not None:
                private_events.append(
                    {
                        "type": "draw_observation",
                        "rawEventIndex": event.event_index,
                        "availableAtPublicEventCount": raw_cutoffs[event.event_index + 1],
                        "tile": _tile_record(event.raw),
                    }
                )
        for position, event in enumerate(parsed.events):
            if event.kind not in {"discard", "kakan"} or event.seat == seat:
                continue
            resolution = response_resolutions[event.event_index]
            exact_action: dict[str, Any] | None = None
            status = "censored"
            if resolution["kind"] == "pass":
                status = "exact"
                exact_action = {"kind": "pass"}
            elif resolution["kind"] == "ron" and seat in resolution["winnerSeats"]:
                status = "exact"
                exact_action = {"kind": "ron"}
            elif resolution["kind"] in {"chi", "pon", "daiminkan"} and seat == resolution["seat"]:
                status = "exact"
                exact_action = resolution["action"]
            private_events.append(
                {
                    "type": "response_observation",
                    "afterRawEventIndex": event.event_index,
                    "availableAtPublicEventCount": response_available[event.event_index],
                    "status": status,
                    "action": exact_action,
                    "constraint": None if status == "exact" else "joint_public_resolution",
                }
            )
        private_events.sort(key=lambda item: (item["availableAtPublicEventCount"], item.get("rawEventIndex", 10**9)))
        private_records.append(
            {
                "schemaVersion": PRIVATE_ROUND_SCHEMA,
                "roundId": round_id,
                "seat": seat,
                "initialHand": [_tile_record(raw) for raw in sorted(parsed.initial_hands[seat], key=_pair)],
                "events": private_events,
            }
        )
    return RoundProjection(public_record, tuple(private_records), raw_cutoffs, response_boundaries, response_resolutions)


def build_inference_prefix(
    projection: RoundProjection,
    *,
    seat: int,
    raw_event_cutoff: int,
) -> dict[str, Any]:
    """行動前までに対象家が知る公開列と私有列だけを切り出す。"""
    if raw_event_cutoff not in projection.raw_cutoffs:
        raise ValueError(f"raw_event_cutoffがない: {raw_event_cutoff}")
    public_count = int(projection.raw_cutoffs[raw_event_cutoff])
    private = projection.private_records[seat]
    public_payload = {
        "initial": projection.public_record["initial"],
        "events": projection.public_record["events"][:public_count],
    }
    private_events = [
        event for event in private["events"] if int(event["availableAtPublicEventCount"]) <= public_count
    ]
    private_payload = {"seat": seat, "initialHand": private["initialHand"], "events": private_events}
    payload = {"public": public_payload, "private": private_payload}
    return {
        "schemaVersion": INFERENCE_PREFIX_SCHEMA,
        "seat": seat,
        "rawEventCutoff": raw_event_cutoff,
        "publicEventCount": public_count,
        "privateEventCount": len(private_events),
        "publicStateHash": _record_hash(public_payload),
        "privatePrefixHash": _record_hash(private_payload),
        "informationStateHash": _record_hash(payload),
        "payload": payload,
    }


def _semantic_self_action(state: RoundState, action: SelfAction) -> dict[str, Any]:
    if action.kind in {"discard", "riichi_discard"}:
        assert action.tile_id is not None
        tile = state.tile_by_id[action.tile_id]
        return {
            "kind": action.kind,
            "tile34": tile.tile34,
            "isRed": tile.is_red,
            "origin": "drawn" if action.tile_id == state.drawn_tile_id else "concealed",
        }
    if action.kind == "ankan":
        tiles = [state.tile_by_id[tile_id] for tile_id in action.tile_ids]
        return {"kind": "ankan", "tile34": tiles[0].tile34, "redCount": sum(tile.is_red for tile in tiles)}
    if action.kind == "kakan":
        assert action.tile_id is not None
        tile = state.tile_by_id[action.tile_id]
        return {"kind": "kakan", "tile34": tile.tile34, "addedIsRed": tile.is_red}
    return {"kind": "tsumo"}


def _deduplicate_actions(actions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = {_canonical(action): action for action in actions}
    return [values[key] for key in sorted(values)]


def semantic_self_actions(state: RoundState) -> tuple[list[dict[str, Any]], str, str | None]:
    """意味上の自己行動を物理コピーから縮約し、和了判定異常を別に返す。"""
    actions = [_semantic_self_action(state, action) for action in state.legal_self_actions(include_tsumo=False)]
    score_status = "known"
    score_reason = None
    if state.phase == PHASE_SELF_ACTION:
        try:
            state._score_tsumo(state.turn_seat)
        except IllegalAction:
            pass
        except ScoringDependencyError:
            raise
        except HandScoringError as error:
            score_status = "hold"
            score_reason = error.code
        else:
            actions.append({"kind": "tsumo"})
    return _deduplicate_actions(actions), score_status, score_reason


def _semantic_response_action(state: RoundState, claim: ResponseClaim) -> dict[str, Any]:
    if claim.kind == "ron":
        return {"kind": "ron"}
    consumed = [
        {"tile34": state.tile_by_id[tile_id].tile34, "isRed": state.tile_by_id[tile_id].is_red}
        for tile_id in claim.consumed_tile_ids
    ]
    return {"kind": claim.kind, "consumed": sorted(consumed, key=_canonical)}


def semantic_response_actions(
    state: RoundState, seat: int, *, chankan: bool = False
) -> tuple[list[dict[str, Any]], str, str | None]:
    actions: list[dict[str, Any]] = [{"kind": "pass"}]
    if not chankan:
        actions.extend(_semantic_response_action(state, claim) for claim in state.legal_call_claims(seat))
        if state.pending_discard is None:
            raise RuntimeReplayError("pending_discardがない")
        tile_id = state.pending_discard[1].tile_id
    else:
        if state.pending_kakan is None:
            raise RuntimeReplayError("pending_kakanがない")
        tile_id = state.pending_kakan.tile_id
    score_status = "known"
    score_reason = None
    try:
        state._score_ron(seat, tile_id, chankan=chankan)
    except IllegalAction:
        pass
    except ScoringDependencyError:
        raise
    except HandScoringError as error:
        score_status = "hold"
        score_reason = error.code
    else:
        actions.append({"kind": "ron"})
    return _deduplicate_actions(actions), score_status, score_reason


def recorded_self_action(parsed: ParsedRound, position: int, state: RoundState) -> dict[str, Any] | None:
    next_event = parsed.events[position + 1] if position + 1 < len(parsed.events) else None
    if next_event is not None and next_event.seat == state.turn_seat:
        if next_event.kind == "discard" and next_event.raw is not None:
            return {
                "kind": "riichi_discard" if next_event.is_riichi else "discard",
                **_tile_record(next_event.raw),
                "origin": "drawn" if next_event.is_tsumogiri else "concealed",
            }
        if next_event.kind == "ankan":
            _, all_tiles, _, _ = call_parts(next_event.token)
            return {
                "kind": "ankan",
                "tile34": tile34(all_tiles[0]),
                "redCount": sum(is_red(raw) for raw in all_tiles),
            }
        if next_event.kind == "kakan" and next_event.raw is not None:
            index, red = _pair(next_event.raw)
            return {"kind": "kakan", "tile34": index, "addedIsRed": red}
    if position == len(parsed.events) - 1 and _terminal_kind(parsed) == "tsumo":
        winners = {item["winnerSeat"] for item in _result_details(parsed.result)}
        if state.turn_seat in winners:
            return {"kind": "tsumo"}
    return None


def _response_priority(kind: str) -> int:
    return {"ron": 0, "pon": 1, "daiminkan": 1, "chi": 2, "pass": 3}[kind]


def resolve_joint_response(
    discarder: int,
    choices: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """各家の同時希望を頭ハネと鳴き優先順位で公開結果へ縮約する。"""
    non_pass = [(seat, dict(action)) for seat, action in choices.items() if action["kind"] != "pass"]
    if not non_pass:
        return {"kind": "pass"}
    winner_seat, winner_action = min(
        non_pass,
        key=lambda item: (_response_priority(str(item[1]["kind"])), (item[0] - discarder) % 4),
    )
    if winner_action["kind"] == "ron":
        return {"kind": "ron", "winnerSeats": [winner_seat]}
    return {"kind": winner_action["kind"], "seat": winner_seat, "action": winner_action}


def compatible_joint_responses(
    discarder: int,
    legal_by_seat: Mapping[int, Sequence[Mapping[str, Any]]],
    observed_resolution: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], ...]:
    """公開解決と両立する全同時希望を列挙する。学習確率はまだ付けない。"""
    seats = sorted(legal_by_seat)
    compatible = []
    for actions in product(*(legal_by_seat[seat] for seat in seats)):
        choices = {seat: dict(action) for seat, action in zip(seats, actions)}
        if resolve_joint_response(discarder, choices) == dict(observed_resolution):
            compatible.append(choices)
    return tuple(compatible)


def _recorded_action_matches(actions: Sequence[Mapping[str, Any]], observed: Mapping[str, Any] | None) -> bool:
    return observed is not None and _canonical(observed) in {_canonical(action) for action in actions}


def _window_id(round_id: str, raw_event_index: int, phase: str) -> str:
    return f"{round_id}:e{raw_event_index}:{phase}"


def _response_teacher_window(
    parsed: ParsedRound,
    projection: RoundProjection,
    state: RoundState,
    event: RecordedEvent,
    round_id: str,
    *,
    chankan: bool,
) -> dict[str, Any]:
    discarder = event.seat
    responders = [seat for seat in range(4) if seat != discarder]
    legal_by_seat: dict[str, Any] = {}
    any_score_hold = False
    for seat in responders:
        actions, score_status, score_reason = semantic_response_actions(state, seat, chankan=chankan)
        legal_by_seat[str(seat)] = {
            "actions": actions,
            "winActionStatus": score_status,
            "winActionReason": score_reason,
        }
        any_score_hold = any_score_hold or score_status != "known"
    resolution = projection.response_resolutions[event.event_index]
    labels = []
    for seat in responders:
        exact = None
        status = "censored"
        if resolution["kind"] == "pass":
            status, exact = "exact", {"kind": "pass"}
        elif resolution["kind"] == "ron" and seat in resolution["winnerSeats"]:
            status, exact = "exact", {"kind": "ron"}
        elif resolution["kind"] in {"chi", "pon", "daiminkan"} and seat == resolution["seat"]:
            status, exact = "exact", resolution["action"]
        labels.append(
            {
                "seat": seat,
                "status": status,
                "action": exact,
                "constraint": None if status == "exact" else "joint_public_resolution",
            }
        )
    return {
        "schemaVersion": TEACHER_WINDOW_SCHEMA,
        "windowId": _window_id(round_id, event.event_index, "chankan_response" if chankan else "discard_response"),
        "roundId": round_id,
        "rawEventIndex": event.event_index,
        "phase": "chankan_response" if chankan else "discard_response",
        "actorSeat": discarder,
        "publicEventCount": projection.response_boundaries[event.event_index],
        "privateEventCounts": {
            str(seat): sum(
                int(item["availableAtPublicEventCount"]) <= projection.response_boundaries[event.event_index]
                for item in projection.private_records[seat]["events"]
            )
            for seat in responders
        },
        "legalBySeat": legal_by_seat,
        "observation": {
            "mass": 1.0,
            "resolution": resolution,
            "perSeatLabels": labels,
            "representation": "joint_resolution_constraint",
        },
        "learningMask": {"jointKind": not any_score_hold, "conditionalDetail": True},
    }


def _self_teacher_window(
    parsed: ParsedRound,
    projection: RoundProjection,
    state: RoundState,
    position: int,
    round_id: str,
    *,
    phase: str,
    label_source: ParsedRound,
) -> dict[str, Any]:
    event = parsed.events[position]
    actions, score_status, score_reason = semantic_self_actions(state)
    # 実行prefixが途中で切れた局でも、次の記録イベントは切る前の牌譜から読む。
    # prefixだけを渡すと、最後の自摸窓に局末のツモ和了を誤って割り当てる（D.3.2b）。
    observed = recorded_self_action(label_source, position, state)
    matched = _recorded_action_matches(actions, observed)
    public_count = projection.raw_cutoffs[event.event_index + 1]
    seat = state.turn_seat
    private_count = sum(
        int(item["availableAtPublicEventCount"]) <= public_count
        for item in projection.private_records[seat]["events"]
    )
    return {
        "schemaVersion": TEACHER_WINDOW_SCHEMA,
        "windowId": _window_id(round_id, event.event_index, phase),
        "roundId": round_id,
        "rawEventIndex": event.event_index,
        "phase": phase,
        "actorSeat": seat,
        "publicEventCount": public_count,
        "privateEventCounts": {str(seat): private_count},
        "legalActions": actions,
        "winActionStatus": score_status,
        "winActionReason": score_reason,
        "observation": {"mass": 1.0, "status": "exact" if matched else "hold", "action": observed},
        "learningMask": {
            "kind": matched and score_status == "known",
            "conditionalDetail": matched and observed is not None and observed["kind"] != "tsumo",
        },
        "holdReason": None if matched else "observed_action_not_in_legal_set",
    }


def _capacity(pair: tuple[int, bool]) -> int:
    if pair[0] in {4, 13, 22}:
        return 1 if pair[1] else 3
    return 4


def runtime_prefix(parsed: ParsedRound) -> tuple[ParsedRound | None, dict[str, Any] | None]:
    """牌在庫が壊れる直前までに切り、正常なprefixを局ごと捨てない。"""
    counts: Counter[tuple[int, bool]] = Counter()

    def add(raw: int) -> bool:
        key = _pair(raw)
        counts[key] += 1
        return counts[key] <= _capacity(key)

    for hand in parsed.initial_hands:
        for raw in hand:
            if not add(raw):
                return None, {"reason": "initial_inventory_invalid", "rawEventIndex": None}
    if not parsed.dora_raw or not add(parsed.dora_raw[0]):
        return None, {"reason": "initial_dora_inventory_invalid", "rawEventIndex": None}
    dora_seen = [parsed.dora_raw[0]]
    kan_count = 0
    for position, event in enumerate(parsed.events):
        additions = [event.raw] if event.kind == "draw" and event.raw is not None else []
        if event.kind in {"ankan", "kakan", "daiminkan"}:
            kan_count += 1
            if kan_count >= len(parsed.dora_raw):
                return replace(parsed, events=parsed.events[:position], dora_raw=tuple(dora_seen), ura_raw=()), {
                    "reason": "additional_dora_missing",
                    "rawEventIndex": event.event_index,
                }
            additions.append(parsed.dora_raw[kan_count])
        work = counts.copy()
        valid = True
        for raw in additions:
            key = _pair(raw)
            work[key] += 1
            valid = valid and work[key] <= _capacity(key)
        if not valid:
            return replace(parsed, events=parsed.events[:position], dora_raw=tuple(dora_seen), ura_raw=()), {
                "reason": "observed_inventory_invalid",
                "rawEventIndex": event.event_index,
            }
        counts = work
        if event.kind in {"ankan", "kakan", "daiminkan"}:
            dora_seen.append(parsed.dora_raw[kan_count])
    return replace(parsed, dora_raw=tuple(dora_seen), ura_raw=()), None


def extract_teacher_windows(
    parsed: ParsedRound,
    projection: RoundProjection,
    round_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """正常prefixの自己行動窓と共同応答窓を局エンジンから抽出する。"""
    runtime_parsed, prefix_rejection = runtime_prefix(parsed)
    if runtime_parsed is None:
        return [], prefix_rejection
    world = build_runtime_world(runtime_parsed)
    state = world.state
    windows: list[dict[str, Any]] = []
    for position, event in enumerate(runtime_parsed.events):
        try:
            if event.kind == "draw":
                apply_recorded_event(state, event, world.draw_tile_ids)
                windows.append(
                    _self_teacher_window(
                        runtime_parsed,
                        projection,
                        state,
                        position,
                        round_id,
                        phase="self_action_after_" + str(event.draw_source),
                        label_source=parsed,
                    )
                )
            elif event.kind == "discard":
                apply_recorded_event(state, event, world.draw_tile_ids)
                windows.append(
                    _response_teacher_window(
                        runtime_parsed, projection, state, event, round_id, chankan=False
                    )
                )
            elif event.kind in {"chi", "pon", "daiminkan"}:
                apply_recorded_event(state, event, world.draw_tile_ids)
                if event.kind in {"chi", "pon"}:
                    windows.append(
                        _self_teacher_window(
                            runtime_parsed,
                            projection,
                            state,
                            position,
                            round_id,
                            phase="self_action_after_call",
                            label_source=parsed,
                        )
                    )
            elif event.kind == "ankan":
                apply_recorded_event(state, event, world.draw_tile_ids)
            elif event.kind == "kakan":
                apply_recorded_event(state, event, world.draw_tile_ids)
                windows.append(
                    _response_teacher_window(
                        runtime_parsed, projection, state, event, round_id, chankan=True
                    )
                )
        except ScoringDependencyError:
            raise
        except (HandScoringError, IllegalAction, RuntimeReplayError, ValueError) as error:
            return windows, {
                "reason": f"{type(error).__name__}:{error}",
                "rawEventIndex": event.event_index,
            }
    return windows, prefix_rejection


def _split(season: str) -> str:
    if season in {"2018-19", "2019-20", "2020-21", "2021-22", "2022-23"}:
        return "train"
    return {"2023-24": "selection", "2024-25": "calibration", "2025-26": "developmentConfirmation"}.get(
        season, "forbidden"
    )


def _round_id(row: Mapping[str, Any], log_index: int) -> str:
    return f"mleague:{row['season']}:{row['gameId']}:{int(row['roundIndex'])}:{log_index}"


class _GzipJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.final_path = path
        self.temp_path = path.with_name(f".{path.name}.tmp")
        self.temp_path.unlink(missing_ok=True)
        self.raw = self.temp_path.open("wb")
        self.gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        self.text = io.TextIOWrapper(self.gzip, encoding="utf-8", newline="\n")
        self.lines = 0

    def write(self, value: Mapping[str, Any]) -> None:
        self.text.write(_canonical(value) + "\n")
        self.lines += 1

    def close(self, *, commit: bool) -> None:
        if not self.text.closed:
            self.text.close()
        if not self.raw.closed:
            self.raw.close()
        if commit:
            self.temp_path.replace(self.final_path)
        else:
            self.temp_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _manifest(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    files = []
    aggregate = hashlib.sha256()
    for path in sorted(paths):
        entry = {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        files.append(entry)
        aggregate.update(_canonical(entry).encode("utf-8"))
    return {"files": files, "aggregateSha256": aggregate.hexdigest()}


def _code_manifest() -> dict[str, Any]:
    """生成時に実際に使った推移的なコードと設計を固定する。"""
    return _manifest([APP_ROOT / relative for relative in CODE_DEPENDENCIES], APP_ROOT)


def _input_manifest_matches(d1_summary: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
    left = [(item["season"], item["path"], item["bytes"], item["sha256"]) for item in d1_summary["input"]["files"]]
    right = [(item["season"], item["path"], item["bytes"], item["sha256"]) for item in selected["files"]]
    return left == right and d1_summary["input"]["selectedSeasons"] == selected["selectedSeasons"]


def extract_opponent_dataset(
    vault_root: Path,
    reservation_path: Path,
    d1_dataset_dir: Path,
    output_dir: Path,
    *,
    selected_seasons: set[str] | None = None,
    maximum_rounds: int | None = None,
) -> dict[str, Any]:
    """許可済み旧期間からD.3.1の正規化データを生成する。"""
    started = time.perf_counter()
    selected = select_development_inputs(vault_root, reservation_path, selected_seasons)
    d1_verification = verify_policy_dataset(d1_dataset_dir)
    d1_summary = json.loads((d1_dataset_dir / "extraction-summary.json").read_text(encoding="utf-8"))
    if d1_verification["status"] != "pass" or not _input_manifest_matches(d1_summary, selected):
        raise ValueError("D.1 manifestまたは検証結果が現在の許可入力と一致しない")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "public": output_dir / "public-events.jsonl.gz",
        "private": output_dir / "private-events.jsonl.gz",
        "teacher": output_dir / "teacher-windows.jsonl.gz",
        "prefix": output_dir / "inference-prefixes.jsonl.gz",
        "outcome": output_dir / "observed-outcomes.jsonl.gz",
        "rejection": output_dir / "rejections.jsonl.gz",
    }
    writers = {name: _GzipJsonlWriter(path) for name, path in paths.items()}
    d1_by_source: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for decision in _iter_jsonl(d1_dataset_dir / "policy-decisions.jsonl"):
        source = decision["source"]
        d1_by_source[(str(source["file"]), int(source["line"]), int(source["logIndex"]))].append(decision)
    totals = Counter()
    phases = Counter()
    labels = Counter()
    reasons = Counter()
    seasons = Counter()
    completed = False
    try:
        for input_entry in selected["files"]:
            source_file = str(input_entry["path"])
            source_path = vault_root / source_file
            with source_path.open(encoding="utf-8") as stream:
                for source_line, raw in enumerate(stream, 1):
                    row = json.loads(raw)
                    season = str(row.get("season"))
                    if season != input_entry["season"]:
                        raise ValueError("ファイル名とレコードのシーズンが一致しない")
                    for log_index, round_log in enumerate(row.get("paifu", {}).get("log", [])):
                        if maximum_rounds is not None and totals["roundsSeen"] >= maximum_rounds:
                            completed = True
                            break
                        totals["roundsSeen"] += 1
                        seasons[season] += 1
                        source = {"file": source_file, "line": source_line, "logIndex": log_index, "season": season}
                        try:
                            parsed = parse_recorded_round(round_log)
                            round_id = _round_id(row, log_index)
                            projection = project_round(parsed, round_id)
                        except (RecordedChronologyError, ValueError) as error:
                            reason = f"{type(error).__name__}:{error}"
                            reasons[reason] += 1
                            writers["rejection"].write(
                                {"schemaVersion": REJECTION_SCHEMA, "level": "round", "source": source, "reason": reason}
                            )
                            continue
                        writers["public"].write(projection.public_record)
                        for private in projection.private_records:
                            writers["private"].write(private)
                        totals["publicRounds"] += 1
                        totals["privateRoundSeats"] += 4

                        source_key = (source_file, source_line, log_index)
                        for decision in d1_by_source.get(source_key, []):
                            if int(decision["publicState"]["kyoku"]) >= 7:
                                totals["heldSouth4OrLaterD1"] += 1
                                continue
                            prefix = build_inference_prefix(
                                projection,
                                seat=int(decision["playerView"]["seat"]),
                                raw_event_cutoff=int(decision["source"]["eventIndex"]),
                            )
                            prefix.pop("payload")
                            prefix.update(
                                {
                                    "decisionId": decision["decisionId"],
                                    "roundId": round_id,
                                    "developmentSplit": decision["developmentSplit"],
                                    "source": source,
                                }
                            )
                            writers["prefix"].write(prefix)
                            totals["inferencePrefixes"] += 1

                        if parsed.kyoku >= 7:
                            totals["heldSouth4OrLaterRounds"] += 1
                            reasons["south4_or_later_outside_initial_policy_scope"] += 1
                            writers["rejection"].write(
                                {
                                    "schemaVersion": REJECTION_SCHEMA,
                                    "level": "teacher_round",
                                    "source": source,
                                    "reason": "south4_or_later_outside_initial_policy_scope",
                                }
                            )
                            continue
                        windows, suffix = extract_teacher_windows(parsed, projection, round_id)
                        for window in windows:
                            window["developmentSplit"] = _split(season)
                            window["source"] = source
                            writers["teacher"].write(window)
                            totals["teacherWindows"] += 1
                            phases[window["phase"]] += 1
                            if window["phase"] in {"discard_response", "chankan_response"}:
                                for label in window["observation"]["perSeatLabels"]:
                                    labels[label["status"]] += 1
                            else:
                                labels[window["observation"]["status"]] += 1
                        if suffix is not None:
                            reason = str(suffix["reason"])
                            reasons[reason] += 1
                            writers["rejection"].write(
                                {
                                    "schemaVersion": REJECTION_SCHEMA,
                                    "level": "round_suffix",
                                    "source": source,
                                    **suffix,
                                }
                            )
                        writers["outcome"].write(
                            {
                                "schemaVersion": OUTCOME_SCHEMA,
                                "roundId": round_id,
                                "developmentSplit": _split(season),
                                "source": source,
                                "resultKind": _terminal_kind(parsed),
                                "scoreDelta": _result_delta(parsed.result),
                                "winDetails": _result_details(parsed.result),
                            }
                        )
                        totals["outcomes"] += 1
                    if completed:
                        break
            if completed:
                break
        for writer in writers.values():
            writer.close(commit=True)
        completed = True
    finally:
        if not completed:
            for writer in writers.values():
                writer.close(commit=False)

    output_manifest = _manifest(list(paths.values()), output_dir)
    dependency_version = None
    try:
        dependency_version = package_version("mahjong")
    except PackageNotFoundError:
        pass
    summary = {
        "schemaVersion": OBSERVATION_SCHEMA,
        "phase": "D.3.1",
        "status": "debug_limit" if maximum_rounds is not None else "generated_pending_verification",
        "eligibleForModelFit": False,
        "labelDefinition": {
            "version": "joint-public-resolution-v1",
            "changedFromDesign": False,
            "exactNoClaimMeansAllPass": True,
            "priorityHiddenChoicesAreCensored": True,
            "individualPassImputedForCensoredChoice": False,
            "observationMass": 1.0,
        },
        "input": selected,
        "d1AggregateSha256": d1_verification["aggregateSha256"],
        "maximumRounds": maximum_rounds,
        "totals": dict(sorted(totals.items())),
        "phases": dict(sorted(phases.items())),
        "labelStatuses": dict(sorted(labels.items())),
        "rejectionReasons": dict(sorted(reasons.items())),
        "seasons": dict(sorted(seasons.items())),
        "generatedFiles": output_manifest,
        "runtimeEnvironment": {
            "pythonVersion": platform.python_version(),
            "scoringPackageVersion": dependency_version,
        },
        "codeManifest": _code_manifest(),
        "runtimeSeconds": time.perf_counter() - started,
    }
    (output_dir / "extraction-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def verify_opponent_dataset(output_dir: Path) -> dict[str, Any]:
    """D.3.1成果物のハッシュ、参照、禁止ラベル、prefix件数を検証する。"""
    summary = json.loads((output_dir / "extraction-summary.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for entry in summary["generatedFiles"]["files"]:
        path = output_dir / entry["path"]
        if not path.is_file():
            errors.append(f"missing:{entry['path']}")
        elif path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            errors.append(f"hash:{entry['path']}")
    current_code = _code_manifest()
    if current_code != summary.get("codeManifest"):
        errors.append("code_manifest")
    counts = Counter()
    prefixes = list(_iter_jsonl(output_dir / "inference-prefixes.jsonl.gz"))
    wanted_prefix_rounds = {row["roundId"] for row in prefixes}
    prefix_public: dict[str, dict[str, Any]] = {}
    prefix_private: dict[tuple[str, int], dict[str, Any]] = {}
    round_ids: set[str] = set()
    for row in _iter_jsonl(output_dir / "public-events.jsonl.gz"):
        counts["public"] += 1
        round_id = str(row["roundId"])
        if round_id in round_ids:
            errors.append("duplicate_public_round")
        round_ids.add(round_id)
        if round_id in wanted_prefix_rounds:
            prefix_public[round_id] = row
    for row in _iter_jsonl(output_dir / "private-events.jsonl.gz"):
        counts["private"] += 1
        if row["roundId"] not in round_ids:
            errors.append("orphan_private")
        if row["roundId"] in wanted_prefix_rounds:
            prefix_private[(str(row["roundId"]), int(row["seat"]))] = row
    for row in _iter_jsonl(output_dir / "teacher-windows.jsonl.gz"):
        counts["teacher"] += 1
        if row["roundId"] not in round_ids:
            errors.append("orphan_teacher")
        if row["developmentSplit"] == "forbidden":
            errors.append("forbidden_teacher_split")
        observation = row["observation"]
        if observation.get("mass") != 1.0:
            errors.append("observation_mass")
        if row["phase"] in {"discard_response", "chankan_response"}:
            for label in observation["perSeatLabels"]:
                if label["status"] == "censored" and label["action"] is not None:
                    errors.append("censored_action_imputed")
                if label["status"] == "censored" and label["constraint"] != "joint_public_resolution":
                    errors.append("censored_constraint_missing")
    for row in prefixes:
        counts["prefix"] += 1
        if row["roundId"] not in round_ids:
            errors.append("orphan_prefix")
        if row["developmentSplit"] == "forbidden":
            errors.append("forbidden_prefix_split")
        public = prefix_public.get(str(row["roundId"]))
        private = prefix_private.get((str(row["roundId"]), int(row["seat"])))
        if public is None or private is None:
            errors.append("prefix_payload_missing")
            continue
        public_count = int(row["publicEventCount"])
        public_payload = {"initial": public["initial"], "events": public["events"][:public_count]}
        private_events = [
            item
            for item in private["events"]
            if int(item["availableAtPublicEventCount"]) <= public_count
        ]
        private_payload = {"seat": int(row["seat"]), "initialHand": private["initialHand"], "events": private_events}
        if len(private_events) != int(row["privateEventCount"]):
            errors.append("private_prefix_count")
        if _record_hash(public_payload) != row["publicStateHash"]:
            errors.append("public_prefix_hash")
        if _record_hash(private_payload) != row["privatePrefixHash"]:
            errors.append("private_prefix_hash")
        if _record_hash({"public": public_payload, "private": private_payload}) != row["informationStateHash"]:
            errors.append("information_prefix_hash")
    for row in _iter_jsonl(output_dir / "observed-outcomes.jsonl.gz"):
        counts["outcome"] += 1
        if row["roundId"] not in round_ids:
            errors.append("orphan_outcome")
        if row["developmentSplit"] == "forbidden":
            errors.append("forbidden_outcome_split")
    expected_prefixes = int(summary["totals"].get("inferencePrefixes", 0))
    if counts["prefix"] != expected_prefixes:
        errors.append("prefix_count")
    expected_counts = {
        "public": int(summary["totals"].get("publicRounds", 0)),
        "private": int(summary["totals"].get("privateRoundSeats", 0)),
        "teacher": int(summary["totals"].get("teacherWindows", 0)),
        "outcome": int(summary["totals"].get("outcomes", 0)),
    }
    for name, expected in expected_counts.items():
        if counts[name] != expected:
            errors.append(f"{name}_count")
    status = "pass" if not errors and summary["maximumRounds"] is None else "debug_pass" if not errors else "fail"
    report = {
        "schemaVersion": "ev-policy-observation-verification/v1",
        "status": status,
        "errors": dict(sorted(Counter(errors).items())),
        "counts": dict(sorted(counts.items())),
        "inputAggregateSha256": summary["generatedFiles"]["aggregateSha256"],
    }
    (output_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    summary["status"] = "verified" if status == "pass" else summary["status"] if status == "debug_pass" else "fail"
    summary["eligibleForModelFit"] = status == "pass"
    (output_dir / "extraction-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report
