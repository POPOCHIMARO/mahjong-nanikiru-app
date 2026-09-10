"""D.3.0の実牌譜イベント変換と局エンジン入口監査。

既存のD.1抽出器を呼び戻さず、原牌譜tokenから判断直前の局面を復元する。
同じ正規化イベントはRoundStateへも適用でき、採点不能は監査理由として外へ返す。
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__:
    from .ev_calibration_state import (
        call_marker,
        call_parts,
        call_source_matches,
        discard_physical_raw,
        is_red,
        shanten,
        tile34,
    )
    from .ev_policy_round import (
        PHASE_CHANKAN_RESPONSES,
        PHASE_DISCARD_RESPONSES,
        PHASE_ENDED,
        ResponseClaim,
        RoundState,
    )
    from .ev_policy_scoring import ScoringDependencyError
    from .ev_policy_state import canonical_tile_set, select_development_inputs, verify_policy_dataset
else:
    from ev_calibration_state import (
        call_marker,
        call_parts,
        call_source_matches,
        discard_physical_raw,
        is_red,
        shanten,
        tile34,
    )
    from ev_policy_round import (
        PHASE_CHANKAN_RESPONSES,
        PHASE_DISCARD_RESPONSES,
        PHASE_ENDED,
        ResponseClaim,
        RoundState,
    )
    from ev_policy_scoring import ScoringDependencyError
    from ev_policy_state import canonical_tile_set, select_development_inputs, verify_policy_dataset


REPLAY_AUDIT_SCHEMA = "ev-policy-runtime-audit/v1"


class RecordedChronologyError(ValueError):
    """原牌譜の時系列を一意に復元できない。"""


class RuntimeReplayError(RuntimeError):
    """正規化イベントを局エンジンへ適用できない。"""


@dataclass(frozen=True)
class RecordedEvent:
    event_index: int
    kind: str
    seat: int
    raw: int | None = None
    token: str | None = None
    discard_index: int | None = None
    is_tsumogiri: bool = False
    is_riichi: bool = False
    draw_source: str | None = None
    from_seat: int | None = None
    consumed_raw: tuple[int, ...] = ()


@dataclass(frozen=True)
class RecordedSnapshot:
    event_index: int
    seat: int
    discard_index: int
    total_draws: int
    current_draw: int
    hand_before_draw: tuple[int, ...]
    hand_before_action: tuple[int, ...]
    scores: tuple[int, int, int, int]
    active_riichi: tuple[tuple[int, int], ...]
    rivers: tuple[tuple[tuple[int, bool, bool, bool, int], ...], ...]
    public_melds: tuple[tuple[tuple[str, tuple[tuple[int, bool], ...], int], ...], ...]


@dataclass(frozen=True)
class ParsedRound:
    events: tuple[RecordedEvent, ...]
    snapshots: tuple[RecordedSnapshot, ...]
    initial_hands: tuple[tuple[int, ...], ...]
    dealer_seat: int
    kyoku: int
    honba: int
    kyoutaku: int
    start_scores: tuple[int, int, int, int]
    dora_raw: tuple[int, ...]
    ura_raw: tuple[int, ...]
    result: Any


def _pair(raw: int) -> tuple[int, bool]:
    index = tile34(raw)
    if index is None:
        raise RecordedChronologyError(f"牌tokenが不正: {raw!r}")
    return index, is_red(raw)


def _sorted_raw(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(values, key=lambda raw: (*_pair(raw), raw)))


def _remove_raw(hand: list[int], raw: int, *, same_kind: bool = False) -> None:
    if raw in hand:
        hand.remove(raw)
        return
    if same_kind:
        wanted = tile34(raw)
        for candidate in sorted(hand):
            if tile34(candidate) == wanted:
                hand.remove(candidate)
                return
    raise RecordedChronologyError(f"手牌にない牌token: {raw}")


def _vector4(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        return None
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def _snapshot(
    *,
    event_index: int,
    seat: int,
    discard_index: int,
    total_draws: int,
    current_draw: int,
    hands: Sequence[Sequence[int]],
    scores: Sequence[int],
    active_riichi: Mapping[int, int],
    rivers: Sequence[Sequence[tuple[int, bool, bool, bool, int]]],
    melds: Sequence[Sequence[tuple[str, tuple[int, ...], int]]],
) -> RecordedSnapshot:
    before_draw = list(hands[seat])
    _remove_raw(before_draw, current_draw)
    public_melds = tuple(
        tuple((kind, tuple(_pair(raw) for raw in raws), index) for kind, raws, index in seat_melds)
        for seat_melds in melds
    )
    return RecordedSnapshot(
        event_index=event_index,
        seat=seat,
        discard_index=discard_index,
        total_draws=total_draws,
        current_draw=current_draw,
        hand_before_draw=_sorted_raw(before_draw),
        hand_before_action=_sorted_raw(hands[seat]),
        scores=tuple(int(value) for value in scores),  # type: ignore[arg-type]
        active_riichi=tuple(sorted(active_riichi.items())),
        rivers=tuple(tuple(items) for items in rivers),
        public_melds=public_melds,
    )


def parse_recorded_round(round_log: Sequence[Any]) -> ParsedRound:
    """原牌譜1局を、D.1抽出器と独立した正規化イベントへ変換する。"""
    if len(round_log) < 17:
        raise RecordedChronologyError("round_log_too_short")
    info = round_log[0] if isinstance(round_log[0], list) else []
    dealer = int(info[0]) % 4 if info else 0
    kyoku = int(info[0]) if info else 0
    honba = int(info[1]) if len(info) > 1 else 0
    kyoutaku = int(info[2]) if len(info) > 2 else 0
    scores = _vector4(round_log[1])
    if scores is None:
        raise RecordedChronologyError("start_scores_invalid")
    start_scores = scores
    score_work = list(scores)
    initial: list[tuple[int, ...]] = []
    hands: list[list[int]] = []
    for seat in range(4):
        value = round_log[4 + seat * 3]
        if not isinstance(value, list) or len(value) != 13 or any(not isinstance(raw, int) for raw in value):
            raise RecordedChronologyError(f"initial_hand_invalid:{seat}")
        raws = tuple(int(raw) for raw in value)
        for raw in raws:
            _pair(raw)
        initial.append(raws)
        hands.append(list(raws))

    draw_cursor = [0, 0, 0, 0]
    discard_cursor = [0, 0, 0, 0]
    active_riichi: dict[int, int] = {}
    rivers: list[list[tuple[int, bool, bool, bool, int]]] = [[], [], [], []]
    melds: list[list[tuple[str, tuple[int, ...], int]]] = [[], [], [], []]
    events: list[RecordedEvent] = []
    snapshots: list[RecordedSnapshot] = []
    current = dealer
    needs_draw = True
    next_draw_is_rinshan = False
    current_draw: int | None = None
    total_draws = 0
    event_index = 0
    expected_discards = sum(
        1
        for seat in range(4)
        for raw in round_log[6 + seat * 3]
        if call_marker(raw) not in {"a", "k"}
    )
    processed_discards = 0
    guard = sum(len(round_log[5 + seat * 3]) + len(round_log[6 + seat * 3]) for seat in range(4)) * 3 + 20

    for _ in range(guard):
        draws = round_log[5 + current * 3]
        discards = round_log[6 + current * 3]
        if not isinstance(draws, list) or not isinstance(discards, list):
            raise RecordedChronologyError("draw_or_discard_array_invalid")
        if needs_draw:
            if draw_cursor[current] >= len(draws):
                break
            raw_draw = draws[draw_cursor[current]]
            if call_marker(raw_draw) is not None:
                raise RecordedChronologyError("unexpected_call_at_draw")
            if not isinstance(raw_draw, int):
                raise RecordedChronologyError("unparseable_draw")
            _pair(raw_draw)
            source = "rinshan" if next_draw_is_rinshan else "live"
            events.append(RecordedEvent(event_index, "draw", current, raw=raw_draw, draw_source=source))
            event_index += 1
            hands[current].append(raw_draw)
            current_draw = raw_draw
            draw_cursor[current] += 1
            total_draws += 1
            next_draw_is_rinshan = False
        else:
            current_draw = None

        if discard_cursor[current] >= len(discards):
            break
        discard_index = discard_cursor[current]
        raw_discard = discards[discard_index]
        marker = call_marker(raw_discard)
        if marker in {"a", "k"}:
            _, all_tiles, consumed, _ = call_parts(raw_discard)
            if not all_tiles:
                raise RecordedChronologyError(f"kan_token_invalid:{raw_discard}")
            added_raw: int | None = None
            if marker == "a":
                for raw in consumed:
                    _remove_raw(hands[current], raw, same_kind=True)
                kind = "ankan"
                melds[current].append((kind, tuple(all_tiles), event_index))
            else:
                # 加槓では手牌から増加分一枚だけが移る。牌譜tokenの牌種で既存ポンを特定する。
                kind = "kakan"
                wanted = tile34(all_tiles[0])
                matching = next(
                    (item for item in melds[current] if item[0] == "pon" and tile34(item[1][0]) == wanted),
                    None,
                )
                if matching is None:
                    raise RecordedChronologyError("kakan_without_pon")
                added = next((raw for raw in consumed if raw in hands[current] and tile34(raw) == wanted), None)
                if added is None:
                    added = next((raw for raw in hands[current] if tile34(raw) == wanted), None)
                if added is None:
                    raise RecordedChronologyError("kakan_tile_missing")
                _remove_raw(hands[current], added)
                added_raw = added
                melds[current].remove(matching)
                melds[current].append((kind, tuple(all_tiles), event_index))
            events.append(
                RecordedEvent(
                    event_index,
                    kind,
                    current,
                    raw=added_raw,
                    token=str(raw_discard),
                    consumed_raw=tuple(consumed),
                )
            )
            event_index += 1
            discard_cursor[current] += 1
            needs_draw = True
            next_draw_is_rinshan = True
            continue

        actual_raw = discard_physical_raw(raw_discard, current_draw)
        if actual_raw is None:
            raise RecordedChronologyError("unparseable_discard")
        _pair(actual_raw)
        if current_draw is not None:
            snapshots.append(
                _snapshot(
                    event_index=event_index,
                    seat=current,
                    discard_index=discard_index,
                    total_draws=total_draws,
                    current_draw=current_draw,
                    hands=hands,
                    scores=score_work,
                    active_riichi=active_riichi,
                    rivers=rivers,
                    melds=melds,
                )
            )
        _remove_raw(hands[current], actual_raw)
        tsumogiri = raw_discard in (60, "r60")
        riichi = isinstance(raw_discard, str) and raw_discard.startswith("r")
        events.append(
            RecordedEvent(
                event_index,
                "discard",
                current,
                raw=actual_raw,
                token=str(raw_discard),
                discard_index=discard_index,
                is_tsumogiri=tsumogiri,
                is_riichi=riichi,
            )
        )
        rivers[current].append((*_pair(actual_raw), tsumogiri, riichi, event_index))
        discard_cursor[current] += 1
        processed_discards += 1
        event_index += 1
        if riichi:
            active_riichi[current] = event_index - 1
            score_work[current] -= 1000

        callers: list[tuple[int, str, str, list[int], list[int]]] = []
        for seat in range(4):
            if seat == current:
                continue
            candidate_draws = round_log[5 + seat * 3]
            if draw_cursor[seat] >= len(candidate_draws):
                continue
            token = candidate_draws[draw_cursor[seat]]
            call_kind, all_tiles, consumed, called_raw = call_parts(token)
            if (
                call_kind in {"c", "p", "m"}
                and tile34(called_raw) == tile34(actual_raw)
                and call_source_matches(token, seat, current)
            ):
                callers.append((seat, call_kind, str(token), all_tiles, consumed))
        if callers:
            callers.sort(key=lambda item: ({"m": 0, "p": 1, "c": 2}[item[1]], (item[0] - current) % 4))
            caller, call_kind, token, all_tiles, consumed = callers[0]
            for raw in consumed:
                _remove_raw(hands[caller], raw, same_kind=True)
            kind = {"c": "chi", "p": "pon", "m": "daiminkan"}[call_kind]
            melds[caller].append((kind, tuple(all_tiles), event_index))
            events.append(
                RecordedEvent(
                    event_index,
                    kind,
                    caller,
                    token=token,
                    from_seat=current,
                    consumed_raw=tuple(consumed),
                )
            )
            event_index += 1
            draw_cursor[caller] += 1
            current = caller
            needs_draw = call_kind == "m"
            next_draw_is_rinshan = call_kind == "m"
        else:
            current = (current + 1) % 4
            needs_draw = True
            next_draw_is_rinshan = False
    else:
        raise RecordedChronologyError("guard_limit_reached")

    if processed_discards != expected_discards:
        # ツモ和了では最後の自摸後に打牌がないため、打牌数だけを比較すれば一致する。
        raise RecordedChronologyError("discard_count_mismatch")
    dora = round_log[2] if isinstance(round_log[2], list) else []
    ura = round_log[3] if isinstance(round_log[3], list) else []
    if any(not isinstance(raw, int) or tile34(raw) is None for raw in (*dora, *ura)):
        raise RecordedChronologyError("indicator_invalid")
    return ParsedRound(
        events=tuple(events),
        snapshots=tuple(snapshots),
        initial_hands=tuple(initial),
        dealer_seat=dealer,
        kyoku=kyoku,
        honba=honba,
        kyoutaku=kyoutaku,
        start_scores=start_scores,
        dora_raw=tuple(int(raw) for raw in dora),
        ura_raw=tuple(int(raw) for raw in ura),
        result=round_log[-1],
    )


def _expected_snapshot(snapshot: RecordedSnapshot, parsed: ParsedRound) -> dict[str, Any]:
    return {
        "publicState": {
            "dealer_seat": parsed.dealer_seat,
            "kyoku": parsed.kyoku,
            "honba": parsed.honba,
            "riichi_sticks": parsed.kyoutaku + len(snapshot.active_riichi),
            "scores": list(snapshot.scores),
            "turn_seat": snapshot.seat,
            "remaining_live_wall_tiles": 70 - snapshot.total_draws,
            "active_riichi": [list(item) for item in snapshot.active_riichi],
            "dora_indicators": [list(_pair(parsed.dora_raw[0]))],
            "rivers": [[list(item) for item in river] for river in snapshot.rivers],
            "public_melds": [
                [[kind, [list(tile) for tile in tiles], event_index] for kind, tiles, event_index in seat_melds]
                for seat_melds in snapshot.public_melds
            ],
        },
        "playerView": {
            "seat": snapshot.seat,
            "seat_wind_index": (snapshot.seat - parsed.dealer_seat) % 4,
            "concealed_tiles_before_draw": [list(_pair(raw)) for raw in snapshot.hand_before_draw],
            "drawn_tile": list(_pair(snapshot.current_draw)),
            "hand_before_action": [list(_pair(raw)) for raw in snapshot.hand_before_action],
            "shanten_before_draw": shanten(_counts34(snapshot.hand_before_draw), 0),
            "shanten_before_discard": shanten(_counts34(snapshot.hand_before_action), 0),
        },
    }


def _counts34(raws: Sequence[int]) -> tuple[int, ...]:
    values = [0] * 34
    for raw in raws:
        index = tile34(raw)
        if index is None:
            raise RecordedChronologyError("tile34_missing")
        values[index] += 1
    return tuple(values)


def compare_policy_decision(decision: Mapping[str, Any], parsed: ParsedRound) -> tuple[str, ...]:
    """独立復元した局面とD.1の公開状態・自家視点を比較する。"""
    source = decision["source"]
    target = (
        int(source["eventIndex"]),
        int(decision["playerView"]["seat"]),
        int(source["discardIndex"]),
    )
    snapshot = next(
        (
            item
            for item in parsed.snapshots
            if (item.event_index, item.seat, item.discard_index) == target
        ),
        None,
    )
    if snapshot is None:
        return ("decision_snapshot_missing",)
    expected = _expected_snapshot(snapshot, parsed)
    mismatches = []
    for field in ("publicState", "playerView"):
        actual = json.loads(json.dumps(decision[field], ensure_ascii=False))
        if expected[field] != actual:
            mismatches.append(f"{field}_differs")
    return tuple(mismatches)


class _TileAllocator:
    def __init__(self) -> None:
        self.available: dict[tuple[int, bool], list[int]] = defaultdict(list)
        for tile in canonical_tile_set():
            self.available[(tile.tile34, tile.is_red)].append(tile.tile_id)

    def take(self, raw: int) -> int:
        key = _pair(raw)
        if not self.available[key]:
            raise RuntimeReplayError(f"牌が4枚を超える: tile34={key[0]}, red={key[1]}")
        return self.available[key].pop(0)

    def remaining(self, *, reverse: bool) -> list[int]:
        values = [tile_id for ids in self.available.values() for tile_id in ids]
        return sorted(values, reverse=reverse)


@dataclass(frozen=True)
class RuntimeWorld:
    state: RoundState
    draw_tile_ids: Mapping[int, int]


def build_runtime_world(parsed: ParsedRound, *, reverse_completion: bool = False) -> RuntimeWorld:
    """記録済み牌を固定し、未記録位置だけを決定的に補完した136牌状態を作る。"""
    allocator = _TileAllocator()
    hands = [[allocator.take(raw) for raw in hand] for hand in parsed.initial_hands]
    draw_ids: dict[int, int] = {}
    live_front: list[int] = []
    rinshan_known: list[int] = []
    for event in parsed.events:
        if event.kind != "draw" or event.raw is None:
            continue
        tile_id = allocator.take(event.raw)
        draw_ids[event.event_index] = tile_id
        if event.draw_source == "rinshan":
            rinshan_known.append(tile_id)
        else:
            live_front.append(tile_id)
    if len(rinshan_known) > 4 or len(parsed.dora_raw) > 5 or len(parsed.ura_raw) > 5:
        raise RuntimeReplayError("王牌の記録位置が上限を超える")
    dora_known = [allocator.take(raw) for raw in parsed.dora_raw]
    ura_known = [allocator.take(raw) for raw in parsed.ura_raw]
    remaining = allocator.remaining(reverse=reverse_completion)

    def fill(known: list[int], size: int) -> list[int]:
        missing = size - len(known)
        if missing < 0 or len(remaining) < missing:
            raise RuntimeReplayError("未記録位置を補完できない")
        extra = remaining[:missing]
        del remaining[:missing]
        return [*known, *extra]

    rinshan = fill(rinshan_known, 4)
    dora = fill(dora_known, 5)
    ura = fill(ura_known, 5)
    if len(live_front) > 70 or len(remaining) != 70 - len(live_front):
        raise RuntimeReplayError("生牌70枚へ補完できない")
    live_wall = [*live_front, *remaining]
    dead_wall = [*rinshan, *dora, *ura]
    round_wind = 27 + min(parsed.kyoku // 4, 3)
    state = RoundState.from_parts(
        hands,
        live_wall,
        dead_wall,
        scores=parsed.start_scores,
        dealer_seat=parsed.dealer_seat,
        round_wind_tile34=round_wind,
        kyoku=parsed.kyoku,
        honba=parsed.honba,
        kyoutaku=parsed.kyoutaku,
    )
    # 原牌譜のura配列は得点へ現れた表示牌だけを持つ局がある。
    # 未記録slotの任意補完を得点へ混ぜると、補完順で点差が変わるため参照させない。
    state.ura_indicator_slots = tuple(ura_known)
    state.validate()
    return RuntimeWorld(state, draw_ids)


def _matching_ids(state: RoundState, seat: int, raws: Sequence[int]) -> tuple[int, ...]:
    available = list(state.hands[seat])
    result = []
    for raw in raws:
        wanted = _pair(raw)
        match = next(
            (tile_id for tile_id in available if (state.tile_by_id[tile_id].tile34, state.tile_by_id[tile_id].is_red) == wanted),
            None,
        )
        if match is None:
            raise RuntimeReplayError(f"鳴き牌を手牌から解決できない: seat={seat}, raw={raw}")
        available.remove(match)
        result.append(match)
    return tuple(result)


def _resolve_pass_before_next_event(state: RoundState) -> None:
    if state.phase == PHASE_DISCARD_RESPONSES:
        state.resolve_discard_responses([])


def apply_recorded_event(state: RoundState, event: RecordedEvent, draw_tile_ids: Mapping[int, int]) -> None:
    """正規化済みの公開イベント一件をRoundStateへ厳密に適用する。"""
    if event.kind == "draw":
        if event.draw_source == "rinshan":
            if state.phase == PHASE_CHANKAN_RESPONSES:
                state.resolve_chankan_responses([])
            actual = state.drawn_tile_id
        else:
            _resolve_pass_before_next_event(state)
            actual = state.draw()
        expected = draw_tile_ids[event.event_index]
        if actual != expected or state.turn_seat != event.seat:
            raise RuntimeReplayError("自摸牌または手番が原牌譜と一致しない")
        return
    if event.kind == "discard":
        if state.turn_seat != event.seat or event.raw is None:
            raise RuntimeReplayError("打牌者または打牌tokenが不正")
        if event.is_tsumogiri:
            tile_id = state.drawn_tile_id
            if tile_id is None or (state.tile_by_id[tile_id].tile34, state.tile_by_id[tile_id].is_red) != _pair(event.raw):
                raise RuntimeReplayError("ツモ切り牌が原牌譜と一致しない")
        else:
            tile_id = _matching_ids(state, event.seat, (event.raw,))[0]
        state.discard(tile_id, declare_riichi=event.is_riichi)
        return
    if event.kind in {"chi", "pon", "daiminkan"}:
        if event.from_seat is None:
            raise RuntimeReplayError("鳴き元がない")
        consumed = _matching_ids(state, event.seat, event.consumed_raw)
        state.resolve_discard_responses([ResponseClaim(event.kind, event.seat, consumed)])  # type: ignore[arg-type]
        return
    if event.kind == "ankan":
        ids = _matching_ids(state, event.seat, event.consumed_raw)
        state.declare_ankan(ids)
        return
    if event.kind == "kakan":
        wanted = tile34(event.raw)
        meld_index = next(
            (
                index
                for index, meld in enumerate(state.melds[event.seat])
                if meld.kind == "pon" and state.tile_by_id[meld.tile_ids[0]].tile34 == wanted
            ),
            None,
        )
        if meld_index is None:
            raise RuntimeReplayError("加槓元のポンを解決できない")
        tile_id = _matching_ids(state, event.seat, (event.raw,))[0] if event.raw is not None else None
        if tile_id is None:
            raise RuntimeReplayError("加槓牌を解決できない")
        state.propose_kakan(meld_index, tile_id)
        return
    raise RuntimeReplayError(f"未対応の記録イベント: {event.kind}")


def _result_details(result: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(result, list):
        return ()
    details = []
    for item in result[2:]:
        if isinstance(item, list) and len(item) >= 3 and isinstance(item[0], int) and isinstance(item[1], int):
            details.append((int(item[0]), int(item[1])))
    return tuple(details)


def _result_delta(result: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(result, list):
        return None
    vectors = [vector for item in result[1:] if (vector := _vector4(item)) is not None]
    if not vectors:
        return None
    return tuple(sum(vector[seat] for vector in vectors) for seat in range(4))  # type: ignore[return-value]


def _result_kind(result: Any) -> str:
    if not isinstance(result, list) or not result:
        return "unknown"
    if result[0] == "和了":
        details = _result_details(result)
        return "tsumo" if details and all(winner == loser for winner, loser in details) else "ron"
    if result[0] == "流局":
        return "exhaustive_draw"
    return "unsupported"


def _expected_end_scores(parsed: ParsedRound) -> tuple[int, int, int, int] | None:
    delta = _result_delta(parsed.result)
    if delta is None:
        return None
    declared = {event.seat for event in parsed.events if event.kind == "discard" and event.is_riichi}
    winners = {winner for winner, _ in _result_details(parsed.result)}
    return tuple(
        parsed.start_scores[seat] + delta[seat] - (1000 if seat in declared & winners else 0)
        for seat in range(4)
    )  # type: ignore[return-value]


def score_delta_check(parsed: ParsedRound) -> tuple[str, str | None]:
    """result差分の形と、供託を含む総和規則を独立に検査する。"""
    delta = _result_delta(parsed.result)
    kind = _result_kind(parsed.result)
    if delta is None:
        return "hold", "settlement_delta_missing"
    declarations = {event.seat for event in parsed.events if event.kind == "discard" and event.is_riichi}
    winners = {winner for winner, _ in _result_details(parsed.result)}
    if kind in {"ron", "tsumo"}:
        expected_sum = 1000 * (parsed.kyoutaku + len(declarations & winners))
    elif kind == "exhaustive_draw":
        expected_sum = -1000 * len(declarations)
    else:
        return "hold", "result_kind_unsupported"
    if sum(delta) != expected_sum:
        return "mismatch", "settlement_sum_differs"
    return "pass", None


def replay_with_round_state(parsed: ParsedRound, *, reverse_completion: bool = False) -> dict[str, Any]:
    """記録イベントをRoundStateへ適用し、終局種別と局末点を照合する。"""
    world = build_runtime_world(parsed, reverse_completion=reverse_completion)
    state = world.state
    for event in parsed.events:
        apply_recorded_event(state, event, world.draw_tile_ids)
    details = _result_details(parsed.result)
    expected_kind = _result_kind(parsed.result)
    if state.phase != PHASE_ENDED:
        if expected_kind == "tsumo" and details:
            winner = details[0][0]
            if state.turn_seat != winner:
                raise RuntimeReplayError("ツモ和了者と手番が一致しない")
            state.declare_tsumo()
        elif expected_kind == "ron" and details:
            claims = [ResponseClaim("ron", winner) for winner, _ in details]
            if state.phase == PHASE_CHANKAN_RESPONSES:
                state.resolve_chankan_responses(claims)
            elif state.phase == PHASE_DISCARD_RESPONSES:
                state.resolve_discard_responses(claims)
            else:
                raise RuntimeReplayError("ロン応答phaseではない")
        elif expected_kind == "exhaustive_draw":
            _resolve_pass_before_next_event(state)
        else:
            raise RuntimeReplayError("終局結果を適用できない")
    if state.result is None or state.phase != PHASE_ENDED:
        raise RuntimeReplayError("局エンジンが終局しない")
    expected_scores = _expected_end_scores(parsed)
    return {
        "status": "pass" if state.result.kind == expected_kind and tuple(state.scores) == expected_scores else "mismatch",
        "recordedKind": expected_kind,
        "runtimeKind": state.result.kind,
        "recordedEndScores": expected_scores,
        "runtimeEndScores": tuple(state.scores),
        "eventsApplied": len(parsed.events),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            yield json.loads(raw)


def _manifest_matches(summary: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
    left = [(item["season"], item["path"], item["bytes"], item["sha256"]) for item in summary["input"]["files"]]
    right = [(item["season"], item["path"], item["bytes"], item["sha256"]) for item in selected["files"]]
    return left == right and summary["input"]["selectedSeasons"] == selected["selectedSeasons"]


def _scoring_package_version() -> str | None:
    try:
        return package_version("mahjong")
    except PackageNotFoundError:
        return None


def audit_policy_runtime(
    dataset_dir: Path,
    vault_root: Path,
    reservation_path: Path,
    *,
    runtime_limit: int | None = None,
) -> dict[str, Any]:
    """D.1全入口と、参照局の実行器適用を理由別に監査する。"""
    started = time.perf_counter()
    selected = select_development_inputs(vault_root, reservation_path)
    dataset_verification = verify_policy_dataset(dataset_dir)
    summary = json.loads((dataset_dir / "extraction-summary.json").read_text(encoding="utf-8"))
    manifest_match = _manifest_matches(summary, selected)
    decisions = list(_iter_jsonl(dataset_dir / "policy-decisions.jsonl"))
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    scope = Counter()
    seasons = Counter()
    for decision in decisions:
        source = decision["source"]
        groups[(str(source["file"]), int(source["line"]), int(source["logIndex"]))].append(decision)
        south4 = int(decision["publicState"]["kyoku"]) >= 7
        scope["south4OrLater" if south4 else "beforeSouth4"] += 1
        if south4 and decision["isPrimaryWithinSeatRound"]:
            scope["south4OrLaterPrimary"] += 1
        seasons[str(source["season"])] += 1

    required: dict[str, set[int]] = defaultdict(set)
    for source_file, line, _ in groups:
        required[source_file].add(line)
    entry_results = Counter()
    entry_reasons = Counter()
    score_results = Counter()
    score_reasons = Counter()
    runtime_results = Counter()
    runtime_reasons = Counter()
    runtime_checked = 0
    runtime_seasons = Counter()
    runtime_south4 = 0
    completion_mismatches = 0
    runtime_examples: list[dict[str, Any]] = []
    missing_sources: set[str] = set()
    ordered_runtime_keys = sorted(
        groups,
        key=lambda key: hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
    if runtime_limit is not None:
        ordered_runtime_keys = ordered_runtime_keys[:runtime_limit]
    runtime_keys = set(ordered_runtime_keys)

    for source_file, lines in required.items():
        path = vault_root / source_file
        if not path.is_file():
            missing_sources.add(source_file)
            for key, grouped in groups.items():
                if key[0] == source_file:
                    entry_results["hold"] += len(grouped)
                    entry_reasons["source_missing"] += len(grouped)
            continue
        by_line: dict[int, list[tuple[int, list[dict[str, Any]]]]] = defaultdict(list)
        for (group_file, line, log_index), grouped in groups.items():
            if group_file == source_file:
                by_line[line].append((log_index, grouped))
        found: set[int] = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if line_number not in lines:
                    if line_number >= max(lines):
                        break
                    continue
                found.add(line_number)
                row = json.loads(raw)
                logs = row.get("paifu", {}).get("log", [])
                for log_index, grouped in by_line[line_number]:
                    source_key = (source_file, line_number, log_index)
                    if log_index not in range(len(logs)):
                        entry_results["hold"] += len(grouped)
                        entry_reasons["source_log_missing"] += len(grouped)
                        continue
                    try:
                        parsed = parse_recorded_round(logs[log_index])
                    except (RecordedChronologyError, ValueError) as error:
                        entry_results["hold"] += len(grouped)
                        entry_reasons[f"chronology:{error}"] += len(grouped)
                        continue
                    for decision in grouped:
                        mismatches = compare_policy_decision(decision, parsed)
                        if mismatches:
                            entry_results["mismatch"] += 1
                            entry_reasons.update(mismatches)
                        else:
                            entry_results["pass"] += 1
                    score_status, score_reason = score_delta_check(parsed)
                    score_results[score_status] += 1
                    if score_reason:
                        score_reasons[score_reason] += 1
                    if source_key not in runtime_keys:
                        continue
                    runtime_checked += 1
                    runtime_seasons[str(grouped[0]["source"]["season"])] += 1
                    runtime_south4 += int(parsed.kyoku >= 7)
                    try:
                        normal = replay_with_round_state(parsed, reverse_completion=False)
                        reverse = replay_with_round_state(parsed, reverse_completion=True)
                        if normal != reverse:
                            completion_mismatches += 1
                            runtime_results["mismatch"] += 1
                            runtime_reasons["unknown_completion_changes_result"] += 1
                            if len(runtime_examples) < 20:
                                runtime_examples.append(
                                    {"source": list(source_key), "status": "mismatch", "reason": "unknown_completion_changes_result"}
                                )
                        else:
                            runtime_results[normal["status"]] += 1
                            if normal["status"] != "pass":
                                runtime_reasons["round_result_or_score_differs"] += 1
                                if len(runtime_examples) < 20:
                                    runtime_examples.append(
                                        {
                                            "source": list(source_key),
                                            "status": normal["status"],
                                            "reason": "round_result_or_score_differs",
                                            "comparison": normal,
                                        }
                                    )
                    except ScoringDependencyError as error:
                        runtime_results["error"] += 1
                        reason = f"{type(error).__name__}:{error}"
                        runtime_reasons[reason] += 1
                        if len(runtime_examples) < 20:
                            runtime_examples.append({"source": list(source_key), "status": "error", "reason": reason})
                    except Exception as error:  # 牌譜または未対応裁定は理由を集計し、学習入口へ進ませない。
                        runtime_results["hold"] += 1
                        reason = f"{type(error).__name__}:{error}"
                        runtime_reasons[reason] += 1
                        if len(runtime_examples) < 20:
                            runtime_examples.append({"source": list(source_key), "status": "hold", "reason": reason})
            for missing in lines - found:
                missing_count = sum(len(grouped) for _, grouped in by_line[missing])
                entry_results["hold"] += missing_count
                entry_reasons["source_line_missing"] += missing_count

    declared_count = int(summary["records"]["decisions"])
    entry_pass = (
        dataset_verification["status"] == "pass"
        and manifest_match
        and len(decisions) == declared_count
        and entry_results["pass"] == len(decisions)
    )
    status = "pass_with_runtime_holds" if entry_pass else "fail"
    if runtime_results["error"]:
        status = "fail"
    if entry_pass and runtime_results["hold"] == 0 and runtime_results["mismatch"] == 0:
        status = "pass" if runtime_results["error"] == 0 else "fail"
    code_paths = [Path(__file__), Path(__file__).with_name("ev_policy_round.py"), Path(__file__).with_name("ev_policy_scoring.py")]
    return {
        "schemaVersion": REPLAY_AUDIT_SCHEMA,
        "phase": "D.3.0",
        "status": status,
        "eligibleForLearning": False,
        "entryGate": {
            "status": "pass" if entry_pass else "fail",
            "reservationManifestMatchesD1": manifest_match,
            "datasetVerificationStatus": dataset_verification["status"],
            "declaredDecisions": declared_count,
            "auditedDecisions": len(decisions),
            "results": dict(sorted(entry_results.items())),
            "reasons": dict(sorted(entry_reasons.items())),
            "missingSources": sorted(missing_sources),
        },
        "scope": {
            "allD1Decisions": len(decisions),
            "beforeSouth4": scope["beforeSouth4"],
            "south4OrLater": scope["south4OrLater"],
            "south4OrLaterPrimary": scope["south4OrLaterPrimary"],
            "sourceRoundLogs": len(groups),
            "seasons": dict(sorted(seasons.items())),
        },
        "modelEntryScope": {
            "status": "held_until_D3_1",
            "eligibleBeforeSouth4": scope["beforeSouth4"],
            "heldSouth4OrLater": scope["south4OrLater"],
            "reasons": {"south4_or_later_outside_initial_policy_scope": scope["south4OrLater"]},
        },
        "scoreDeltaConformance": {
            "results": dict(sorted(score_results.items())),
            "reasons": dict(sorted(score_reasons.items())),
        },
        "roundRuntimeConformance": {
            "checkedRoundLogs": runtime_checked,
            "runtimeLimit": runtime_limit,
            "selectionMethod": "sha256_order_without_replacement" if runtime_limit is not None else "all_source_round_logs",
            "sampleSeasons": dict(sorted(runtime_seasons.items())),
            "sampleSouth4OrLater": runtime_south4,
            "results": dict(sorted(runtime_results.items())),
            "reasons": dict(sorted(runtime_reasons.items())),
            "examples": runtime_examples,
            "unknownCompletionMismatches": completion_mismatches,
            "unobservedResponseHistory": "held_separately_from_recorded_public_result",
        },
        "acceptanceEvidence": {
            "D30-01": [
                "tests/test_ev_policy_replay.py::RecordedEventAdapterTest",
                "tests/test_ev_policy_replay.py::RoundRuntimeFixtureTest",
                "tests/test_ev_policy_round.py::PolicyRoundTest",
            ],
            "D30-02": "tests/test_ev_policy_round.py::ScoringErrorBoundaryTest",
            "D30-03": "entryGate and scope in this report",
        },
        "runtimeEnvironment": {
            "pythonVersion": platform.python_version(),
            "scoringPackage": "mahjong",
            "scoringPackageVersion": _scoring_package_version(),
        },
        "codeSha256": {path.name: _sha256(path) for path in code_paths},
        "runtimeSeconds": time.perf_counter() - started,
    }
