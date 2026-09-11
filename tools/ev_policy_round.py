"""フェーズD.2bの決定的な一局状態機械。

牌譜の未来や方策を参照せず、合法な一局の進行だけを扱う。
得点はev_policy_scoringへ委ね、各遷移後に136牌と点棒を検査する。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from itertools import combinations
from typing import Literal, Sequence

from tools.ev_calibration_state import shanten
from tools.ev_policy_scoring import (
    BaseScore,
    HandFlags,
    HandScore,
    HandScoreRequest,
    HandScoringError,
    MeldInput,
    PaoMeld,
    PaoResponsibility,
    Settlement,
    detect_pao,
    score_hand,
    settle_win,
)
from tools.ev_policy_state import TileInstance, TileLocation, WorldState, canonical_tile_set


ROUND_RULE_PROFILE_ID = "mleague-round-v2"
NON_WINNING_SCORING_CODES = frozenset({"no_yaku"})
PHASE_DRAW = "draw"
PHASE_SELF_ACTION = "self_action"
PHASE_DISCARD_RESPONSES = "discard_responses"
PHASE_CALL_DISCARD = "call_discard"
PHASE_CHANKAN_RESPONSES = "chankan_responses"
PHASE_ENDED = "ended"


class IllegalAction(ValueError):
    """現在の状態では実行できない行動。"""


@dataclass
class RiverTile:
    tile_id: int
    is_tsumogiri: bool
    is_riichi_declaration: bool
    event_index: int
    called: bool = False


@dataclass
class RoundMeld:
    kind: Literal["chi", "pon", "daiminkan", "ankan", "shouminkan"]
    tile_ids: tuple[int, ...]
    called_tile_id: int | None
    from_seat: int | None


@dataclass(frozen=True)
class RiichiStatus:
    declaration_event_index: int
    is_double: bool
    base_counts: tuple[int, ...]


@dataclass(frozen=True)
class ResponseClaim:
    kind: Literal["ron", "pon", "chi", "daiminkan"]
    seat: int
    consumed_tile_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SelfAction:
    kind: Literal["discard", "riichi_discard", "tsumo", "ankan", "kakan"]
    tile_id: int | None = None
    tile_ids: tuple[int, ...] = ()
    meld_index: int | None = None


@dataclass(frozen=True)
class PendingKakan:
    seat: int
    meld_index: int
    tile_id: int


@dataclass(frozen=True)
class RoundResult:
    kind: Literal["ron", "tsumo", "exhaustive_draw"]
    winner: int | None
    loser: int | None
    settlement: Settlement | None
    tenpai_seats: tuple[int, ...]
    score_deltas: tuple[int, int, int, int]
    next_honba: int
    kyoutaku_after: int


def _validate_seat(seat: int) -> None:
    if seat not in range(4):
        raise ValueError("seatは0から3が必要")


def _counts(tile_ids: Sequence[int], tile_by_id: dict[int, TileInstance]) -> tuple[int, ...]:
    result = [0] * 34
    for tile_id in tile_ids:
        result[tile_by_id[tile_id].tile34] += 1
    return tuple(result)


@lru_cache(maxsize=500_000)
def waiting_tile34(counts: tuple[int, ...], fixed_melds: int = 0) -> tuple[int, ...]:
    """13枚相当の形を完成させる牌種を返す。役や場の残り枚数は見ない。"""
    if len(counts) != 34 or any(value < 0 or value > 4 for value in counts):
        raise ValueError("countsは各0〜4の34要素が必要")
    if shanten(counts, fixed_melds) != 0:
        return ()
    waits = []
    for tile34, count in enumerate(counts):
        if count == 4:
            continue
        work = list(counts)
        work[tile34] += 1
        if shanten(tuple(work), fixed_melds) == -1:
            waits.append(tile34)
    return tuple(waits)


def noten_deltas(tenpai_seats: Sequence[int]) -> tuple[int, int, int, int]:
    """流局時の3,000点を均等に移す。"""
    seats = tuple(sorted(set(tenpai_seats)))
    if any(seat not in range(4) for seat in seats):
        raise ValueError("tenpai_seatsが不正")
    if len(seats) in (0, 4):
        return (0, 0, 0, 0)
    gain = 3000 // len(seats)
    loss = 3000 // (4 - len(seats))
    return tuple(gain if seat in seats else -loss for seat in range(4))  # type: ignore[return-value]


def riichi_timing_legal(draw_source: str | None, is_last_live_draw: bool) -> bool:
    """海底の通常ツモだけを禁止し、嶺上ツモ後の宣言を区別する。"""
    if draw_source not in ("live", "rinshan"):
        return False
    return not (draw_source == "live" and is_last_live_draw)


def kuikae_forbidden_tile34(kind: str, called_tile34: int, consumed_tile34: Sequence[int]) -> frozenset[int]:
    """鳴いた直後に切れない牌種。同種の赤・通常は区別しない。"""
    if kind == "pon":
        return frozenset({called_tile34})
    if kind != "chi" or len(consumed_tile34) != 2 or called_tile34 >= 27:
        raise ValueError("食い換え判定用の鳴きが不正")
    suit_start = (called_tile34 // 9) * 9
    pair = tuple(sorted(consumed_tile34))
    forbidden = set()
    for candidate in range(suit_start, suit_start + 9):
        values = sorted((*pair, candidate))
        if values[0] + 1 == values[1] and values[1] + 1 == values[2]:
            forbidden.add(candidate)
    if called_tile34 not in forbidden:
        raise ValueError("チーの構成が順子にならない")
    return frozenset(forbidden)


def _complete_decompositions(counts: tuple[int, ...], melds_needed: int) -> tuple[tuple[int, tuple[tuple[str, int], ...]], ...]:
    """標準形の雀頭と面子を列挙する。固定済み面子はmelds_neededから除く。"""
    if sum(counts) != melds_needed * 3 + 2:
        return ()
    results: set[tuple[int, tuple[tuple[str, int], ...]]] = set()

    def visit(work: list[int], pair: int | None, groups: list[tuple[str, int]]) -> None:
        try:
            index = next(i for i, value in enumerate(work) if value)
        except StopIteration:
            if pair is not None and len(groups) == melds_needed:
                results.add((pair, tuple(sorted(groups))))
            return
        if pair is None and work[index] >= 2:
            work[index] -= 2
            visit(work, index, groups)
            work[index] += 2
        if len(groups) < melds_needed and work[index] >= 3:
            work[index] -= 3
            visit(work, pair, [*groups, ("triplet", index)])
            work[index] += 3
        if len(groups) < melds_needed and index < 27 and index % 9 <= 6:
            if work[index + 1] and work[index + 2]:
                for offset in range(3):
                    work[index + offset] -= 1
                visit(work, pair, [*groups, ("sequence", index)])
                for offset in range(3):
                    work[index + offset] += 1

    visit(list(counts), None, [])
    return tuple(sorted(results))


def _decomposition_signatures(
    base_counts: tuple[int, ...], wait: int, melds_needed: int, remove_triplet: int | None
) -> tuple[tuple[int, tuple[tuple[str, int], ...], tuple[str, int]], ...] | None:
    completed = list(base_counts)
    completed[wait] += 1
    signatures = set()
    for pair, groups in _complete_decompositions(tuple(completed), melds_needed):
        removable = ("triplet", remove_triplet) if remove_triplet is not None else None
        if removable is not None and removable not in groups:
            return None
        roles: list[tuple[str, int]] = []
        if pair == wait:
            roles.append(("pair", wait))
        roles.extend(group for group in groups if group[0] == "triplet" and group[1] == wait)
        roles.extend(group for group in groups if group[0] == "sequence" and group[1] <= wait <= group[1] + 2)
        for role in roles:
            if removable is not None and role == removable:
                return None
            remaining = list(groups)
            if removable is not None:
                remaining.remove(removable)
            signatures.add((pair, tuple(sorted(remaining)), role))
    return tuple(sorted(signatures))


def riichi_ankan_legal(base_counts: tuple[int, ...], candidate_tile34: int, fixed_melds: int = 0) -> bool:
    """リーチ時の全和了分解を保つ暗槓だけを許可する。"""
    if len(base_counts) != 34 or candidate_tile34 not in range(34) or base_counts[candidate_tile34] != 3:
        return False
    before_waits = waiting_tile34(base_counts, fixed_melds)
    if not before_waits:
        return False
    reduced = list(base_counts)
    reduced[candidate_tile34] -= 3
    after_waits = waiting_tile34(tuple(reduced), fixed_melds + 1)
    if before_waits != after_waits:
        return False
    before_melds_needed = 4 - fixed_melds
    after_melds_needed = before_melds_needed - 1
    for wait in before_waits:
        before = _decomposition_signatures(base_counts, wait, before_melds_needed, candidate_tile34)
        after = _decomposition_signatures(tuple(reduced), wait, after_melds_needed, None)
        if before is None or not before or before != after:
            return False
    return True


@dataclass
class RoundState:
    tiles: tuple[TileInstance, ...]
    hands: list[list[int]]
    live_wall: list[int]
    dead_wall: list[int]
    rinshan_tiles: list[int]
    dora_indicator_slots: tuple[int, ...]
    ura_indicator_slots: tuple[int, ...]
    scores: list[int]
    dealer_seat: int = 0
    round_wind_tile34: int = 27
    kyoku: int = 0
    honba: int = 0
    kyoutaku: int = 0
    turn_seat: int = 0
    phase: str = PHASE_DRAW
    melds: list[list[RoundMeld]] = field(default_factory=lambda: [[], [], [], []])
    rivers: list[list[RiverTile]] = field(default_factory=lambda: [[], [], [], []])
    active_riichi: dict[int, RiichiStatus] = field(default_factory=dict)
    ippatsu_eligible: set[int] = field(default_factory=set)
    temporary_furiten: set[int] = field(default_factory=set)
    riichi_furiten: set[int] = field(default_factory=set)
    pao: list[list[PaoResponsibility]] = field(default_factory=lambda: [[], [], [], []])
    total_kans: int = 0
    revealed_dora_count: int = 1
    drawn_tile_id: int | None = None
    draw_source: Literal["live", "rinshan"] | None = None
    is_last_live_draw: bool = False
    pending_discard: tuple[int, RiverTile] | None = None
    pending_riichi: bool = False
    pending_kakan: PendingKakan | None = None
    kuikae_forbidden: frozenset[int] = frozenset()
    event_index: int = 0
    calls_occurred: bool = False
    result: RoundResult | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    _initial_ledger_total: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for seat in range(4):
            _validate_seat(seat)
        _validate_seat(self.dealer_seat)
        _validate_seat(self.turn_seat)
        if self.round_wind_tile34 not in range(27, 31):
            raise ValueError("round_wind_tile34が不正")
        if len(self.scores) != 4 or self.honba < 0 or self.kyoutaku < 0:
            raise ValueError("点数または場情報が不正")
        self._initial_ledger_total = sum(self.scores) + 1000 * self.kyoutaku
        self.validate()

    @classmethod
    def from_parts(
        cls,
        hands: Sequence[Sequence[int]],
        live_wall: Sequence[int],
        dead_wall: Sequence[int],
        *,
        scores: Sequence[int] = (25000, 25000, 25000, 25000),
        dealer_seat: int = 0,
        round_wind_tile34: int = 27,
        kyoku: int = 0,
        honba: int = 0,
        kyoutaku: int = 0,
    ) -> "RoundState":
        if len(hands) != 4 or any(len(hand) != 13 for hand in hands):
            raise ValueError("開始手牌は各家13枚が必要")
        if len(live_wall) != 70 or len(dead_wall) != 14:
            raise ValueError("開始時は生牌70枚、王牌14枚が必要")
        dead = list(dead_wall)
        return cls(
            tiles=canonical_tile_set(),
            hands=[list(hand) for hand in hands],
            live_wall=list(live_wall),
            dead_wall=dead,
            rinshan_tiles=list(dead[:4]),
            dora_indicator_slots=tuple(dead[4:9]),
            ura_indicator_slots=tuple(dead[9:14]),
            scores=list(scores),
            dealer_seat=dealer_seat,
            round_wind_tile34=round_wind_tile34,
            kyoku=kyoku,
            honba=honba,
            kyoutaku=kyoutaku,
            turn_seat=dealer_seat,
        )

    @classmethod
    def shuffled(cls, seed: int, **kwargs: object) -> "RoundState":
        ids = list(range(136))
        random.Random(seed).shuffle(ids)
        hands = [ids[seat * 13:(seat + 1) * 13] for seat in range(4)]
        live_wall = ids[52:122]
        dead_wall = ids[122:136]
        return cls.from_parts(hands, live_wall, dead_wall, **kwargs)

    @cached_property
    def tile_by_id(self) -> dict[int, TileInstance]:
        # 牌集合は局中不変なので、全合法手判定で同じ辞書を作り直さない。
        return {tile.tile_id: tile for tile in self.tiles}

    @property
    def current_dora_indicators(self) -> tuple[TileInstance, ...]:
        by_id = self.tile_by_id
        return tuple(by_id[tile_id] for tile_id in self.dora_indicator_slots[:self.revealed_dora_count])

    def _record(self, event_type: str, **data: object) -> None:
        self.events.append({"eventIndex": self.event_index, "type": event_type, **data})
        self.event_index += 1

    def validate(self) -> None:
        if len(self.dead_wall) != 14:
            raise ValueError("王牌は常に14枚必要")
        if not set(self.rinshan_tiles).issubset(self.dead_wall):
            raise ValueError("未使用の嶺上牌は王牌内に必要")
        if not set(self.dora_indicator_slots + self.ura_indicator_slots).issubset(self.dead_wall):
            raise ValueError("ドラ表示牌は王牌内に必要")
        if self.revealed_dora_count != self.total_kans + 1 or self.revealed_dora_count > 5:
            raise ValueError("槓回数とドラ表示枚数が不一致")
        locations: list[TileLocation] = []
        locations.extend(TileLocation(tile_id, "live_wall") for tile_id in self.live_wall)
        locations.extend(TileLocation(tile_id, "dead_wall") for tile_id in self.dead_wall)
        for seat, hand in enumerate(self.hands):
            locations.extend(TileLocation(tile_id, "hand", seat) for tile_id in hand)
        for seat, seat_melds in enumerate(self.melds):
            for meld in seat_melds:
                locations.extend(TileLocation(tile_id, "meld", seat) for tile_id in meld.tile_ids)
        for seat, river in enumerate(self.rivers):
            locations.extend(TileLocation(item.tile_id, "river", seat) for item in river if not item.called)
        WorldState(self.tiles, tuple(locations)).validate()
        if self.phase != PHASE_ENDED:
            for seat in range(4):
                expected = 13 - 3 * len(self.melds[seat])
                if self.phase in (PHASE_SELF_ACTION, PHASE_CALL_DISCARD) and seat == self.turn_seat:
                    expected += 1
                if self.phase == PHASE_CHANKAN_RESPONSES and self.pending_kakan is not None and seat == self.pending_kakan.seat:
                    expected += 1
                if len(self.hands[seat]) != expected:
                    raise ValueError(f"seat {seat}の局面別手牌枚数が不正")
        if sum(self.scores) + 1000 * self.kyoutaku != self._initial_ledger_total:
            raise ValueError("点数と供託の保存則に違反")
        if self.total_kans not in range(5):
            raise ValueError("槓は全体4回まで")

    def draw(self) -> int:
        if self.phase != PHASE_DRAW or not self.live_wall:
            raise IllegalAction("通常ツモを実行できない")
        tile_id = self.live_wall.pop(0)
        self.hands[self.turn_seat].append(tile_id)
        self.drawn_tile_id = tile_id
        self.draw_source = "live"
        self.is_last_live_draw = not self.live_wall
        self.phase = PHASE_SELF_ACTION
        self._record("draw", seat=self.turn_seat, tileId=tile_id, source="live")
        self.validate()
        return tile_id

    def _is_closed(self, seat: int) -> bool:
        return all(meld.kind == "ankan" for meld in self.melds[seat])

    def _counts_in_hand(self, seat: int, without: int | None = None) -> tuple[int, ...]:
        ids = list(self.hands[seat])
        if without is not None:
            ids.remove(without)
        return _counts(ids, self.tile_by_id)

    def _can_declare_riichi(self, seat: int, discard_tile_id: int) -> bool:
        if seat in self.active_riichi or not self._is_closed(seat):
            return False
        if self.phase != PHASE_SELF_ACTION or not riichi_timing_legal(self.draw_source, self.is_last_live_draw):
            return False
        counts = self._counts_in_hand(seat, discard_tile_id)
        return shanten(counts, len(self.melds[seat])) == 0

    def discard(self, tile_id: int, *, declare_riichi: bool = False) -> RiverTile:
        if self.phase not in (PHASE_SELF_ACTION, PHASE_CALL_DISCARD):
            raise IllegalAction("現在は打牌できない")
        seat = self.turn_seat
        if tile_id not in self.hands[seat]:
            raise IllegalAction("手牌にない牌は切れない")
        tile34 = self.tile_by_id[tile_id].tile34
        if tile34 in self.kuikae_forbidden:
            raise IllegalAction("食い換えになる牌は切れない")
        if seat in self.active_riichi and tile_id != self.drawn_tile_id:
            raise IllegalAction("リーチ後はツモ切りだけ可能")
        if declare_riichi and not self._can_declare_riichi(seat, tile_id):
            raise IllegalAction("この打牌ではリーチできない")
        if self.phase == PHASE_CALL_DISCARD and declare_riichi:
            raise IllegalAction("鳴き直後にリーチできない")
        is_tsumogiri = tile_id == self.drawn_tile_id
        self.hands[seat].remove(tile_id)
        river_tile = RiverTile(tile_id, is_tsumogiri, declare_riichi, self.event_index)
        self.rivers[seat].append(river_tile)
        if seat in self.temporary_furiten and seat not in self.active_riichi:
            self.temporary_furiten.remove(seat)
        if seat in self.active_riichi:
            self.ippatsu_eligible.discard(seat)
        self.pending_discard = (seat, river_tile)
        self.pending_riichi = declare_riichi
        self.drawn_tile_id = None
        self.draw_source = None
        self.is_last_live_draw = False
        self.kuikae_forbidden = frozenset()
        self.phase = PHASE_DISCARD_RESPONSES
        self._record("discard", seat=seat, tileId=tile_id, riichi=declare_riichi)
        self.validate()
        return river_tile

    def _waits(self, seat: int) -> tuple[int, ...]:
        return waiting_tile34(self._counts_in_hand(seat), len(self.melds[seat]))

    def _discard_furiten(self, seat: int, waits: Sequence[int] | None = None) -> bool:
        waits_set = set(self._waits(seat) if waits is None else waits)
        own_discards = {self.tile_by_id[item.tile_id].tile34 for item in self.rivers[seat]}
        return bool(waits_set.intersection(own_discards))

    def _is_furiten(self, seat: int, waits: Sequence[int] | None = None) -> bool:
        return (
            seat in self.temporary_furiten
            or seat in self.riichi_furiten
            or self._discard_furiten(seat, waits)
        )

    def furiten_reasons(self, seat: int) -> tuple[str, ...]:
        _validate_seat(seat)
        reasons = []
        if self._discard_furiten(seat):
            reasons.append("own_discard")
        if seat in self.temporary_furiten:
            reasons.append("temporary")
        if seat in self.riichi_furiten:
            reasons.append("riichi_pass")
        return tuple(reasons)

    def _score_request(
        self,
        seat: int,
        win_tile_id: int,
        *,
        is_tsumo: bool,
        is_chankan: bool = False,
        is_houtei: bool = False,
    ) -> HandScoreRequest:
        by_id = self.tile_by_id
        tile_ids = list(self.hands[seat])
        if not is_tsumo:
            tile_ids.append(win_tile_id)
        for meld in self.melds[seat]:
            tile_ids.extend(meld.tile_ids)
        status = self.active_riichi.get(seat)
        flags = HandFlags(
            is_tsumo=is_tsumo,
            is_riichi=status is not None and not status.is_double,
            is_double_riichi=status is not None and status.is_double,
            is_ippatsu=seat in self.ippatsu_eligible,
            is_rinshan=is_tsumo and self.draw_source == "rinshan",
            is_chankan=is_chankan,
            is_haitei=is_tsumo and self.draw_source == "live" and self.is_last_live_draw,
            is_houtei=is_houtei,
        )
        meld_inputs = []
        for meld in self.melds[seat]:
            kind = "kan" if meld.kind in ("daiminkan", "ankan") else meld.kind
            meld_inputs.append(
                MeldInput(
                    kind=kind,  # type: ignore[arg-type]
                    tile_ids=meld.tile_ids,
                    opened=meld.kind != "ankan",
                    called_tile_id=meld.called_tile_id,
                )
            )
        ura = ()
        if status is not None:
            ura = tuple(by_id[tile_id] for tile_id in self.ura_indicator_slots[:self.revealed_dora_count])
        return HandScoreRequest(
            tiles=tuple(by_id[tile_id] for tile_id in tile_ids),
            win_tile_id=win_tile_id,
            seat_wind_tile34=27 + ((seat - self.dealer_seat) % 4),
            round_wind_tile34=self.round_wind_tile34,
            melds=tuple(meld_inputs),
            dora_indicators=self.current_dora_indicators,
            ura_dora_indicators=ura,
            flags=flags,
        )

    def _score_ron(self, seat: int, tile_id: int, *, chankan: bool = False) -> HandScore:
        waits = self._waits(seat)
        if self.tile_by_id[tile_id].tile34 not in waits or self._is_furiten(seat, waits):
            raise IllegalAction("ロンできない待ちまたはフリテン")
        try:
            return score_hand(
                self._score_request(
                    seat,
                    tile_id,
                    is_tsumo=False,
                    is_chankan=chankan,
                    is_houtei=not chankan and not self.live_wall,
                )
            )
        except HandScoringError as error:
            if error.code in NON_WINNING_SCORING_CODES:
                raise IllegalAction(f"ロンできない: {error.code}") from error
            raise

    def _score_tsumo(self, seat: int) -> HandScore:
        if self.drawn_tile_id is None or self.tile_by_id[self.drawn_tile_id].tile34 not in self._waits_before_draw(seat):
            raise IllegalAction("ツモ和了形ではない")
        try:
            return score_hand(self._score_request(seat, self.drawn_tile_id, is_tsumo=True))
        except HandScoringError as error:
            if error.code in NON_WINNING_SCORING_CODES:
                raise IllegalAction(f"ツモできない: {error.code}") from error
            raise

    def _waits_before_draw(self, seat: int) -> tuple[int, ...]:
        if self.drawn_tile_id is None:
            return ()
        return waiting_tile34(self._counts_in_hand(seat, self.drawn_tile_id), len(self.melds[seat]))

    def _matching_pao(self, seat: int, hand_score: HandScore) -> tuple[PaoResponsibility, ...]:
        yakuman_names = {item.name for item in hand_score.yaku if item.is_yakuman}
        return tuple(item for item in self.pao[seat] if item.yaku in yakuman_names)

    def _apply_win(self, hand_score: HandScore, winner: int, loser: int | None) -> RoundResult:
        settlement = settle_win(
            hand_score.base,
            winner=winner,
            dealer=self.dealer_seat,
            loser=loser,
            honba=self.honba,
            kyoutaku=self.kyoutaku,
            pao=self._matching_pao(winner, hand_score),
        )
        for seat, delta in enumerate(settlement.deltas):
            self.scores[seat] += delta
        self.kyoutaku = settlement.kyoutaku_after
        kind = "tsumo" if hand_score.base.is_tsumo else "ron"
        result = RoundResult(
            kind=kind,
            winner=winner,
            loser=loser,
            settlement=settlement,
            tenpai_seats=(),
            score_deltas=settlement.deltas,
            next_honba=self.honba + 1 if winner == self.dealer_seat else 0,
            kyoutaku_after=self.kyoutaku,
        )
        self.result = result
        self.phase = PHASE_ENDED
        self.pending_discard = None
        self.pending_kakan = None
        self._record(kind, winner=winner, loser=loser, score=hand_score.to_record())
        self.validate()
        return result

    def declare_tsumo(self) -> RoundResult:
        if self.phase != PHASE_SELF_ACTION:
            raise IllegalAction("現在はツモ和了を宣言できない")
        return self._apply_win(self._score_tsumo(self.turn_seat), self.turn_seat, None)

    def _mark_passed_win(self, seat: int) -> None:
        if seat in self.active_riichi:
            self.riichi_furiten.add(seat)
        else:
            self.temporary_furiten.add(seat)

    def _commit_pending_riichi(self, seat: int, river_tile: RiverTile) -> None:
        if not self.pending_riichi:
            return
        counts = self._counts_in_hand(seat)
        is_double = len(self.rivers[seat]) == 1 and not self.calls_occurred
        self.scores[seat] -= 1000
        self.kyoutaku += 1
        self.active_riichi[seat] = RiichiStatus(river_tile.event_index, is_double, counts)
        self.ippatsu_eligible.add(seat)
        self.pending_riichi = False
        self._record("riichi_committed", seat=seat, double=is_double)

    def legal_call_claims(self, seat: int) -> tuple[ResponseClaim, ...]:
        if self.phase != PHASE_DISCARD_RESPONSES or self.pending_discard is None:
            return ()
        discarder, river_tile = self.pending_discard
        if seat == discarder or seat in self.active_riichi or not self.live_wall:
            return ()
        called_type = self.tile_by_id[river_tile.tile_id].tile34
        matching = [tile_id for tile_id in self.hands[seat] if self.tile_by_id[tile_id].tile34 == called_type]
        claims = [ResponseClaim("pon", seat, tuple(ids)) for ids in combinations(matching, 2)]
        if self.total_kans < 4:
            claims.extend(ResponseClaim("daiminkan", seat, tuple(ids)) for ids in combinations(matching, 3))
        if seat == (discarder + 1) % 4 and called_type < 27:
            suit_start = (called_type // 9) * 9
            by_type: dict[int, list[int]] = {}
            for tile_id in self.hands[seat]:
                tile_type = self.tile_by_id[tile_id].tile34
                by_type.setdefault(tile_type, []).append(tile_id)
            for start in range(max(suit_start, called_type - 2), min(suit_start + 6, called_type) + 1):
                sequence = {start, start + 1, start + 2}
                if called_type not in sequence:
                    continue
                needed = sorted(sequence - {called_type})
                if all(by_type.get(tile_type) for tile_type in needed):
                    for left in by_type[needed[0]]:
                        for right in by_type[needed[1]]:
                            consumed = (left, right)
                            forbidden = kuikae_forbidden_tile34(
                                "chi",
                                called_type,
                                tuple(self.tile_by_id[tile_id].tile34 for tile_id in consumed),
                            )
                            remaining = (
                                tile_id for tile_id in self.hands[seat] if tile_id not in consumed
                            )
                            if any(self.tile_by_id[tile_id].tile34 not in forbidden for tile_id in remaining):
                                claims.append(ResponseClaim("chi", seat, consumed))
        return tuple(claims)

    def legal_response_claims(self, seat: int) -> tuple[ResponseClaim, ...]:
        """現在の捨牌に対して、その家が選べるロンと鳴きを返す。"""
        if self.phase != PHASE_DISCARD_RESPONSES or self.pending_discard is None:
            return ()
        discarder, river_tile = self.pending_discard
        if seat == discarder:
            return ()
        claims = list(self.legal_call_claims(seat))
        try:
            self._score_ron(seat, river_tile.tile_id)
        except IllegalAction:
            pass
        else:
            claims.insert(0, ResponseClaim("ron", seat))
        return tuple(claims)

    def _validate_call(self, claim: ResponseClaim) -> None:
        if not any(
            candidate.kind == claim.kind
            and candidate.seat == claim.seat
            and set(candidate.consumed_tile_ids) == set(claim.consumed_tile_ids)
            for candidate in self.legal_call_claims(claim.seat)
        ):
            raise IllegalAction("不正な鳴き希望")

    def _pao_melds(self, seat: int) -> tuple[PaoMeld, ...]:
        return tuple(
            PaoMeld(
                meld.kind,
                self.tile_by_id[meld.tile_ids[0]].tile34,
                meld.from_seat if meld.kind in ("pon", "daiminkan") else None,
            )
            for meld in self.melds[seat]
            if meld.kind in ("pon", "daiminkan", "ankan", "shouminkan")
        )

    def _commit_call(self, claim: ResponseClaim, discarder: int, river_tile: RiverTile) -> None:
        self._validate_call(claim)
        seat = claim.seat
        for tile_id in claim.consumed_tile_ids:
            self.hands[seat].remove(tile_id)
        river_tile.called = True
        kind = claim.kind
        new_meld = RoundMeld(kind, tuple((*claim.consumed_tile_ids, river_tile.tile_id)), river_tile.tile_id, discarder)
        if kind in ("pon", "daiminkan"):
            new_pao = detect_pao(
                self._pao_melds(seat),
                PaoMeld(kind, self.tile_by_id[river_tile.tile_id].tile34, discarder),
            )
            self.pao[seat].extend(new_pao)
        self.melds[seat].append(new_meld)
        self.calls_occurred = True
        self.ippatsu_eligible.clear()
        self.turn_seat = seat
        self._record(kind, seat=seat, fromSeat=discarder, tileId=river_tile.tile_id)
        if kind == "chi":
            consumed_types = [self.tile_by_id[tile_id].tile34 for tile_id in claim.consumed_tile_ids]
            self.kuikae_forbidden = kuikae_forbidden_tile34(
                "chi", self.tile_by_id[river_tile.tile_id].tile34, consumed_types
            )
        elif kind == "pon":
            self.kuikae_forbidden = frozenset({self.tile_by_id[river_tile.tile_id].tile34})
        self.pending_discard = None
        if kind == "daiminkan":
            self._after_kan_committed()
        else:
            self.phase = PHASE_CALL_DISCARD
            self.drawn_tile_id = None
            self.draw_source = None
        self.validate()

    def resolve_discard_responses(self, claims: Sequence[ResponseClaim]) -> RoundResult | None:
        if self.phase != PHASE_DISCARD_RESPONSES or self.pending_discard is None:
            raise IllegalAction("捨牌応答を解決できない")
        discarder, river_tile = self.pending_discard
        if len({claim.seat for claim in claims}) != len(claims):
            raise IllegalAction("同じ家から複数の応答は出せない")
        ron_scores: dict[int, HandScore] = {}
        for claim in claims:
            _validate_seat(claim.seat)
            if claim.seat == discarder:
                raise IllegalAction("打牌者は自分の捨牌へ応答できない")
            if claim.kind == "ron":
                ron_scores[claim.seat] = self._score_ron(claim.seat, river_tile.tile_id)
            else:
                self._validate_call(claim)
        if ron_scores:
            winner = min(ron_scores, key=lambda seat: (seat - discarder) % 4)
            self.pending_riichi = False
            return self._apply_win(ron_scores[winner], winner, discarder)

        called_type = self.tile_by_id[river_tile.tile_id].tile34
        for seat in range(4):
            if seat == discarder or any(claim.seat == seat and claim.kind == "ron" for claim in claims):
                continue
            waits = self._waits(seat)
            if called_type in waits and not self._is_furiten(seat, waits):
                self._mark_passed_win(seat)
        self._commit_pending_riichi(discarder, river_tile)

        call_claims = [claim for claim in claims if claim.kind != "ron"]
        if call_claims:
            priority = {"pon": 0, "daiminkan": 0, "chi": 1}
            selected = min(call_claims, key=lambda item: (priority[item.kind], (item.seat - discarder) % 4))
            self._commit_call(selected, discarder, river_tile)
            return None
        self.pending_discard = None
        self.pending_riichi = False
        if not self.live_wall:
            return self.resolve_exhaustive_draw()
        self.turn_seat = (discarder + 1) % 4
        self.phase = PHASE_DRAW
        self.validate()
        return None

    def _can_start_kan(self) -> bool:
        return self.phase == PHASE_SELF_ACTION and self.total_kans < 4 and bool(self.live_wall) and bool(self.rinshan_tiles)

    def declare_ankan(self, tile_ids: Sequence[int]) -> None:
        if not self._can_start_kan() or len(tile_ids) != 4 or len(set(tile_ids)) != 4:
            raise IllegalAction("暗槓できない")
        seat = self.turn_seat
        if any(tile_id not in self.hands[seat] for tile_id in tile_ids):
            raise IllegalAction("暗槓牌が手牌にない")
        types = {self.tile_by_id[tile_id].tile34 for tile_id in tile_ids}
        if len(types) != 1:
            raise IllegalAction("暗槓は同じ牌種4枚が必要")
        tile_type = next(iter(types))
        if seat in self.active_riichi:
            status = self.active_riichi[seat]
            if not riichi_ankan_legal(status.base_counts, tile_type, len(self.melds[seat])):
                raise IllegalAction("リーチ後の手牌構成を変える暗槓")
            reduced = list(status.base_counts)
            reduced[tile_type] -= 3
            self.active_riichi[seat] = RiichiStatus(
                status.declaration_event_index,
                status.is_double,
                tuple(reduced),
            )
        for tile_id in tile_ids:
            self.hands[seat].remove(tile_id)
        self.melds[seat].append(RoundMeld("ankan", tuple(tile_ids), None, None))
        self.calls_occurred = True
        self._record("ankan", seat=seat, tileIds=tuple(tile_ids))
        self._after_kan_committed()

    def propose_kakan(self, meld_index: int, tile_id: int) -> None:
        if not self._can_start_kan() or self.turn_seat in self.active_riichi:
            raise IllegalAction("加槓できない")
        if meld_index not in range(len(self.melds[self.turn_seat])):
            raise IllegalAction("加槓元の面子がない")
        meld = self.melds[self.turn_seat][meld_index]
        if meld.kind != "pon" or tile_id not in self.hands[self.turn_seat]:
            raise IllegalAction("加槓元はポン、追加牌は手牌に必要")
        if self.tile_by_id[tile_id].tile34 != self.tile_by_id[meld.tile_ids[0]].tile34:
            raise IllegalAction("加槓牌の牌種が違う")
        self.pending_kakan = PendingKakan(self.turn_seat, meld_index, tile_id)
        self.phase = PHASE_CHANKAN_RESPONSES
        self._record("kakan_proposed", seat=self.turn_seat, tileId=tile_id)
        self.validate()

    def resolve_chankan_responses(self, claims: Sequence[ResponseClaim]) -> RoundResult | None:
        if self.phase != PHASE_CHANKAN_RESPONSES or self.pending_kakan is None:
            raise IllegalAction("搶槓応答を解決できない")
        pending = self.pending_kakan
        if any(claim.kind != "ron" or claim.seat == pending.seat for claim in claims):
            raise IllegalAction("加槓にはロンだけ応答できる")
        if len({claim.seat for claim in claims}) != len(claims):
            raise IllegalAction("ロン応答が重複")
        scores = {
            claim.seat: self._score_ron(claim.seat, pending.tile_id, chankan=True)
            for claim in claims
        }
        if scores:
            winner = min(scores, key=lambda seat: (seat - pending.seat) % 4)
            return self._apply_win(scores[winner], winner, pending.seat)
        tile_type = self.tile_by_id[pending.tile_id].tile34
        for seat in range(4):
            if seat == pending.seat:
                continue
            waits = self._waits(seat)
            if tile_type in waits and not self._is_furiten(seat, waits):
                self._mark_passed_win(seat)
        meld = self.melds[pending.seat][pending.meld_index]
        self.hands[pending.seat].remove(pending.tile_id)
        self.melds[pending.seat][pending.meld_index] = RoundMeld(
            "shouminkan", tuple((*meld.tile_ids, pending.tile_id)), meld.called_tile_id, meld.from_seat
        )
        self.pending_kakan = None
        self.calls_occurred = True
        self._record("kakan_committed", seat=pending.seat, tileId=pending.tile_id)
        self._after_kan_committed()
        return None

    def legal_chankan_claims(self, seat: int) -> tuple[ResponseClaim, ...]:
        if self.phase != PHASE_CHANKAN_RESPONSES or self.pending_kakan is None:
            return ()
        if seat == self.pending_kakan.seat:
            return ()
        try:
            self._score_ron(seat, self.pending_kakan.tile_id, chankan=True)
        except IllegalAction:
            return ()
        return (ResponseClaim("ron", seat),)

    def _after_kan_committed(self) -> None:
        if not self.live_wall or not self.rinshan_tiles or self.total_kans >= 4:
            raise IllegalAction("槓を成立させられない")
        self.total_kans += 1
        self.revealed_dora_count += 1
        self.ippatsu_eligible.clear()
        self._record("kan_committed", seat=self.turn_seat, doraCount=self.revealed_dora_count)
        rinshan = self.rinshan_tiles.pop(0)
        self.dead_wall.remove(rinshan)
        replacement = self.live_wall.pop()
        self.dead_wall.append(replacement)
        self.hands[self.turn_seat].append(rinshan)
        self.drawn_tile_id = rinshan
        self.draw_source = "rinshan"
        self.is_last_live_draw = False
        self.phase = PHASE_SELF_ACTION
        self._record("draw", seat=self.turn_seat, tileId=rinshan, source="rinshan")
        self.validate()

    def legal_self_actions(self, *, include_tsumo: bool = True) -> tuple[SelfAction, ...]:
        if self.phase not in (PHASE_SELF_ACTION, PHASE_CALL_DISCARD):
            return ()
        seat = self.turn_seat
        actions: list[SelfAction] = []
        discard_ids = [self.drawn_tile_id] if seat in self.active_riichi else list(self.hands[seat])
        riichi_by_tile_type: dict[int, bool] = {}
        for tile_id in discard_ids:
            if tile_id is None or self.tile_by_id[tile_id].tile34 in self.kuikae_forbidden:
                continue
            actions.append(SelfAction("discard", tile_id=tile_id))
            tile_type = self.tile_by_id[tile_id].tile34
            if tile_type not in riichi_by_tile_type:
                riichi_by_tile_type[tile_type] = self._can_declare_riichi(seat, tile_id)
            if riichi_by_tile_type[tile_type]:
                actions.append(SelfAction("riichi_discard", tile_id=tile_id))
        if self.phase == PHASE_SELF_ACTION:
            if include_tsumo:
                try:
                    self._score_tsumo(seat)
                except IllegalAction:
                    pass
                else:
                    actions.append(SelfAction("tsumo", tile_id=self.drawn_tile_id))
            if self._can_start_kan():
                by_type: dict[int, list[int]] = {}
                for tile_id in self.hands[seat]:
                    by_type.setdefault(self.tile_by_id[tile_id].tile34, []).append(tile_id)
                for tile_type, ids in by_type.items():
                    if len(ids) == 4:
                        if seat not in self.active_riichi or riichi_ankan_legal(
                            self.active_riichi[seat].base_counts, tile_type, len(self.melds[seat])
                        ):
                            actions.append(SelfAction("ankan", tile_ids=tuple(ids)))
                if seat not in self.active_riichi:
                    for index, meld in enumerate(self.melds[seat]):
                        if meld.kind != "pon":
                            continue
                        tile_type = self.tile_by_id[meld.tile_ids[0]].tile34
                        for tile_id in by_type.get(tile_type, []):
                            actions.append(SelfAction("kakan", tile_id=tile_id, meld_index=index))
        return tuple(actions)

    def _is_tenpai_at_exhaustion(self, seat: int) -> bool:
        waits = self._waits(seat)
        if not waits:
            return False
        owned = Counter(self.tile_by_id[tile_id].tile34 for tile_id in self.hands[seat])
        for meld in self.melds[seat]:
            owned.update(self.tile_by_id[tile_id].tile34 for tile_id in meld.tile_ids)
        return any(owned[wait] < 4 for wait in waits)

    def resolve_exhaustive_draw(self) -> RoundResult:
        if self.live_wall or self.phase not in (PHASE_DISCARD_RESPONSES, PHASE_DRAW):
            raise IllegalAction("山が残っているため流局にできない")
        tenpai = tuple(seat for seat in range(4) if self._is_tenpai_at_exhaustion(seat))
        deltas = noten_deltas(tenpai)
        for seat, delta in enumerate(deltas):
            self.scores[seat] += delta
        result = RoundResult(
            kind="exhaustive_draw",
            winner=None,
            loser=None,
            settlement=None,
            tenpai_seats=tenpai,
            score_deltas=deltas,
            next_honba=self.honba + 1,
            kyoutaku_after=self.kyoutaku,
        )
        self.result = result
        self.phase = PHASE_ENDED
        self.pending_discard = None
        self._record("exhaustive_draw", tenpaiSeats=tenpai, scoreDeltas=deltas)
        self.validate()
        return result
