"""フェーズD.2cの決定的方策、合成世界、牌譜prefix再検証。

合成世界は牌保存と状態機械のデバッグ専用であり、現実の局収支予測には使わない。
方策へ渡す値は公開情報、自家手牌、合法行動から作り、他家手牌と未来山を含めない。
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__:
    from .ev_calibration_state import classify_danger, shanten, ukeire
    from .ev_policy_round import (
        PHASE_DISCARD_RESPONSES,
        PHASE_DRAW,
        PHASE_ENDED,
        PHASE_SELF_ACTION,
        ResponseClaim,
        RiichiStatus,
        RiverTile,
        RoundState,
        SelfAction,
    )
    from .ev_policy_state import RULE_PROFILE_ID, canonical_tile_set, policy_decision_record
else:
    from ev_calibration_state import classify_danger, shanten, ukeire
    from ev_policy_round import (
        PHASE_DISCARD_RESPONSES,
        PHASE_DRAW,
        PHASE_ENDED,
        PHASE_SELF_ACTION,
        ResponseClaim,
        RiichiStatus,
        RiverTile,
        RoundState,
        SelfAction,
    )
    from ev_policy_state import RULE_PROFILE_ID, canonical_tile_set, policy_decision_record


PUSH_POLICY = "closed_push_v1"
FOLD_POLICY = "closed_fold_v1"
SYNTHETIC_BELIEF_VERSION = "uniform_visible_tiles_v1"
SYNTHETIC_OPPONENT_VERSION = "closed_push_v1"
SIMULATION_SCHEMA = "ev-policy-synthetic-debug/v1"
REPLAY_SCHEMA = "ev-policy-replay-verification/v1"


class UnsupportedSimulation(ValueError):
    """D.2cのデバッグ範囲外であることを理由付きで表す。"""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PolicyActionView:
    """方策が比較する、物理牌IDを含まない合法行動。"""

    kind: str
    tile34: int | None = None
    is_red: bool = False
    shanten_after_discard: int = 99
    ukeire_count: int = 0
    remaining_dora: int = 0
    danger_levels: tuple[int, ...] = ()

    @property
    def max_danger(self) -> int:
        return max(self.danger_levels, default=0)

    @property
    def sum_danger(self) -> int:
        return sum(self.danger_levels)


@dataclass(frozen=True)
class SimulationPlayerView:
    """一手を選ぶための情報。隠れ状態や出典IDは持たない。"""

    seat: int
    active_riichi_seats: tuple[int, ...]
    hand: tuple[tuple[int, bool], ...]
    drawn_tile: tuple[int, bool] | None
    actions: tuple[PolicyActionView, ...]


def _dora_tile34(indicator: int) -> int:
    if indicator < 27:
        start = indicator // 9 * 9
        return start + (indicator - start + 1) % 9
    if indicator <= 30:
        return 27 + (indicator - 27 + 1) % 4
    return 31 + (indicator - 31 + 1) % 3


def _normal_before_red(action: PolicyActionView) -> int:
    return int(action.is_red)


def _attack_key(action: PolicyActionView) -> tuple[int, int, int, int, int, int, int]:
    # テンパイ時は同じ牌のダマよりリーチを選ぶ。
    riichi_rank = 0 if action.kind == "riichi_discard" else 1
    return (
        action.shanten_after_discard,
        -action.ukeire_count,
        -action.remaining_dora,
        action.max_danger,
        action.tile34 if action.tile34 is not None else 99,
        _normal_before_red(action),
        riichi_rank,
    )


def _defense_key(action: PolicyActionView) -> tuple[int, int, int, int, int, int, int]:
    return (
        action.max_danger,
        action.sum_danger,
        action.shanten_after_discard,
        -action.ukeire_count,
        -action.remaining_dora,
        action.tile34 if action.tile34 is not None else 99,
        _normal_before_red(action),
    )


def choose_policy_action(policy_version: str, view: SimulationPlayerView) -> PolicyActionView:
    """PlayerViewだけから、仕様順で一つの合法行動を決める。"""
    wins = [action for action in view.actions if action.kind == "tsumo"]
    if wins:
        return wins[0]
    discards = [action for action in view.actions if action.kind in {"discard", "riichi_discard"}]
    if policy_version == PUSH_POLICY:
        if not discards:
            raise UnsupportedSimulation("push_policy_has_no_legal_discard")
        return min(discards, key=_attack_key)
    if policy_version == FOLD_POLICY:
        discards = [action for action in discards if action.kind == "discard"]
        if not discards:
            raise UnsupportedSimulation("fold_policy_has_no_non_riichi_discard")
        return min(discards, key=_defense_key)
    raise ValueError(f"未対応の方策: {policy_version}")


def initial_player_view(
    decision: Mapping[str, Any], actions: Sequence[Mapping[str, Any]], policy_version: str
) -> SimulationPlayerView:
    """D.1レコードを、初手方策用の小さな視点へ変換する。"""
    public = decision["publicState"]
    player = decision["playerView"]
    selected: list[PolicyActionView] = []
    for action in actions:
        if str(action["decisionId"]) != str(decision["decisionId"]):
            continue
        shanten_after = int(action["shantenAfterDiscard"])
        is_riichi = bool(action.get("riichiDeclaration", False))
        if policy_version == PUSH_POLICY and shanten_after not in (0, 1):
            continue
        if policy_version == FOLD_POLICY and is_riichi:
            continue
        dangers = (int(action.get("dangerLevel", 0)),) if public["active_riichi"] else ()
        selected.append(
            PolicyActionView(
                kind="riichi_discard" if is_riichi else "discard",
                tile34=int(action["discardTile34"]),
                is_red=bool(action.get("discardsRed", False)),
                shanten_after_discard=shanten_after,
                ukeire_count=int(action.get("ukeireCount", 0)),
                remaining_dora=_remaining_dora_from_record(decision, action),
                danger_levels=dangers,
            )
        )
    if not selected:
        raise UnsupportedSimulation(f"{policy_version}_has_no_initial_candidate")
    drawn = tuple(player["drawn_tile"])
    return SimulationPlayerView(
        seat=int(player["seat"]),
        active_riichi_seats=tuple(int(item[0]) for item in public["active_riichi"]),
        hand=tuple((int(item[0]), bool(item[1])) for item in player["hand_before_action"]),
        drawn_tile=(int(drawn[0]), bool(drawn[1])),
        actions=tuple(selected),
    )


def _remaining_dora_from_record(decision: Mapping[str, Any], action: Mapping[str, Any]) -> int:
    hand = [(int(tile[0]), bool(tile[1])) for tile in decision["playerView"]["hand_before_action"]]
    discard = (int(action["discardTile34"]), bool(action.get("discardsRed", False)))
    hand.remove(discard)
    indicators = [int(tile[0]) for tile in decision["publicState"]["dora_indicators"]]
    dora_types = Counter(_dora_tile34(index) for index in indicators)
    return sum(dora_types[index] + int(is_red) for index, is_red in hand)


def _public_rivers(state: RoundState) -> list[list[dict[str, Any]]]:
    by_id = state.tile_by_id
    return [
        [
            {
                "tile34": by_id[item.tile_id].tile34,
                "isRed": by_id[item.tile_id].is_red,
                "called": item.called,
            }
            for item in river
        ]
        for river in state.rivers
    ]


def round_player_view(state: RoundState, seat: int) -> SimulationPlayerView:
    """状態機械から公開情報と自家情報だけを読み、継続方策の入力を作る。"""
    if seat != state.turn_seat or state.phase != PHASE_SELF_ACTION:
        raise UnsupportedSimulation("not_self_action")
    by_id = state.tile_by_id
    rivers = _public_rivers(state)
    seen: Counter[int] = Counter(by_id[tile_id].tile34 for tile_id in state.hands[seat])
    for river in rivers:
        for tile in river:
            if not tile["called"]:
                seen[int(tile["tile34"])] += 1
    for melds in state.melds:
        for meld in melds:
            seen.update(by_id[tile_id].tile34 for tile_id in meld.tile_ids)
    for indicator in state.current_dora_indicators:
        seen[indicator.tile34] += 1
    dora_types = Counter(_dora_tile34(tile.tile34) for tile in state.current_dora_indicators)
    active = tuple(sorted(state.active_riichi))
    action_views: list[PolicyActionView] = []
    for legal in state.legal_self_actions():
        if legal.kind == "tsumo":
            action_views.append(PolicyActionView("tsumo"))
            continue
        if legal.kind not in {"discard", "riichi_discard"} or legal.tile_id is None:
            continue
        tile = by_id[legal.tile_id]
        remaining_ids = list(state.hands[seat])
        remaining_ids.remove(legal.tile_id)
        counts = Counter(by_id[tile_id].tile34 for tile_id in remaining_ids)
        counts_tuple = tuple(counts[index] for index in range(34))
        after = shanten(counts_tuple, len(state.melds[seat]))
        reception = ukeire(counts_tuple, after, seen)
        dangers = tuple(
            int(classify_danger(tile.tile34, riichi_seat, rivers, seen)["dangerLevel"])
            for riichi_seat in active
        )
        remaining_dora = sum(
            dora_types[by_id[tile_id].tile34] + int(by_id[tile_id].is_red) for tile_id in remaining_ids
        )
        action_views.append(
            PolicyActionView(
                kind=legal.kind,
                tile34=tile.tile34,
                is_red=tile.is_red,
                shanten_after_discard=after,
                ukeire_count=int(reception["ukeireCount"]),
                remaining_dora=remaining_dora,
                danger_levels=dangers,
            )
        )
    drawn = None
    if state.drawn_tile_id is not None:
        tile = by_id[state.drawn_tile_id]
        drawn = (tile.tile34, tile.is_red)
    return SimulationPlayerView(
        seat=seat,
        active_riichi_seats=active,
        hand=tuple(sorted((by_id[tile_id].tile34, by_id[tile_id].is_red) for tile_id in state.hands[seat])),
        drawn_tile=drawn,
        actions=tuple(action_views),
    )


def _resolve_action(state: RoundState, choice: PolicyActionView) -> SelfAction:
    by_id = state.tile_by_id
    for legal in state.legal_self_actions():
        if legal.kind != choice.kind:
            continue
        if choice.kind == "tsumo":
            return legal
        if legal.tile_id is not None:
            tile = by_id[legal.tile_id]
            if (tile.tile34, tile.is_red) == (choice.tile34, choice.is_red):
                return legal
    raise UnsupportedSimulation("semantic_action_not_legal_in_world")


def _apply_self_action(state: RoundState, action: SelfAction) -> None:
    if action.kind == "tsumo":
        state.declare_tsumo()
    elif action.kind in {"discard", "riichi_discard"} and action.tile_id is not None:
        state.discard(action.tile_id, declare_riichi=action.kind == "riichi_discard")
    else:
        raise UnsupportedSimulation(f"synthetic_policy_does_not_select_{action.kind}")


def run_synthetic_round(
    initial_state: RoundState,
    *,
    target_seat: int,
    target_policy: str,
    forced_initial: PolicyActionView | None = None,
    max_steps: int = 600,
) -> dict[str, Any]:
    """固定された一世界を、鳴き・槓をしない単純方策で局末まで進める。"""
    state = copy.deepcopy(initial_state)
    initial_scores = tuple(state.scores)
    first = True
    for step in range(max_steps):
        if state.phase == PHASE_ENDED:
            assert state.result is not None
            return {
                "status": "synthetic_debug",
                "steps": step,
                "result": state.result.kind,
                "winner": state.result.winner,
                "loser": state.result.loser,
                "scoreDeltas": [state.scores[index] - initial_scores[index] for index in range(4)],
                "targetPoints": state.scores[target_seat] - initial_scores[target_seat],
            }
        if state.phase == PHASE_DRAW:
            state.draw()
            continue
        if state.phase == PHASE_SELF_ACTION:
            seat = state.turn_seat
            view = round_player_view(state, seat)
            if first and seat == target_seat and forced_initial is not None:
                choice = forced_initial
            else:
                choice = choose_policy_action(target_policy if seat == target_seat else PUSH_POLICY, view)
            _apply_self_action(state, _resolve_action(state, choice))
            first = False
            continue
        if state.phase == PHASE_DISCARD_RESPONSES:
            discarder = state.pending_discard[0] if state.pending_discard else -1
            claims: list[ResponseClaim] = []
            for seat in range(4):
                if seat == discarder:
                    continue
                ron = next((item for item in state.legal_response_claims(seat) if item.kind == "ron"), None)
                if ron is not None:
                    claims.append(ron)
            state.resolve_discard_responses(claims)
            continue
        raise UnsupportedSimulation(f"unsupported_round_phase:{state.phase}")
    raise UnsupportedSimulation("round_step_limit_exceeded")


class _TileAllocator:
    def __init__(self) -> None:
        self.tiles = canonical_tile_set()
        self.available: dict[tuple[int, bool], list[int]] = defaultdict(list)
        for tile in self.tiles:
            self.available[(tile.tile34, tile.is_red)].append(tile.tile_id)

    def take(self, pair: Sequence[Any]) -> int:
        key = (int(pair[0]), bool(pair[1]))
        if not self.available[key]:
            raise UnsupportedSimulation(f"visible_tile_count_exceeded:{key[0]}:{int(key[1])}")
        return self.available[key].pop(0)

    def rest(self) -> list[int]:
        return [tile_id for ids in self.available.values() for tile_id in ids]


def synthetic_world_from_decision(decision: Mapping[str, Any], seed: int) -> RoundState:
    """D.1局面の可視牌を固定し、未知牌だけを一様に配るデバッグ世界を作る。"""
    if decision.get("ruleProfileId") != RULE_PROFILE_ID:
        raise UnsupportedSimulation("rule_profile_mismatch")
    public = decision["publicState"]
    player = decision["playerView"]
    if any(public["public_melds"]):
        raise UnsupportedSimulation("public_meld_not_supported_by_uniform_debug_world")
    if int(public["turn_seat"]) != int(player["seat"]):
        raise UnsupportedSimulation("turn_seat_mismatch")
    allocator = _TileAllocator()
    concealed = [allocator.take(tile) for tile in player["concealed_tiles_before_draw"]]
    drawn = allocator.take(player["drawn_tile"])
    hands: list[list[int]] = [[], [], [], []]
    target = int(player["seat"])
    hands[target] = concealed + [drawn]
    rivers: list[list[RiverTile]] = [[], [], [], []]
    for seat, source_river in enumerate(public["rivers"]):
        for item in source_river:
            tile_id = allocator.take(item[:2])
            rivers[seat].append(
                RiverTile(
                    tile_id=tile_id,
                    is_tsumogiri=bool(item[2]),
                    is_riichi_declaration=bool(item[3]),
                    event_index=int(item[4]),
                )
            )
    dora_pairs = list(public["dora_indicators"])
    if len(dora_pairs) != 1:
        raise UnsupportedSimulation("synthetic_world_requires_one_visible_dora")
    first_dora = allocator.take(dora_pairs[0])
    unknown = allocator.rest()
    random.Random(seed).shuffle(unknown)
    cursor = 0
    for seat in range(4):
        if seat == target:
            continue
        hands[seat] = unknown[cursor : cursor + 13]
        cursor += 13
    remaining = int(public["remaining_live_wall_tiles"])
    live_wall = unknown[cursor : cursor + remaining]
    cursor += remaining
    dead_unknown = unknown[cursor:]
    if len(dead_unknown) != 13:
        raise UnsupportedSimulation(f"tile_ledger_mismatch:dead_unknown={len(dead_unknown)}")
    dead_wall = dead_unknown[:4] + [first_dora] + dead_unknown[4:]
    by_id = {tile.tile_id: tile for tile in allocator.tiles}
    active: dict[int, RiichiStatus] = {}
    for seat_value, declaration_event in public["active_riichi"]:
        seat = int(seat_value)
        counts = Counter(by_id[tile_id].tile34 for tile_id in hands[seat])
        active[seat] = RiichiStatus(
            int(declaration_event),
            False,
            tuple(counts[index] for index in range(34)),
        )
    event_index = 1 + max((item.event_index for river in rivers for item in river), default=-1)
    return RoundState(
        tiles=allocator.tiles,
        hands=hands,
        live_wall=live_wall,
        dead_wall=dead_wall,
        rinshan_tiles=list(dead_wall[:4]),
        dora_indicator_slots=tuple(dead_wall[4:9]),
        ura_indicator_slots=tuple(dead_wall[9:14]),
        scores=[int(value) for value in public["scores"]],
        dealer_seat=int(public["dealer_seat"]),
        round_wind_tile34=27 + min(int(public["kyoku"] or 0) // 4, 3),
        kyoku=int(public["kyoku"] or 0),
        honba=int(public["honba"]),
        kyoutaku=int(public["riichi_sticks"]),
        turn_seat=target,
        phase=PHASE_SELF_ACTION,
        rivers=rivers,
        active_riichi=active,
        drawn_tile_id=drawn,
        draw_source="live",
        is_last_live_draw=remaining == 0,
        event_index=event_index,
    )


def evaluate_synthetic_debug(
    decision: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    policies: Sequence[str] = (PUSH_POLICY, FOLD_POLICY),
) -> dict[str, Any]:
    """同じseedの初期世界を全初手候補で共有し、デバッグ局収支を集計する。"""
    started = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    target = int(decision["playerView"]["seat"])
    worlds = {seed: synthetic_world_from_decision(decision, seed) for seed in seeds}
    for policy in policies:
        view = initial_player_view(decision, actions, policy)
        for initial in view.actions:
            outcomes = []
            errors: Counter[str] = Counter()
            for seed in seeds:
                try:
                    outcomes.append(
                        run_synthetic_round(
                            worlds[seed], target_seat=target, target_policy=policy, forced_initial=initial
                        )
                    )
                except (ValueError, RuntimeError) as error:
                    errors[f"{type(error).__name__}:{error}"] += 1
            points = [int(item["targetPoints"]) for item in outcomes]
            candidates.append(
                {
                    "policyVersion": policy,
                    "initialAction": asdict(initial),
                    "completedTrials": len(outcomes),
                    "unscorableTrials": sum(errors.values()),
                    "unscorableReasons": dict(sorted(errors.items())),
                    "meanTargetPoints": sum(points) / len(points) if points else None,
                    "outcomeCounts": dict(sorted(Counter(item["result"] for item in outcomes).items())),
                }
            )
    elapsed = time.perf_counter() - started
    return {
        "schemaVersion": SIMULATION_SCHEMA,
        "scopeStatus": "synthetic_debug",
        "holdReasons": [
            "unknown_tiles_are_uniform_not_history_conditioned",
            "riichi_hidden_hands_are_not_conditioned_to_tenpai",
            "opponents_never_call_or_kan",
            "not_eligible_for_app_scoring",
        ],
        "ruleVersion": "mleague-round-v2",
        "policyVersions": list(policies),
        "beliefVersion": SYNTHETIC_BELIEF_VERSION,
        "opponentVersion": SYNTHETIC_OPPONENT_VERSION,
        "decisionId": decision["decisionId"],
        "codeSha256": _file_sha256(Path(__file__)),
        "inputDecisionSha256": _record_sha256(decision),
        "inputActionsSha256": _record_sha256(list(actions)),
        "publicStateId": hashlib.sha256(
            json.dumps(decision["publicState"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "worldSeeds": list(seeds),
        "candidates": candidates,
        "runtime": {
            "seconds": elapsed,
            "attemptedTrials": len(seeds) * len(candidates),
            "trialsPerSecond": len(seeds) * len(candidates) / elapsed if elapsed else None,
        },
    }


def replay_policy_decision(decision: Mapping[str, Any], raw_row: Mapping[str, Any]) -> dict[str, Any]:
    """原牌譜の記録済みprefixを既存extractorで再構築し、D.1スナップショットと照合する。"""
    if __package__:
        from .calibrate_ev import extract_round
    else:
        from calibrate_ev import extract_round

    source = decision["source"]
    logs = raw_row.get("paifu", {}).get("log", [])
    log_index = int(source["logIndex"])
    if log_index not in range(len(logs)):
        return {"status": "unscorable", "reason": "source_log_index_missing"}
    extracted = extract_round(
        dict(raw_row),
        logs[log_index],
        log_index,
        str(source["file"]),
        int(source["line"]),
        "all",
        "all",
        "pre_action",
    )
    rebuilt = {
        record["decisionId"]: record
        for record in (policy_decision_record(item) for item in extracted["decisions"])
    }
    actual = rebuilt.get(decision["decisionId"])
    if actual is None:
        return {
            "status": "unscorable",
            "reason": "decision_not_reconstructed",
            "extractorStats": extracted["stats"],
        }
    if _record_sha256(actual) != _record_sha256(decision):
        return {"status": "mismatch", "reason": "policy_snapshot_differs"}
    return {
        "status": "pass",
        "recordedPrefixEventIndex": int(source["eventIndex"]),
        "futureWallStatus": "unobserved_not_fabricated",
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify_policy_replay(
    dataset_dir: Path, vault_root: Path, *, limit: int | None = None
) -> dict[str, Any]:
    """D.1判断を原牌譜行へ戻し、記録済みprefixを決定的に全件または指定件数照合する。"""
    started = time.perf_counter()
    decision_count = 0
    by_source: dict[tuple[str, int, int], list[tuple[str, str]]] = defaultdict(list)
    for decision in _iter_jsonl(dataset_dir / "policy-decisions.jsonl"):
        source = decision["source"]
        key = (str(source["file"]), int(source["line"]), int(source["logIndex"]))
        by_source[key].append((str(decision["decisionId"]), _record_sha256(decision)))
        decision_count += 1
        if limit is not None and decision_count >= limit:
            break
    required_lines: dict[str, set[int]] = defaultdict(set)
    for source_file, line, _ in by_source:
        required_lines[source_file].add(line)
    missing_sources: list[str] = []
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    if __package__:
        from .calibrate_ev import extract_round
    else:
        from calibrate_ev import extract_round
    for source_file, line_numbers in required_lines.items():
        path = vault_root / source_file
        groups_by_line: dict[int, list[tuple[int, list[tuple[str, str]]]]] = defaultdict(list)
        for (group_file, group_line, log_index), grouped in by_source.items():
            if group_file == source_file:
                groups_by_line[group_line].append((log_index, grouped))
        if not path.is_file():
            missing_sources.append(source_file)
            missing_count = sum(len(grouped) for values in groups_by_line.values() for _, grouped in values)
            statuses["unscorable"] += missing_count
            reasons["source_row_missing"] += missing_count
            continue
        found_lines: set[int] = set()
        with path.open(encoding="utf-8") as stream:
            for source_line, raw in enumerate(stream, 1):
                if source_line not in line_numbers:
                    if source_line >= max(line_numbers):
                        break
                    continue
                found_lines.add(source_line)
                row = json.loads(raw)
                for log_index, grouped in groups_by_line[source_line]:
                    logs = row.get("paifu", {}).get("log", [])
                    if log_index not in range(len(logs)):
                        statuses["unscorable"] += len(grouped)
                        reasons["source_log_index_missing"] += len(grouped)
                        continue
                    extracted = extract_round(
                        row,
                        logs[log_index],
                        log_index,
                        source_file,
                        source_line,
                        "all",
                        "all",
                        "pre_action",
                    )
                    rebuilt = {
                        record["decisionId"]: record
                        for record in (policy_decision_record(item) for item in extracted["decisions"])
                    }
                    for decision_id, expected_sha256 in grouped:
                        actual = rebuilt.get(decision_id)
                        if actual is None:
                            statuses["unscorable"] += 1
                            reasons["decision_not_reconstructed"] += 1
                        elif _record_sha256(actual) != expected_sha256:
                            statuses["mismatch"] += 1
                            reasons["policy_snapshot_differs"] += 1
                        else:
                            statuses["pass"] += 1
                if source_line >= max(line_numbers):
                    break
        for source_line in line_numbers - found_lines:
            missing_count = sum(len(grouped) for _, grouped in groups_by_line[source_line])
            statuses["unscorable"] += missing_count
            reasons["source_row_missing"] += missing_count
    return {
        "schemaVersion": REPLAY_SCHEMA,
        "status": "pass" if statuses["mismatch"] == 0 and statuses["unscorable"] == 0 else "hold",
        "scope": "recorded_prefix_only",
        "futureWallStatus": "unobserved_not_fabricated",
        "codeSha256": _file_sha256(Path(__file__)),
        "policyDecisionsSha256": _file_sha256(dataset_dir / "policy-decisions.jsonl"),
        "requestedDecisions": decision_count,
        "sourceRoundLogs": len(by_source),
        "runtimeSeconds": time.perf_counter() - started,
        "results": dict(sorted(statuses.items())),
        "reasons": dict(sorted(reasons.items())),
        "missingSourceFiles": sorted(set(missing_sources)),
    }


def load_decision_bundle(dataset_dir: Path, decision_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decision = next(
        (row for row in _iter_jsonl(dataset_dir / "policy-decisions.jsonl") if row["decisionId"] == decision_id),
        None,
    )
    if decision is None:
        raise ValueError(f"decisionIdがない: {decision_id}")
    actions = [
        row for row in _iter_jsonl(dataset_dir / "policy-actions.jsonl") if row["decisionId"] == decision_id
    ]
    if not actions:
        raise ValueError(f"行動候補がない: {decision_id}")
    return decision, actions
