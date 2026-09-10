"""フェーズDの公開状態、プレイヤー視点、入力台帳を定義する。

D.1では対局を進めない。方策へ渡せる情報の境界と、将来評価用の
シーズンを開発入力へ混ぜない規則を、実行可能な形で固定する。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POLICY_RECORD_SCHEMA = "ev-policy-decision/v1"
POLICY_ACTION_SCHEMA = "ev-policy-action/v1"
RULE_PROFILE_ID = "mleague-phase-d-v1"

FORBIDDEN_POLICY_KEYS = frozenset(
    {
        "actor",
        "actualActionId",
        "actualDiscardTile34",
        "actualShantenAfterDiscard",
        "isActual",
        "observedOutcomeId",
        "outcomeObserved",
        "outcome",
        "result",
        "resultClass",
        "rewardPoints",
        "roundEndScores",
        "futureDraws",
        "uraDoraIndicators",
    }
)


@dataclass(frozen=True)
class RuleProfile:
    """D.1で確定した、局面とリーチ可否に必要なルール境界。"""

    profile_id: str = RULE_PROFILE_ID
    players: int = 4
    red_fives_per_suit: int = 1
    dead_wall_tiles: int = 14
    initial_live_wall_tiles: int = 70
    allow_riichi_without_next_draw: bool = True
    prohibit_riichi_on_haitei_draw: bool = True
    allow_negative_score_after_riichi_deposit: bool = True

    def can_declare_riichi(
        self,
        *,
        closed: bool,
        shanten_after_discard: int,
        after_normal_draw: bool,
        live_wall_tiles_after_draw: int,
    ) -> bool:
        if not closed or shanten_after_discard != 0 or not after_normal_draw:
            return False
        if live_wall_tiles_after_draw < 0:
            raise ValueError("live_wall_tiles_after_drawは0以上が必要")
        if self.prohibit_riichi_on_haitei_draw and live_wall_tiles_after_draw == 0:
            return False
        return self.allow_riichi_without_next_draw or live_wall_tiles_after_draw >= 4

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TileInstance:
    tile_id: int
    tile34: int
    is_red: bool


@dataclass(frozen=True)
class TileLocation:
    tile_id: int
    zone: str
    seat: int | None = None


@dataclass(frozen=True)
class PublicState:
    dealer_seat: int
    kyoku: int | None
    honba: int
    riichi_sticks: int
    scores: tuple[int, int, int, int]
    turn_seat: int
    remaining_live_wall_tiles: int
    active_riichi: tuple[tuple[int, int], ...]
    dora_indicators: tuple[tuple[int, bool], ...]
    rivers: tuple[tuple[tuple[int, bool, bool, bool, int], ...], ...]
    public_melds: tuple[tuple[tuple[str, tuple[tuple[int, bool], ...], int], ...], ...]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlayerView:
    seat: int
    seat_wind_index: int
    concealed_tiles_before_draw: tuple[tuple[int, bool], ...]
    drawn_tile: tuple[int, bool]
    hand_before_action: tuple[tuple[int, bool], ...]
    shanten_before_draw: int
    shanten_before_discard: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorldState:
    """D.2以降のルール実行器だけが受け取る完全状態。"""

    tiles: tuple[TileInstance, ...]
    locations: tuple[TileLocation, ...]

    def validate(self) -> None:
        validate_world_inventory(self.tiles, self.locations)


def canonical_tile_set() -> tuple[TileInstance, ...]:
    """各牌種4枚、各色の5に赤1枚を割り当てた136枚を返す。"""
    tiles: list[TileInstance] = []
    tile_id = 0
    red_types = {4, 13, 22}
    for tile_index in range(34):
        for copy_index in range(4):
            tiles.append(TileInstance(tile_id, tile_index, tile_index in red_types and copy_index == 0))
            tile_id += 1
    return tuple(tiles)


def validate_world_inventory(tiles: Sequence[TileInstance], locations: Sequence[TileLocation]) -> None:
    if len(tiles) != 136 or len(locations) != 136:
        raise ValueError("WorldStateは136枚すべての牌と所在を必要とする")
    tile_ids = [tile.tile_id for tile in tiles]
    location_ids = [location.tile_id for location in locations]
    if len(set(tile_ids)) != 136 or set(tile_ids) != set(range(136)):
        raise ValueError("tile_idは0から135まで一意である必要がある")
    if len(set(location_ids)) != 136 or set(location_ids) != set(tile_ids):
        raise ValueError("各tile_idの所在はちょうど一つ必要")
    expected = canonical_tile_set()
    actual = sorted((tile.tile_id, tile.tile34, tile.is_red) for tile in tiles)
    canonical = sorted((tile.tile_id, tile.tile34, tile.is_red) for tile in expected)
    if actual != canonical:
        raise ValueError("牌種または赤牌の構成がMリーグ用136枚と一致しない")


def _tile_pair(tile: Mapping[str, Any]) -> tuple[int, bool]:
    return int(tile["tile34"]), bool(tile.get("isRed", False))


def public_state_from_decision(decision: Mapping[str, Any]) -> PublicState:
    active_riichi = tuple(
        (int(item["seat"]), int(item["declarationEventIndex"])) for item in decision["activeRiichi"]
    )
    dora = tuple(_tile_pair(tile) for tile in decision["publicDoraIndicators"])
    rivers = tuple(
        tuple(
            (
                int(item["tile34"]),
                bool(item.get("isRed", False)),
                bool(item.get("isTsumogiri", False)),
                bool(item.get("isRiichiDeclaration", False)),
                int(item["eventIndex"]),
            )
            for item in river
        )
        for river in decision["rivers"]
    )
    melds = tuple(
        tuple(
            (
                str(meld["type"]),
                tuple(_tile_pair(tile) for tile in meld["tiles"]),
                int(meld["eventIndex"]),
            )
            for meld in seat_melds
        )
        for seat_melds in decision["publicMelds"]
    )
    scores = tuple(int(value) for value in decision["scoresAtDecision"])
    if len(scores) != 4:
        raise ValueError("scoresAtDecisionは4人分必要")
    return PublicState(
        dealer_seat=int(decision["dealerSeat"]),
        kyoku=None if decision.get("kyoku") is None else int(decision["kyoku"]),
        honba=int(decision.get("honba") or 0),
        riichi_sticks=int(decision["riichiSticksAtDecision"]),
        scores=scores,  # type: ignore[arg-type]
        turn_seat=int(decision["seat"]),
        remaining_live_wall_tiles=int(decision["remainingWallTiles"]),
        active_riichi=active_riichi,
        dora_indicators=dora,
        rivers=rivers,
        public_melds=melds,
    )


def player_view_from_decision(decision: Mapping[str, Any]) -> PlayerView:
    drawn = _tile_pair(decision["drawnTile"])
    return PlayerView(
        seat=int(decision["seat"]),
        seat_wind_index=int(decision["seatWindIndex"]),
        concealed_tiles_before_draw=tuple(_tile_pair(tile) for tile in decision["concealedTilesBeforeDraw"]),
        drawn_tile=drawn,
        hand_before_action=tuple(_tile_pair(tile) for tile in decision["handBeforeAction"]),
        shanten_before_draw=int(decision["shantenBeforeDraw"]),
        shanten_before_discard=int(decision["shantenBeforeDiscard"]),
    )


def policy_action_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """観測された実打牌を示す印を除き、方策が比較できる行動へ変換する。"""
    allowed = {
        "actionId",
        "discardRaw",
        "discardTile34",
        "discardTile",
        "discardsRed",
        "copiesInHand",
        "riichiDeclaration",
        "riichiLegal",
        "shantenAfterDiscard",
        "ukeireKinds",
        "ukeireCount",
        "ukeireTiles",
        "furitenAfterDiscard",
        "riichiSeat",
        "dangerClass",
        "dangerLevel",
        "safetyGroup",
    }
    action = {key: candidate[key] for key in allowed if key in candidate}
    return {"schemaVersion": POLICY_ACTION_SCHEMA, "decisionId": candidate["decisionId"], **action}


def policy_decision_record(decision: Mapping[str, Any]) -> dict[str, Any]:
    source = decision["source"]
    split = "developmentConfirmation" if decision["split"] == "finalTest" else decision["split"]
    record = {
        "schemaVersion": POLICY_RECORD_SCHEMA,
        "decisionId": decision["decisionId"],
        "developmentSplit": split,
        "source": {
            "dataset": source["dataset"],
            "file": source["file"],
            "line": int(source["line"]),
            "season": source["season"],
            "gameId": source["gameId"],
            "roundIndex": int(source["roundIndex"]),
            "logIndex": int(source["logIndex"]),
            "eventIndex": int(source["eventIndex"]),
            "discardIndex": int(source["discardIndex"]),
        },
        "date": decision.get("date"),
        "stage": decision.get("stage"),
        "roundName": decision.get("roundName"),
        "isPrimaryWithinSeatRound": bool(decision["isPrimaryWithinSeatRound"]),
        "ruleProfileId": RULE_PROFILE_ID,
        "publicState": public_state_from_decision(decision).to_record(),
        "playerView": player_view_from_decision(decision).to_record(),
    }
    validate_policy_payload(record)
    return record


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


def validate_policy_payload(value: Mapping[str, Any]) -> None:
    leaked = sorted(set(_walk_keys(value)).intersection(FORBIDDEN_POLICY_KEYS))
    if leaked:
        raise ValueError("方策入力に観測行動または未来情報が混入: " + ", ".join(leaked))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_development_inputs(
    vault_root: Path,
    reservation_path: Path,
    selected_seasons: set[str] | None = None,
) -> dict[str, Any]:
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    if reservation.get("schemaVersion") != "ev-policy-future-evaluation-reservation/v1":
        raise ValueError("未対応の将来評価台帳schemaVersion")
    development = reservation["development"]
    allowed = set(map(str, development["allowedSeasons"]))
    forbidden = set(map(str, development["forbiddenSeasons"]))
    if allowed.intersection(forbidden):
        raise ValueError("開発用と将来評価用のシーズンが重複している")
    available_paths = sorted((vault_root / "data" / "mleague" / "paifu").glob("*.jsonl"))
    available = {path.stem: path for path in available_paths}
    requested = allowed if selected_seasons is None else set(selected_seasons)
    rejected = requested - allowed
    if rejected:
        raise ValueError("開発入力として許可されていないシーズン: " + ", ".join(sorted(rejected)))
    missing = requested - set(available)
    if missing:
        raise ValueError("指定シーズンの牌譜がない: " + ", ".join(sorted(missing)))
    if not requested:
        raise ValueError("開発用の牌譜が選択されていない")
    files = []
    for season in sorted(requested):
        path = available[season]
        files.append(
            {
                "season": season,
                "path": f"data/mleague/paifu/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schemaVersion": "ev-policy-input-validation/v1",
        "status": "pass",
        "reservationId": reservation["reservationId"],
        "reservationStatus": reservation["status"],
        "selectedSeasons": sorted(requested),
        "allowedSeasons": sorted(allowed),
        "forbiddenSeasons": sorted(forbidden),
        "availableUnlistedSeasons": sorted(set(available) - allowed - forbidden),
        "availableForbiddenSeasons": sorted(set(available).intersection(forbidden)),
        "defaultForUnlistedSeason": development["defaultForUnlistedSeason"],
        "files": files,
    }


def verify_policy_dataset(dataset_dir: Path) -> dict[str, Any]:
    """D.1出力を全件走査し、ハッシュ、参照、情報境界を検証する。"""
    summary = json.loads((dataset_dir / "extraction-summary.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    aggregate = hashlib.sha256()
    for entry in summary["generatedFiles"]["files"]:
        path = dataset_dir / entry["path"]
        if not path.is_file():
            errors.append(f"missing:{entry['path']}")
            continue
        digest = _sha256(path)
        if path.stat().st_size != entry["bytes"]:
            errors.append(f"bytes:{entry['path']}")
        if digest != entry["sha256"]:
            errors.append(f"sha256:{entry['path']}")
        aggregate.update(entry["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if aggregate.hexdigest() != summary["generatedFiles"]["aggregateSha256"]:
        errors.append("aggregateSha256")

    decisions: dict[str, int] = {}
    decision_rows = 0
    with (dataset_dir / "policy-decisions.jsonl").open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            row = json.loads(raw)
            validate_policy_payload(row)
            decision_id = str(row["decisionId"])
            if decision_id in decisions:
                errors.append(f"duplicateDecisionId:{decision_id}")
            decisions[decision_id] = int(row["publicState"]["remaining_live_wall_tiles"])
            if row["ruleProfileId"] != RULE_PROFILE_ID:
                errors.append(f"ruleProfileId:{decision_id}")
            if len(row["playerView"]["concealed_tiles_before_draw"]) != 13:
                errors.append(f"concealedTileCount:{decision_id}")
            if len(row["playerView"]["hand_before_action"]) != 14:
                errors.append(f"actionHandTileCount:{decision_id}")
            decision_rows = line_number

    action_ids: set[str] = set()
    action_counts: dict[str, int] = {decision_id: 0 for decision_id in decisions}
    in_scope_counts: dict[str, int] = {decision_id: 0 for decision_id in decisions}
    riichi_with_no_next_draw = 0
    action_rows = 0
    with (dataset_dir / "policy-actions.jsonl").open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            row = json.loads(raw)
            validate_policy_payload(row)
            decision_id = str(row["decisionId"])
            action_id = str(row["actionId"])
            if action_id in action_ids:
                errors.append(f"duplicateActionId:{action_id}")
            action_ids.add(action_id)
            if decision_id not in decisions:
                errors.append(f"orphanAction:{action_id}")
                continue
            action_counts[decision_id] += 1
            if int(row["shantenAfterDiscard"]) in (0, 1):
                in_scope_counts[decision_id] += 1
            if row["riichiDeclaration"]:
                remaining = decisions[decision_id]
                if not row["riichiLegal"] or row["shantenAfterDiscard"] != 0 or remaining == 0:
                    errors.append(f"illegalRiichiAction:{action_id}")
                if remaining in (1, 2, 3):
                    riichi_with_no_next_draw += 1
            action_rows = line_number
    for decision_id in decisions:
        if action_counts[decision_id] == 0:
            errors.append(f"missingActions:{decision_id}")
        if in_scope_counts[decision_id] == 0:
            errors.append(f"missingInScopeAction:{decision_id}")
    if decision_rows != summary["records"]["decisions"]:
        errors.append("decisionCount")
    if action_rows != summary["records"]["actions"]:
        errors.append("actionCount")
    if errors:
        raise ValueError("D.1データセット検証に失敗: " + ", ".join(errors[:20]))
    return {
        "schemaVersion": "ev-policy-dataset-verification/v1",
        "status": "pass",
        "aggregateSha256": summary["generatedFiles"]["aggregateSha256"],
        "decisions": decision_rows,
        "actions": action_rows,
        "riichiActionsWithOneToThreeLiveWallTiles": riichi_with_no_next_draw,
        "forbiddenKeys": 0,
        "duplicateDecisionIds": 0,
        "duplicateActionIds": 0,
        "orphanActions": 0,
        "decisionsWithoutInScopeAction": 0,
    }
