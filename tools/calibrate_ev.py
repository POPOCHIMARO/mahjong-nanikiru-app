#!/usr/bin/env python3
"""実牌譜EV較正パイプラインのCLI入口。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

if __package__:
    from .ev_calibration_state import (
        call_marker,
        call_parts,
        call_source_matches,
        classify_danger,
        counter34,
        discard_physical_raw,
        is_red,
        shanten,
        tile34,
        tile_name,
        tile_record,
        ukeire,
        visible_counts,
    )
else:
    from ev_calibration_state import (
        call_marker,
        call_parts,
        call_source_matches,
        classify_danger,
        counter34,
        discard_physical_raw,
        is_red,
        shanten,
        tile34,
        tile_name,
        tile_record,
        ukeire,
        visible_counts,
    )


APP_ROOT = Path(__file__).resolve().parents[1]
# `python tools/calibrate_ev.py` でも `tools.*` を一意のmodule名で読む。
# 同じ例外classの二重loadはexcept境界をすり抜けるため、CLI入口で統一する。
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
DEFAULT_VAULT_ROOT = APP_ROOT / "麻雀強者の考え方"
DEFAULT_OUTPUT = APP_ROOT / "calibration" / "data-audit.json"
DEFAULT_DATASET_DIR = APP_ROOT / "calibration" / "dataset"
DEFAULT_MODEL_DIR = APP_ROOT / "calibration" / "model"
DEFAULT_OBSERVED_DATASET_DIR = APP_ROOT / "calibration" / "dataset-observed"
DEFAULT_OBSERVED_MODEL_DIR = APP_ROOT / "calibration" / "model-c1"
DEFAULT_POLICY_DATASET_DIR = APP_ROOT / "calibration" / "dataset-policy"
DEFAULT_POLICY_RESERVATION = APP_ROOT / "calibration" / "future-evaluation-reservation.json"
DEFAULT_POLICY_INPUT_REPORT = APP_ROOT / "calibration" / "policy-input-validation.json"
DEFAULT_POLICY_VERIFICATION = DEFAULT_POLICY_DATASET_DIR / "verification.json"
DEFAULT_POLICY_REPLAY_VERIFICATION = DEFAULT_POLICY_DATASET_DIR / "replay-verification.json"
DEFAULT_SYNTHETIC_DEBUG_OUTPUT = APP_ROOT / "calibration" / "synthetic-debug.json"
DEFAULT_POLICY_RUNTIME_AUDIT = APP_ROOT / "calibration" / "policy-runtime-audit.json"
DEFAULT_OPPONENT_DATASET_DIR = APP_ROOT / "calibration" / "dataset-opponent-v3"
DEFAULT_OPPONENT_FEATURE_DIR = APP_ROOT / "calibration" / "features-opponent-v3"
DEFAULT_OPPONENT_MODEL_DIR = APP_ROOT / "calibration" / "model-opponent-v3"
DEFAULT_OPPONENT_FEATURE_PROBE_DIR = APP_ROOT / "calibration" / "probes" / "opponent-features-v3"

REQUIRED_PAIFU_FIELDS = ("season", "date", "gameId", "roundIndex", "roundName", "paifu")
REQUIRED_DECISION_FIELDS = (
    "season",
    "gameId",
    "roundIndex",
    "logIndex",
    "actionIndex",
    "seat",
    "opponentRiichiCount",
    "openMelds",
    "shantenBeforeDiscard",
    "shantenAfterDiscard",
    "isRiichiDeclaration",
)


def _distribution(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=lambda value: str(value))}


def _key_dict(key: tuple[str, str, int, int]) -> dict[str, Any]:
    return {"season": key[0], "gameId": key[1], "roundIndex": key[2], "logIndex": key[3]}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _vector4(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    converted = [_as_int(item) for item in value]
    if any(item is None for item in converted):
        return None
    return [int(item) for item in converted]


def _round_key(row: dict[str, Any], log_index: int = 0) -> tuple[str, str, int, int] | None:
    season = row.get("season")
    game_id = row.get("gameId")
    round_index = _as_int(row.get("roundIndex"))
    if not isinstance(season, str) or not isinstance(game_id, str) or round_index is None:
        return None
    return season, game_id, round_index, log_index


def _decision_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    fields = ("season", "gameId", "roundIndex", "logIndex", "actionIndex", "seat")
    if any(field not in row for field in fields):
        return None
    return tuple(row[field] for field in fields)


def _jsonl_rows(path: Path, errors: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    errors.append({"file": path.name, "line": line_number, "error": str(exc)})
                    continue
                if not isinstance(value, dict):
                    errors.append({"file": path.name, "line": line_number, "error": "JSON objectではない"})
                    continue
                yield value
    except OSError as exc:
        errors.append({"file": path.name, "line": None, "error": str(exc)})


def _file_manifest(paths: Iterable[Path], root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    total_lines = 0
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest = hashlib.sha256()
        byte_count = 0
        line_count = 0
        last_byte = b""
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                line_count += chunk.count(b"\n")
                last_byte = chunk[-1:]
        if byte_count and last_byte != b"\n":
            line_count += 1
        relative = path.relative_to(root).as_posix()
        file_digest = digest.hexdigest()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_digest.encode("ascii"))
        aggregate.update(b"\n")
        files.append({"path": relative, "bytes": byte_count, "lines": line_count, "sha256": file_digest})
        total_bytes += byte_count
        total_lines += line_count
    return {
        "fileCount": len(files),
        "bytes": total_bytes,
        "lines": total_lines,
        "aggregateSha256": aggregate.hexdigest(),
        "files": files,
    }


def _settlement_delta(result: Any) -> tuple[list[int] | None, int]:
    """和了時の複数精算を含め、全4要素点数ベクトルを合算する。"""
    if not isinstance(result, list) or len(result) < 2:
        return None, 0
    vectors = [vector for item in result[1:] if (vector := _vector4(item)) is not None]
    if not vectors:
        return None, 0
    return [sum(vector[seat] for vector in vectors) for seat in range(4)], len(vectors)


def _riichi_declarations(round_log: list[Any]) -> list[int]:
    declarations: list[int] = []
    for seat in range(4):
        discard_index = 6 + seat * 3
        discards = round_log[discard_index] if len(round_log) > discard_index else []
        if isinstance(discards, list) and any(isinstance(item, str) and item.startswith("r") for item in discards):
            declarations.append(seat)
    return declarations


def _winner_seats(result: Any) -> list[int]:
    if not isinstance(result, list) or not result or result[0] != "和了":
        return []
    winners: set[int] = set()
    for detail in result[2:]:
        if not isinstance(detail, list) or len(detail) < 3:
            continue
        winner = _as_int(detail[0])
        loser = _as_int(detail[1])
        if winner is not None and loser is not None and 0 <= winner < 4 and 0 <= loser < 4:
            winners.add(winner)
    return sorted(winners)


def _audit_paifu(paths: list[Path]) -> tuple[dict[str, Any], dict[tuple[str, str, int, int], dict[str, Any]], list[dict[str, Any]]]:
    parse_errors: list[dict[str, Any]] = []
    missing = Counter()
    log_count_distribution: Counter[int] = Counter()
    aka_distribution: Counter[Any] = Counter()
    result_types: Counter[Any] = Counter()
    settlement_vector_counts: Counter[int] = Counter()
    settlement_sums: Counter[int] = Counter()
    records: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    duplicate_keys: list[dict[str, Any]] = []
    seasons: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "roundLogs": 0, "gameIds": set(), "dates": [], "stages": set()}
    )

    for path in paths:
        for row in _jsonl_rows(path, parse_errors):
            season_name = row.get("season") if isinstance(row.get("season"), str) else path.stem
            season_stats = seasons[season_name]
            season_stats["rows"] += 1
            if isinstance(row.get("gameId"), str):
                season_stats["gameIds"].add(row["gameId"])
            if isinstance(row.get("date"), str):
                season_stats["dates"].append(row["date"])
            if isinstance(row.get("stage"), str):
                season_stats["stages"].add(row["stage"])
            for field in REQUIRED_PAIFU_FIELDS:
                if field not in row:
                    missing[field] += 1

            paifu = row.get("paifu")
            if not isinstance(paifu, dict):
                missing["paifu.object"] += 1
                continue
            rule = paifu.get("rule")
            aka = rule.get("aka") if isinstance(rule, dict) else None
            aka_distribution[aka if aka is not None else "missing"] += 1
            logs = paifu.get("log")
            if not isinstance(logs, list):
                missing["paifu.log"] += 1
                continue
            log_count_distribution[len(logs)] += 1
            season_stats["roundLogs"] += len(logs)

            for log_index, round_log in enumerate(logs):
                key = _round_key(row, log_index)
                if key is None:
                    missing["roundKey"] += 1
                    continue
                if key in records:
                    duplicate_keys.append(_key_dict(key))
                    continue
                if not isinstance(round_log, list) or len(round_log) < 17:
                    missing["paifu.log.roundShape"] += 1
                    records[key] = {"valid": False}
                    continue
                round_info = round_log[0]
                scores = _vector4(round_log[1])
                result = round_log[-1]
                delta, vector_count = _settlement_delta(result)
                if scores is None:
                    missing["scoresBefore"] += 1
                if delta is None:
                    missing["settlementDelta"] += 1
                result_type = result[0] if isinstance(result, list) and result else "missing"
                result_types[result_type] += 1
                settlement_vector_counts[vector_count] += 1
                if delta is not None:
                    settlement_sums[sum(delta)] += 1
                riichi_sticks = None
                honba = None
                if isinstance(round_info, list) and len(round_info) >= 3:
                    honba = _as_int(round_info[1])
                    riichi_sticks = _as_int(round_info[2])
                else:
                    missing["roundInfo"] += 1
                records[key] = {
                    "valid": scores is not None and delta is not None,
                    "scoresBefore": scores,
                    "settlementDelta": delta,
                    "settlementVectorCount": vector_count,
                    "riichiDeclarationSeats": _riichi_declarations(round_log),
                    "winnerSeats": _winner_seats(result),
                    "riichiSticksBefore": riichi_sticks,
                    "honba": honba,
                }

    season_output: dict[str, Any] = {}
    for season, stats in sorted(seasons.items()):
        dates = stats["dates"]
        season_output[season] = {
            "rows": stats["rows"],
            "roundLogs": stats["roundLogs"],
            "matches": len(stats["gameIds"]),
            "dateFrom": min(dates) if dates else None,
            "dateTo": max(dates) if dates else None,
            "stages": sorted(stats["stages"]),
        }

    audit = {
        "seasons": season_output,
        "totals": {
            "rows": sum(item["rows"] for item in season_output.values()),
            "roundLogs": sum(item["roundLogs"] for item in season_output.values()),
            "matches": sum(item["matches"] for item in season_output.values()),
        },
        "requiredRowFields": list(REQUIRED_PAIFU_FIELDS),
        "missingByField": _distribution(missing),
        "malformedJson": parse_errors,
        "duplicateRoundKeys": duplicate_keys,
        "paifuLogCountDistribution": _distribution(log_count_distribution),
        "akaRuleDistribution": _distribution(aka_distribution),
        "resultTypeDistribution": _distribution(result_types),
        "settlementVectorCountDistribution": _distribution(settlement_vector_counts),
        "settlementDeltaSumDistribution": _distribution(settlement_sums),
    }
    return audit, records, parse_errors


def _audit_issues(paths: list[Path], known_rounds: set[tuple[str, str, int, int]]) -> dict[str, Any]:
    parse_errors: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    affected_rounds: set[tuple[str, str, int, int]] = set()
    identifiers: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    seen_events: set[tuple[Any, ...]] = set()
    duplicate_events = 0
    for path in paths:
        for row in _jsonl_rows(path, parse_errors):
            key = _round_key(row, _as_int(row.get("logIndex")) or 0)
            issue = str(row.get("issue", "missing"))
            issue_counts[issue] += 1
            event_key = (
                *(key or (row.get("season"), row.get("gameId"), row.get("roundIndex"), row.get("logIndex", 0))),
                row.get("turnIndex"),
                row.get("seat"),
                issue,
            )
            if event_key in seen_events:
                duplicate_events += 1
            seen_events.add(event_key)
            identifier = {
                "season": row.get("season"),
                "gameId": row.get("gameId"),
                "roundIndex": row.get("roundIndex"),
                "logIndex": row.get("logIndex", 0),
                "turnIndex": row.get("turnIndex"),
                "seat": row.get("seat"),
                "issue": issue,
            }
            identifiers.append(identifier)
            if key is None or key not in known_rounds:
                unmatched.append(identifier)
            else:
                affected_rounds.add(key)
    identifiers.sort(key=lambda item: tuple(str(item.get(field)) for field in ("season", "gameId", "roundIndex", "logIndex", "turnIndex", "seat", "issue")))
    return {
        "policy": "該当する局全体を学習・評価候補から隔離する",
        "issueEvents": len(identifiers),
        "affectedRounds": len(affected_rounds),
        "issueCounts": _distribution(issue_counts),
        "duplicateIssueEvents": duplicate_events,
        "malformedJson": parse_errors,
        "unmatchedIssueIds": unmatched,
        "ids": identifiers,
        "affectedRoundKeys": [_key_dict(key) for key in sorted(affected_rounds)],
    }


def _audit_score_ledger(records: dict[tuple[str, str, int, int], dict[str, Any]]) -> dict[str, Any]:
    conservation_checked = 0
    conservation_violations: list[dict[str, Any]] = []
    riichi_rounds = 0
    riichi_declarations = 0
    valid_rounds = 0
    for key, record in sorted(records.items()):
        if not record.get("valid"):
            continue
        valid_rounds += 1
        declarations = record["riichiDeclarationSeats"]
        if declarations:
            riichi_rounds += 1
            riichi_declarations += len(declarations)
        scores = record["scoresBefore"]
        sticks = record.get("riichiSticksBefore")
        if scores is not None and sticks is not None:
            conservation_checked += 1
            conserved = sum(scores) + sticks * 1000
            if conserved != 100000:
                conservation_violations.append({**_key_dict(key), "scoresPlusKyotaku": conserved})

    by_game: dict[tuple[str, str], list[tuple[tuple[str, str, int, int], dict[str, Any]]]] = defaultdict(list)
    for key, record in records.items():
        by_game[(key[0], key[1])].append((key, record))

    transition_checked = 0
    transition_exact = 0
    raw_delta_exact = 0
    double_deduct_exact = 0
    transition_mismatches: list[dict[str, Any]] = []
    index_gaps: list[dict[str, Any]] = []
    for game_rows in by_game.values():
        game_rows.sort(key=lambda item: (item[0][2], item[0][3]))
        for (current_key, current), (next_key, following) in zip(game_rows, game_rows[1:]):
            if (next_key[2], next_key[3]) <= (current_key[2], current_key[3]):
                continue
            if next_key[2] != current_key[2] + 1:
                index_gaps.append({"from": _key_dict(current_key), "to": _key_dict(next_key)})
            if not current.get("valid") or not following.get("valid"):
                continue
            transition_checked += 1
            raw_expected = [
                current["scoresBefore"][seat]
                + current["settlementDelta"][seat]
                for seat in range(4)
            ]
            winner_riichi_seats = sorted(
                set(current["riichiDeclarationSeats"]).intersection(current.get("winnerSeats", []))
            )
            expected = [
                raw_expected[seat] - (1000 if seat in winner_riichi_seats else 0)
                for seat in range(4)
            ]
            double_deduct_expected = [
                raw_expected[seat] - (1000 if seat in current["riichiDeclarationSeats"] else 0)
                for seat in range(4)
            ]
            actual = following["scoresBefore"]
            if expected == actual:
                transition_exact += 1
            if raw_expected == actual:
                raw_delta_exact += 1
            if double_deduct_expected == actual:
                double_deduct_exact += 1
            if expected != actual:
                transition_mismatches.append(
                    {
                        "from": _key_dict(current_key),
                        "to": _key_dict(next_key),
                        "expectedNextScores": expected,
                        "actualNextScores": actual,
                        "riichiDeclarationSeats": current["riichiDeclarationSeats"],
                        "winnerSeats": current.get("winnerSeats", []),
                        "winnerRiichiDeductionSeats": winner_riichi_seats,
                        "settlementVectorCount": current["settlementVectorCount"],
                    }
                )

    quarantined: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for item in conservation_violations:
        key = (item["season"], item["gameId"], item["roundIndex"], item["logIndex"])
        quarantined[key].add("score_conservation_violation")
    for item in transition_mismatches:
        for side in ("from", "to"):
            value = item[side]
            key = (value["season"], value["gameId"], value["roundIndex"], value["logIndex"])
            quarantined[key].add("between_round_continuity_mismatch")

    return {
        "rewardDefinition": "R_t = 復元した局末の自分の点数 - 判断時点の自分の点数。局末点は局開始点 + result精算差分で得る",
        "settlementRule": "result内の数値4要素ベクトルをすべて合算する。和了者自身が当該局でリーチした場合だけ、その和了者から1000点を控除する",
        "validRoundLedgers": valid_rounds,
        "riichiDeclarationRounds": riichi_rounds,
        "riichiDeclarations": riichi_declarations,
        "scoreConservation": {
            "invariant": "sum(scoresBefore) + 1000 * riichiSticksBefore == 100000",
            "checked": conservation_checked,
            "violations": len(conservation_violations),
            "examples": conservation_violations[:20],
        },
        "betweenRoundContinuity": {
            "formula": "nextScores = scoresBefore + aggregateSettlementDelta - winningPlayerOwnRiichiDeposit",
            "checked": transition_checked,
            "exactMatches": transition_exact,
            "mismatches": len(transition_mismatches),
            "examples": transition_mismatches[:20],
        },
        "riichiDepositHypothesisCheck": {
            "primary": {
                "formula": "和了者自身の当該局リーチだけを1000点控除する",
                "exactMatches": transition_exact,
            },
            "alternatives": {
                "result精算差分をそのまま使う": raw_delta_exact,
                "全リーチ宣言者を1000点ずつ再控除する": double_deduct_exact,
            },
            "conclusion": "牌譜のresult差分は、非和了者のリーチ支出と獲得供託を含むが、和了者自身の当該局リーチ支出は別に控除する",
        },
        "roundIndexGaps": {"count": len(index_gaps), "examples": index_gaps[:20]},
        "quarantinePolicy": "台帳異常の両側にある局を学習・評価候補から隔離する。並べ替えによる推測修復はしない",
        "quarantinedRounds": [
            {**_key_dict(key), "reasons": sorted(reasons)} for key, reasons in sorted(quarantined.items())
        ],
    }


def _audit_eligibility(path: Path) -> tuple[dict[str, Any], set[tuple[str, str, int, int]]]:
    parse_errors: list[dict[str, Any]] = []
    missing = Counter()
    seen_keys: set[tuple[Any, ...]] = set()
    duplicates = 0
    total_rows = 0
    total_rounds: set[tuple[str, str, int, int]] = set()
    eligible_rows = 0
    eligible_rounds: set[tuple[str, str, int, int]] = set()
    by_season: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "rounds": set(), "eligibleRows": 0, "eligibleRounds": set()})

    if not path.exists():
        parse_errors.append({"file": path.name, "line": None, "error": "ファイルが存在しない"})
    else:
        for row in _jsonl_rows(path, parse_errors):
            total_rows += 1
            season = str(row.get("season", "missing"))
            stats = by_season[season]
            stats["rows"] += 1
            for field in REQUIRED_DECISION_FIELDS:
                if field not in row:
                    missing[field] += 1
            decision_key = _decision_key(row)
            if decision_key is not None:
                if decision_key in seen_keys:
                    duplicates += 1
                seen_keys.add(decision_key)
            round_key = _round_key(row, _as_int(row.get("logIndex")) or 0)
            if round_key is not None:
                total_rounds.add(round_key)
                stats["rounds"].add(round_key)

            eligible = (
                row.get("opponentRiichiCount") == 1
                and row.get("openMelds") == 0
                and row.get("shantenBeforeDiscard") == 1
                and row.get("shantenAfterDiscard") in (0, 1)
                and row.get("isRiichiDeclaration") is False
            )
            if eligible:
                eligible_rows += 1
                stats["eligibleRows"] += 1
                if round_key is not None:
                    eligible_rounds.add(round_key)
                    stats["eligibleRounds"].add(round_key)

    seasons = {
        season: {
            "decisionRows": stats["rows"],
            "independentRounds": len(stats["rounds"]),
            "provisionalEligibleDecisionRows": stats["eligibleRows"],
            "provisionalEligibleIndependentRounds": len(stats["eligibleRounds"]),
        }
        for season, stats in sorted(by_season.items())
    }
    return {
        "status": "provisional",
        "reason": "既存派生表には自家リーチ前状態と完全な公開局面がないため、最終適格性は原牌譜のイベント再構成後に確定する",
        "provisionalFilter": {
            "opponentRiichiCount": 1,
            "openMelds": 0,
            "shantenBeforeDiscard": 1,
            "shantenAfterDiscard": [0, 1],
            "isRiichiDeclaration": False,
        },
        "missingForFinalEligibility": ["ownRiichiBefore", "completePublicState", "decisionTimeHand", "decisionTimeRivers"],
        "requiredDecisionFields": list(REQUIRED_DECISION_FIELDS),
        "missingByField": _distribution(missing),
        "malformedJson": parse_errors,
        "duplicateDecisionKeys": duplicates,
        "totals": {
            "decisionRows": total_rows,
            "independentRounds": len(total_rounds),
            "provisionalEligibleDecisionRows": eligible_rows,
            "provisionalEligibleIndependentRounds": len(eligible_rounds),
        },
        "seasons": seasons,
    }, eligible_rounds


def _field_mapping() -> list[dict[str, Any]]:
    return [
        {"field": "roundKey", "source": "paifu row: season, gameId, roundIndex + paifu.log index", "status": "available"},
        {"field": "scoresBefore", "source": "paifu.log[*][1]", "status": "available"},
        {"field": "honba/riichiSticksBefore", "source": "paifu.log[*][0][1:3]", "status": "available"},
        {"field": "roundSettlementDelta", "source": "paifu.log[*][-1]内の全数値4要素ベクトル", "status": "derived"},
        {"field": "riichiDeposits", "source": "各家の捨て牌列にあるr接頭辞", "status": "derived"},
        {"field": "redTileRule", "source": "paifu.rule.aka", "status": "available"},
        {"field": "doraIndicatorsAtRoundStart", "source": "paifu.log[*][2]", "status": "available"},
        {"field": "uraIndicators", "source": "paifu.log[*][3]", "status": "resultOnly", "use": "特徴量へ入れない"},
        {"field": "opponentRiichiCount/openMelds/shanten", "source": "danger/against_riichi_decisions_with_danger.jsonl", "status": "availableDerived"},
        {"field": "ownRiichiBefore", "source": "原牌譜の時系列", "status": "requiresReconstruction"},
        {"field": "decisionTimeHand", "source": "原牌譜の初期手牌・ツモ・打牌・副露", "status": "requiresReconstruction"},
        {"field": "decisionTimeRivers", "source": "原牌譜の各家捨て牌列", "status": "requiresReconstruction"},
        {"field": "furiten/publicState", "source": "判断時点までのイベント", "status": "requiresReconstruction"},
        {"field": "observedRoundReward", "source": "判断時点点数と復元した局末点数", "status": "requiresReconstruction"},
    ]


def _remove_physical(hand: Counter[int], raw: int | None, *, allow_same_kind: bool = False) -> bool:
    if raw is None:
        return False
    if hand[raw] > 0:
        hand[raw] -= 1
        if hand[raw] == 0:
            del hand[raw]
        return True
    if allow_same_kind:
        wanted = tile34(raw)
        for candidate in sorted(hand):
            if hand[candidate] > 0 and tile34(candidate) == wanted:
                hand[candidate] -= 1
                if hand[candidate] == 0:
                    del hand[candidate]
                return True
    return False


def _physical_hand_records(hand: Counter[int]) -> list[dict[str, Any]]:
    values = [tile_record(raw) for raw, count in hand.items() for _ in range(count)]
    return sorted(values, key=lambda item: (item["tile34"], item["isRed"], item["raw"]))


def _win_details(result: Any) -> list[dict[str, int]]:
    if not isinstance(result, list) or not result or result[0] != "和了":
        return []
    details: list[dict[str, int]] = []
    for value in result[2:]:
        if not isinstance(value, list) or len(value) < 3:
            continue
        winner = _as_int(value[0])
        loser = _as_int(value[1])
        if winner is not None and loser is not None and 0 <= winner < 4 and 0 <= loser < 4:
            details.append({"winnerSeat": winner, "loserSeat": loser})
    return details


def _outcome_class(seat: int, result_type: str, details: list[dict[str, int]]) -> str:
    if result_type == "流局":
        return "draw"
    if result_type != "和了" or not details:
        return "unclassified"
    own_wins = [item for item in details if item["winnerSeat"] == seat]
    if own_wins:
        return "self_tsumo" if any(item["loserSeat"] == seat for item in own_wins) else "self_ron"
    own_deal_ins = [item for item in details if item["loserSeat"] == seat and item["winnerSeat"] != seat]
    if own_deal_ins:
        return "self_deal_in_multi" if len(own_deal_ins) > 1 else "self_deal_in"
    ron_details = [item for item in details if item["winnerSeat"] != item["loserSeat"]]
    if len(ron_details) > 1:
        return "other_players_multi_ron"
    if ron_details:
        return "other_players_ron"
    return "opponent_tsumo"


def _round_end_scores(round_log: list[Any]) -> tuple[list[int] | None, list[int] | None, list[dict[str, int]]]:
    start = _vector4(round_log[1]) if len(round_log) > 1 else None
    result = round_log[-1] if round_log else None
    delta, _ = _settlement_delta(result)
    details = _win_details(result)
    if start is None or delta is None:
        return None, delta, details
    declared = set(_riichi_declarations(round_log))
    winners = {item["winnerSeat"] for item in details}
    end = [start[seat] + delta[seat] - (1000 if seat in declared & winners else 0) for seat in range(4)]
    return end, delta, details


def _split_for_season(season: str) -> str:
    for name, values in _split_assignment([season]).items():
        if values:
            return name
    return "unassigned"


def _decision_id(row: dict[str, Any], log_index: int, event_index: int, seat: int) -> str:
    return (
        f"mleague:{row['season']}:{row['gameId']}:{int(row['roundIndex'])}:"
        f"{log_index}:e{event_index}:s{seat}"
    )


def _riichi_legal(
    after_shanten: int,
    score: int,
    remaining_wall: int,
    policy_rules: bool,
) -> bool:
    if not policy_rules:
        return after_shanten == 0 and score >= 1000 and remaining_wall >= 4
    if __package__:
        from .ev_policy_state import RuleProfile
    else:
        from ev_policy_state import RuleProfile
    return RuleProfile().can_declare_riichi(
        closed=True,
        shanten_after_discard=after_shanten,
        after_normal_draw=True,
        live_wall_tiles_after_draw=remaining_wall,
    )


def _build_candidates(
    decision_id: str,
    seat: int,
    hand: Counter[int],
    rivers: list[list[dict[str, Any]]],
    dora_raw: list[int],
    opponent_riichi_seat: int,
    score: int,
    remaining_wall: int,
    actual_raw: int,
    actual_riichi: bool,
    policy_rules: bool = False,
) -> tuple[list[dict[str, Any]], int, Counter[int]]:
    seen = visible_counts(hand, rivers, dora_raw)
    candidates: list[dict[str, Any]] = []
    actual_count = 0
    own_discards = {int(item["tile34"]) for item in rivers[seat] if item.get("tile34") is not None}
    for raw in sorted(hand, key=lambda value: (tile34(value) if tile34(value) is not None else 99, is_red(value), value)):
        if hand[raw] <= 0:
            continue
        after = hand.copy()
        _remove_physical(after, raw)
        counts = counter34(after)
        after_shanten = shanten(counts, 0)
        reception = ukeire(counts, after_shanten, seen)
        index = tile34(raw)
        if index is None:
            continue
        danger = classify_danger(index, opponent_riichi_seat, rivers, seen)
        wait_indices = [item["tile34"] for item in reception["ukeireTiles"]] if after_shanten == 0 else []
        furiten = bool(set(wait_indices).intersection(own_discards | {index})) if after_shanten == 0 else None
        riichi_legal = _riichi_legal(after_shanten, score, remaining_wall, policy_rules)
        for declare_riichi in ([False, True] if riichi_legal else [False]):
            action_id = f"{decision_id}:d{raw}:r{int(declare_riichi)}"
            is_actual = raw == actual_raw and declare_riichi == actual_riichi
            actual_count += int(is_actual)
            candidates.append(
                {
                    "schemaVersion": "ev-calibration-candidate/v1",
                    "decisionId": decision_id,
                    "actionId": action_id,
                    "discardRaw": raw,
                    "discardTile34": index,
                    "discardTile": tile_name(index),
                    "discardsRed": is_red(raw),
                    "copiesInHand": hand[raw],
                    "riichiDeclaration": declare_riichi,
                    "riichiLegal": riichi_legal,
                    "shantenAfterDiscard": after_shanten,
                    **reception,
                    "furitenAfterDiscard": furiten,
                    **danger,
                    "isActual": is_actual,
                    "outcomeObserved": is_actual,
                }
            )
    return candidates, actual_count, seen


def _build_actual_candidate(
    decision_id: str,
    seat: int,
    hand: Counter[int],
    rivers: list[list[dict[str, Any]]],
    dora_raw: list[int],
    opponent_riichi_seat: int,
    score: int,
    remaining_wall: int,
    actual_raw: int,
    actual_riichi: bool,
    policy_rules: bool = False,
) -> tuple[list[dict[str, Any]], int, Counter[int]]:
    """実際の打牌だけを評価し、全候補展開の計算量と出力量を避ける。"""
    seen = visible_counts(hand, rivers, dora_raw)
    if hand.get(actual_raw, 0) <= 0:
        return [], 0, seen
    after = hand.copy()
    if not _remove_physical(after, actual_raw):
        return [], 0, seen
    counts = counter34(after)
    after_shanten = shanten(counts, 0)
    reception = ukeire(counts, after_shanten, seen)
    index = tile34(actual_raw)
    if index is None:
        return [], 0, seen
    danger = classify_danger(index, opponent_riichi_seat, rivers, seen)
    own_discards = {int(item["tile34"]) for item in rivers[seat] if item.get("tile34") is not None}
    wait_indices = [item["tile34"] for item in reception["ukeireTiles"]] if after_shanten == 0 else []
    furiten = bool(set(wait_indices).intersection(own_discards | {index})) if after_shanten == 0 else None
    riichi_legal = _riichi_legal(after_shanten, score, remaining_wall, policy_rules)
    if actual_riichi and not riichi_legal:
        return [], 0, seen
    action_id = f"{decision_id}:d{actual_raw}:r{int(actual_riichi)}"
    return [
        {
            "schemaVersion": "ev-calibration-candidate/v1",
            "decisionId": decision_id,
            "actionId": action_id,
            "discardRaw": actual_raw,
            "discardTile34": index,
            "discardTile": tile_name(index),
            "discardsRed": is_red(actual_raw),
            "copiesInHand": hand[actual_raw],
            "riichiDeclaration": actual_riichi,
            "riichiLegal": riichi_legal,
            "shantenAfterDiscard": after_shanten,
            **reception,
            "furitenAfterDiscard": furiten,
            **danger,
            "isActual": True,
            "outcomeObserved": True,
        }
    ], 1, seen


def extract_round(
    row: dict[str, Any],
    round_log: list[Any],
    log_index: int,
    source_file: str,
    source_line: int,
    decision_unit: str = "primary",
    candidate_projection: str = "all",
    eligibility_mode: str = "observed_action",
) -> dict[str, Any]:
    """1局を時系列復元し、採用判断・候補・結果・除外理由を返す。"""
    if candidate_projection not in {"all", "actual"}:
        raise ValueError(f"未対応のcandidate_projection: {candidate_projection}")
    if eligibility_mode not in {"observed_action", "pre_action"}:
        raise ValueError(f"未対応のeligibility_mode: {eligibility_mode}")
    if eligibility_mode == "pre_action" and candidate_projection != "all":
        raise ValueError("pre_actionでは全候補の抽出が必要")
    include_outcomes = eligibility_mode == "observed_action"
    decisions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    if not isinstance(round_log, list) or len(round_log) < 17:
        return {"decisions": [], "candidates": [], "outcomes": [], "rejections": [], "stats": {"round_log_too_short": 1}}

    names = row.get("paifu", {}).get("name", [])
    rule = row.get("paifu", {}).get("rule", {})
    round_info = round_log[0] if isinstance(round_log[0], list) else []
    dealer = int(round_info[0]) % 4 if round_info else 0
    honba = _as_int(round_info[1]) if len(round_info) > 1 else None
    riichi_sticks_start = _as_int(round_info[2]) if len(round_info) > 2 else None
    start_scores = _vector4(round_log[1])
    end_scores, settlement_delta, result_details = _round_end_scores(round_log)
    result = round_log[-1]
    result_type = str(result[0]) if isinstance(result, list) and result else "unknown"
    first_dora = [round_log[2][0]] if isinstance(round_log[2], list) and round_log[2] else []

    hands: list[Counter[int]] = []
    for seat in range(4):
        initial = round_log[4 + seat * 3]
        hand = Counter(raw for raw in initial if isinstance(raw, int) and tile34(raw) is not None)
        hands.append(hand)
        if sum(hand.values()) != 13:
            stats["initial_hand_invalid"] += 1
    if eligibility_mode == "pre_action" and stats["initial_hand_invalid"]:
        return {
            "decisions": [],
            "candidates": [],
            "outcomes": [],
            "rejections": [
                {
                    "schemaVersion": "ev-calibration-rejection/v1",
                    "level": "round",
                    "source": {
                        "file": source_file,
                        "line": source_line,
                        "season": row.get("season"),
                        "gameId": row.get("gameId"),
                        "roundIndex": row.get("roundIndex"),
                        "logIndex": log_index,
                    },
                    "reasons": ["initial_hand_invalid"],
                }
            ],
            "stats": dict(stats),
        }

    rivers: list[list[dict[str, Any]]] = [[], [], [], []]
    public_melds: list[list[dict[str, Any]]] = [[], [], [], []]
    draw_cursor = [0, 0, 0, 0]
    discard_cursor = [0, 0, 0, 0]
    active_riichi: set[int] = set()
    riichi_event: dict[int, int] = {}
    scores_at_event = start_scores[:] if start_scores is not None else None
    current = dealer
    needs_draw = True
    current_draw: int | None = None
    event_index = 0
    total_draws = 0
    complex_state = False
    chronology_error: str | None = None
    primary_seen: set[int] = set()
    expected_discards = sum(
        1
        for seat in range(4)
        for raw in round_log[6 + seat * 3]
        if call_marker(raw) not in {"a", "k"}
    )
    processed_discards = 0
    guard_limit = sum(len(round_log[5 + seat * 3]) + len(round_log[6 + seat * 3]) for seat in range(4)) * 3 + 20

    for _ in range(guard_limit):
        draw_arr = round_log[5 + current * 3]
        discard_arr = round_log[6 + current * 3]
        if needs_draw:
            if draw_cursor[current] >= len(draw_arr):
                break
            raw_draw = draw_arr[draw_cursor[current]]
            if call_marker(raw_draw):
                chronology_error = "unexpected_call_at_normal_draw"
                break
            if not isinstance(raw_draw, int) or tile34(raw_draw) is None:
                chronology_error = "unparseable_draw"
                break
            hands[current][raw_draw] += 1
            current_draw = raw_draw
            draw_cursor[current] += 1
            total_draws += 1
            event_index += 1
        else:
            current_draw = None

        if discard_cursor[current] >= len(discard_arr):
            break
        discard_index = discard_cursor[current]
        raw_discard = discard_arr[discard_index]
        marker = call_marker(raw_discard)
        if marker in {"a", "k"}:
            complex_state = True
            _, all_tiles, consumed, _ = call_parts(raw_discard)
            for raw in consumed:
                _remove_physical(hands[current], raw, allow_same_kind=True)
            public_melds[current].append(
                {"type": "ankan" if marker == "a" else "kakan", "tiles": [tile_record(raw) for raw in all_tiles], "eventIndex": event_index}
            )
            discard_cursor[current] += 1
            event_index += 1
            needs_draw = True
            continue

        actual_raw = discard_physical_raw(raw_discard, current_draw)
        is_riichi_declaration = isinstance(raw_discard, str) and raw_discard.startswith("r")
        before_counts = counter34(hands[current])
        before_shanten = shanten(before_counts, 0)
        after_hand = hands[current].copy()
        actual_removed = _remove_physical(after_hand, actual_raw)
        after_shanten = shanten(counter34(after_hand), 0) if actual_removed else None
        hand_before_draw = hands[current].copy()
        draw_removed = _remove_physical(hand_before_draw, current_draw) if current_draw is not None else False
        before_draw_shanten = shanten(counter34(hand_before_draw), 0) if draw_removed else None
        opponents = sorted(seat for seat in active_riichi if seat != current)
        decision_event = event_index

        if opponents:
            reasons: list[str] = []
            if current in active_riichi:
                reasons.append("own_riichi_already_active")
            if len(opponents) != 1:
                reasons.append("multiple_opponent_riichi")
            if complex_state:
                reasons.append("prior_call_or_kan")
            if sum(hands[current].values()) != 14 or current_draw is None:
                reasons.append("decision_not_after_normal_draw")
            if before_draw_shanten != 1:
                reasons.append("shanten_before_draw_not_one")
            if eligibility_mode == "observed_action" and after_shanten not in (0, 1):
                reasons.append("actual_shanten_after_out_of_scope")
            if rule.get("aka") != 1:
                reasons.append("red_rule_not_supported")
            metadata_incomplete = len(names) != 4 or start_scores is None
            if include_outcomes:
                metadata_incomplete = metadata_incomplete or end_scores is None or settlement_delta is None
            else:
                metadata_incomplete = (
                    metadata_incomplete
                    or honba is None
                    or riichi_sticks_start is None
                    or not first_dora
                )
            if metadata_incomplete:
                reasons.append("round_metadata_incomplete")
            if (
                not include_outcomes
                and scores_at_event is not None
                and riichi_sticks_start is not None
                and sum(scores_at_event) + 1000 * (riichi_sticks_start + len(active_riichi)) != 100000
            ):
                reasons.append("score_conservation_violation")
            if decision_unit == "primary" and current in primary_seen:
                reasons.append("not_first_eligible_for_seat_round")

            decision_id = _decision_id(row, log_index, decision_event, current)
            remaining_wall = max(0, 70 - total_draws)
            rows_for_decision: list[dict[str, Any]] = []
            actual_candidate_count = 0
            seen: Counter[int] = Counter()
            if not reasons and actual_raw is not None and scores_at_event is not None:
                builder = _build_actual_candidate if candidate_projection == "actual" else _build_candidates
                rows_for_decision, actual_candidate_count, seen = builder(
                    decision_id,
                    current,
                    hands[current],
                    rivers,
                    first_dora,
                    opponents[0],
                    scores_at_event[current],
                    remaining_wall,
                    actual_raw,
                    is_riichi_declaration,
                    eligibility_mode == "pre_action",
                )
                if any(count > 4 for count in seen.values()):
                    reasons.append("visible_tile_count_over_four")
                if actual_candidate_count != 1:
                    reasons.append("actual_action_not_unique_in_candidates")
                if eligibility_mode == "pre_action" and not any(
                    candidate["shantenAfterDiscard"] in (0, 1) for candidate in rows_for_decision
                ):
                    reasons.append("no_in_scope_candidate")

            if reasons:
                stats.update(f"rejected_{reason}" for reason in reasons)
                rejections.append(
                    {
                        "schemaVersion": "ev-calibration-rejection/v1",
                        "source": {
                            "file": source_file,
                            "line": source_line,
                            "season": row.get("season"),
                            "gameId": row.get("gameId"),
                            "roundIndex": row.get("roundIndex"),
                            "logIndex": log_index,
                            "eventIndex": decision_event,
                            "seat": current,
                        },
                        "reasons": reasons,
                    }
                )
            else:
                river_snapshot = copy.deepcopy(rivers)
                meld_snapshot = copy.deepcopy(public_melds)
                active_snapshot = [
                    {"seat": seat, "declarationEventIndex": riichi_event[seat]} for seat in sorted(active_riichi)
                ]
                actual_action = next(item for item in rows_for_decision if item["isActual"])
                is_primary = current not in primary_seen
                primary_seen.add(current)
                decision = {
                    "schemaVersion": "ev-calibration-decision/v1",
                    "decisionId": decision_id,
                    "split": _split_for_season(str(row["season"])),
                    "source": {
                        "dataset": "mleague-tenhou-jsonl",
                        "file": source_file,
                        "line": source_line,
                        "season": row["season"],
                        "gameId": row["gameId"],
                        "roundIndex": int(row["roundIndex"]),
                        "logIndex": log_index,
                        "eventIndex": decision_event,
                        "discardIndex": discard_index,
                    },
                    "date": row.get("date"),
                    "stage": row.get("stage"),
                    "roundName": row.get("roundName"),
                    "seat": current,
                    "actor": names[current],
                    "dealerSeat": dealer,
                    "seatWindIndex": (current - dealer) % 4,
                    "kyoku": round_info[0] if round_info else None,
                    "honba": honba,
                    "isPrimaryWithinSeatRound": is_primary,
                    "ownTurnIndex": discard_index,
                    "wallDrawsSeen": total_draws,
                    "remainingWallTiles": remaining_wall,
                    "scoresAtDecision": scores_at_event[:],
                    "riichiSticksAtDecision": (riichi_sticks_start or 0) + len(active_riichi),
                    "activeRiichi": active_snapshot,
                    "opponentRiichiSeats": opponents,
                    "concealedTilesBeforeDraw": _physical_hand_records(hand_before_draw),
                    "drawnTile": tile_record(current_draw),
                    "handBeforeAction": _physical_hand_records(hands[current]),
                    "publicDoraIndicators": [tile_record(raw) for raw in first_dora],
                    "rivers": river_snapshot,
                    "publicMelds": meld_snapshot,
                    "visibleCounts": [seen[index] for index in range(34)],
                    "shantenBeforeDraw": before_draw_shanten,
                    "shantenBeforeDiscard": before_shanten,
                    "actualDiscardTile34": tile34(actual_raw),
                    "actualShantenAfterDiscard": after_shanten,
                    "actualActionId": actual_action["actionId"],
                }
                decisions.append(decision)
                candidates.extend(rows_for_decision)
                if include_outcomes:
                    outcome_class = _outcome_class(current, result_type, result_details)
                    winners = sorted({item["winnerSeat"] for item in result_details})
                    losers = sorted(
                        {item["loserSeat"] for item in result_details if item["winnerSeat"] != item["loserSeat"]}
                    )
                    last_normal_discard = max(
                        index
                        for index, value in enumerate(discard_arr)
                        if call_marker(value) not in {"a", "k"}
                    )
                    immediate_ron = current in losers and discard_index == last_normal_discard
                    outcome_id = f"{decision_id}:outcome"
                    for candidate in rows_for_decision:
                        candidate["observedOutcomeId"] = outcome_id if candidate["isActual"] else None
                    outcomes.append(
                        {
                            "schemaVersion": "ev-calibration-outcome/v1",
                            "outcomeId": outcome_id,
                            "decisionId": decision_id,
                            "actualActionId": actual_action["actionId"],
                            "resultType": result_type,
                            "resultClass": outcome_class,
                            "winnerSeats": winners,
                            "loserSeats": losers,
                            "immediateRon": immediate_ron,
                            "immediateRonByActiveRiichi": immediate_ron and bool(set(winners).intersection(opponents)),
                            "scoreAtDecision": scores_at_event[current],
                            "scoreAtRoundEnd": end_scores[current],
                            "rewardPoints": end_scores[current] - scores_at_event[current],
                            "aggregateSettlementDelta": settlement_delta[current],
                            "roundEndScores": end_scores,
                        }
                    )
                stats["accepted_decisions"] += 1
                stats["candidate_rows"] += len(rows_for_decision)
                stats["primary_decisions"] += int(is_primary)

        if not actual_removed:
            chronology_error = "discard_tile_missing"
            break
        hands[current] = after_hand
        river_item = {
            **tile_record(int(actual_raw)),
            "isTsumogiri": raw_discard in (60, "r60"),
            "isRiichiDeclaration": is_riichi_declaration,
            "eventIndex": event_index,
            "called": False,
        }
        rivers[current].append(river_item)
        discard_cursor[current] += 1
        processed_discards += 1
        event_index += 1
        if is_riichi_declaration:
            active_riichi.add(current)
            riichi_event[current] = event_index - 1
            if scores_at_event is not None:
                scores_at_event[current] -= 1000

        callers: list[tuple[int, str, Any, list[int], list[int]]] = []
        discarded_index = tile34(actual_raw)
        for seat in range(4):
            if seat == current:
                continue
            candidate_draws = round_log[5 + seat * 3]
            if draw_cursor[seat] >= len(candidate_draws):
                continue
            call_raw = candidate_draws[draw_cursor[seat]]
            call_kind, all_tiles, consumed, called_raw = call_parts(call_raw)
            if (
                call_kind in {"c", "p", "m"}
                and tile34(called_raw) == discarded_index
                and call_source_matches(call_raw, seat, current)
            ):
                callers.append((seat, call_kind, call_raw, all_tiles, consumed))
        if callers:
            callers.sort(key=lambda item: ({"m": 0, "p": 1, "c": 2}[item[1]], (item[0] - current) % 4))
            caller, call_kind, _call_raw, all_tiles, consumed = callers[0]
            complex_state = True
            rivers[current][-1]["called"] = True
            for raw in consumed:
                _remove_physical(hands[caller], raw, allow_same_kind=True)
            public_melds[caller].append(
                {"type": {"c": "chi", "p": "pon", "m": "minkan"}[call_kind], "tiles": [tile_record(raw) for raw in all_tiles], "eventIndex": event_index}
            )
            draw_cursor[caller] += 1
            event_index += 1
            current = caller
            needs_draw = call_kind == "m"
        else:
            current = (current + 1) % 4
            needs_draw = True
    else:
        chronology_error = "guard_limit_reached"

    if processed_discards != expected_discards and chronology_error is None:
        chronology_error = "discard_count_mismatch"
    if chronology_error is not None:
        stats[f"chronology_{chronology_error}"] += 1
        discarded_count = len(decisions) if eligibility_mode == "observed_action" else 0
        stats["discarded_accepted_decisions_due_to_round_chronology"] += discarded_count
        rejections.append(
            {
                "schemaVersion": "ev-calibration-rejection/v1",
                "level": "round" if eligibility_mode == "observed_action" else "round_suffix",
                "source": {
                    "file": source_file,
                    "line": source_line,
                    "season": row.get("season"),
                    "gameId": row.get("gameId"),
                    "roundIndex": row.get("roundIndex"),
                    "logIndex": log_index,
                },
                "reasons": [f"chronology_{chronology_error}"],
            }
        )
        if eligibility_mode == "observed_action":
            decisions = []
            candidates = []
            outcomes = []
    stats["processed_discards"] = processed_discards
    stats["expected_discards"] = expected_discards
    return {
        "decisions": decisions,
        "candidates": candidates,
        "outcomes": outcomes,
        "rejections": rejections,
        "stats": dict(stats),
    }


def _split_assignment(seasons: Iterable[str]) -> dict[str, list[str]]:
    available = set(seasons)
    plan = {
        "train": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
        "selection": ["2023-24"],
        "calibration": ["2024-25"],
        "finalTest": ["2025-26"],
    }
    return {name: [season for season in values if season in available] for name, values in plan.items()}


def build_audit(vault_root: Path) -> dict[str, Any]:
    data_root = vault_root / "data" / "mleague"
    paifu_paths = sorted((data_root / "paifu").glob("*.jsonl"))
    issue_paths = sorted((data_root / "validation" / "hand_reconstruction_issues").glob("*.jsonl"))
    danger_path = data_root / "danger" / "against_riichi_decisions_with_danger.jsonl"

    paifu_audit, records, _ = _audit_paifu(paifu_paths)
    issues = _audit_issues(issue_paths, set(records))
    score_ledger = _audit_score_ledger(records)
    eligibility, eligible_round_keys = _audit_eligibility(danger_path)

    exclusion_reasons: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for item in issues["affectedRoundKeys"]:
        key = (item["season"], item["gameId"], item["roundIndex"], item["logIndex"])
        exclusion_reasons[key].add("hand_reconstruction_issue")
    for item in score_ledger["quarantinedRounds"]:
        key = (item["season"], item["gameId"], item["roundIndex"], item["logIndex"])
        exclusion_reasons[key].update(item["reasons"])
    known_exclusion_keys = set(exclusion_reasons)
    eligible_after_exclusions = eligible_round_keys - known_exclusion_keys
    eligibility["totals"]["knownExcludedIndependentRounds"] = len(eligible_round_keys & known_exclusion_keys)
    eligibility["totals"]["provisionalEligibleAfterKnownExclusions"] = len(eligible_after_exclusions)
    for season, stats in eligibility["seasons"].items():
        stats["knownExcludedIndependentRounds"] = sum(
            1 for key in eligible_round_keys & known_exclusion_keys if key[0] == season
        )
        stats["provisionalEligibleAfterKnownExclusions"] = sum(
            1 for key in eligible_after_exclusions if key[0] == season
        )
    known_exclusions = {
        "policy": "手牌復元または得点台帳に異常がある局は、フェーズB以降の学習・選択・較正・最終テストから除外する",
        "uniqueRounds": len(known_exclusion_keys),
        "rounds": [
            {**_key_dict(key), "reasons": sorted(reasons)}
            for key, reasons in sorted(exclusion_reasons.items())
        ],
    }

    source_paths = paifu_paths + issue_paths + ([danger_path] if danger_path.exists() else [])
    source = {
        "root": vault_root.name,
        "readOnlyContract": True,
        "allInputs": _file_manifest(source_paths, vault_root),
        "paifu": _file_manifest(paifu_paths, vault_root),
        "validationIssues": _file_manifest(issue_paths, vault_root),
        "dangerDecisions": _file_manifest([danger_path], vault_root) if danger_path.exists() else _file_manifest([], vault_root),
    }

    errors: list[str] = []
    warnings: list[str] = []
    if not paifu_paths:
        errors.append("牌譜JSONLが見つからない")
    if paifu_audit["malformedJson"]:
        errors.append("牌譜JSONLに解析不能行がある")
    if paifu_audit["duplicateRoundKeys"]:
        errors.append("牌譜に重複局キーがある")
    if paifu_audit["missingByField"]:
        errors.append("牌譜の必須項目または得点台帳に欠損がある")
    if issues["malformedJson"] or issues["unmatchedIssueIds"] or issues["duplicateIssueEvents"]:
        errors.append("復元例外台帳を牌譜へ一意に対応できない")
    if score_ledger["scoreConservation"]["violations"]:
        warnings.append("開始点と供託の保存則に違反する局は次フェーズで隔離する")
    if score_ledger["betweenRoundContinuity"]["mismatches"]:
        warnings.append("次局開始点と連続しない局間の両側は次フェーズで隔離する")
    if eligibility["malformedJson"]:
        errors.append("対リーチ判断JSONLを完全に解析できない")
    if eligibility["missingByField"]:
        errors.append("暫定適格性の判定項目に欠損がある")
    if eligibility["duplicateDecisionKeys"]:
        errors.append("対リーチ判断キーが重複している")
    if issues["affectedRounds"]:
        warnings.append(f"手牌復元例外{issues['issueEvents']}件を含む{issues['affectedRounds']}局は次フェーズで隔離する")
    warnings.append("適格件数は暫定値。自家リーチ前状態と完全な公開局面は次フェーズで原牌譜から復元する")

    seasons = paifu_audit["seasons"]
    gate_status = "fail" if errors else ("pass_with_exclusions" if known_exclusion_keys else "pass")
    return {
        "schemaVersion": "ev-calibration-data-audit/v1",
        "phase": "A-data-audit",
        "source": source,
        "coverage": paifu_audit,
        "splitAssignment": _split_assignment(seasons),
        "exceptions": issues,
        "knownExclusions": known_exclusions,
        "scoreLedger": score_ledger,
        "eligibility": eligibility,
        "fieldMapping": _field_mapping(),
        "qualityGate": {
            "status": gate_status,
            "errors": errors,
            "warnings": warnings,
            "knownExcludedRoundCount": len(known_exclusion_keys),
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON objectではありません: {path}")
    return value


def _write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _crosscheck_shanten(
    vault_root: Path,
    seasons: Iterable[str],
    targets: dict[tuple[str, str, int, int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    remaining = dict(targets)
    checked = 0
    mismatch_count = 0
    mismatches: list[dict[str, Any]] = []
    parse_error_count = 0
    for season in sorted(seasons):
        path = vault_root / "data" / "mleague" / "shanten" / f"{season}.jsonl"
        if not path.exists():
            continue
        errors: list[dict[str, Any]] = []
        for row in _jsonl_rows(path, errors):
            key = (
                str(row.get("season")),
                str(row.get("gameId")),
                int(row.get("roundIndex", -1)),
                int(row.get("logIndex", 0)),
                int(row.get("seat", -1)),
                int(row.get("discardIndex", -1)),
            )
            expected = remaining.pop(key, None)
            if expected is None:
                continue
            checked += 1
            actual = {
                "shantenBeforeDiscard": row.get("shantenBeforeDiscard"),
                "shantenAfterDiscard": row.get("shantenAfterDiscard"),
                "discardTile34": row.get("discardTile34"),
            }
            if actual != expected:
                mismatch_count += 1
                if len(mismatches) < 20:
                    mismatches.append({"key": list(key), "extracted": expected, "existingDerived": actual})
        parse_error_count += len(errors)
    return {
        "contract": "採用判断のシャンテン前後と実打牌を既存の独立派生行へ照合する。派生行は特徴量には使用しない",
        "targets": len(targets),
        "checked": checked,
        "missing": len(remaining),
        "mismatches": mismatch_count,
        "parseErrors": parse_error_count,
        "mismatchExamples": mismatches,
        "missingExamples": [list(key) for key in sorted(remaining)[:20]],
    }


def extract_dataset(
    vault_root: Path,
    audit_path: Path,
    output_dir: Path,
    selected_seasons: set[str] | None = None,
    decision_unit: str = "primary",
) -> dict[str, Any]:
    """監査済み原牌譜からv1のdecision/candidate/outcomeをストリーム生成する。"""
    stored_audit = _read_json(audit_path)
    live_audit = build_audit(vault_root)
    stored_hash = stored_audit.get("source", {}).get("allInputs", {}).get("aggregateSha256")
    live_hash = live_audit.get("source", {}).get("allInputs", {}).get("aggregateSha256")
    if stored_hash != live_hash:
        raise ValueError("data-audit.jsonと現在のVault入力ハッシュが一致しません。先にauditを再実行してください")
    if live_audit.get("qualityGate", {}).get("status") == "fail":
        raise ValueError("データ監査のqualityGateがfailです")

    exclusion_reasons: dict[tuple[str, str, int, int], list[str]] = {}
    for item in live_audit.get("knownExclusions", {}).get("rounds", []):
        key = (str(item["season"]), str(item["gameId"]), int(item["roundIndex"]), int(item["logIndex"]))
        exclusion_reasons[key] = list(item["reasons"])

    paifu_paths = sorted((vault_root / "data" / "mleague" / "paifu").glob("*.jsonl"))
    if selected_seasons:
        paifu_paths = [path for path in paifu_paths if path.stem in selected_seasons]
    if not paifu_paths:
        raise ValueError("抽出対象の牌譜JSONLがありません")

    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("decisions", "candidates", "outcomes", "rejections")
    final_paths = {name: output_dir / f"{name}.jsonl" for name in names}
    temporary_paths = {name: output_dir / f".{name}.jsonl.tmp" for name in names}
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    totals: Counter[str] = Counter()
    seasons: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    decision_ids: set[str] = set()
    duplicate_decisions = 0
    actual_candidate_errors = 0
    unclassified_outcomes = 0
    adopted_rounds: set[tuple[str, str, int, int]] = set()
    adopted_matches: set[tuple[str, str]] = set()
    shanten_targets: dict[tuple[str, str, int, int, int, int], dict[str, Any]] = {}
    shanten_after_counts: Counter[str] = Counter()

    handles: dict[str, Any] = {}
    try:
        handles = {
            name: path.open("w", encoding="utf-8", newline="\n") for name, path in temporary_paths.items()
        }
        for paifu_path in paifu_paths:
            with paifu_path.open("r", encoding="utf-8") as source:
                for source_line, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    season = str(row.get("season", paifu_path.stem))
                    for log_index, round_log in enumerate(row.get("paifu", {}).get("log", [])):
                        key = (season, str(row.get("gameId")), int(row.get("roundIndex")), log_index)
                        totals["roundsSeen"] += 1
                        seasons[season]["roundsSeen"] += 1
                        if key in exclusion_reasons:
                            totals["knownExcludedRounds"] += 1
                            seasons[season]["knownExcludedRounds"] += 1
                            rejection = {
                                "schemaVersion": "ev-calibration-rejection/v1",
                                "level": "round",
                                "source": {
                                    "file": f"data/mleague/paifu/{paifu_path.name}",
                                    "line": source_line,
                                    "season": season,
                                    "gameId": key[1],
                                    "roundIndex": key[2],
                                    "logIndex": log_index,
                                },
                                "reasons": exclusion_reasons[key],
                            }
                            _write_jsonl_row(handles["rejections"], rejection)
                            for reason in rejection["reasons"]:
                                rejection_reasons[reason] += 1
                            continue

                        bundle = extract_round(
                            row,
                            round_log,
                            log_index,
                            f"data/mleague/paifu/{paifu_path.name}",
                            source_line,
                            decision_unit,
                        )
                        for stat, count in bundle["stats"].items():
                            totals[stat] += int(count)
                            seasons[season][stat] += int(count)
                        if any(name.startswith("chronology_") for name in bundle["stats"]):
                            totals["chronologyExcludedRounds"] += 1
                            seasons[season]["chronologyExcludedRounds"] += 1

                        for rejection in bundle["rejections"]:
                            rejection.setdefault("level", "decision")
                            _write_jsonl_row(handles["rejections"], rejection)
                            if rejection["level"] == "decision":
                                totals["rejectedDecisionRows"] += 1
                                seasons[season]["rejectedDecisionRows"] += 1
                            for reason in rejection["reasons"]:
                                rejection_reasons[reason] += 1
                        for decision in bundle["decisions"]:
                            if decision["decisionId"] in decision_ids:
                                duplicate_decisions += 1
                            decision_ids.add(decision["decisionId"])
                            _write_jsonl_row(handles["decisions"], decision)
                            totals["decisions"] += 1
                            seasons[season]["decisions"] += 1
                            shanten_after_counts[str(decision["actualShantenAfterDiscard"])] += 1
                            adopted_rounds.add(key)
                            adopted_matches.add((season, key[1]))
                            source_key = decision["source"]
                            shanten_targets[
                                (
                                    season,
                                    key[1],
                                    key[2],
                                    log_index,
                                    int(decision["seat"]),
                                    int(source_key["discardIndex"]),
                                )
                            ] = {
                                "shantenBeforeDiscard": decision["shantenBeforeDiscard"],
                                "shantenAfterDiscard": decision["actualShantenAfterDiscard"],
                                "discardTile34": decision["actualDiscardTile34"],
                            }
                        candidate_counts: Counter[str] = Counter()
                        actual_counts: Counter[str] = Counter()
                        for candidate in bundle["candidates"]:
                            candidate_counts[candidate["decisionId"]] += 1
                            actual_counts[candidate["decisionId"]] += int(candidate["isActual"])
                            _write_jsonl_row(handles["candidates"], candidate)
                            totals["candidates"] += 1
                            seasons[season]["candidates"] += 1
                        actual_candidate_errors += sum(
                            count != 1 for decision_id, count in actual_counts.items() if candidate_counts[decision_id]
                        )
                        for outcome in bundle["outcomes"]:
                            unclassified_outcomes += int(outcome["resultClass"] == "unclassified")
                            _write_jsonl_row(handles["outcomes"], outcome)
                            totals["outcomes"] += 1
                            seasons[season]["outcomes"] += 1
        for handle in handles.values():
            handle.close()
        handles = {}
        for name in names:
            temporary_paths[name].replace(final_paths[name])
    finally:
        for handle in handles.values():
            handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    errors: list[str] = []
    if duplicate_decisions:
        errors.append("decisionIdが重複している")
    if totals["decisions"] != totals["outcomes"]:
        errors.append("decisionとoutcomeが1対1ではない")
    if actual_candidate_errors:
        errors.append("実選択候補が一意でないdecisionがある")
    if unclassified_outcomes:
        errors.append("未分類の局末結果がある")
    shanten_crosscheck = _crosscheck_shanten(vault_root, (path.stem for path in paifu_paths), shanten_targets)
    if shanten_crosscheck["missing"]:
        errors.append("採用判断に対応する既存シャンテン派生行がない")
    if shanten_crosscheck["mismatches"]:
        errors.append("シャンテンまたは実打牌が既存派生行と一致しない")
    if shanten_crosscheck["parseErrors"]:
        errors.append("既存シャンテン派生JSONLに解析不能行がある")
    generated_manifest = _file_manifest(final_paths.values(), output_dir)
    summary = {
        "schemaVersion": "ev-calibration-extraction-summary/v1",
        "phase": "B-training-data-extraction",
        "scope": {
            "players": 4,
            "redRule": True,
            "opponentRiichiCount": 1,
            "selfClosedAndNotRiichi": True,
            "shantenBeforeDraw": 1,
            "shantenAfterActualDiscard": [0, 1],
            "priorCallsOrKans": False,
            "unit": (
                "各局各家の最初の適格判断"
                if decision_unit == "primary"
                else "全適格判断。isPrimaryWithinSeatRoundで各局各家の最初の判断を識別する"
            ),
        },
        "source": {
            "auditPath": "calibration/data-audit.json",
            "auditInputAggregateSha256": stored_hash,
            "seasons": [path.stem for path in paifu_paths],
        },
        "records": {
            "decisions": totals["decisions"],
            "candidates": totals["candidates"],
            "outcomes": totals["outcomes"],
            "primaryDecisions": totals["primary_decisions"],
            "adoptedIndependentRounds": len(adopted_rounds),
            "adoptedMatches": len(adopted_matches),
            "rejectedDecisionRows": totals["rejectedDecisionRows"],
            "knownExcludedRounds": totals["knownExcludedRounds"],
            "chronologyExcludedRounds": totals["chronologyExcludedRounds"],
        },
        "rejectionReasons": _distribution(rejection_reasons),
        "shantenAfterDistribution": _distribution(shanten_after_counts),
        "seasons": {season: dict(sorted(stats.items())) for season, stats in sorted(seasons.items())},
        "quality": {
            "status": "fail" if errors else "pass_with_exclusions",
            "errors": errors,
            "duplicateDecisionIds": duplicate_decisions,
            "actualCandidateErrors": actual_candidate_errors,
            "unclassifiedOutcomes": unclassified_outcomes,
            "decisionOutcomeOneToOne": totals["decisions"] == totals["outcomes"],
            "shantenCrosscheck": shanten_crosscheck,
            "futureInformationContract": {
                "decisionAndCandidateFeatures": "判断時点までのイベント、初期ドラ表示牌、公開情報だけを使用",
                "outcomeOnly": "局末result、局末点、和了者・放銃者",
                "forbiddenInFeatures": ["uraDoraIndicators", "futureDraws", "finalWaits", "roundResult"],
            },
        },
        "generatedFiles": generated_manifest,
    }
    write_json(output_dir / "extraction-summary.json", summary)
    return summary


def extract_observed_dataset(
    vault_root: Path,
    audit_path: Path,
    output_dir: Path,
    selected_seasons: set[str] | None = None,
) -> dict[str, Any]:
    """全適格打牌の実選択だけを、1判断1行の軽量データへ投影する。"""
    if __package__:
        from .ev_calibration_model import compact_observed_record
    else:
        from ev_calibration_model import compact_observed_record

    stored_audit = _read_json(audit_path)
    live_audit = build_audit(vault_root)
    stored_hash = stored_audit.get("source", {}).get("allInputs", {}).get("aggregateSha256")
    live_hash = live_audit.get("source", {}).get("allInputs", {}).get("aggregateSha256")
    if stored_hash != live_hash:
        raise ValueError("data-audit.jsonと現在のVault入力ハッシュが一致しません。先にauditを再実行してください")
    if live_audit.get("qualityGate", {}).get("status") == "fail":
        raise ValueError("データ監査のqualityGateがfailです")

    exclusion_reasons: dict[tuple[str, str, int, int], list[str]] = {}
    for item in live_audit.get("knownExclusions", {}).get("rounds", []):
        key = (str(item["season"]), str(item["gameId"]), int(item["roundIndex"]), int(item["logIndex"]))
        exclusion_reasons[key] = list(item["reasons"])

    paifu_paths = sorted((vault_root / "data" / "mleague" / "paifu").glob("*.jsonl"))
    if selected_seasons:
        paifu_paths = [path for path in paifu_paths if path.stem in selected_seasons]
    if not paifu_paths:
        raise ValueError("抽出対象の牌譜JSONLがありません")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "observed": output_dir / "observed-actions.jsonl",
        "excluded": output_dir / "excluded-rounds.jsonl",
    }
    temporary_paths = {name: path.with_name(f".{path.name}.tmp") for name, path in final_paths.items()}
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    totals: Counter[str] = Counter()
    seasons: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    decision_ids: set[str] = set()
    duplicate_decisions = 0
    adopted_rounds: set[tuple[str, str, int, int]] = set()
    adopted_matches: set[tuple[str, str]] = set()
    shanten_targets: dict[tuple[str, str, int, int, int, int], dict[str, Any]] = {}
    split_counts: Counter[str] = Counter()
    shanten_after_counts: Counter[str] = Counter()
    handles: dict[str, Any] = {}
    try:
        handles = {
            name: path.open("w", encoding="utf-8", newline="\n") for name, path in temporary_paths.items()
        }
        for paifu_path in paifu_paths:
            with paifu_path.open("r", encoding="utf-8") as source:
                for source_line, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    season = str(row.get("season", paifu_path.stem))
                    for log_index, round_log in enumerate(row.get("paifu", {}).get("log", [])):
                        key = (season, str(row.get("gameId")), int(row.get("roundIndex")), log_index)
                        totals["roundsSeen"] += 1
                        seasons[season]["roundsSeen"] += 1
                        if key in exclusion_reasons:
                            totals["knownExcludedRounds"] += 1
                            seasons[season]["knownExcludedRounds"] += 1
                            rejection = {
                                "schemaVersion": "ev-calibration-rejection/v1",
                                "level": "round",
                                "source": {
                                    "file": f"data/mleague/paifu/{paifu_path.name}",
                                    "line": source_line,
                                    "season": season,
                                    "gameId": key[1],
                                    "roundIndex": key[2],
                                    "logIndex": log_index,
                                },
                                "reasons": exclusion_reasons[key],
                            }
                            _write_jsonl_row(handles["excluded"], rejection)
                            totals["excludedRoundRows"] += 1
                            for reason in rejection["reasons"]:
                                rejection_reasons[reason] += 1
                            continue

                        bundle = extract_round(
                            row,
                            round_log,
                            log_index,
                            f"data/mleague/paifu/{paifu_path.name}",
                            source_line,
                            decision_unit="all",
                            candidate_projection="actual",
                        )
                        for stat, count in bundle["stats"].items():
                            seasons[season][stat] += int(count)
                        for rejection in bundle["rejections"]:
                            rejection.setdefault("level", "decision")
                            for reason in rejection["reasons"]:
                                rejection_reasons[reason] += 1
                            if rejection["level"] == "decision":
                                totals["rejectedDecisionRows"] += 1
                                seasons[season]["rejectedDecisionRows"] += 1
                            else:
                                totals["chronologyExcludedRounds"] += 1
                                seasons[season]["chronologyExcludedRounds"] += 1
                                totals["excludedRoundRows"] += 1
                                _write_jsonl_row(handles["excluded"], rejection)

                        candidate_by_id = {item["decisionId"]: item for item in bundle["candidates"]}
                        outcome_by_id = {item["decisionId"]: item for item in bundle["outcomes"]}
                        if len(candidate_by_id) != len(bundle["decisions"]) or len(outcome_by_id) != len(bundle["decisions"]):
                            raise ValueError("軽量投影前のdecision/candidate/outcomeが1対1ではない")
                        for decision in bundle["decisions"]:
                            decision_id = decision["decisionId"]
                            if decision_id in decision_ids:
                                duplicate_decisions += 1
                            decision_ids.add(decision_id)
                            candidate = candidate_by_id[decision_id]
                            outcome = outcome_by_id[decision_id]
                            compact = compact_observed_record(decision, candidate, outcome)
                            _write_jsonl_row(handles["observed"], compact)
                            totals["observedActions"] += 1
                            totals["primaryDecisions"] += int(decision["isPrimaryWithinSeatRound"])
                            totals["immediateRonEvents"] += int(outcome["immediateRonByActiveRiichi"])
                            seasons[season]["observedActions"] += 1
                            seasons[season]["primaryDecisions"] += int(decision["isPrimaryWithinSeatRound"])
                            seasons[season]["immediateRonEvents"] += int(outcome["immediateRonByActiveRiichi"])
                            split_counts[decision["split"]] += 1
                            shanten_after_counts[str(decision["actualShantenAfterDiscard"])] += 1
                            adopted_rounds.add(key)
                            adopted_matches.add((season, key[1]))
                            source_key = decision["source"]
                            shanten_targets[
                                (
                                    season,
                                    key[1],
                                    key[2],
                                    log_index,
                                    int(decision["seat"]),
                                    int(source_key["discardIndex"]),
                                )
                            ] = {
                                "shantenBeforeDiscard": decision["shantenBeforeDiscard"],
                                "shantenAfterDiscard": decision["actualShantenAfterDiscard"],
                                "discardTile34": decision["actualDiscardTile34"],
                            }
        for handle in handles.values():
            handle.close()
        handles = {}
        for name in final_paths:
            temporary_paths[name].replace(final_paths[name])
    finally:
        for handle in handles.values():
            handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    errors = []
    if duplicate_decisions:
        errors.append("decisionIdが重複している")
    shanten_crosscheck = _crosscheck_shanten(vault_root, (path.stem for path in paifu_paths), shanten_targets)
    if shanten_crosscheck["missing"] or shanten_crosscheck["mismatches"] or shanten_crosscheck["parseErrors"]:
        errors.append("既存シャンテン派生行との照合に失敗")
    generated_manifest = _file_manifest(final_paths.values(), output_dir)
    summary = {
        "schemaVersion": "ev-calibration-observed-extraction-summary/v1",
        "phase": "C1-all-observed-actions",
        "scope": {
            "unit": "全適格打牌の実選択のみ",
            "candidateProjection": "actual-only",
            "players": 4,
            "redRule": True,
            "opponentRiichiCount": 1,
            "selfClosedAndNotRiichi": True,
            "shantenBeforeDraw": 1,
            "shantenAfterActualDiscard": [0, 1],
            "priorCallsOrKans": False,
        },
        "source": {
            "auditPath": "calibration/data-audit.json",
            "auditInputAggregateSha256": stored_hash,
            "seasons": [path.stem for path in paifu_paths],
        },
        "records": {
            "decisions": totals["observedActions"],
            "observedActions": totals["observedActions"],
            "primaryDecisions": totals["primaryDecisions"],
            "adoptedIndependentRounds": len(adopted_rounds),
            "adoptedMatches": len(adopted_matches),
            "immediateRonEvents": totals["immediateRonEvents"],
            "rejectedDecisionRows": totals["rejectedDecisionRows"],
            "knownExcludedRounds": totals["knownExcludedRounds"],
            "chronologyExcludedRounds": totals["chronologyExcludedRounds"],
            "excludedRoundRows": totals["excludedRoundRows"],
        },
        "splitCounts": _distribution(split_counts),
        "shantenAfterDistribution": _distribution(shanten_after_counts),
        "rejectionReasons": _distribution(rejection_reasons),
        "seasons": {season: dict(sorted(stats.items())) for season, stats in sorted(seasons.items())},
        "quality": {
            "status": "fail" if errors else "pass_with_exclusions",
            "errors": errors,
            "duplicateDecisionIds": duplicate_decisions,
            "observedActionOneRowPerDecision": True,
            "shantenCrosscheck": shanten_crosscheck,
            "futureInformationContract": {
                "features": "判断時点までの情報だけをfeaturesへ格納",
                "label": "直後のリーチ者への放銃だけをlabelへ格納",
                "forbidden": ["rewardPoints", "outcomeClass", "roundEndScores", "futureDraws", "uraDoraIndicators"],
            },
        },
        "generatedFiles": generated_manifest,
    }
    write_json(output_dir / "extraction-summary.json", summary)
    return summary


def validate_policy_inputs(
    vault_root: Path,
    reservation_path: Path,
    output_path: Path,
    selected_seasons: set[str] | None = None,
) -> dict[str, Any]:
    """Dの開発用シーズンだけを台帳から選び、入力ハッシュを保存する。"""
    if __package__:
        from .ev_policy_state import select_development_inputs
    else:
        from ev_policy_state import select_development_inputs
    report = select_development_inputs(vault_root, reservation_path, selected_seasons)
    write_json(output_path, report)
    return report


def extract_policy_dataset(
    vault_root: Path,
    reservation_path: Path,
    output_dir: Path,
    selected_seasons: set[str] | None = None,
) -> dict[str, Any]:
    """実選択や局末結果に依存しない、D用の選択前局面と全行動を抽出する。"""
    if __package__:
        from .ev_policy_state import (
            RULE_PROFILE_ID,
            policy_action_from_candidate,
            policy_decision_record,
            select_development_inputs,
            validate_policy_payload,
        )
    else:
        from ev_policy_state import (
            RULE_PROFILE_ID,
            policy_action_from_candidate,
            policy_decision_record,
            select_development_inputs,
            validate_policy_payload,
        )

    input_report = select_development_inputs(vault_root, reservation_path, selected_seasons)
    selected = set(input_report["selectedSeasons"])
    paifu_paths = sorted(
        path for path in (vault_root / "data" / "mleague" / "paifu").glob("*.jsonl") if path.stem in selected
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "decisions": output_dir / "policy-decisions.jsonl",
        "actions": output_dir / "policy-actions.jsonl",
        "rejections": output_dir / "rejections.jsonl",
    }
    temporary_paths = {name: path.with_name(f".{path.name}.tmp") for name, path in final_paths.items()}
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    totals: Counter[str] = Counter()
    seasons: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    decision_ids: set[str] = set()
    action_ids: set[str] = set()
    primary_rounds: set[tuple[str, str, int, int]] = set()
    matches: set[tuple[str, str]] = set()
    errors: list[str] = []
    handles: dict[str, Any] = {}
    try:
        handles = {name: path.open("w", encoding="utf-8", newline="\n") for name, path in temporary_paths.items()}
        for paifu_path in paifu_paths:
            with paifu_path.open("r", encoding="utf-8") as source:
                for source_line, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    season = str(row.get("season", paifu_path.stem))
                    if season not in selected:
                        raise ValueError(f"ファイル名とレコードのシーズンが台帳選択に一致しない: {season}")
                    for log_index, round_log in enumerate(row.get("paifu", {}).get("log", [])):
                        totals["roundsSeen"] += 1
                        seasons[season]["roundsSeen"] += 1
                        bundle = extract_round(
                            row,
                            round_log,
                            log_index,
                            f"data/mleague/paifu/{paifu_path.name}",
                            source_line,
                            decision_unit="all",
                            candidate_projection="all",
                            eligibility_mode="pre_action",
                        )
                        for stat, count in bundle["stats"].items():
                            seasons[season][stat] += int(count)
                        for rejection in bundle["rejections"]:
                            for reason in rejection["reasons"]:
                                rejection_reasons[reason] += 1
                            _write_jsonl_row(handles["rejections"], rejection)
                            totals["rejections"] += 1

                        candidates_by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
                        for candidate in bundle["candidates"]:
                            candidates_by_decision[candidate["decisionId"]].append(candidate)
                        for decision in bundle["decisions"]:
                            decision_id = decision["decisionId"]
                            if decision_id in decision_ids:
                                errors.append(f"duplicateDecisionId:{decision_id}")
                                continue
                            decision_ids.add(decision_id)
                            policy_decision = policy_decision_record(decision)
                            validate_policy_payload(policy_decision)
                            _write_jsonl_row(handles["decisions"], policy_decision)
                            totals["decisions"] += 1
                            totals["primaryDecisions"] += int(decision["isPrimaryWithinSeatRound"])
                            seasons[season]["decisions"] += 1
                            if decision["isPrimaryWithinSeatRound"]:
                                primary_rounds.add(
                                    (
                                        season,
                                        str(row["gameId"]),
                                        int(row["roundIndex"]),
                                        log_index,
                                    )
                                )
                            matches.add((season, str(row["gameId"])))
                            actions = candidates_by_decision.get(decision_id, [])
                            if not actions or not any(item["shantenAfterDiscard"] in (0, 1) for item in actions):
                                errors.append(f"missingInScopeAction:{decision_id}")
                            for candidate in actions:
                                action = policy_action_from_candidate(candidate)
                                validate_policy_payload(action)
                                action_id = str(action["actionId"])
                                if action_id in action_ids:
                                    errors.append(f"duplicateActionId:{action_id}")
                                action_ids.add(action_id)
                                _write_jsonl_row(handles["actions"], action)
                                totals["actions"] += 1
                                seasons[season]["actions"] += 1
        for handle in handles.values():
            handle.close()
        handles = {}
        for name in final_paths:
            temporary_paths[name].replace(final_paths[name])
    finally:
        for handle in handles.values():
            handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    manifest = _file_manifest(final_paths.values(), output_dir)
    summary = {
        "schemaVersion": "ev-policy-extraction-summary/v1",
        "phase": "D1-policy-input",
        "quality": {"status": "fail" if errors else "pass_with_exclusions", "errors": errors},
        "ruleProfileId": RULE_PROFILE_ID,
        "selectionContract": {
            "eligibility": "ツモ前13枚が1シャンテンで、選択前の14枚に打牌後0/1シャンテンとなる候補がある",
            "actualActionMayLeaveScope": True,
            "futureChronologyFailureKeepsValidPrefix": True,
            "actualActionAndOutcomeExcludedFromPolicyPayload": True,
        },
        "input": input_report,
        "records": {
            "decisions": totals["decisions"],
            "actions": totals["actions"],
            "primaryDecisions": totals["primaryDecisions"],
            "primaryIndependentRounds": len(primary_rounds),
            "matches": len(matches),
            "rejections": totals["rejections"],
        },
        "rejectionReasons": _distribution(rejection_reasons),
        "seasons": {season: dict(sorted(values.items())) for season, values in sorted(seasons.items())},
        "generatedFiles": manifest,
    }
    write_json(output_dir / "extraction-summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="実牌譜EV較正データを監査する")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="入力データ、例外、得点台帳、暫定適格件数を監査する")
    audit.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT, help="麻雀強者の考え方 vault のパス")
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="監査JSONの出力先")
    extract = subparsers.add_parser("extract", help="監査済み原牌譜から学習用JSONLを生成する")
    extract.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT, help="麻雀強者の考え方 vault のパス")
    extract.add_argument("--audit", type=Path, default=DEFAULT_OUTPUT, help="auditで生成したJSON")
    extract.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR, help="学習用JSONLの出力先")
    extract.add_argument("--season", action="append", help="対象シーズン。省略時は全シーズン")
    extract.add_argument(
        "--decision-unit",
        choices=("primary", "all"),
        default="primary",
        help="primaryは各局各家の最初の適格判断、allは全適格判断",
    )
    observed = subparsers.add_parser("extract-observed", help="全適格打牌の実選択だけを軽量JSONLへ抽出する")
    observed.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT, help="麻雀強者の考え方 vault のパス")
    observed.add_argument("--audit", type=Path, default=DEFAULT_OUTPUT, help="auditで生成したJSON")
    observed.add_argument("--output-dir", type=Path, default=DEFAULT_OBSERVED_DATASET_DIR, help="軽量JSONLの出力先")
    observed.add_argument("--season", action="append", help="対象シーズン。省略時は全シーズン")
    validate_policy = subparsers.add_parser(
        "validate-policy-input", help="Dの予約台帳に従い、開発へ使える牌譜だけを検証する"
    )
    validate_policy.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT, help="牌譜vaultのパス")
    validate_policy.add_argument(
        "--reservation", type=Path, default=DEFAULT_POLICY_RESERVATION, help="将来評価予約台帳"
    )
    validate_policy.add_argument(
        "--output", type=Path, default=DEFAULT_POLICY_INPUT_REPORT, help="入力検証レポート"
    )
    validate_policy.add_argument("--season", action="append", help="開発対象シーズン。省略時は台帳の許可期間")
    policy = subparsers.add_parser(
        "extract-policy", help="実打牌と局末結果に依存しないD用選択前局面を抽出する"
    )
    policy.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT, help="牌譜vaultのパス")
    policy.add_argument("--reservation", type=Path, default=DEFAULT_POLICY_RESERVATION, help="将来評価予約台帳")
    policy.add_argument("--output-dir", type=Path, default=DEFAULT_POLICY_DATASET_DIR, help="D用JSONLの出力先")
    policy.add_argument("--season", action="append", help="開発対象シーズン。省略時は台帳の許可期間")
    verify_policy = subparsers.add_parser(
        "verify-policy-dataset", help="D.1データのハッシュ、参照、情報遮断を全件検証する"
    )
    verify_policy.add_argument("--dataset-dir", type=Path, default=DEFAULT_POLICY_DATASET_DIR, help="D.1出力先")
    verify_policy.add_argument("--output", type=Path, default=DEFAULT_POLICY_VERIFICATION, help="検証結果JSON")
    replay_policy = subparsers.add_parser(
        "verify-policy-replay", help="D.1判断を原牌譜の記録済みprefixから再構築して照合する"
    )
    replay_policy.add_argument("--dataset-dir", type=Path, default=DEFAULT_POLICY_DATASET_DIR, help="D.1出力先")
    replay_policy.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT, help="牌譜vaultのパス")
    replay_policy.add_argument("--output", type=Path, default=DEFAULT_POLICY_REPLAY_VERIFICATION, help="照合結果JSON")
    replay_policy.add_argument("--limit", type=int, default=0, help="0は全件、正数は先頭からの確認件数")
    simulate_policy = subparsers.add_parser(
        "simulate-policy-debug", help="D.1の一局面を一様未知牌の合成世界でデバッグ実行する"
    )
    simulate_policy.add_argument("--dataset-dir", type=Path, default=DEFAULT_POLICY_DATASET_DIR, help="D.1出力先")
    simulate_policy.add_argument("--decision-id", required=True, help="評価するD.1 decisionId")
    simulate_policy.add_argument("--trials", type=int, default=16, help="合成世界seedの数")
    simulate_policy.add_argument("--seed", type=int, default=20260907, help="先頭seed")
    simulate_policy.add_argument("--output", type=Path, default=DEFAULT_SYNTHETIC_DEBUG_OUTPUT, help="デバッグ結果JSON")
    runtime_audit = subparsers.add_parser(
        "audit-policy-runtime", help="D.1全入口を独立復元し、実牌譜イベントを局エンジンへ照合する"
    )
    runtime_audit.add_argument("--dataset-dir", type=Path, default=DEFAULT_POLICY_DATASET_DIR, help="D.1出力先")
    runtime_audit.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT, help="牌譜vaultのパス")
    runtime_audit.add_argument(
        "--reservation", type=Path, default=DEFAULT_POLICY_RESERVATION, help="将来評価予約台帳"
    )
    runtime_audit.add_argument("--output", type=Path, default=DEFAULT_POLICY_RUNTIME_AUDIT, help="D.3.0監査JSON")
    runtime_audit.add_argument(
        "--runtime-limit", type=int, default=0, help="局エンジン照合数。0はD.1参照局を全件照合"
    )
    opponent = subparsers.add_parser(
        "extract-opponent-events", help="D.3.1の公開・私有イベント、教師窓、推論prefixを抽出する"
    )
    opponent.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT_ROOT, help="牌譜vaultのパス")
    opponent.add_argument("--reservation", type=Path, default=DEFAULT_POLICY_RESERVATION, help="将来評価予約台帳")
    opponent.add_argument("--d1-dataset-dir", type=Path, default=DEFAULT_POLICY_DATASET_DIR, help="検証済みD.1出力先")
    opponent.add_argument("--output-dir", type=Path, default=DEFAULT_OPPONENT_DATASET_DIR, help="D.3.1出力先")
    opponent.add_argument("--season", action="append", help="開発対象シーズン。省略時は台帳の許可期間")
    opponent.add_argument("--max-rounds", type=int, default=0, help="0は全件、正数はデバッグ用の先頭局数")
    verify_opponent = subparsers.add_parser(
        "verify-opponent-dataset", help="D.3.1データのハッシュ、参照、禁止ラベルを検証する"
    )
    verify_opponent.add_argument("--output-dir", type=Path, default=DEFAULT_OPPONENT_DATASET_DIR, help="D.3.1出力先")
    probe_opponent_features = subparsers.add_parser(
        "probe-opponent-features", help="固定標本でD.3.2a特徴のcold、warm、永続読込性能を測る"
    )
    probe_opponent_features.add_argument("--dataset-dir", type=Path, default=DEFAULT_OPPONENT_DATASET_DIR)
    probe_opponent_features.add_argument("--output-dir", type=Path, default=DEFAULT_OPPONENT_FEATURE_PROBE_DIR)
    probe_opponent_features.add_argument(
        "--max-windows", type=int, default=20_000, help="既定20000。小さい値はデバッグ測定"
    )
    build_opponent_features = subparsers.add_parser(
        "build-opponent-features", help="D.3.1教師窓からD.3.2aの厳密特徴cacheを生成する"
    )
    build_opponent_features.add_argument("--dataset-dir", type=Path, default=DEFAULT_OPPONENT_DATASET_DIR)
    build_opponent_features.add_argument("--output-dir", type=Path, default=DEFAULT_OPPONENT_FEATURE_DIR)
    build_opponent_features.add_argument("--resume", action="store_true", help="検証済みの完了shardから再開する")
    build_opponent_features.add_argument("--max-windows", type=int, default=0, help="各splitのデバッグ上限。0は全件")
    build_opponent_features.add_argument("--shard-windows", type=int, default=10_000, help="1shardの最大窓数")
    verify_opponent_features = subparsers.add_parser(
        "verify-opponent-features", help="D.3.2a特徴cacheのhash、件数、schemaを検証する"
    )
    verify_opponent_features.add_argument("--dataset-dir", type=Path, default=DEFAULT_OPPONENT_DATASET_DIR)
    verify_opponent_features.add_argument("--feature-dir", type=Path, default=DEFAULT_OPPONENT_FEATURE_DIR)
    fit_opponent = subparsers.add_parser(
        "fit-opponent", help="検証済みD.3.2a特徴cacheからv2相手行動モデルを学習する"
    )
    fit_opponent.add_argument("--dataset-dir", type=Path, default=DEFAULT_OPPONENT_DATASET_DIR, help="検証済みD.3.1出力先")
    fit_opponent.add_argument("--feature-dir", type=Path, default=DEFAULT_OPPONENT_FEATURE_DIR, help="検証済みD.3.2b特徴cache")
    fit_opponent.add_argument("--output-dir", type=Path, default=DEFAULT_OPPONENT_MODEL_DIR, help="D.3.2b v3モデル出力先")
    fit_opponent.add_argument(
        "--max-windows", type=int, default=0,
        help="0は全件。正数は各splitの先頭件数だけを使うデバッグ実行",
    )
    evaluate_opponent = subparsers.add_parser(
        "evaluate-opponent", help="保存済み相手行動モデルを固定期間別に再評価する"
    )
    evaluate_opponent.add_argument("--dataset-dir", type=Path, default=DEFAULT_OPPONENT_DATASET_DIR, help="検証済みD.3.1出力先")
    evaluate_opponent.add_argument("--feature-dir", type=Path, default=DEFAULT_OPPONENT_FEATURE_DIR, help="fitに使ったD.3.2b特徴cache")
    evaluate_opponent.add_argument("--model-dir", type=Path, default=DEFAULT_OPPONENT_MODEL_DIR, help="fit-opponentの出力先")
    evaluate_opponent.add_argument(
        "--max-windows", type=int, default=0,
        help="0は全件。正数は各splitの先頭件数だけを使うデバッグ実行",
    )
    fit = subparsers.add_parser("fit", help="固定した開発期間で観測結果モデルを学習・較正する")
    fit.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="フェーズBのJSONL出力先")
    fit.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR, help="モデルと学習summaryの出力先")
    evaluate = subparsers.add_parser("evaluate", help="封印した2025-26で較正モデルを最終評価する")
    evaluate.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="フェーズBのJSONL出力先")
    evaluate.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="fitで生成したモデルの場所")
    evaluate.add_argument("--bootstrap-replicates", type=int, default=2000, help="対局単位ブートストラップ反復数")
    evaluate.add_argument("--seed", type=int, default=20260906, help="ブートストラップ乱数seed")
    evaluate.add_argument(
        "--test-role",
        choices=("reused-confirmatory", "sealed"),
        default="reused-confirmatory",
        help="2025-26は既に開封済みのため既定は確認用。新規の未参照データだけsealedを指定する",
    )
    fit_ron = subparsers.add_parser("fit-ron", help="C.1軽量データで直後放銃モデルだけを学習・較正する")
    fit_ron.add_argument("--dataset-dir", type=Path, default=DEFAULT_OBSERVED_DATASET_DIR, help="extract-observedの出力先")
    fit_ron.add_argument("--output-dir", type=Path, default=DEFAULT_OBSERVED_MODEL_DIR, help="C.1モデルの出力先")
    evaluate_ron = subparsers.add_parser("evaluate-ron", help="C.1直後放銃モデルを2025-26で確認評価する")
    evaluate_ron.add_argument("--dataset-dir", type=Path, default=DEFAULT_OBSERVED_DATASET_DIR, help="extract-observedの出力先")
    evaluate_ron.add_argument("--model-dir", type=Path, default=DEFAULT_OBSERVED_MODEL_DIR, help="fit-ronのモデル出力先")
    evaluate_ron.add_argument("--bootstrap-replicates", type=int, default=2000, help="対局単位ブートストラップ反復数")
    evaluate_ron.add_argument("--seed", type=int, default=20260906, help="ブートストラップ乱数seed")
    evaluate_ron.add_argument(
        "--test-role",
        choices=("reused-confirmatory", "sealed"),
        default="reused-confirmatory",
        help="2025-26は既に開封済みのため既定は確認用",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        report = build_audit(args.vault_root.resolve())
        write_json(args.output.resolve(), report)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "qualityGate": report["qualityGate"]["status"],
                    "rounds": report["coverage"]["totals"]["roundLogs"],
                    "eligibleRounds": report["eligibility"]["totals"]["provisionalEligibleIndependentRounds"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["qualityGate"]["status"] != "fail" else 2
    if args.command == "extract":
        try:
            summary = extract_dataset(
                args.vault_root.resolve(),
                args.audit.resolve(),
                args.output_dir.resolve(),
                set(args.season) if args.season else None,
                args.decision_unit,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "outputDir": str(args.output_dir.resolve()),
                    "quality": summary["quality"]["status"],
                    **summary["records"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if summary["quality"]["status"] != "fail" else 2
    if args.command == "extract-observed":
        try:
            summary = extract_observed_dataset(
                args.vault_root.resolve(),
                args.audit.resolve(),
                args.output_dir.resolve(),
                set(args.season) if args.season else None,
            )
        except (ImportError, OSError, ValueError, KeyError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"outputDir": str(args.output_dir.resolve()), "quality": summary["quality"]["status"], **summary["records"]},
                ensure_ascii=False,
            )
        )
        return 0 if summary["quality"]["status"] != "fail" else 2
    if args.command == "validate-policy-input":
        try:
            report = validate_policy_inputs(
                args.vault_root.resolve(),
                args.reservation.resolve(),
                args.output.resolve(),
                set(args.season) if args.season else None,
            )
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "status": report["status"],
                    "selectedSeasons": report["selectedSeasons"],
                    "files": len(report["files"]),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "extract-policy":
        try:
            summary = extract_policy_dataset(
                args.vault_root.resolve(),
                args.reservation.resolve(),
                args.output_dir.resolve(),
                set(args.season) if args.season else None,
            )
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "outputDir": str(args.output_dir.resolve()),
                    "quality": summary["quality"]["status"],
                    **summary["records"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if summary["quality"]["status"] != "fail" else 2
    if args.command == "verify-policy-dataset":
        try:
            if __package__:
                from .ev_policy_state import verify_policy_dataset
            else:
                from ev_policy_state import verify_policy_dataset
            report = verify_policy_dataset(args.dataset_dir.resolve())
            write_json(args.output.resolve(), report)
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({"output": str(args.output.resolve()), **report}, ensure_ascii=False))
        return 0
    if args.command == "verify-policy-replay":
        if args.limit < 0:
            print(json.dumps({"error": "limitは0以上が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            if __package__:
                from .ev_policy_simulation import verify_policy_replay
            else:
                from ev_policy_simulation import verify_policy_replay
            report = verify_policy_replay(
                args.dataset_dir.resolve(),
                args.vault_root.resolve(),
                limit=None if args.limit == 0 else args.limit,
            )
            write_json(args.output.resolve(), report)
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({"output": str(args.output.resolve()), **report}, ensure_ascii=False))
        return 0 if report["status"] == "pass" else 2
    if args.command == "simulate-policy-debug":
        if args.trials < 1:
            print(json.dumps({"error": "trialsは1以上が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            if __package__:
                from .ev_policy_simulation import evaluate_synthetic_debug, load_decision_bundle
            else:
                from ev_policy_simulation import evaluate_synthetic_debug, load_decision_bundle
            decision, actions = load_decision_bundle(args.dataset_dir.resolve(), args.decision_id)
            report = evaluate_synthetic_debug(
                decision,
                actions,
                seeds=tuple(args.seed + index for index in range(args.trials)),
            )
            write_json(args.output.resolve(), report)
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "scopeStatus": report["scopeStatus"],
                    "candidates": len(report["candidates"]),
                    **report["runtime"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "audit-policy-runtime":
        if args.runtime_limit < 0:
            print(json.dumps({"error": "runtime-limitは0以上が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            if __package__:
                from .ev_policy_replay import audit_policy_runtime
            else:
                from ev_policy_replay import audit_policy_runtime
            report = audit_policy_runtime(
                args.dataset_dir.resolve(),
                args.vault_root.resolve(),
                args.reservation.resolve(),
                runtime_limit=None if args.runtime_limit == 0 else args.runtime_limit,
            )
            write_json(args.output.resolve(), report)
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "status": report["status"],
                    "entryGate": report["entryGate"]["status"],
                    "decisions": report["scope"]["allD1Decisions"],
                    "runtimeRoundLogs": report["roundRuntimeConformance"]["checkedRoundLogs"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["status"] != "fail" else 2
    if args.command == "extract-opponent-events":
        if args.max_rounds < 0:
            print(json.dumps({"error": "max-roundsは0以上が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            from tools.ev_policy_observation import extract_opponent_dataset
            summary = extract_opponent_dataset(
                args.vault_root.resolve(),
                args.reservation.resolve(),
                args.d1_dataset_dir.resolve(),
                args.output_dir.resolve(),
                selected_seasons=set(args.season) if args.season else None,
                maximum_rounds=None if args.max_rounds == 0 else args.max_rounds,
            )
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "outputDir": str(args.output_dir.resolve()),
                    "status": summary["status"],
                    **summary["totals"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "verify-opponent-dataset":
        try:
            from tools.ev_policy_observation import verify_opponent_dataset
            report = verify_opponent_dataset(args.output_dir.resolve())
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({"outputDir": str(args.output_dir.resolve()), **report}, ensure_ascii=False))
        return 0 if report["status"] in {"pass", "debug_pass"} else 2
    if args.command == "probe-opponent-features":
        if not 1 <= args.max_windows <= 20_000:
            print(json.dumps({"error": "max-windowsは1〜20000が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            from tools.ev_policy_opponent import probe_opponent_features
            report = probe_opponent_features(
                args.dataset_dir.resolve(), args.output_dir.resolve(), benchmark_windows=args.max_windows
            )
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({
            "outputDir": str(args.output_dir.resolve()),
            "status": report["status"],
            "coldSeconds": report["cold"]["seconds"],
            "warmSeconds": report["warm"]["seconds"],
            "projectedFullBuildSeconds": report["projection"]["fullBuildSecondsFromColdMean"],
        }, ensure_ascii=False))
        return 0 if report["status"] != "feature_budget_exceeded" else 2
    if args.command == "build-opponent-features":
        if args.max_windows < 0 or not 1 <= args.shard_windows <= 10_000:
            print(json.dumps({"error": "max-windowsは0以上、shard-windowsは1〜10000が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            from tools.ev_policy_opponent import build_opponent_feature_cache
            manifest = build_opponent_feature_cache(
                args.dataset_dir.resolve(),
                args.output_dir.resolve(),
                maximum_windows=None if args.max_windows == 0 else args.max_windows,
                windows_per_shard=args.shard_windows,
                resume=args.resume,
            )
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({
            "outputDir": str(args.output_dir.resolve()),
            "status": manifest["status"],
            "windows": manifest["totalWindows"],
            "candidates": manifest["totalCandidates"],
        }, ensure_ascii=False))
        return 0
    if args.command == "verify-opponent-features":
        try:
            from tools.ev_policy_opponent import verify_opponent_feature_cache
            manifest = verify_opponent_feature_cache(args.dataset_dir.resolve(), args.feature_dir.resolve())
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({
            "featureDir": str(args.feature_dir.resolve()),
            "status": manifest["verification"]["status"],
            "windows": manifest["verification"]["windows"],
            "candidates": manifest["verification"]["candidates"],
        }, ensure_ascii=False))
        return 0
    if args.command == "fit-opponent":
        if args.max_windows < 0:
            print(json.dumps({"error": "max-windowsは0以上が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            from tools.ev_policy_opponent import fit_opponent_model
            summary = fit_opponent_model(
                args.dataset_dir.resolve(),
                args.feature_dir.resolve(),
                args.output_dir.resolve(),
                maximum_windows=None if args.max_windows == 0 else args.max_windows,
            )
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "outputDir": str(args.output_dir.resolve()),
                    "status": summary["status"],
                    "selectedLambda": summary["selectedLambda"],
                    "holds": summary["holds"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "evaluate-opponent":
        if args.max_windows < 0:
            print(json.dumps({"error": "max-windowsは0以上が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            from tools.ev_policy_opponent import evaluate_opponent_model
            evaluation = evaluate_opponent_model(
                args.dataset_dir.resolve(),
                args.feature_dir.resolve(),
                args.model_dir.resolve(),
                maximum_windows=None if args.max_windows == 0 else args.max_windows,
            )
        except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "modelDir": str(args.model_dir.resolve()),
                    "status": evaluation["status"],
                    "eligibleForD33": evaluation["eligibleForD33"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "fit":
        try:
            if __package__:
                from .ev_calibration_model import fit_models
            else:
                from ev_calibration_model import fit_models
            summary = fit_models(args.dataset_dir.resolve(), args.output_dir.resolve())
        except (ImportError, OSError, ValueError, KeyError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(json.dumps({"outputDir": str(args.output_dir.resolve()), **summary["selected"]}, ensure_ascii=False))
        return 0
    if args.command == "evaluate":
        if args.bootstrap_replicates < 100:
            print(json.dumps({"error": "bootstrap-replicatesは100以上が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            if __package__:
                from .ev_calibration_model import evaluate_models
            else:
                from ev_calibration_model import evaluate_models
            evaluation = evaluate_models(
                args.dataset_dir.resolve(),
                args.model_dir.resolve(),
                args.bootstrap_replicates,
                args.seed,
                args.test_role,
            )
        except (ImportError, OSError, ValueError, KeyError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "modelDir": str(args.model_dir.resolve()),
                    "status": evaluation["status"],
                    "finalTest": evaluation["finalTest"],
                    "immediateRon": evaluation["immediateRon"]["comparison"],
                    "reward": evaluation["reward"]["comparison"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "fit-ron":
        try:
            if __package__:
                from .ev_calibration_model import fit_immediate_ron_model
            else:
                from ev_calibration_model import fit_immediate_ron_model
            summary = fit_immediate_ron_model(args.dataset_dir.resolve(), args.output_dir.resolve())
        except (ImportError, OSError, ValueError, KeyError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "outputDir": str(args.output_dir.resolve()),
                    "selectedLambda": summary["selectedLambda"],
                    "selectionMetrics": summary["selectionMetrics"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "evaluate-ron":
        if args.bootstrap_replicates < 100:
            print(json.dumps({"error": "bootstrap-replicatesは100以上が必要"}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            if __package__:
                from .ev_calibration_model import evaluate_immediate_ron_model
            else:
                from ev_calibration_model import evaluate_immediate_ron_model
            evaluation = evaluate_immediate_ron_model(
                args.dataset_dir.resolve(),
                args.model_dir.resolve(),
                args.bootstrap_replicates,
                args.seed,
                args.test_role,
            )
        except (ImportError, OSError, ValueError, KeyError) as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "modelDir": str(args.model_dir.resolve()),
                    "status": evaluation["status"],
                    "confirmation": evaluation["confirmation"],
                    "immediateRon": evaluation["immediateRon"]["comparison"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
