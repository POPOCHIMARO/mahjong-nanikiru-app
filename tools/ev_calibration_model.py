"""フェーズC: 実牌譜から観測確率と局収支を較正する。

外部の学習ライブラリへ依存せず、NumPyだけで固定分割・正則化・
確率較正・対局単位ブートストラップを再現する。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODEL_SCHEMA = "ev-observation-calibration-model/v1"
EVALUATION_SCHEMA = "ev-observation-calibration-evaluation/v1"
RON_MODEL_SCHEMA = "ev-immediate-ron-calibration-model/v1"
RON_EVALUATION_SCHEMA = "ev-immediate-ron-calibration-evaluation/v1"
LAMBDA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
OUTCOME_CLASSES = (
    "draw",
    "opponent_tsumo",
    "other_players_ron",
    "self_deal_in",
    "self_ron",
    "self_tsumo",
)
DANGER_RATES = {
    "genbutsu": 0.0,
    "honor_3_visible": 0.05,
    "honor_2_visible": 0.6,
    "honor_1_visible": 1.6,
    "honor_live": 3.2,
    "suji_19": 2.2,
    "suji_28": 3.1,
    "suji_37": 3.8,
    "suji_456": 4.1,
    "double_suji_middle": 2.0,
    "one_chance_19": 3.0,
    "one_chance_28": 3.0,
    "no_chance_19": 2.2,
    "no_chance_28": 2.2,
    "one_chance": 3.0,
    "no_chance": 2.2,
    "non_suji_19": 3.4,
    # フェーズB/Vault互換名。現行engine.jsの non_suji_19 と同じ外側無スジ。
    "terminal_non_suji": 3.4,
    "non_suji_28": 4.3,
    "non_suji_37": 4.9,
    "non_suji_456": 5.7,
}
NUMERIC_FEATURES = (
    "turn",
    "remaining_wall",
    "shanten_after",
    "ukeire_count",
    "ukeire_kinds",
    "visible_discard_count",
    "danger_level",
    "old_danger_probability",
    "self_dealer",
    "opponent_dealer",
    "score",
    "score_lead",
    "honba",
    "riichi_sticks",
    "discards_red",
    "declares_riichi",
    "dora_count_after",
    "aka_count_after",
    "tsumogiri",
)
CATEGORICAL_FEATURES = ("danger_class", "safety_group", "turn_bucket", "tile_band", "seat_wind")
COMPACT_FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    summary_path = dataset_dir / "extraction-summary.json"
    summary = _json(summary_path)
    errors = []
    aggregate = hashlib.sha256()
    for entry in summary["generatedFiles"]["files"]:
        path = dataset_dir / entry["path"]
        if not path.is_file():
            errors.append(f"missing:{entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            errors.append(f"size:{entry['path']}")
        else:
            digest = _sha256(path)
            if digest != entry["sha256"]:
                errors.append(f"sha256:{entry['path']}")
            aggregate.update(entry["path"].encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\n")
    if aggregate.hexdigest() != summary["generatedFiles"]["aggregateSha256"]:
        errors.append("aggregateSha256")
    if errors:
        raise ValueError("抽出データの整合性検査に失敗: " + ", ".join(errors))
    return summary


def _dora_from_indicator(tile: int) -> int:
    if tile < 27:
        base = (tile // 9) * 9
        return base + ((tile - base + 1) % 9)
    if tile <= 30:
        return 27 + ((tile - 27 + 1) % 4)
    return 31 + ((tile - 31 + 1) % 3)


def _turn_bucket(turn: int) -> str:
    if turn <= 6:
        return "early"
    if turn <= 12:
        return "middle"
    return "late"


def _tile_band(tile: int) -> str:
    if tile >= 27:
        return "honor"
    number = tile % 9 + 1
    if number in (1, 9):
        return "terminal"
    if number in (2, 8):
        return "outer"
    return "middle"


def _old_push_ev(row: dict[str, Any]) -> float:
    turn = row["turn"]
    remain = max(1, 18 - turn)
    shanten_after = int(row["shanten_after"])
    effective_rate = min(0.35, max(0.0, row["ukeire_count"]) / 70.0)
    reach_effective = 1.0 - (1.0 - effective_rate) ** remain
    if shanten_after == 0:
        hand_win = min(0.58, max(0.04, reach_effective * 0.62))
    else:
        win_after_advance = min(0.24, 0.08 + 0.010 * remain)
        hand_win = min(0.34, max(0.02, reach_effective * win_after_advance))
    if row["ukeire_count"] <= 0:
        hand_win = 0.0
    p_now = row["old_danger_probability"]
    p_win = (1.0 - p_now) * hand_win
    p_future = min(0.14, 0.014 * remain) if shanten_after == 0 else min(0.20, 0.022 * remain)
    p_deal = p_now + (1.0 - p_now) * (1.0 - hand_win) * p_future
    p_opp_win = min(0.52, 0.045 * remain)
    deal_loss = 7700 if row["opponent_dealer"] else 5300
    tsumo_pay = 2300 if row["opponent_dealer"] else (2800 if row["self_dealer"] else 1400)
    win_gain = row["old_own_value"] + 1000
    unresolved = max(0.0, 1.0 - p_win - p_deal)
    p_tsumo_push = unresolved * (p_opp_win * 0.85) * 0.4
    return float(round(p_win * win_gain - p_deal * deal_loss - p_tsumo_push * tsumo_pay))


def _make_row(decision: dict[str, Any], candidate: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    seat = int(decision["seat"])
    riichi_seat = int(candidate["riichiSeat"])
    discard = int(candidate["discardTile34"])
    hand = decision["handBeforeAction"]
    indicator = int(decision["publicDoraIndicators"][0]["tile34"])
    dora = _dora_from_indicator(indicator)
    removed = False
    after = []
    for tile in hand:
        if not removed and int(tile["raw"]) == int(candidate["discardRaw"]):
            removed = True
            continue
        after.append(tile)
    if not removed:
        after = []
        for tile in hand:
            if not removed and int(tile["tile34"]) == discard:
                removed = True
                continue
            after.append(tile)
    if not removed or len(after) != len(hand) - 1:
        raise ValueError(f"実打牌を手牌から1枚だけ除去できない: {decision['decisionId']}")
    aka_after = sum(bool(tile["isRed"]) for tile in after)
    dora_after = sum(int(tile["tile34"] == dora) for tile in after) + aka_after
    self_dealer = seat == int(decision["dealerSeat"])
    opponent_dealer = riichi_seat == int(decision["dealerSeat"])
    child_value = min(12000, 4500 + 1500 * dora_after)
    old_own_value = round(child_value * (1.5 if self_dealer else 1.0))
    scores = decision["scoresAtDecision"]
    score = int(scores[seat])
    other_mean = (sum(scores) - score) / 3.0
    old_danger = DANGER_RATES[candidate["dangerClass"]] / 100.0
    turn = int(decision["ownTurnIndex"])
    row = {
        "decision_id": decision["decisionId"],
        "split": decision["split"],
        "season": decision["source"]["season"],
        "match_id": decision["source"]["gameId"],
        "round_key": ":".join(
            str(x)
            for x in (
                decision["source"]["season"],
                decision["source"]["gameId"],
                decision["source"]["roundIndex"],
                decision["source"]["logIndex"],
            )
        ),
        "turn": turn,
        "remaining_wall": int(decision["remainingWallTiles"]),
        "shanten_after": int(candidate["shantenAfterDiscard"]),
        "ukeire_count": int(candidate["ukeireCount"]),
        "ukeire_kinds": int(candidate["ukeireKinds"]),
        "visible_discard_count": int(decision["visibleCounts"][discard]),
        "danger_level": int(candidate["dangerLevel"]),
        "old_danger_probability": old_danger,
        "self_dealer": int(self_dealer),
        "opponent_dealer": int(opponent_dealer),
        "score": score / 10000.0,
        "score_lead": (score - other_mean) / 10000.0,
        "honba": int(decision["honba"]),
        "riichi_sticks": int(decision["riichiSticksAtDecision"]),
        "discards_red": int(bool(candidate["discardsRed"])),
        "declares_riichi": int(bool(candidate["riichiDeclaration"])),
        "dora_count_after": int(dora_after),
        "aka_count_after": int(aka_after),
        "tsumogiri": int(int(decision["drawnTile"]["raw"]) == int(candidate["discardRaw"])),
        "danger_class": candidate["dangerClass"],
        "safety_group": candidate["safetyGroup"],
        "turn_bucket": _turn_bucket(turn),
        "tile_band": _tile_band(discard),
        "seat_wind": str(decision["seatWindIndex"]),
        "immediate_ron": int(bool(outcome["immediateRonByActiveRiichi"])),
        "reward": float(outcome["rewardPoints"]),
        "outcome_class": outcome["resultClass"],
        "old_own_value": float(old_own_value),
    }
    row["old_push_ev"] = _old_push_ev(row)
    return row


def compact_observed_record(
    decision: dict[str, Any], candidate: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    """実選択1件を、学習入力・教師・診断値の境界を保って圧縮する。"""
    row = _make_row(decision, candidate, outcome)
    return {
        "schemaVersion": "ev-calibration-immediate-ron-action/v1",
        "decisionId": row["decision_id"],
        "split": row["split"],
        "source": {
            "season": row["season"],
            "gameId": row["match_id"],
            "roundKey": row["round_key"],
            "roundIndex": int(decision["source"]["roundIndex"]),
            "logIndex": int(decision["source"]["logIndex"]),
            "eventIndex": int(decision["source"]["eventIndex"]),
            "discardIndex": int(decision["source"]["discardIndex"]),
            "seat": int(decision["seat"]),
            "isPrimaryWithinSeatRound": bool(decision["isPrimaryWithinSeatRound"]),
        },
        "features": {name: row[name] for name in COMPACT_FEATURES},
        "label": {"immediateRonByActiveRiichi": bool(row["immediate_ron"])},
    }


def _row_from_compact(record: dict[str, Any]) -> dict[str, Any]:
    if "labels" not in record:
        raise ValueError("直後放銃専用データにはfit-ronを使用してください")
    features = record["features"]
    labels = record["labels"]
    source = record["source"]
    missing = [name for name in COMPACT_FEATURES if name not in features]
    if missing:
        raise ValueError("compact observed-actionの特徴量不足: " + ", ".join(missing))
    return {
        "decision_id": record["decisionId"],
        "split": record["split"],
        "season": source["season"],
        "match_id": source["gameId"],
        "round_key": source["roundKey"],
        **features,
        "immediate_ron": int(bool(labels["immediateRonByActiveRiichi"])),
        "reward": float(labels["rewardPoints"]),
        "outcome_class": labels["outcomeClass"],
        "old_own_value": float(record["diagnostics"]["currentEngineOwnValue"]),
        "old_push_ev": float(record["diagnostics"]["currentEnginePushEv"]),
    }


def _ron_row_from_compact(record: dict[str, Any]) -> dict[str, Any]:
    features = record["features"]
    source = record["source"]
    missing = [name for name in COMPACT_FEATURES if name not in features]
    if missing:
        raise ValueError("compact immediate-ronの特徴量不足: " + ", ".join(missing))
    return {
        "decision_id": record["decisionId"],
        "split": record["split"],
        "season": source["season"],
        "match_id": source["gameId"],
        "round_key": source["roundKey"],
        **features,
        "immediate_ron": int(bool(record["label"]["immediateRonByActiveRiichi"])),
    }


def load_rows(dataset_dir: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    with (dataset_dir / "decisions.jsonl").open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["split"] in allowed_splits:
                decisions[row["decisionId"]] = row
    candidates: dict[str, dict[str, Any]] = {}
    with (dataset_dir / "candidates.jsonl").open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["isActual"] and row["decisionId"] in decisions:
                candidates[row["decisionId"]] = row
    outcomes: dict[str, dict[str, Any]] = {}
    with (dataset_dir / "outcomes.jsonl").open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            if row["decisionId"] in decisions:
                outcomes[row["decisionId"]] = row
    ids = sorted(decisions)
    if set(ids) != set(candidates) or set(ids) != set(outcomes):
        raise ValueError("decision/candidate/outcomeの結合が1対1ではない")
    rows = [_make_row(decisions[key], candidates[key], outcomes[key]) for key in ids]
    if any(row["outcome_class"] not in OUTCOME_CLASSES for row in rows):
        raise ValueError("未対応の結果クラスがある")
    return rows


def load_compact_rows(dataset_dir: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    rows = []
    decision_ids = set()
    with (dataset_dir / "observed-actions.jsonl").open(encoding="utf-8") as stream:
        for raw in stream:
            record = json.loads(raw)
            if record["split"] not in allowed_splits:
                continue
            if record["decisionId"] in decision_ids:
                raise ValueError("compact observed-actionのdecisionIdが重複している")
            decision_ids.add(record["decisionId"])
            row = _row_from_compact(record)
            if row["outcome_class"] not in OUTCOME_CLASSES:
                raise ValueError("未対応の結果クラスがある")
            rows.append(row)
    return sorted(rows, key=lambda row: row["decision_id"])


def load_compact_ron_rows(dataset_dir: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    rows = []
    decision_ids = set()
    with (dataset_dir / "observed-actions.jsonl").open(encoding="utf-8") as stream:
        for raw in stream:
            record = json.loads(raw)
            if record["split"] not in allowed_splits:
                continue
            if record["decisionId"] in decision_ids:
                raise ValueError("compact immediate-ronのdecisionIdが重複している")
            decision_ids.add(record["decisionId"])
            rows.append(_ron_row_from_compact(record))
    return sorted(rows, key=lambda row: row["decision_id"])


def _load_model_rows(dataset_dir: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    if (dataset_dir / "observed-actions.jsonl").is_file():
        return load_compact_rows(dataset_dir, allowed_splits)
    return load_rows(dataset_dir, allowed_splits)


@dataclass
class FeatureSpec:
    means: dict[str, float]
    scales: dict[str, float]
    levels: dict[str, list[str]]

    @classmethod
    def fit(cls, rows: list[dict[str, Any]]) -> "FeatureSpec":
        means = {name: float(np.mean([row[name] for row in rows])) for name in NUMERIC_FEATURES}
        scales = {}
        for name in NUMERIC_FEATURES:
            scale = float(np.std([row[name] for row in rows]))
            scales[name] = scale if scale > 1e-9 else 1.0
        levels = {name: sorted({str(row[name]) for row in rows}) for name in CATEGORICAL_FEATURES}
        return cls(means, scales, levels)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "FeatureSpec":
        return cls(value["means"], value["scales"], value["levels"])

    def to_json(self) -> dict[str, Any]:
        return {"means": self.means, "scales": self.scales, "levels": self.levels, "names": self.names()}

    def names(self) -> list[str]:
        names = ["intercept", *NUMERIC_FEATURES]
        for field in CATEGORICAL_FEATURES:
            names.extend(f"{field}={level}" for level in self.levels[field][1:])
        return names

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        matrix = np.zeros((len(rows), len(self.names())), dtype=float)
        matrix[:, 0] = 1.0
        column = 1
        for name in NUMERIC_FEATURES:
            matrix[:, column] = [(float(row[name]) - self.means[name]) / self.scales[name] for row in rows]
            column += 1
        for field in CATEGORICAL_FEATURES:
            for level in self.levels[field][1:]:
                matrix[:, column] = [float(str(row[field]) == level) for row in rows]
                column += 1
        return matrix


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_binary_logistic(x: np.ndarray, y: np.ndarray, l2: float, max_iter: int = 100) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.ones(x.shape[1], dtype=float) * l2
    penalty[0] = 0.0
    for _ in range(max_iter):
        probability = _sigmoid(x @ beta)
        weight = np.maximum(probability * (1.0 - probability), 1e-7)
        gradient = x.T @ (probability - y) + penalty * beta
        hessian = (x.T * weight) @ x + np.diag(penalty + 1e-8)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return beta


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(x.T @ x + penalty + np.eye(x.shape[1]) * 1e-10, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(x.T @ x + penalty) @ (x.T @ y)


def _binary_logloss(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _multiclass_logloss(y_index: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability[np.arange(len(y_index)), y_index], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _fit_ovr(x: np.ndarray, y_index: np.ndarray, l2: float) -> np.ndarray:
    return np.column_stack(
        [fit_binary_logistic(x, (y_index == index).astype(float), l2) for index in range(len(OUTCOME_CLASSES))]
    )


def _predict_ovr(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    raw = np.clip(_sigmoid(x @ coefficients), 1e-9, 1.0)
    return raw / raw.sum(axis=1, keepdims=True)


def _temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    adjusted = np.clip(probability, 1e-12, 1.0) ** (1.0 / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _fit_conditional_rewards(
    x: np.ndarray, y_index: np.ndarray, reward: np.ndarray, l2: float
) -> tuple[np.ndarray, list[list[float]]]:
    coefficients = []
    bounds = []
    for index in range(len(OUTCOME_CLASSES)):
        mask = y_index == index
        coefficients.append(fit_ridge(x[mask], reward[mask], l2))
        bounds.append([float(np.min(reward[mask])), float(np.max(reward[mask]))])
    return np.column_stack(coefficients), bounds


def _predict_reward(
    x: np.ndarray, class_probability: np.ndarray, coefficients: np.ndarray, bounds: list[list[float]]
) -> np.ndarray:
    conditional = x @ coefficients
    for index, (low, high) in enumerate(bounds):
        conditional[:, index] = np.clip(conditional[:, index], low, high)
    return np.sum(class_probability * conditional, axis=1)


def _metric_dict(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "meanError": float(np.mean(error)),
    }


def _probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(np.mean((probability - y) ** 2)),
        "logLoss": _binary_logloss(y, probability),
        "meanPredicted": float(np.mean(probability)),
        "observedRate": float(np.mean(y)),
        "auc": _auc(y, probability),
    }


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((np.sum(ranks[y == 1]) - positive * (positive + 1) / 2.0) / (positive * negative))


def _coarse_key(row: dict[str, Any], depth: int) -> str:
    parts = [str(row["shanten_after"]), str(row["opponent_dealer"]), row["turn_bucket"]]
    return "|".join(parts[:depth])


def _fit_coarse_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_mean = float(np.mean([row["reward"] for row in rows]))
    levels: dict[str, dict[str, dict[str, float]]] = {}
    parent_mean = global_mean
    for depth in (1, 2, 3):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[_coarse_key(row, depth)].append(row["reward"])
        current = {}
        for key, values in grouped.items():
            if depth == 1:
                fallback = global_mean
            else:
                parent_key = "|".join(key.split("|")[:-1])
                fallback = levels[str(depth - 1)][parent_key]["mean"]
            count = len(values)
            weight = count / (count + 50.0)
            current[key] = {"count": count, "mean": float(weight * np.mean(values) + (1.0 - weight) * fallback)}
        levels[str(depth)] = current
        parent_mean = float(np.mean([entry["mean"] for entry in current.values()]))
    return {"globalMean": global_mean, "shrinkage": 50, "levels": levels}


def _predict_coarse(rows: list[dict[str, Any]], model: dict[str, Any]) -> np.ndarray:
    predictions = []
    for row in rows:
        value = model["globalMean"]
        for depth in (1, 2, 3):
            entry = model["levels"][str(depth)].get(_coarse_key(row, depth))
            if entry and entry["count"] >= 200:
                value = entry["mean"]
        predictions.append(value)
    return np.asarray(predictions, dtype=float)


def _coefficient_rows(names: list[str], coefficients: Iterable[float]) -> list[dict[str, Any]]:
    return [{"feature": name, "coefficient": float(value)} for name, value in zip(names, coefficients)]


def fit_models(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    extraction = verify_dataset(dataset_dir)
    phase = (
        "C1-all-observed-actions-calibration"
        if extraction.get("phase") == "C1-all-observed-actions"
        else "C-observed-outcome-calibration"
    )
    rows = _load_model_rows(dataset_dir, {"train", "selection", "calibration"})
    split_rows = {name: [row for row in rows if row["split"] == name] for name in ("train", "selection", "calibration")}
    expected = {
        season: values.get("decisions", values.get("observedActions", 0))
        for season, values in extraction["seasons"].items()
    }
    fixed_splits = {
        "train": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
        "selection": ["2023-24"],
        "calibration": ["2024-25"],
    }
    for split, seasons in fixed_splits.items():
        if len(split_rows[split]) != sum(expected[season] for season in seasons):
            raise ValueError(f"固定分割の件数がsummaryと一致しない: {split}")

    spec = FeatureSpec.fit(split_rows["train"])
    x_train = spec.transform(split_rows["train"])
    x_selection = spec.transform(split_rows["selection"])
    x_calibration = spec.transform(split_rows["calibration"])
    y_train = np.asarray([row["immediate_ron"] for row in split_rows["train"]], dtype=float)
    y_selection = np.asarray([row["immediate_ron"] for row in split_rows["selection"]], dtype=float)
    y_calibration = np.asarray([row["immediate_ron"] for row in split_rows["calibration"]], dtype=float)

    binary_trials = []
    for l2 in LAMBDA_GRID:
        beta = fit_binary_logistic(x_train, y_train, l2)
        probability = _sigmoid(x_selection @ beta)
        binary_trials.append({"lambda": l2, **_probability_metrics(y_selection, probability)})
    binary_lambda = min(binary_trials, key=lambda row: (row["brier"], row["logLoss"]))["lambda"]
    development_rows = split_rows["train"] + split_rows["selection"]
    x_development = spec.transform(development_rows)
    y_development = np.asarray([row["immediate_ron"] for row in development_rows], dtype=float)
    binary_beta = fit_binary_logistic(x_development, y_development, binary_lambda)
    calibration_base = np.clip(_sigmoid(x_calibration @ binary_beta), 1e-7, 1.0 - 1e-7)
    calibration_logit = np.log(calibration_base / (1.0 - calibration_base))
    platt_x = np.column_stack([np.ones(len(calibration_logit)), calibration_logit])
    platt_beta = fit_binary_logistic(platt_x, y_calibration, 1e-6)

    class_index = {name: index for index, name in enumerate(OUTCOME_CLASSES)}
    outcome_train = np.asarray([class_index[row["outcome_class"]] for row in split_rows["train"]], dtype=int)
    outcome_selection = np.asarray([class_index[row["outcome_class"]] for row in split_rows["selection"]], dtype=int)
    reward_train = np.asarray([row["reward"] for row in split_rows["train"]], dtype=float)
    reward_selection = np.asarray([row["reward"] for row in split_rows["selection"]], dtype=float)
    reward_trials = []
    for l2 in LAMBDA_GRID:
        ovr = _fit_ovr(x_train, outcome_train, l2)
        conditional, bounds = _fit_conditional_rewards(x_train, outcome_train, reward_train, l2)
        probability = _predict_ovr(x_selection, ovr)
        prediction = _predict_reward(x_selection, probability, conditional, bounds)
        reward_trials.append(
            {
                "lambda": l2,
                "outcomeLogLoss": _multiclass_logloss(outcome_selection, probability),
                **_metric_dict(reward_selection, prediction),
            }
        )
    reward_lambda = min(reward_trials, key=lambda row: (row["rmse"], row["outcomeLogLoss"]))["lambda"]
    development_outcome = np.asarray([class_index[row["outcome_class"]] for row in development_rows], dtype=int)
    development_reward = np.asarray([row["reward"] for row in development_rows], dtype=float)
    ovr = _fit_ovr(x_development, development_outcome, reward_lambda)
    conditional, bounds = _fit_conditional_rewards(x_development, development_outcome, development_reward, reward_lambda)
    calibration_outcome = np.asarray([class_index[row["outcome_class"]] for row in split_rows["calibration"]], dtype=int)
    calibration_probability = _predict_ovr(x_calibration, ovr)
    temperatures = np.linspace(0.5, 2.0, 61)
    temperature_trials = [
        {"temperature": float(value), "logLoss": _multiclass_logloss(calibration_outcome, _temperature(calibration_probability, value))}
        for value in temperatures
    ]
    chosen_temperature = min(temperature_trials, key=lambda row: row["logLoss"])["temperature"]

    # 比較基準も新モデルと同じ学習＋方式選択期間だけで推定する。
    # 2024-25は確率較正専用とし、収支ラベルを基準だけへ追加しない。
    coarse = _fit_coarse_baseline(development_rows)
    coarse["trainingSplits"] = ["train", "selection"]
    names = spec.names()
    model = {
        "schemaVersion": MODEL_SCHEMA,
        "phase": phase,
        "dataset": {
            "sourcePhase": extraction.get("phase"),
            "extractionAggregateSha256": extraction["generatedFiles"]["aggregateSha256"],
            "records": extraction["records"],
        },
        "splitLedger": {
            "train": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
            "selection": ["2023-24"],
            "calibration": ["2024-25"],
            "finalTest": ["2025-26"],
            "finalTestUsedDuringFit": False,
            "counts": {name: len(values) for name, values in split_rows.items()},
            "finalTestCountFromExtractionSummaryOnly": expected["2025-26"],
        },
        "featureSpec": spec.to_json(),
        "immediateRon": {
            "target": "immediateRonByActiveRiichi",
            "baseTrainingSplits": ["train", "selection"],
            "plattCalibrationSplit": "calibration",
            "selectedLambda": binary_lambda,
            "selectionTrials": binary_trials,
            "baseCoefficients": _coefficient_rows(names, binary_beta),
            "platt": {"intercept": float(platt_beta[0]), "slope": float(platt_beta[1])},
            "calibrationCount": len(split_rows["calibration"]),
        },
        "reward": {
            "target": "rewardPoints",
            "method": "normalized-one-vs-rest outcome probabilities × class-conditional ridge reward",
            "baseTrainingSplits": ["train", "selection"],
            "probabilityCalibrationSplit": "calibration",
            "classes": list(OUTCOME_CLASSES),
            "selectedLambda": reward_lambda,
            "selectionTrials": reward_trials,
            "ovrCoefficients": {
                name: _coefficient_rows(names, ovr[:, index]) for index, name in enumerate(OUTCOME_CLASSES)
            },
            "conditionalRewardCoefficients": {
                name: _coefficient_rows(names, conditional[:, index]) for index, name in enumerate(OUTCOME_CLASSES)
            },
            "conditionalRewardBounds": {name: bounds[index] for index, name in enumerate(OUTCOME_CLASSES)},
            "temperature": chosen_temperature,
            "temperatureTrials": temperature_trials,
        },
        "coarseRewardBaseline": coarse,
        "currentEngineReference": {
            "dangerRatesPercent": DANGER_RATES,
            "pushEvDefinition": "engine.js evaluatePushFold相当。観測局収支と目的が異なるため診断専用",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "model.json", model)
    fit_summary = {
        "schemaVersion": "ev-observation-calibration-fit-summary/v1",
        "phase": phase,
        "quality": "pass",
        "modelPath": "model.json",
        "modelSha256": _sha256(output_dir / "model.json"),
        "datasetAggregateSha256": extraction["generatedFiles"]["aggregateSha256"],
        "splitLedger": model["splitLedger"],
        "selected": {
            "immediateRonLambda": binary_lambda,
            "rewardLambda": reward_lambda,
            "outcomeTemperature": chosen_temperature,
        },
        "selectionMetrics": {
            "immediateRon": next(row for row in binary_trials if row["lambda"] == binary_lambda),
            "reward": next(row for row in reward_trials if row["lambda"] == reward_lambda),
        },
        "finalTestMetrics": None,
    }
    _write_json(output_dir / "fit-summary.json", fit_summary)
    return fit_summary


def _coefficients(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["coefficient"] for row in rows], dtype=float)


def _cluster_bootstrap_difference(
    truth: np.ndarray,
    new: np.ndarray,
    baseline: np.ndarray,
    match_ids: list[str],
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, match_id in enumerate(match_ids):
        groups[match_id].append(index)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)

    def score(indices: np.ndarray, prediction: np.ndarray) -> float:
        error = prediction[indices] - truth[indices]
        if metric == "brier":
            return float(np.mean(error**2))
        if metric == "rmse":
            return float(np.sqrt(np.mean(error**2)))
        raise ValueError(metric)

    all_indices = np.arange(len(truth))
    observed = score(all_indices, new) - score(all_indices, baseline)
    samples = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.choice(keys, size=len(keys), replace=True)
        indices = np.fromiter((i for key in selected for i in groups[key]), dtype=int)
        samples[replicate] = score(indices, new) - score(indices, baseline)
    low, high = np.quantile(samples, [0.025, 0.975])
    status = "pass" if high < 0 else ("fail" if low > 0 else "inconclusive")
    return {
        "metric": metric,
        "differenceNewMinusBaseline": observed,
        "cluster": "gameId",
        "matches": len(keys),
        "replicates": replicates,
        "seed": seed,
        "ci95": [float(low), float(high)],
        "criterion": "95%上限が0未満",
        "status": status,
    }


def _cluster_mean_error_ci(
    truth: np.ndarray, prediction: np.ndarray, match_ids: list[str], indices: list[int], replicates: int, seed: int
) -> list[float]:
    by_match: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_match[match_ids[index]].append(index)
    keys = sorted(by_match)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.choice(keys, size=len(keys), replace=True)
        sample = np.fromiter((i for key in selected for i in by_match[key]), dtype=int)
        values[replicate] = float(np.mean(prediction[sample] - truth[sample]))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _reliability_rows(y: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    edges = np.asarray([0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.10, 0.20, 1.000001])
    rows = []
    for model, probability in predictions.items():
        bins = np.clip(np.digitize(probability, edges) - 1, 0, len(edges) - 2)
        for index in range(len(edges) - 1):
            mask = bins == index
            if not np.any(mask):
                continue
            rows.append(
                {
                    "model": model,
                    "binLow": float(edges[index]),
                    "binHigh": float(min(edges[index + 1], 1.0)),
                    "count": int(np.sum(mask)),
                    "meanPredicted": float(np.mean(probability[mask])),
                    "observedRate": float(np.mean(y[mask])),
                }
            )
    return rows


def _write_reliability_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_segments_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "dimension", "value", "count", "observedMean", "predictedMean", "meanError",
            "ci95Low", "ci95High", "withinTarget150",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_reliability_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 760, 560
    left, top, plot = 80, 50, 430
    maximum = max(0.06, max(max(row["meanPredicted"], row["observedRate"]) for row in rows) * 1.08)
    maximum = min(0.25, maximum)
    colors = {"calibrated": "#2563eb", "currentDangerTable": "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="80" y="28" font-family="sans-serif" font-size="20" font-weight="700">Immediate ron reliability — 2025-26 confirmation</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        x = left + plot * tick / 5
        y = top + plot - plot * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.3f}</text>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top}" stroke="#9ca3af" stroke-dasharray="5 5"/>')
    for model, color in colors.items():
        points = []
        for row in rows:
            if row["model"] != model:
                continue
            x = left + plot * min(row["meanPredicted"], maximum) / maximum
            y = top + plot - plot * min(row["observedRate"], maximum) / maximum
            radius = min(10.0, 3.0 + math.sqrt(row["count"]) / 8.0)
            points.append((x, y))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.72"><title>n={row["count"]}, predicted={row["meanPredicted"]:.4f}, observed={row["observedRate"]:.4f}</title></circle>')
        if points:
            parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.extend(
        [
            f'<text x="{left+plot/2}" y="{height-25}" text-anchor="middle" font-family="sans-serif" font-size="14">Mean predicted probability</text>',
            f'<text x="20" y="{top+plot/2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 {top+plot/2})">Observed rate</text>',
            '<circle cx="555" cy="90" r="6" fill="#2563eb"/><text x="570" y="95" font-family="sans-serif" font-size="13">Calibrated model</text>',
            '<circle cx="555" cy="120" r="6" fill="#dc2626"/><text x="570" y="125" font-family="sans-serif" font-size="13">Current danger table</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def evaluate_models(
    dataset_dir: Path,
    model_dir: Path,
    replicates: int = 2000,
    seed: int = 20260906,
    test_role: str = "reused-confirmatory",
) -> dict[str, Any]:
    extraction = verify_dataset(dataset_dir)
    model_path = model_dir / "model.json"
    model = _json(model_path)
    if model["schemaVersion"] != MODEL_SCHEMA:
        raise ValueError("未対応のモデルschema")
    if model["dataset"]["extractionAggregateSha256"] != extraction["generatedFiles"]["aggregateSha256"]:
        raise ValueError("モデルと抽出データのハッシュが一致しない")
    rows = _load_model_rows(dataset_dir, {"finalTest"})
    if len(rows) != model["splitLedger"]["finalTestCountFromExtractionSummaryOnly"]:
        raise ValueError("最終テスト件数が固定台帳と一致しない")
    spec = FeatureSpec.from_json(model["featureSpec"])
    x = spec.transform(rows)
    y_ron = np.asarray([row["immediate_ron"] for row in rows], dtype=float)
    reward = np.asarray([row["reward"] for row in rows], dtype=float)
    match_ids = [row["match_id"] for row in rows]

    binary_beta = _coefficients(model["immediateRon"]["baseCoefficients"])
    base_probability = np.clip(_sigmoid(x @ binary_beta), 1e-7, 1.0 - 1e-7)
    logit = np.log(base_probability / (1.0 - base_probability))
    platt = model["immediateRon"]["platt"]
    calibrated_probability = _sigmoid(platt["intercept"] + platt["slope"] * logit)
    current_probability = np.asarray([row["old_danger_probability"] for row in rows], dtype=float)

    ovr = np.column_stack([_coefficients(model["reward"]["ovrCoefficients"][name]) for name in OUTCOME_CLASSES])
    conditional = np.column_stack(
        [_coefficients(model["reward"]["conditionalRewardCoefficients"][name]) for name in OUTCOME_CLASSES]
    )
    bounds = [model["reward"]["conditionalRewardBounds"][name] for name in OUTCOME_CLASSES]
    class_probability = _temperature(_predict_ovr(x, ovr), model["reward"]["temperature"])
    reward_prediction = _predict_reward(x, class_probability, conditional, bounds)
    coarse_prediction = _predict_coarse(rows, model["coarseRewardBaseline"])
    old_ev = np.asarray([row["old_push_ev"] for row in rows], dtype=float)

    immediate_comparison = _cluster_bootstrap_difference(
        y_ron, calibrated_probability, current_probability, match_ids, "brier", replicates, seed
    )
    reward_comparison = _cluster_bootstrap_difference(
        reward, reward_prediction, coarse_prediction, match_ids, "rmse", replicates, seed + 1
    )
    reliability = _reliability_rows(
        y_ron, {"calibrated": calibrated_probability, "currentDangerTable": current_probability}
    )

    segment_rows = []
    segment_specs = {
        "shantenAfter": lambda row: str(row["shanten_after"]),
        "opponentDealer": lambda row: str(row["opponent_dealer"]),
        "turnBucket": lambda row: row["turn_bucket"],
        "safetyGroup": lambda row: row["safety_group"],
    }
    segment_counter = 0
    for dimension, getter in segment_specs.items():
        values = sorted({getter(row) for row in rows})
        for value in values:
            indices = [index for index, row in enumerate(rows) if getter(row) == value]
            if len(indices) < 200:
                continue
            idx = np.asarray(indices, dtype=int)
            ci = _cluster_mean_error_ci(reward, reward_prediction, match_ids, indices, replicates, seed + 100 + segment_counter)
            segment_counter += 1
            segment_rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "count": len(indices),
                    "observedMean": float(np.mean(reward[idx])),
                    "predictedMean": float(np.mean(reward_prediction[idx])),
                    "meanError": float(np.mean(reward_prediction[idx] - reward[idx])),
                    "ci95Low": ci[0],
                    "ci95High": ci[1],
                    "withinTarget150": ci[0] >= -150 and ci[1] <= 150,
                }
            )

    class_index = {name: index for index, name in enumerate(OUTCOME_CLASSES)}
    observed_class = np.asarray([class_index[row["outcome_class"]] for row in rows], dtype=int)
    system_target = all(row["withinTarget150"] for row in segment_rows)
    statistical_pass = immediate_comparison["status"] == reward_comparison["status"] == "pass" and system_target
    overall = "pass" if statistical_pass and test_role == "sealed" else "hold"
    evaluation = {
        "schemaVersion": EVALUATION_SCHEMA,
        "phase": model["phase"],
        "status": overall,
        "testIndependence": {
            "role": test_role,
            "eligibleForNewAdoptionDecision": test_role == "sealed",
            "reason": (
                "この評価より前に結果を参照していない"
                if test_role == "sealed"
                else "2025-26は過去のフェーズC評価で開封済み。確認用であり新しい封印テストではない"
            ),
        },
        "interpretation": "観測行動後の直後放銃と局収支の評価。未選択行動や継続方策のEVではない",
        "modelSha256": _sha256(model_path),
        "datasetAggregateSha256": extraction["generatedFiles"]["aggregateSha256"],
        "finalTest": {
            "season": "2025-26",
            "decisions": len(rows),
            "rounds": len({row["round_key"] for row in rows}),
            "matches": len(set(match_ids)),
            "immediateRonEvents": int(np.sum(y_ron)),
            "outcomes": dict(sorted(Counter(row["outcome_class"] for row in rows).items())),
            "shantenAfterDistribution": dict(
                sorted(Counter(str(row["shanten_after"]) for row in rows).items())
            ),
        },
        "scopeWarnings": [
            "最初の適格判断を使った最終テストは全件が打牌後1シャンテン。テンパイ打牌へ一般化しない"
        ] if {row["shanten_after"] for row in rows} == {1} else [],
        "immediateRon": {
            "calibrated": _probability_metrics(y_ron, calibrated_probability),
            "currentDangerTable": _probability_metrics(y_ron, current_probability),
            "comparison": immediate_comparison,
            "smallGroupRule": "独立表示には200判断かつ20イベント以上が必要。満たさない群は全体モデルへ縮約",
        },
        "reward": {
            "calibratedOutcomeModel": _metric_dict(reward, reward_prediction),
            "coarseShrunkMeanBaseline": _metric_dict(reward, coarse_prediction),
            "comparison": reward_comparison,
            "outcomeLogLoss": _multiclass_logloss(observed_class, class_probability),
            "systematicErrorTarget": {
                "criterion": "標本200件以上の主層で平均誤差95%区間が±150点内",
                "segmentsChecked": len(segment_rows),
                "allWithinTarget": system_target,
            },
        },
        "currentDisplayedEvDiagnostic": {
            **_metric_dict(reward, old_ev),
            "meanDisplayedPushEv": float(np.mean(old_ev)),
            "meanObservedReward": float(np.mean(reward)),
            "adoptionCriterion": False,
            "reason": "現行push EVと観測局収支は継続方策・点数調整の定義が異なる",
        },
        "artifacts": {
            "reliabilityCsv": "reliability.csv",
            "reliabilitySvg": "reliability.svg",
            "segmentErrorsCsv": "segment-errors.csv",
        },
    }
    _write_json(model_dir / "evaluation.json", evaluation)
    _write_reliability_csv(model_dir / "reliability.csv", reliability)
    _write_segments_csv(model_dir / "segment-errors.csv", segment_rows)
    _write_reliability_svg(model_dir / "reliability.svg", reliability)
    return evaluation


def fit_immediate_ron_model(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """全適格打牌を使い、直後放銃だけを学習するC.1専用入口。"""
    extraction = verify_dataset(dataset_dir)
    if extraction.get("phase") != "C1-all-observed-actions":
        raise ValueError("fit-ronにはextract-observedのC.1データが必要")
    rows = load_compact_ron_rows(dataset_dir, {"train", "selection", "calibration"})
    split_rows = {name: [row for row in rows if row["split"] == name] for name in ("train", "selection", "calibration")}
    for split, expected in extraction["splitCounts"].items():
        if split != "finalTest" and len(split_rows[split]) != expected:
            raise ValueError(f"固定分割の件数がsummaryと一致しない: {split}")

    spec = FeatureSpec.fit(split_rows["train"])
    x_train = spec.transform(split_rows["train"])
    x_selection = spec.transform(split_rows["selection"])
    x_calibration = spec.transform(split_rows["calibration"])
    y_train = np.asarray([row["immediate_ron"] for row in split_rows["train"]], dtype=float)
    y_selection = np.asarray([row["immediate_ron"] for row in split_rows["selection"]], dtype=float)
    y_calibration = np.asarray([row["immediate_ron"] for row in split_rows["calibration"]], dtype=float)

    trials = []
    for l2 in LAMBDA_GRID:
        beta = fit_binary_logistic(x_train, y_train, l2)
        probability = _sigmoid(x_selection @ beta)
        trials.append({"lambda": l2, **_probability_metrics(y_selection, probability)})
    selected_lambda = min(trials, key=lambda row: (row["brier"], row["logLoss"]))["lambda"]
    development = split_rows["train"] + split_rows["selection"]
    x_development = spec.transform(development)
    y_development = np.asarray([row["immediate_ron"] for row in development], dtype=float)
    beta = fit_binary_logistic(x_development, y_development, selected_lambda)
    calibration_base = np.clip(_sigmoid(x_calibration @ beta), 1e-7, 1.0 - 1e-7)
    calibration_logit = np.log(calibration_base / (1.0 - calibration_base))
    platt_x = np.column_stack([np.ones(len(calibration_logit)), calibration_logit])
    platt_beta = fit_binary_logistic(platt_x, y_calibration, 1e-6)

    model = {
        "schemaVersion": RON_MODEL_SCHEMA,
        "phase": "C1-all-observed-actions-immediate-ron",
        "target": "immediateRonByActiveRiichi",
        "dataset": {
            "sourcePhase": extraction["phase"],
            "extractionAggregateSha256": extraction["generatedFiles"]["aggregateSha256"],
            "records": extraction["records"],
        },
        "splitLedger": {
            "train": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
            "selection": ["2023-24"],
            "calibration": ["2024-25"],
            "confirmation": ["2025-26"],
            "confirmationUsedDuringFit": False,
            "counts": {name: len(values) for name, values in split_rows.items()},
            "confirmationCountFromExtractionSummaryOnly": extraction["splitCounts"]["finalTest"],
        },
        "featureSpec": spec.to_json(),
        "selectedLambda": selected_lambda,
        "selectionTrials": trials,
        "baseTrainingSplits": ["train", "selection"],
        "baseCoefficients": _coefficient_rows(spec.names(), beta),
        "plattCalibrationSplit": "calibration",
        "platt": {"intercept": float(platt_beta[0]), "slope": float(platt_beta[1])},
        "currentEngineReference": {"dangerRatesPercent": DANGER_RATES},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("segment-errors.csv",):
        (output_dir / stale).unlink(missing_ok=True)
    _write_json(output_dir / "model.json", model)
    summary = {
        "schemaVersion": "ev-immediate-ron-calibration-fit-summary/v1",
        "phase": model["phase"],
        "quality": "pass",
        "modelPath": "model.json",
        "modelSha256": _sha256(output_dir / "model.json"),
        "datasetAggregateSha256": extraction["generatedFiles"]["aggregateSha256"],
        "splitLedger": model["splitLedger"],
        "selectedLambda": selected_lambda,
        "selectionMetrics": next(row for row in trials if row["lambda"] == selected_lambda),
        "confirmationMetrics": None,
    }
    _write_json(output_dir / "fit-summary.json", summary)
    return summary


def evaluate_immediate_ron_model(
    dataset_dir: Path,
    model_dir: Path,
    replicates: int = 2000,
    seed: int = 20260906,
    test_role: str = "reused-confirmatory",
) -> dict[str, Any]:
    extraction = verify_dataset(dataset_dir)
    model_path = model_dir / "model.json"
    model = _json(model_path)
    if model.get("schemaVersion") != RON_MODEL_SCHEMA:
        raise ValueError("evaluate-ronにはfit-ronで生成したモデルが必要")
    if model["dataset"]["extractionAggregateSha256"] != extraction["generatedFiles"]["aggregateSha256"]:
        raise ValueError("モデルと抽出データのハッシュが一致しない")
    rows = load_compact_ron_rows(dataset_dir, {"finalTest"})
    if len(rows) != model["splitLedger"]["confirmationCountFromExtractionSummaryOnly"]:
        raise ValueError("確認評価件数が固定台帳と一致しない")
    spec = FeatureSpec.from_json(model["featureSpec"])
    x = spec.transform(rows)
    truth = np.asarray([row["immediate_ron"] for row in rows], dtype=float)
    base = np.clip(_sigmoid(x @ _coefficients(model["baseCoefficients"])), 1e-7, 1.0 - 1e-7)
    logit = np.log(base / (1.0 - base))
    calibrated = _sigmoid(model["platt"]["intercept"] + model["platt"]["slope"] * logit)
    current = np.asarray([row["old_danger_probability"] for row in rows], dtype=float)
    match_ids = [row["match_id"] for row in rows]
    comparison = _cluster_bootstrap_difference(truth, calibrated, current, match_ids, "brier", replicates, seed)
    reliability = _reliability_rows(truth, {"calibrated": calibrated, "currentDangerTable": current})
    statistical_pass = comparison["status"] == "pass"
    status = "pass" if statistical_pass and test_role == "sealed" else "hold"
    evaluation = {
        "schemaVersion": RON_EVALUATION_SCHEMA,
        "phase": model["phase"],
        "status": status,
        "testIndependence": {
            "role": test_role,
            "eligibleForNewAdoptionDecision": test_role == "sealed",
            "reason": (
                "この評価より前に結果を参照していない"
                if test_role == "sealed"
                else "2025-26は過去のフェーズC評価で開封済み。確認用であり新しい封印テストではない"
            ),
        },
        "interpretation": "全適格実打牌の直後放銃だけを評価。局収支と未選択行動のEVは対象外",
        "modelSha256": _sha256(model_path),
        "datasetAggregateSha256": extraction["generatedFiles"]["aggregateSha256"],
        "confirmation": {
            "season": "2025-26",
            "decisions": len(rows),
            "rounds": len({row["round_key"] for row in rows}),
            "matches": len(set(match_ids)),
            "immediateRonEvents": int(np.sum(truth)),
            "shantenAfterDistribution": dict(sorted(Counter(str(row["shanten_after"]) for row in rows).items())),
        },
        "immediateRon": {
            "calibrated": _probability_metrics(truth, calibrated),
            "currentDangerTable": _probability_metrics(truth, current),
            "comparison": comparison,
            "smallGroupRule": "独立表示には200判断かつ20イベント以上が必要。満たさない群は全体モデルへ縮約",
        },
        "excludedTargets": {
            "rewardPoints": "全打牌では同じ局収支を重複計上するためC.1の学習・評価対象外",
            "unselectedActions": "観測結果がないため対象外",
        },
        "artifacts": {"reliabilityCsv": "reliability.csv", "reliabilitySvg": "reliability.svg"},
    }
    _write_json(model_dir / "evaluation.json", evaluation)
    _write_reliability_csv(model_dir / "reliability.csv", reliability)
    _write_reliability_svg(model_dir / "reliability.svg", reliability)
    return evaluation
