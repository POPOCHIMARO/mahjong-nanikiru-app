#!/usr/bin/env python3
"""D.3.2a: 候補別の厳密なシャンテンと受け入れ特徴。

このモジュールへ渡すのは、判断する家の手牌と判断時点の公開情報だけである。
教師ラベルや未来の自摸を受け取らない境界にして、特徴量への情報漏洩を防ぐ。
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .ev_calibration_state import TERMINAL_HONORS, _group_shapes, _standard_shanten, shanten
    from .ev_policy_round import kuikae_forbidden_tile34
except ImportError:
    from ev_calibration_state import TERMINAL_HONORS, _group_shapes, _standard_shanten, shanten
    from ev_policy_round import kuikae_forbidden_tile34


CALCULATION_VERSION = "exact-candidate-ukeire/v1"
MAX_SHAPE_CACHE_ENTRIES = 100_000
FEATURE_CACHE_SCHEMA = "ev-policy-opponent-feature-cache/v2"


@dataclass(frozen=True)
class ShapeResult:
    """公開残数を適用する前の牌姿計算結果。"""

    shanten: int
    improving_mask: int


@dataclass(frozen=True)
class ActionView:
    """一つの候補を評価するために許可された判断時点の情報。"""

    counts34: tuple[int, ...]
    fixed_melds: int
    visible34: tuple[int, ...]
    called_tile34: int | None = None


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}は整数が必要")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label}は{minimum}以上{maximum}以下が必要: {value}")
    return value


def _counts34(values: Sequence[Any], label: str) -> tuple[int, ...]:
    if len(values) != 34:
        raise ValueError(f"{label}は34要素が必要: {len(values)}")
    return tuple(_integer(value, f"{label}[{index}]", 0, 4) for index, value in enumerate(values))


def _validate_inventory(counts: tuple[int, ...], visible: tuple[int, ...]) -> None:
    for index, (hidden, public) in enumerate(zip(counts, visible)):
        if hidden + public > 4:
            raise ValueError(
                f"牌在庫が4枚を超える: tile34={index}, hand={hidden}, visible={public}"
            )


def make_action_view(
    counts34: Sequence[Any],
    fixed_melds: Any,
    visible34: Sequence[Any],
    called_tile34: Any | None = None,
) -> ActionView:
    """外部入力を検査し、不変な行動投影viewへ変換する。"""

    counts = _counts34(counts34, "counts34")
    visible = _counts34(visible34, "visible34")
    melds = _integer(fixed_melds, "fixed_melds", 0, 4)
    called = None if called_tile34 is None else _integer(called_tile34, "called_tile34", 0, 33)
    _validate_inventory(counts, visible)
    return ActionView(counts, melds, visible, called)


def _coerce_view(view: ActionView | Mapping[str, Any]) -> ActionView:
    if isinstance(view, ActionView):
        # dataclassを直接生成した呼び出し元も同じ検査を通す。
        return make_action_view(view.counts34, view.fixed_melds, view.visible34, view.called_tile34)
    return make_action_view(
        view["counts34"],
        view.get("fixed_melds", view.get("fixedMelds")),
        view.get("visible34", view.get("visible")),
        view.get("called_tile34", view.get("calledTile34")),
    )


@lru_cache(maxsize=MAX_SHAPE_CACHE_ENTRIES)
def _exact_shape_cached(counts: tuple[int, ...], fixed_melds: int) -> ShapeResult:
    groups = (
        _group_shapes(counts[0:9], True),
        _group_shapes(counts[9:18], True),
        _group_shapes(counts[18:27], True),
        _group_shapes(counts[27:34], False),
    )
    current = _shanten_from_groups(counts, fixed_melds, groups)
    mask = 0
    work = list(counts)
    for tile34 in range(34):
        if work[tile34] >= 4:
            continue
        work[tile34] += 1
        group_index = min(3, tile34 // 9)
        starts = (0, 9, 18, 27)
        ends = (9, 18, 27, 34)
        changed_groups = list(groups)
        changed_groups[group_index] = _group_shapes(
            tuple(work[starts[group_index] : ends[group_index]]),
            group_index < 3,
        )
        if _shanten_from_groups(tuple(work), fixed_melds, tuple(changed_groups)) < current:
            mask |= 1 << tile34
        work[tile34] -= 1
    return ShapeResult(current, mask)


def _standard_shanten_from_groups(
    fixed_melds: int,
    groups: tuple[tuple[tuple[int, int, int], ...], ...],
) -> int:
    """四群を段階結合し、同じ合計状態だけを重複排除する。"""

    target_melds = 4 - fixed_melds
    states = {(0, 0, 0)}
    for group in groups:
        combined = set()
        for left in states:
            for right in group:
                melds = left[0] + right[0]
                if melds <= target_melds:
                    combined.add((melds, left[1] + right[1], left[2] + right[2]))
        states = combined
    best = 8
    for melds, taatsu, pairs in states:
        has_pair = int(pairs > 0)
        taatsu += max(0, pairs - 1)
        effective_taatsu = min(taatsu, max(0, target_melds - melds))
        best = min(best, 8 - 2 * (fixed_melds + melds) - effective_taatsu - has_pair)
    return best


def _shanten_from_groups(
    counts: tuple[int, ...],
    fixed_melds: int,
    groups: tuple[tuple[tuple[int, int, int], ...], ...],
) -> int:
    values = [_standard_shanten_from_groups(fixed_melds, groups)]
    if fixed_melds == 0:
        pairs = sum(count >= 2 for count in counts)
        unique = sum(count > 0 for count in counts)
        values.append(6 - pairs + max(0, 7 - unique))
        terminal_unique = sum(counts[index] > 0 for index in TERMINAL_HONORS)
        terminal_pair = any(counts[index] >= 2 for index in TERMINAL_HONORS)
        values.append(13 - terminal_unique - int(terminal_pair))
    return min(values)


def reference_exact_shape(counts34: Sequence[Any], fixed_melds: Any = 0) -> ShapeResult:
    """高速経路の照合に使う、既存shanten全34種列挙。"""

    counts = _counts34(counts34, "counts34")
    melds = _integer(fixed_melds, "fixed_melds", 0, 4)
    expected = 13 - 3 * melds
    if sum(counts) != expected:
        raise ValueError(f"牌数不一致: fixed_melds={melds}, expected={expected}, actual={sum(counts)}")
    current = shanten(counts, melds)
    work = list(counts)
    mask = 0
    for tile34 in range(34):
        if work[tile34] >= 4:
            continue
        work[tile34] += 1
        if shanten(tuple(work), melds) < current:
            mask |= 1 << tile34
        work[tile34] -= 1
    return ShapeResult(current, mask)


def exact_shape(counts34: Sequence[Any], fixed_melds: Any = 0) -> ShapeResult:
    """13枚相当の伏せ牌姿についてシャンテンと改善牌maskを返す。"""

    counts = _counts34(counts34, "counts34")
    melds = _integer(fixed_melds, "fixed_melds", 0, 4)
    expected = 13 - 3 * melds
    if sum(counts) != expected:
        raise ValueError(f"牌数不一致: fixed_melds={melds}, expected={expected}, actual={sum(counts)}")
    return _exact_shape_cached(counts, melds)


def shape_cache_info() -> Any:
    return _exact_shape_cached.cache_info()


def clear_shape_cache() -> None:
    _exact_shape_cached.cache_clear()


def clear_all_shape_caches() -> None:
    """cold性能測定用。通常処理では局所cacheだけを消さない。"""

    clear_shape_cache()
    shanten.cache_clear()
    _standard_shanten.cache_clear()
    _group_shapes.cache_clear()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """同じディレクトリの一時ファイルを書き切ってから正本へ切り替える。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_feature_shard(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """gzip JSONL shardを原子的に保存し、検証用の内容hashを返す。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "windows": len(rows),
        "candidates": sum(int(row["candidateCount"]) for row in rows),
        "firstWindowId": rows[0]["windowId"] if rows else None,
        "lastWindowId": rows[-1]["windowId"] if rows else None,
    }


def iter_feature_shard(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"特徴shardのJSONが不正: {path.name}:{line_number}") from error


def _consume(work: list[int], tile34: int, amount: int, label: str) -> None:
    if work[tile34] < amount:
        raise ValueError(f"存在しない牌を消費する: {label}, tile34={tile34}, amount={amount}")
    work[tile34] -= amount


def _consume_records(work: list[int], records: Sequence[Mapping[str, Any]], label: str) -> None:
    for record in records:
        tile34 = _integer(record.get("tile34"), f"{label}.tile34", 0, 33)
        _consume(work, tile34, 1, label)


def _metrics(
    counts: tuple[int, ...],
    fixed_melds: int,
    visible: tuple[int, ...],
) -> dict[str, Any]:
    _validate_inventory(counts, visible)
    shape = exact_shape(counts, fixed_melds)
    kinds = 0
    total = 0
    for tile34 in range(34):
        if not shape.improving_mask & (1 << tile34):
            continue
        remaining = 4 - counts[tile34] - visible[tile34]
        if remaining < 0:
            raise ValueError(f"公開残数が負: tile34={tile34}, remaining={remaining}")
        if remaining:
            kinds += 1
            total += remaining
    return {
        "shanten": shape.shanten,
        "ukeireCount": total,
        "ukeireKinds": kinds,
        "applicable": 1,
        "counts34": counts,
        "visible34": visible,
    }


def _project_simple(
    view: ActionView,
    action: Mapping[str, Any],
) -> tuple[list[int], int, list[int]]:
    kind = str(action["kind"])
    counts = list(view.counts34)
    visible = list(view.visible34)
    fixed_melds = view.fixed_melds
    if kind in {"discard", "riichi_discard"}:
        tile34 = _integer(action.get("tile34"), "action.tile34", 0, 33)
        _consume(counts, tile34, 1, kind)
        visible[tile34] += 1
    elif kind == "pass":
        pass
    elif kind == "daiminkan":
        consumed = action.get("consumed") or []
        if len(consumed) != 3:
            raise ValueError("daiminkanのconsumedは3枚が必要")
        _consume_records(counts, consumed, kind)
        for tile in consumed:
            visible[int(tile["tile34"])] += 1
        fixed_melds += 1
    elif kind == "ankan":
        tile34 = _integer(action.get("tile34"), "action.tile34", 0, 33)
        _consume(counts, tile34, 4, kind)
        visible[tile34] += 4
        fixed_melds += 1
    elif kind == "kakan":
        tile34 = _integer(action.get("tile34"), "action.tile34", 0, 33)
        _consume(counts, tile34, 1, kind)
        visible[tile34] += 1
    else:
        raise ValueError(f"投影できない行動種別: {kind}")
    if fixed_melds > 4:
        raise ValueError(f"固定面子数が4を超える: {fixed_melds}")
    return counts, fixed_melds, visible


def project_action(
    view: ActionView | Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    """合法候補をコピー上へ投影し、候補固有の厳密特徴を返す。"""

    checked = _coerce_view(view)
    kind = str(action.get("kind"))
    if kind in {"ron", "tsumo"}:
        return {
            "shanten": -1,
            "ukeireCount": 0,
            "ukeireKinds": 0,
            "applicable": 0,
            "counts34": checked.counts34,
            "visible34": checked.visible34,
            "followupTile34": None,
        }

    if kind not in {"chi", "pon"}:
        counts, fixed_melds, visible = _project_simple(checked, action)
        result = _metrics(tuple(counts), fixed_melds, tuple(visible))
        result["followupTile34"] = None
        return result

    consumed = action.get("consumed") or []
    if len(consumed) != 2:
        raise ValueError(f"{kind}のconsumedは2枚が必要")
    if checked.called_tile34 is None:
        raise ValueError(f"{kind}には応答対象牌が必要")
    counts = list(checked.counts34)
    visible = list(checked.visible34)
    _consume_records(counts, consumed, kind)
    for tile in consumed:
        visible[int(tile["tile34"])] += 1
    fixed_melds = checked.fixed_melds + 1
    forbidden = kuikae_forbidden_tile34(
        kind,
        checked.called_tile34,
        tuple(int(tile["tile34"]) for tile in consumed),
    )

    options: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for tile34, amount in enumerate(counts):
        if amount <= 0 or tile34 in forbidden:
            continue
        projected = counts.copy()
        projected[tile34] -= 1
        projected_visible = visible.copy()
        projected_visible[tile34] += 1
        metrics = _metrics(tuple(projected), fixed_melds, tuple(projected_visible))
        metrics["followupTile34"] = tile34
        order = (
            int(metrics["shanten"]),
            -int(metrics["ukeireCount"]),
            -int(metrics["ukeireKinds"]),
            tile34,
        )
        options.append((order, metrics))
    if not options:
        raise ValueError(f"{kind}後の合法な打牌がない")
    return min(options, key=lambda item: item[0])[1]
