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
    from .ev_calibration_model import DANGER_RATES
    from .ev_calibration_state import (
        TERMINAL_HONORS,
        _band,
        _group_shapes,
        _number,
        _safety_group,
        _standard_shanten,
        shanten,
    )
    from .ev_policy_round import kuikae_forbidden_tile34
except ImportError:
    from ev_calibration_model import DANGER_RATES
    from ev_calibration_state import (
        TERMINAL_HONORS,
        _band,
        _group_shapes,
        _number,
        _safety_group,
        _standard_shanten,
        shanten,
    )
    from ev_policy_round import kuikae_forbidden_tile34


CALCULATION_VERSION = "exact-candidate-ukeire/v1"
MAX_SHAPE_CACHE_ENTRIES = 100_000
FEATURE_CACHE_SCHEMA = "ev-policy-opponent-feature-cache/v3"
# D.3.2b：特徴v3で実装した群。採用判定はこの名前で実装済みかを確かめる。
DANGER_FEATURE_GROUP = "danger_multi_riichi_v3"
YAKU_SHAPE_FEATURE_GROUP = "yaku_shape_cues_v1"
IMPLEMENTED_FEATURE_GROUPS = ("exact_candidate_ukeire", DANGER_FEATURE_GROUP, YAKU_SHAPE_FEATURE_GROUP)
# 旧表の最大率（無スジ4・5・6の5.7%）。率は危険度の順位付けにだけ使い、放銃確率とは呼ばない。
DANGER_RATE_SCALE = 0.057
SAFETY_GROUPS = ("safe", "semi_safe", "guarded", "moderate_risk", "high_risk")


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


# ---------------------------------------------------------------------------
# D.3.2b 特徴v3：危険度（設計5節）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiichiOpponentView:
    """行動する家から見た成立済みリーチ者一人分の公開情報。"""

    seat: int
    is_dealer: bool
    safe_tiles: frozenset[int]


def classify_danger_v3(tile34: int, safe_tiles: frozenset[int] | set[int], seen: Sequence[int]) -> str:
    """既存classify_dangerと同じラベルと判定順で、現物とスジの判定元だけを安全牌集合へ替える。

    safe_tilesはリーチ者の河の全牌と、リーチ後に他家から出て見逃された牌の和集合。
    seenは行動する家から見えている枚数（公開済み牌と自分の手牌）。
    """

    index = _integer(tile34, "tile34", 0, 33)
    if index in safe_tiles:
        return "genbutsu"
    if index >= 27:
        return "honor_3_visible" if seen[index] >= 3 else "honor_2_visible" if seen[index] >= 2 else "honor_live"
    number = _number(index)
    neighbor = index + 1 if number in (1, 2) else index - 1 if number in (8, 9) else None
    if neighbor is not None and seen[neighbor] >= 4:
        return f"no_chance_{'19' if number in (1, 9) else '28'}"
    if neighbor is not None and seen[neighbor] >= 3:
        return f"one_chance_{'19' if number in (1, 9) else '28'}"
    suit_start = index // 9 * 9
    hits = sum(
        suit_start <= index + delta < suit_start + 9 and index + delta in safe_tiles
        for delta in (-3, 3)
    )
    if hits:
        return "double_suji_middle" if hits == 2 and _band(index) == "456" else f"suji_{_band(index)}"
    if number in (1, 9):
        return "terminal_non_suji"
    return f"non_suji_{_band(index)}"


def danger_features(
    tile34: int | None,
    riichi_opponents: Sequence[RiichiOpponentView],
    seen: Sequence[int],
) -> dict[str, float]:
    """打牌候補の危険度特徴。リーチ者がいない、または打牌を伴わない行動では全て0。"""

    values = {
        "danger_applicable": 0.0,
        "danger_max_rate": 0.0,
        "danger_sum_rate": 0.0,
        "danger_dealer_rate": 0.0,
        "genbutsu_all": 0.0,
        "genbutsu_any": 0.0,
    }
    values.update({f"danger_group_{group}": 0.0 for group in SAFETY_GROUPS})
    if tile34 is None or not riichi_opponents:
        return values
    rated = []
    for opponent in sorted(riichi_opponents, key=lambda item: item.seat):
        label = classify_danger_v3(tile34, opponent.safe_tiles, seen)
        rated.append((DANGER_RATES[label] / 100.0, opponent, label))
    # 最大率が同じなら座席番号の小さい家を代表にし、結果を入力順に依存させない。
    worst_rate, _, worst_label = max(rated, key=lambda item: (item[0], -item[1].seat))
    values["danger_applicable"] = 1.0
    values["danger_max_rate"] = worst_rate / DANGER_RATE_SCALE
    values["danger_sum_rate"] = sum(rate for rate, _, _ in rated) / (3 * DANGER_RATE_SCALE)
    values["danger_dealer_rate"] = next(
        (rate / DANGER_RATE_SCALE for rate, opponent, _ in rated if opponent.is_dealer), 0.0
    )
    genbutsu = [label == "genbutsu" for _, _, label in rated]
    values["genbutsu_all"] = float(all(genbutsu))
    values["genbutsu_any"] = float(any(genbutsu))
    values[f"danger_group_{_safety_group(worst_label)}"] = 1.0
    return values


# ---------------------------------------------------------------------------
# D.3.2b 特徴v3：役と形の手掛かり（設計6節）。得点器は使わない近似である。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeldView:
    """固定面子一つ。kindはchi、pon、daiminkan、ankan、kakanのいずれか。"""

    kind: str
    tiles34: tuple[int, ...]


def _suit(tile34: int) -> int | None:
    return None if tile34 >= 27 else tile34 // 9


def yaku_shape_features(
    concealed34: Sequence[int],
    melds: Sequence[MeldView],
    seat_wind_tile34: int,
    round_wind_tile34: int,
) -> dict[str, float]:
    """行動後の手中牌と固定面子から、役と形の手掛かりを数える。"""

    counts = _counts34(concealed34, "concealed34")
    yakuhai = {31, 32, 33, seat_wind_tile34, round_wind_tile34}
    fixed = len(melds)
    open_after = any(meld.kind != "ankan" for meld in melds)
    meld_tiles = [tile for meld in melds for tile in meld.tiles34]

    yakuhai_secured = any(
        meld.kind != "chi" and meld.tiles34[0] in yakuhai for meld in melds
    ) or any(counts[tile] >= 3 for tile in yakuhai)
    yakuhai_pairs = sum(counts[tile] == 2 for tile in yakuhai)
    tanyao_path = not any(tile in TERMINAL_HONORS for tile in meld_tiles)
    tanyao_distance = sum(counts[tile] for tile in TERMINAL_HONORS)

    fixed_suits = {_suit(tile) for tile in meld_tiles if _suit(tile) is not None}
    flush_path = len(fixed_suits) <= 1
    if flush_path:
        if fixed_suits:
            target = next(iter(fixed_suits))
        else:
            # 固定面子に数牌がなければ手中の最多色。同数なら萬子、筒子、索子の順。
            sums = [sum(counts[suit * 9 : suit * 9 + 9]) for suit in range(3)]
            target = max(range(3), key=lambda suit: (sums[suit], -suit))
        off_suit = sum(counts[tile] for tile in range(27) if tile // 9 != target)
        flush_distance = off_suit / 14.0
    else:
        flush_distance = 1.0

    triplets = sum(amount >= 3 for amount in counts)
    pairs = sum(amount == 2 for amount in counts)
    toitoi_path = not any(meld.kind == "chi" for meld in melds) and triplets + pairs >= 4 - fixed
    chiitoi_applicable = fixed == 0
    chiitoi_shanten = 0.0
    if chiitoi_applicable:
        pair_kinds = sum(amount >= 2 for amount in counts)
        unique = sum(amount > 0 for amount in counts)
        chiitoi_shanten = (6 - pair_kinds + max(0, 7 - unique)) / 6.0

    no_cue = (
        open_after
        and not yakuhai_secured
        and yakuhai_pairs == 0
        and not tanyao_path
        and not flush_path
        and not toitoi_path
    )
    return {
        "menzen_after": float(not open_after),
        "yakuhai_secured": float(yakuhai_secured),
        "yakuhai_pairs": min(1.0, yakuhai_pairs / 3.0),
        "tanyao_path": float(tanyao_path),
        "tanyao_distance": tanyao_distance / 14.0,
        "flush_path": float(flush_path),
        "flush_distance": flush_distance,
        "toitoi_blocks": min(1.0, (triplets + pairs) / 5.0),
        "toitoi_path": float(toitoi_path),
        "pair_count": pairs / 7.0,
        "chiitoi_applicable": float(chiitoi_applicable),
        "chiitoi_shanten": chiitoi_shanten,
        "open_no_listed_yaku_cue": float(no_cue),
    }


def melds_after_action(
    melds: Sequence[MeldView],
    action: Mapping[str, Any],
    called_tile34: int | None,
) -> tuple[MeldView, ...]:
    """候補を実行した後の固定面子。元の面子列は変更しない。"""

    kind = str(action["kind"])
    result = list(melds)
    if kind in {"chi", "pon", "daiminkan"}:
        if called_tile34 is None:
            raise ValueError(f"{kind}には応答対象牌が必要")
        consumed = tuple(int(tile["tile34"]) for tile in action.get("consumed") or [])
        result.append(MeldView(kind, tuple(sorted(consumed + (called_tile34,)))))
    elif kind == "ankan":
        tile34 = _integer(action.get("tile34"), "action.tile34", 0, 33)
        result.append(MeldView("ankan", (tile34,) * 4))
    elif kind == "kakan":
        tile34 = _integer(action.get("tile34"), "action.tile34", 0, 33)
        index = next(
            (i for i, meld in enumerate(result) if meld.kind == "pon" and meld.tiles34[0] == tile34),
            None,
        )
        if index is None:
            raise ValueError(f"加槓の元になるポンがない: tile34={tile34}")
        result[index] = MeldView("kakan", (tile34,) * 4)
    return tuple(result)
