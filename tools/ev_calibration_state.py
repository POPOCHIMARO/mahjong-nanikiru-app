"""EV較正用の牌表現、シャンテン、受け入れ、危険度計算。"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any


TERMINAL_HONORS = frozenset((0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33))
DANGER_LEVELS = {
    "genbutsu": 0,
    "honor_3_visible": 1,
    "no_chance_19": 2,
    "no_chance_28": 3,
    "double_suji_middle": 4,
    "suji_19": 4,
    "honor_2_visible": 5,
    "one_chance_19": 5,
    "one_chance_28": 6,
    "suji_28": 6,
    "suji_37": 7,
    "suji_456": 8,
    "terminal_non_suji": 9,
    "honor_live": 10,
    "non_suji_28": 11,
    "non_suji_37": 12,
    "non_suji_456": 13,
    "unknown": 14,
}


def raw_codes(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(part) for part in re.findall(r"\d{2}", value)]
    return []


def tile34(value: Any) -> int | None:
    codes = raw_codes(value)
    if not codes:
        return None
    raw = codes[-1]
    if raw == 60:
        return None
    raw = {51: 15, 52: 25, 53: 35}.get(raw, raw)
    suit, number = divmod(raw, 10)
    if suit == 1 and 1 <= number <= 9:
        return number - 1
    if suit == 2 and 1 <= number <= 9:
        return number + 8
    if suit == 3 and 1 <= number <= 9:
        return number + 17
    if suit == 4 and 1 <= number <= 7:
        return number + 26
    return None


def tile_name(index: int | None) -> str | None:
    if index is None:
        return None
    if 0 <= index < 9:
        return f"{index + 1}m"
    if 9 <= index < 18:
        return f"{index - 8}p"
    if 18 <= index < 27:
        return f"{index - 17}s"
    if 27 <= index < 34:
        return ("東", "南", "西", "北", "白", "發", "中")[index - 27]
    return None


def is_red(raw: Any) -> bool:
    return raw in (51, 52, 53)


def tile_record(raw: int) -> dict[str, Any]:
    index = tile34(raw)
    return {"raw": raw, "tile34": index, "tile": tile_name(index), "isRed": is_red(raw)}


def discard_physical_raw(value: Any, paired_draw: int | None) -> int | None:
    if value in (60, "r60"):
        return paired_draw
    codes = raw_codes(value)
    return codes[-1] if codes else None


def call_marker(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    for marker in ("c", "p", "m", "k", "a"):
        if marker in value:
            return marker
    return None


def call_parts(value: Any) -> tuple[str | None, list[int], list[int], int | None]:
    marker = call_marker(value)
    if marker is None or not isinstance(value, str):
        return None, [], [], None
    before_text, after_text = value.split(marker, 1)
    before = [int(part) for part in re.findall(r"\d{2}", before_text)]
    after = [int(part) for part in re.findall(r"\d{2}", after_text)]
    all_tiles = before + after
    if marker in {"a", "k"}:
        return marker, all_tiles, all_tiles, None
    if not before and after:
        return marker, all_tiles, after[1:], after[0]
    if before and not after:
        return marker, all_tiles, before[:-1], before[-1]
    if before and after:
        return marker, all_tiles, before + after[1:], after[0]
    return marker, all_tiles, [], None


def call_source_matches(value: Any, caller: int, discarder: int) -> bool:
    marker = call_marker(value)
    if marker == "c":
        return caller == (discarder + 1) % 4
    if marker not in {"p", "m"} or not isinstance(value, str):
        return False
    before_count = len(re.findall(r"\d{2}", value.split(marker, 1)[0]))
    if marker == "p":
        source_offset = 3 - before_count
    elif before_count == 0:
        source_offset = 3
    elif before_count == 3:
        source_offset = 1
    else:
        source_offset = 2
    return discarder == (caller + source_offset) % 4


def counter34(raw_hand: Counter[int]) -> tuple[int, ...]:
    counts = [0] * 34
    for raw, count in raw_hand.items():
        index = tile34(raw)
        if index is not None:
            counts[index] += count
    return tuple(counts)


@lru_cache(maxsize=500_000)
def _group_shapes(group: tuple[int, ...], suited: bool) -> tuple[tuple[int, int, int], ...]:
    results: set[tuple[int, int, int]] = set()

    def visit(work: list[int], melds: int, taatsu: int, pairs: int) -> None:
        try:
            index = next(i for i, count in enumerate(work) if count)
        except StopIteration:
            results.add((melds, taatsu, pairs))
            return
        if work[index] >= 3:
            work[index] -= 3
            visit(work, melds + 1, taatsu, pairs)
            work[index] += 3
        if suited and index <= len(work) - 3 and work[index + 1] and work[index + 2]:
            for offset in range(3):
                work[index + offset] -= 1
            visit(work, melds + 1, taatsu, pairs)
            for offset in range(3):
                work[index + offset] += 1
        if work[index] >= 2:
            work[index] -= 2
            visit(work, melds, taatsu, pairs + 1)
            visit(work, melds, taatsu + 1, pairs)
            work[index] += 2
        if suited and index <= len(work) - 2 and work[index + 1]:
            work[index] -= 1
            work[index + 1] -= 1
            visit(work, melds, taatsu + 1, pairs)
            work[index] += 1
            work[index + 1] += 1
        if suited and index <= len(work) - 3 and work[index + 2]:
            work[index] -= 1
            work[index + 2] -= 1
            visit(work, melds, taatsu + 1, pairs)
            work[index] += 1
            work[index + 2] += 1
        work[index] -= 1
        visit(work, melds, taatsu, pairs)
        work[index] += 1

    visit(list(group), 0, 0, 0)
    nondominated = set(results)
    for left in results:
        for right in results:
            if left != right and all(r >= l for l, r in zip(left, right)) and sum(right) > sum(left):
                nondominated.discard(left)
                break
    return tuple(sorted(nondominated))


@lru_cache(maxsize=500_000)
def _standard_shanten(counts: tuple[int, ...], open_melds: int) -> int:
    target_melds = 4 - open_melds
    groups = (
        _group_shapes(counts[0:9], True),
        _group_shapes(counts[9:18], True),
        _group_shapes(counts[18:27], True),
        _group_shapes(counts[27:34], False),
    )
    best = 8
    for a in groups[0]:
        for b in groups[1]:
            for c in groups[2]:
                for d in groups[3]:
                    melds = sum(item[0] for item in (a, b, c, d))
                    if melds > target_melds:
                        continue
                    taatsu = sum(item[1] for item in (a, b, c, d))
                    pairs = sum(item[2] for item in (a, b, c, d))
                    has_pair = int(pairs > 0)
                    taatsu += max(0, pairs - 1)
                    effective_taatsu = min(taatsu, max(0, target_melds - melds))
                    best = min(best, 8 - 2 * (open_melds + melds) - effective_taatsu - has_pair)
    return best


@lru_cache(maxsize=500_000)
def shanten(counts: tuple[int, ...], open_melds: int = 0) -> int:
    values = [_standard_shanten(counts, open_melds)]
    if open_melds == 0:
        pairs = sum(count >= 2 for count in counts)
        unique = sum(count > 0 for count in counts)
        values.append(6 - pairs + max(0, 7 - unique))
        terminal_unique = sum(counts[index] > 0 for index in TERMINAL_HONORS)
        terminal_pair = any(counts[index] >= 2 for index in TERMINAL_HONORS)
        values.append(13 - terminal_unique - int(terminal_pair))
    return min(values)


def visible_counts(own_hand: Counter[int], rivers: list[list[dict[str, Any]]], dora_raw: list[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for raw, count in own_hand.items():
        index = tile34(raw)
        if index is not None:
            counts[index] += count
    for river in rivers:
        for discard in river:
            if not discard.get("called") and discard.get("tile34") is not None:
                counts[int(discard["tile34"])] += 1
    for raw in dora_raw:
        index = tile34(raw)
        if index is not None:
            counts[index] += 1
    return counts


def ukeire(counts: tuple[int, ...], current_shanten: int, seen: Counter[int]) -> dict[str, Any]:
    tiles: list[dict[str, Any]] = []
    total = 0
    work = list(counts)
    for index in range(34):
        remaining = max(0, 4 - seen[index])
        if remaining == 0:
            continue
        work[index] += 1
        next_shanten = shanten(tuple(work), 0)
        work[index] -= 1
        if next_shanten < current_shanten:
            tiles.append({"tile34": index, "tile": tile_name(index), "remaining": remaining})
            total += remaining
    return {"ukeireKinds": len(tiles), "ukeireCount": total, "ukeireTiles": tiles}


def _number(index: int) -> int | None:
    return None if index >= 27 else index % 9 + 1


def _band(index: int) -> str:
    number = _number(index)
    if number in (1, 9):
        return "19"
    if number in (2, 8):
        return "28"
    if number in (3, 7):
        return "37"
    return "456"


def _safety_group(label: str) -> str:
    if label == "genbutsu":
        return "safe"
    if label in {"honor_3_visible", "no_chance_19", "no_chance_28", "suji_19", "double_suji_middle"}:
        return "semi_safe"
    if label in {"honor_2_visible", "one_chance_19", "one_chance_28", "suji_28", "suji_37", "suji_456"}:
        return "guarded"
    if label in {"terminal_non_suji", "honor_live"}:
        return "moderate_risk"
    if label.startswith("non_suji_"):
        return "high_risk"
    return "unknown"


def classify_danger(index: int, riichi_seat: int, rivers: list[list[dict[str, Any]]], seen: Counter[int]) -> dict[str, Any]:
    discards = {int(item["tile34"]) for item in rivers[riichi_seat] if item.get("tile34") is not None}
    if index in discards:
        label = "genbutsu"
    elif index >= 27:
        label = "honor_3_visible" if seen[index] >= 3 else "honor_2_visible" if seen[index] >= 2 else "honor_live"
    else:
        number = _number(index)
        neighbor = index + 1 if number in (1, 2) else index - 1 if number in (8, 9) else None
        if neighbor is not None and seen[neighbor] >= 4:
            label = f"no_chance_{'19' if number in (1, 9) else '28'}"
        elif neighbor is not None and seen[neighbor] >= 3:
            label = f"one_chance_{'19' if number in (1, 9) else '28'}"
        else:
            suit_start = index // 9 * 9
            hits = sum(
                suit_start <= index + delta < suit_start + 9 and index + delta in discards
                for delta in (-3, 3)
            )
            if hits:
                label = "double_suji_middle" if hits == 2 and _band(index) == "456" else f"suji_{_band(index)}"
            elif number in (1, 9):
                label = "terminal_non_suji"
            else:
                label = f"non_suji_{_band(index)}"
    return {
        "riichiSeat": riichi_seat,
        "dangerClass": label,
        "dangerLevel": DANGER_LEVELS[label],
        "safetyGroup": _safety_group(label),
    }
