#!/usr/bin/env python3
"""Independently reconstruct riichi wins and check the ura-dora analysis.

This verifier deliberately does not import the analysis implementation. It
reads the original Tenhou JSON arrays and counts physical tiles itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS_PATH = ROOT / "data/mleague/ingest/2026-10-12.paifu-corrections.jsonl"


def normal_tile(value: Any) -> int:
    if isinstance(value, str):
        numbers = re.findall(r"\d{2}", value)
        assert numbers, f"No tile in {value!r}"
        value = int(numbers[-1])
    assert isinstance(value, int) and value != 60, value
    value = {51: 15, 52: 25, 53: 35}.get(value, value)
    assert value in [s * 10 + n for s in (1, 2, 3) for n in range(1, 10)] + list(range(41, 48)), value
    return value


def successor(indicator: int) -> int:
    if indicator < 40:
        return indicator // 10 * 10 + indicator % 10 % 9 + 1
    if indicator <= 44:
        return 41 + (indicator - 40) % 4
    return 45 + (indicator - 44) % 3


def discarded(value: Any, draw: Any) -> int:
    return normal_tile(draw if value in (60, "r60") else value)


def independent_win(row: dict[str, Any], log: list[Any], log_index: int) -> dict[str, Any] | None:
    result = log[-1]
    if result[0] != "和了":
        return None
    assert len(result) == 3, "M-League has one winner per round"
    winner, donor = result[2][:2]
    draws = log[5 + 3 * winner]
    discards = log[6 + 3 * winner]
    if not any(isinstance(v, str) and v.startswith("r") for v in discards):
        return None
    hand = Counter(normal_tile(v) for v in log[4 + 3 * winner])
    assert sum(hand.values()) == 13
    kan_tiles: list[int] = []
    for index, draw in enumerate(draws):
        # A riichi winner can have concealed kans, but cannot have open calls.
        assert isinstance(draw, int), f"Open call in riichi hand: {draw}"
        hand[normal_tile(draw)] += 1
        if index < len(discards):
            discard = discards[index]
            if isinstance(discard, str) and "a" in discard:
                tiles = [normal_tile(int(v)) for v in re.findall(r"\d{2}", discard)]
                assert len(tiles) == 4 and len(set(tiles)) == 1, discard
                kan_tiles.extend(tiles)
                hand.subtract(tiles)
            else:
                hand[discarded(discard, draw)] -= 1
        assert min(hand.values()) >= 0, "A discarded tile was absent"
        assert max(hand.values()) <= 4, "A fifth copy appeared"
    if winner != donor:
        donor_draws = log[5 + 3 * donor]
        donor_discards = log[6 + 3 * donor]
        final_discard = donor_discards[-1]
        # A robbed added kan uses the kan tile, rather than an ordinary discard.
        # A big open kan adds a replacement draw without a matching discard,
        # so the final discard must use the last draw, not the discard index.
        hand[discarded(final_discard, donor_draws[-1])] += 1
    hand.update(kan_tiles)
    assert sum(hand.values()) == 14 + len(kan_tiles) // 4
    assert min(hand.values()) >= 0 and max(hand.values()) <= 4
    indicators = [normal_tile(v) for v in log[3]]
    ura = sum(hand[successor(v)] for v in indicators) if indicators else None
    return {
        "season": row["season"], "gameId": row["gameId"],
        "roundIndex": row["roundIndex"], "logIndex": log_index,
        "seat": winner, "ura": ura, "indicators": indicators,
        "tiles": sorted(hand.elements()), "kanCount": len(kan_tiles) // 4,
        "scoreLabel": result[2][3],
    }


def raw_records(
    paifu_dir: Path, corrections_path: Path = CORRECTIONS_PATH,
) -> tuple[list[dict[str, Any]], dict[str, str], int, int]:
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    round_count = 0
    seen: set[tuple[Any, ...]] = set()
    corrections: dict[tuple[str, int], dict[str, Any]] = {}
    used_corrections: set[tuple[str, int]] = set()
    if corrections_path.exists():
        hashes[corrections_path.name] = hashlib.sha256(corrections_path.read_bytes()).hexdigest()
        with corrections_path.open(encoding="utf-8") as source:
            for line in source:
                correction = json.loads(line)
                key = (correction["matchUrl"].rstrip("/"), correction["roundIndex"])
                assert key not in corrections, f"Duplicate correction: {key}"
                corrections[key] = correction
    for path in sorted(paifu_dir.glob("*.jsonl")):
        assert path.name not in hashes, f"Ambiguous input filename: {path.name}"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                correction_key = (row["matchUrl"].rstrip("/"), row["roundIndex"])
                if correction_key in corrections:
                    corrected = corrections[correction_key]
                    # Historical IDs stay stable when the source fixes a log.
                    assert all(corrected[field] == row[field] for field in ("season", "gameId", "roundIndex")), correction_key
                    assert correction_key not in used_corrections, f"Correction applied more than once: {correction_key}"
                    used_corrections.add(correction_key)
                    row = corrected
                for log_index, log in enumerate(row["paifu"]["log"]):
                    round_count += 1
                    key = (row["season"], row["gameId"], row["roundIndex"], log_index)
                    assert key not in seen, f"Duplicate original round: {key}"
                    seen.add(key)
                    win = independent_win(row, log, log_index)
                    if win:
                        records.append(win)
    assert round_count, "No paifu rounds found"
    assert used_corrections == set(corrections), "Correction rows did not match the original input"
    return records, hashes, round_count, len(used_corrections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paifu-dir", type=Path, default=ROOT / "data/mleague/paifu")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "data/mleague/ura_dora")
    parser.add_argument("--corrections-file", type=Path, default=CORRECTIONS_PATH)
    parser.add_argument("--raw-only", action="store_true", help="Only independently recompute original records")
    args = parser.parse_args()
    records, hashes, round_count, corrections_used = raw_records(args.paifu_dir, args.corrections_file)
    known = [r for r in records if r["ura"] is not None]
    hits = sum(r["ura"] > 0 for r in known)
    result = {
        "rawRounds": round_count, "riichiWins": len(records),
        "knownUraWins": len(known), "missingUraWins": len(records) - len(known),
        "uraHitWins": hits, "uraHitRate": hits / len(known),
        "uraHistogram": dict(sorted(Counter(r["ura"] for r in known).items())),
        "sourceSha256": hashes,
        "correctionsUsed": corrections_used,
    }
    if not args.raw_only:
        verify_outputs(records, hashes, round_count, corrections_used, args.analysis_dir)
        result["outputComparison"] = "passed"
    print(json.dumps(result, ensure_ascii=False, indent=2))


def verify_outputs(
    records: list[dict[str, Any]], hashes: dict[str, str], round_count: int,
    corrections_used: int, analysis_dir: Path,
) -> None:
    """Check every original win and each published probability denominator."""
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    with (analysis_dir / "wins.jsonl").open(encoding="utf-8") as source:
        output = [json.loads(line) for line in source]
    raw_key = lambda r: (r["season"], r["gameId"], r["roundIndex"], r["logIndex"], r["seat"])
    output_key = lambda r: (r["season"], r["gameId"], r["roundIndex"], r["logIndex"], r["winner"])
    originals = {raw_key(r): r for r in records}
    assert len(originals) == len(records)
    assert len({output_key(r) for r in output}) == len(output), "Duplicate output wins"
    assert {output_key(r) for r in output} == set(originals), "Missing or extra output wins"
    for generated in output:
        original = originals[output_key(generated)]
        assert generated["uraCount"] == original["ura"], output_key(generated)
        assert generated["indicatorCount"] == len(original["indicators"])
        assert [normal_tile(v) for v in generated["uraIndicators"]] == original["indicators"]
        assert sorted(normal_tile(v) for v in generated["handRaw"]) == original["tiles"]
        assert len(generated["closedKans"]) == original["kanCount"]
        # Validate the second representation of the same hand independently.
        counts = Counter(original["tiles"])
        tile_order = [s * 10 + n for s in (1, 2, 3) for n in range(1, 10)] + list(range(41, 48))
        assert generated["hand34"] == [counts[t] for t in tile_order]
        assert generated["distinctTileTypes"] == len(counts)
    inputs = {Path(item["path"]).name: item["sha256"] for item in summary["inputs"]}
    assert len(inputs) == len(summary["inputs"])
    assert inputs == hashes, "Published input hashes no longer match the original files"
    assert summary["rounds"] == round_count
    assert summary.get("correctionsUsed", 0) == corrections_used

    def check_rate(published: dict[str, Any], subset: list[dict[str, Any]]) -> None:
        known = [r for r in subset if r["uraCount"] is not None]
        hit = sum(r["uraCount"] >= 1 for r in known)
        expected = {
            "riichiWins": len(subset), "known": len(known),
            "missing": len(subset) - len(known), "hit": hit,
            "rate": hit / len(known) if known else None,
            "meanUra": sum(r["uraCount"] for r in known) / len(known) if known else None,
            "twoOrMore": sum(r["uraCount"] >= 2 for r in known),
            "distribution": {str(k): v for k, v in Counter(r["uraCount"] for r in known).items()},
        }
        for field, value in expected.items():
            actual = published[field]
            assert math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12) if isinstance(value, float) else actual == value, (field, actual, value)

    check_rate(summary["overall"], output)
    check_rate(summary["oneIndicator"], [r for r in output if r["indicatorCount"] == 1])
    check_rate(summary["twoOrMoreIndicators"], [r for r in output if r["indicatorCount"] > 1])
    check_rate(summary["nonYakuman"], [r for r in output if not r["yakuman"]])
    assert set(summary["bySeason"]) == {r["season"] for r in output}
    for season, published in summary["bySeason"].items():
        check_rate(published, [r for r in output if r["season"] == season])
    all_hit = sum(r["uraCount"] is not None and r["uraCount"] > 0 for r in output)
    assert {r["yaku"] for r in summary["byYaku"]} == {y for r in output for y in r["yaku"]}
    for published in summary["byYaku"]:
        subset = [r for r in output if published["yaku"] in r["yaku"]]
        check_rate(published, subset)
        check_rate(published["noKan"], [r for r in subset if r["indicatorCount"] == 1])
        assert math.isclose(published["shareAmongHits"], published["hit"] / all_hit)
    for kinds, published in summary["byDistinctTypes"].items():
        check_rate(published, [r for r in output if r["distinctTileTypes"] == int(kinds) and r["indicatorCount"] == 1])
    lower = all_hit / len(output)
    upper = (all_hit + sum(r["uraCount"] is None for r in output)) / len(output)
    assert all(math.isclose(a, b) for a, b in zip(summary["unknownBounds"], [lower, upper]))


if __name__ == "__main__":
    main()
