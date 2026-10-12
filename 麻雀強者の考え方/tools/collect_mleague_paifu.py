#!/usr/bin/env python3
"""
Collect public M-League Tenhou-format paifu from https://m-league.konoui.dev/.

This script intentionally avoids browser automation and authentication. It reads
public HTML pages, extracts embedded season metadata and Tenhou #json links, and
writes normalized JSONL files for later analysis.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any


BASE_URL = "https://m-league.konoui.dev"
USER_AGENT = "mahjong-llm-wiki-research/0.1 (+local personal research)"


def fetch_text(url: str, delay: float = 0.5, retries: int = 3) -> str:
    time.sleep(delay)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return subprocess.check_output(
                [
                    "curl.exe",
                    "-L",
                    "--ssl-no-revoke",
                    "-A",
                    USER_AGENT,
                    url,
                ],
                stderr=subprocess.DEVNULL,
                timeout=60,
            ).decode("utf-8", errors="replace")
        except Exception as curl_error:
            last_error = curl_error
            if attempt < retries:
                time.sleep(delay * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def extract_next_flight_payload(page_html: str) -> str:
    chunks: list[str] = []
    for match in re.finditer(r"self\.__next_f\.push\((\[.*?\])\)</script>", page_html, re.S):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if len(value) >= 2 and isinstance(value[1], str):
            chunks.append(value[1])
    return "\n".join(chunks)


def extract_balanced_json(text: str, marker: str) -> dict[str, Any]:
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"marker not found: {marker}")
    pos = text.find("{", start)
    if pos < 0:
        raise ValueError(f"json object not found after marker: {marker}")

    depth = 0
    in_string = False
    escape = False
    for idx in range(pos, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[pos : idx + 1])
    raise ValueError(f"unterminated json object after marker: {marker}")


def discover_seasons() -> list[str]:
    text = fetch_text(f"{BASE_URL}/")
    seasons = sorted(set(re.findall(r"/seasons/(\d{4}-\d{2})/?", text)), reverse=True)
    if not seasons:
        raise RuntimeError("No seasons discovered")
    return seasons


def load_season_data(season: str) -> dict[str, Any]:
    page = fetch_text(f"{BASE_URL}/seasons/{season}/")
    payload = extract_next_flight_payload(page)
    return extract_balanced_json(payload, '"seasonData"')


def iter_matches(season_data: dict[str, Any]) -> list[dict[str, Any]]:
    season = season_data["season"]
    rows: list[dict[str, Any]] = []
    for stage in season_data.get("stages", []):
        stage_name = stage["stage"]
        for day in stage.get("days", []):
            month_day = day["date"][5:]
            for table in day.get("tables", []):
                for match in table.get("matches", []):
                    url = (
                        f"{BASE_URL}/seasons/{season}/{stage_name}/"
                        f"{month_day}/{table['tableId']}-{match['matchNumber']}"
                    )
                    rows.append(
                        {
                            "season": season,
                            "stage": stage_name,
                            "date": day["date"],
                            "dayOfWeek": day.get("dayOfWeek"),
                            "tableId": table["tableId"],
                            "teams": table.get("teams", []),
                            "gameId": match["gameId"],
                            "matchNumber": match["matchNumber"],
                            "players": match.get("players", []),
                            "matchUrl": url,
                        }
                    )
    return rows


def extract_tenhou_json_links(page_html: str) -> list[str]:
    links = re.findall(r'href="(https://tenhou\.net/5/#json=[^"]+)"', page_html)
    return [html.unescape(link) for link in links]


def decode_tenhou_link(link: str) -> dict[str, Any]:
    encoded = link.split("#json=", 1)[1]
    return json.loads(urllib.parse.unquote(encoded))


def collect_match(match: dict[str, Any], delay: float) -> list[dict[str, Any]]:
    page = fetch_text(match["matchUrl"], delay=delay)
    links = extract_tenhou_json_links(page)
    rounds: list[dict[str, Any]] = []
    for round_index, link in enumerate(links):
        paifu = decode_tenhou_link(link)
        title = paifu.get("title", [])
        round_name = title[1] if len(title) > 1 else f"round-{round_index + 1}"
        rounds.append(
            {
                **match,
                "roundIndex": round_index,
                "roundName": round_name,
                "tenhouUrl": link,
                "paifu": paifu,
            }
        )
    return rounds


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def existing_game_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    game_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            game_id = row.get("gameId")
            if isinstance(game_id, str):
                game_ids.add(game_id)
    return game_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", action="append", help="Season to collect, e.g. 2024-25. Repeatable.")
    parser.add_argument("--all", action="store_true", help="Collect every discovered season.")
    parser.add_argument("--metadata-only", action="store_true", help="Only collect match lists, not round paifu.")
    parser.add_argument("--overwrite-paifu", action="store_true", help="Overwrite existing paifu JSONL files.")
    parser.add_argument("--limit-matches", type=int, default=0, help="Limit matches per season for testing.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds.")
    parser.add_argument("--out", default="data/mleague", help="Output directory.")
    args = parser.parse_args()

    seasons = discover_seasons() if args.all else args.season
    if not seasons:
        parser.error("Specify --season YYYY-YY or --all")

    out_dir = Path(args.out)
    all_matches: list[dict[str, Any]] = []
    total_rounds = 0

    for season in seasons:
        print(f"[season] {season}", file=sys.stderr)
        season_data = load_season_data(season)
        matches = iter_matches(season_data)
        if args.limit_matches:
            matches = matches[: args.limit_matches]
        write_jsonl(out_dir / "matches" / f"{season}.jsonl", matches)
        all_matches.extend(matches)

        if args.metadata_only:
            continue

        paifu_path = out_dir / "paifu" / f"{season}.jsonl"
        error_path = out_dir / "errors" / f"{season}.jsonl"
        if args.overwrite_paifu and paifu_path.exists():
            paifu_path.unlink()
        done_game_ids = existing_game_ids(paifu_path)
        season_rounds = 0
        for idx, match in enumerate(matches, start=1):
            if match["gameId"] in done_game_ids:
                print(f"  [{idx}/{len(matches)}] skip {match['gameId']}", file=sys.stderr)
                continue
            print(f"  [{idx}/{len(matches)}] {match['matchUrl']}", file=sys.stderr)
            try:
                rounds = collect_match(match, delay=args.delay)
            except Exception as error:
                append_jsonl(error_path, [{**match, "error": str(error)}])
                print(f"    error: {error}", file=sys.stderr)
                continue
            append_jsonl(paifu_path, rounds)
            season_rounds += len(rounds)
        total_rounds += season_rounds

    write_jsonl(out_dir / "matches" / "all.jsonl", all_matches)
    print(f"Collected {len(all_matches)} matches and {total_rounds} rounds.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
