#!/usr/bin/env python3
"""Ingest the currently published M-League season pages and Tenhou JSON links."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

from collect_mleague_paifu import decode_tenhou_link, extract_tenhou_json_links


BASE_URL = "https://m-league.konoui.dev"
USER_AGENT = "mahjong-llm-wiki-research/0.1 (+local personal research)"
MATCH_PATH = re.compile(r"/seasons/(\d{4}-\d{2})/([^/]+)/(\d{2}-\d{2})/([A-Z])-([1-9]\d*)/?$")


def fetch(url: str, delay: float) -> bytes:
    time.sleep(delay)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def plain(markup: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", markup.replace("<!-- -->", ""))).strip()


def month_urls(season_html: str, season: str) -> list[str]:
    pattern = rf'href="(/seasons/{re.escape(season)}/[^/]+/\d{{4}}-\d{{2}}/)"'
    return sorted({urljoin(BASE_URL, path) for path in re.findall(pattern, season_html)})


def match_list(month_html: str, season: str) -> list[dict]:
    rows: list[dict] = []
    groups = re.findall(r'<div class="table-group"><h4>(.*?)</h4>(.*?)</div>', month_html, re.S)
    for heading, body in groups:
        heading_text = plain(heading)
        table_name, _, team_text = heading_text.partition(" - ")
        teams = team_text.split(" vs ")
        if not table_name or len(teams) != 4:
            raise ValueError(f"Cannot parse table heading: {heading_text}")
        links = re.findall(r'<a class="match-link" href="([^"]+)">(.*?)</a>', body, re.S)
        for path, content in links:
            match = MATCH_PATH.fullmatch(path)
            if not match or match.group(1) != season or match.group(4) != table_name[0]:
                raise ValueError(f"Unexpected match path: {path}")
            player_markup = re.search(r'<span class="players">(.*?)</span>', content, re.S)
            players = plain(player_markup.group(1)).split(" vs ") if player_markup else []
            if len(players) != 4:
                raise ValueError(f"Cannot parse four players: {path}")
            rows.append({
                "season": season,
                "stage": match.group(2),
                "date": f"{match.group(3)[:2]}-{match.group(3)[3:]}",
                "tableId": match.group(4),
                "teams": teams,
                "players": players,
                "matchNumber": int(match.group(5)),
                "matchUrl": urljoin(BASE_URL, path),
            })
    return rows


def ranking(match_html: str) -> dict[str, str]:
    body_match = re.search(r"<tbody>(.*?)</tbody>", match_html, re.S)
    if not body_match:
        raise ValueError("No ranking table")
    result = {}
    for row in re.findall(r"<tr>(.*?)</tr>", body_match.group(1), re.S):
        cells = [plain(value) for value in re.findall(r"<td>(.*?)</td>", row, re.S)]
        if len(cells) != 4:
            raise ValueError("Unexpected ranking row")
        result[cells[1]] = cells[2]
    if len(result) != 4:
        raise ValueError("Expected four ranked players")
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True)
    parser.add_argument("--date", default=date.today().isoformat(), help="Acquisition date, YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}", args.season):
        parser.error("--season must be YYYY-YY")
    date.fromisoformat(args.date)

    source_dir = Path("raw/sources/mleague") / args.season / args.date
    matches_dir = Path("data/mleague/matches")
    paifu_dir = Path("data/mleague/paifu")
    season_url = f"{BASE_URL}/seasons/{args.season}/"
    captures = [("season.html", season_url, fetch(season_url, args.delay))]
    rows: list[dict] = []
    for month_url in month_urls(captures[0][2].decode("utf-8"), args.season):
        name = "month-" + month_url.rstrip("/").split("/")[-1] + ".html"
        content = fetch(month_url, args.delay)
        captures.append((name, month_url, content))
        for row in match_list(content.decode("utf-8"), args.season):
            if row["matchUrl"] not in {item["matchUrl"] for item in rows}:
                row["date"] = f"{month_url.rstrip('/').split('/')[-1][:4]}-{row['date']}"
                rows.append(row)
    if not rows:
        raise ValueError("No published matches found; no files written")

    rounds: list[dict] = []
    for row in rows:
        path = MATCH_PATH.fullmatch(urlparse(row["matchUrl"]).path)
        assert path is not None
        row["gameId"] = f"{args.season}_{row['stage']}_{path.group(3)}_{row['tableId']}-{row['matchNumber']}"
        row["gameIdSource"] = "matchUrl (synthetic; site no longer exposes gameId)"
        content = fetch(row["matchUrl"], args.delay)
        captures.append((row["gameId"] + ".html", row["matchUrl"], content))
        page = content.decode("utf-8")
        player_teams = ranking(page)
        if player_teams != dict(zip(row["players"], row["teams"])):
            raise ValueError(f"Player/team mismatch: {row['matchUrl']}")
        links = extract_tenhou_json_links(page)
        if not links:
            raise ValueError(f"No paifu links: {row['matchUrl']}")
        match_rounds = []
        for index, link in enumerate(links):
            paifu = decode_tenhou_link(link)
            if set(paifu.get("name", [])) != set(row["players"]) or len(paifu.get("log", [])) != 1:
                raise ValueError(f"Invalid paifu at round {index}: {row['matchUrl']}")
            title = paifu.get("title", [])
            match_rounds.append({
                **row,
                "roundIndex": index,
                "roundName": title[1] if len(title) > 1 else f"round-{index + 1}",
                "tenhouUrl": link,
                "paifu": paifu,
            })
        rounds.extend(match_rounds)
        print(f"{row['gameId']}: {len(match_rounds)} rounds")

    # Keep a dated, byte-for-byte HTML capture. Existing captures are immutable.
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, url, content in captures:
        target = source_dir / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Existing raw capture changed: {target}")
        if not target.exists():
            target.write_bytes(content)
        manifest.append({"file": name, "url": url, "sha256": hashlib.sha256(content).hexdigest()})
    manifest_path = source_dir / "manifest.json"
    manifest_content = json.dumps({"season": args.season, "acquired": args.date, "captures": manifest}, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_content:
        raise ValueError("Existing raw manifest changed")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_content, encoding="utf-8")

    matches_dir.mkdir(parents=True, exist_ok=True)
    paifu_dir.mkdir(parents=True, exist_ok=True)
    # A site's temporary omission or correction must not silently erase prior research data.
    old_matches_path = matches_dir / f"{args.season}.jsonl"
    old_paifu_path = paifu_dir / f"{args.season}.jsonl"
    if old_matches_path.exists():
        old_matches = [json.loads(line) for line in old_matches_path.read_text(encoding="utf-8").splitlines() if line]
        missing = {item["gameId"] for item in old_matches} - {item["gameId"] for item in rows}
        if missing:
            raise ValueError(f"Previously collected matches missing from site: {sorted(missing)}")
    if old_paifu_path.exists():
        old_rounds = [json.loads(line) for line in old_paifu_path.read_text(encoding="utf-8").splitlines() if line]
        new_by_key = {(item["gameId"], item["roundIndex"]): item["paifu"] for item in rounds}
        for item in old_rounds:
            key = (item["gameId"], item["roundIndex"])
            if key not in new_by_key or new_by_key[key] != item["paifu"]:
                raise ValueError(f"Previously collected round missing or changed: {key}")
    write_jsonl(matches_dir / f"{args.season}.jsonl", rows)
    write_jsonl(paifu_dir / f"{args.season}.jsonl", rounds)
    all_rows = []
    for path in sorted(matches_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].jsonl"), reverse=True):
        all_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    write_jsonl(matches_dir / "all.jsonl", all_rows)
    print(f"Saved {len(rows)} matches, {len(rounds)} rounds; all-season index: {len(all_rows)} matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
