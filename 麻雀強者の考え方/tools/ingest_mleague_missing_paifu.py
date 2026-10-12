#!/usr/bin/env python3
"""Append uncollected public matches, retaining existing IDs and paifu.

The site no longer publishes the old seasonData object. Match URLs (ignoring a
trailing slash) provide the stable identity across the old and new collectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from collect_mleague_paifu import decode_tenhou_link, extract_tenhou_json_links
from ingest_mleague_current_season import BASE_URL, MATCH_PATH, USER_AGENT, match_list, month_urls, ranking, write_jsonl


def canonical_url(url: str) -> str:
    return url.rstrip("/")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fetch_bytes(url: str, delay: float) -> bytes:
    time.sleep(delay)
    # curl uses the OS trust store; --fail prevents saving an HTTP error as HTML.
    return subprocess.check_output(
        ["curl.exe", "--fail", "--location", "--ssl-no-revoke", "--retry", "2",
         "--max-time", "60", "--silent", "--show-error", "-A", USER_AGENT, url],
        timeout=190,
    )


class Capture:
    def __init__(self, directory: Path, season: str | None, acquired: str):
        self.directory = directory
        self.season = season
        self.acquired = acquired
        self.entries: list[dict] = []

    def get(self, name: str, url: str, delay: float) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        # A resumed acquisition uses its original immutable dated snapshot.
        content = path.read_bytes() if path.exists() else fetch_bytes(url, delay)
        if not path.exists():
            path.write_bytes(content)
        self.entries.append({"file": name, "url": url, "sha256": hashlib.sha256(content).hexdigest()})
        return content.decode("utf-8")

    def finish(self) -> dict:
        result = {"season": self.season, "acquired": self.acquired, "captures": self.entries}
        content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        path = self.directory / "manifest.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_entries = {entry["file"]: entry for entry in previous["captures"]}
            if previous.get("season") != self.season or previous.get("acquired") != self.acquired or any(previous_entries.get(entry["file"]) != entry for entry in self.entries):
                raise ValueError(f"Existing raw manifest changed: {path}")
            # Resume can skip matches already added by this dated acquisition.
            self.entries = previous["captures"]
        if not path.exists():
            path.write_text(content, encoding="utf-8", newline="\n")
        for entry in self.entries:
            assert hashlib.sha256((self.directory / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
        return {"path": path.as_posix(), "captureCount": len(self.entries), "hashesVerified": True}


def duplicate_values(values: list) -> list:
    return [key for key, count in Counter(values).items() if count > 1]


def row_hash(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def collect_season(season: str, acquired: str, delay: float) -> dict:
    matches_path = Path("data/mleague/matches") / f"{season}.jsonl"
    paifu_path = Path("data/mleague/paifu") / f"{season}.jsonl"
    old_matches = read_jsonl(matches_path)
    old_rounds = read_jsonl(paifu_path)
    if duplicate_values([canonical_url(row["matchUrl"]) for row in old_matches]):
        raise ValueError(f"Duplicate existing match URLs in {season}")
    if duplicate_values([(canonical_url(row["matchUrl"]), row["roundIndex"]) for row in old_rounds]):
        raise ValueError(f"Duplicate existing round keys in {season}")
    old_by_url = {canonical_url(row["matchUrl"]): row for row in old_matches}
    old_round_by_url: dict[str, list[dict]] = {}
    for row in old_rounds:
        old_round_by_url.setdefault(canonical_url(row["matchUrl"]), []).append(row)

    capture = Capture(Path("raw/sources/mleague") / season / acquired, season, acquired)
    season_url = f"{BASE_URL}/seasons/{season}/"
    season_html = capture.get("season.html", season_url, delay)
    published_by_url: dict[str, dict] = {}
    months = month_urls(season_html, season)
    if not months:
        raise ValueError(f"No month links for {season}")
    for month_url in months:
        stage, month = month_url.rstrip("/").split("/")[-2:]
        page = capture.get(f"month-{stage}-{month}.html", month_url, delay)
        parsed = match_list(page, season)
        # Any match link missed by the parser must stop the ingest.
        if len(parsed) != len(re.findall(r'<a class="match-link"', page)):
            raise ValueError(f"Unparsed match links: {month_url}")
        for row in parsed:
            row["date"] = f"{month[:4]}-{row['date']}"
            key = canonical_url(row["matchUrl"])
            if key in published_by_url and published_by_url[key] != row:
                raise ValueError(f"Inconsistent repeated match metadata: {key}")
            published_by_url[key] = row
    if not published_by_url:
        raise ValueError(f"No published matches in {season}")

    omitted = sorted(set(old_by_url) - set(published_by_url))
    metadata_changes = []
    for key in sorted(set(old_by_url) & set(published_by_url)):
        old, current = old_by_url[key], published_by_url[key]
        changes = {field: {"existing": old.get(field), "published": current.get(field)}
                   for field in ("season", "stage", "date", "tableId", "matchNumber", "players", "teams")
                   if old.get(field) != current.get(field)}
        if changes:
            metadata_changes.append({"matchUrl": key, "gameId": old["gameId"], "changes": changes})

    added_matches, added_rounds, repaired, unavailable = [], [], [], []
    added_paifu_matches = 0
    for key, current in published_by_url.items():
        # Do not re-fetch complete existing matches. This audit detects listing
        # corrections, but intentionally does not claim a full historical paifu comparison.
        if key in old_by_url and key in old_round_by_url:
            continue
        row = old_by_url.get(key, dict(current))
        if key not in old_by_url:
            path = MATCH_PATH.fullmatch(urlparse(row["matchUrl"]).path)
            assert path is not None
            row["gameId"] = f"{season}_{row['stage']}_{path.group(3)}_{row['tableId']}-{row['matchNumber']}"
            row["gameIdSource"] = "matchUrl (synthetic; site no longer exposes gameId)"
        page = capture.get(f"{row['gameId']}.html", current["matchUrl"], delay)
        if ranking(page) != dict(zip(current["players"], current["teams"])):
            raise ValueError(f"Player/team mismatch: {key}")
        links = extract_tenhou_json_links(page)
        if key not in old_by_url:
            added_matches.append(row)
        if not links:
            unavailable.append({"matchUrl": current["matchUrl"], "gameId": row["gameId"],
                                "reason": "Results page exists but contains no Tenhou paifu links"})
            print(f"Unavailable paifu: {key}", flush=True)
            continue
        for index, link in enumerate(links):
            paifu = decode_tenhou_link(link)
            if set(paifu.get("name", [])) != set(current["players"]) or len(paifu.get("log", [])) != 1:
                raise ValueError(f"Invalid paifu round {index}: {key}")
            title = paifu.get("title", [])
            added_rounds.append({**row, "roundIndex": index,
                                 "roundName": title[1] if len(title) > 1 else f"round-{index + 1}",
                                 "tenhouUrl": link, "paifu": paifu})
        added_paifu_matches += 1
        if key in old_by_url:
            repaired.append(key)
        print(f"Added {row['gameId']}: {len(links)} rounds", flush=True)

    manifest = capture.finish()
    combined_matches = old_matches + added_matches
    combined_rounds = old_rounds + added_rounds
    if duplicate_values([row["gameId"] for row in combined_matches]):
        raise ValueError(f"Duplicate match IDs after ingest: {season}")
    if duplicate_values([(row["gameId"], row["roundIndex"]) for row in combined_rounds]):
        raise ValueError(f"Duplicate round IDs after ingest: {season}")
    if {canonical_url(row["matchUrl"]) for row in combined_rounds} - {canonical_url(row["matchUrl"]) for row in combined_matches}:
        raise ValueError(f"Orphan paifu: {season}")
    if added_matches:
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(matches_path, combined_matches)
    if added_rounds:
        paifu_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(paifu_path, combined_rounds)
    # Prove that the decoded historical rows were not modified or removed.
    assert read_jsonl(matches_path)[:len(old_matches)] == old_matches
    assert read_jsonl(paifu_path)[:len(old_rounds)] == old_rounds
    result = {"season": season, "publishedMatches": len(published_by_url),
              "beforeMatches": len(old_matches), "afterMatches": len(combined_matches),
              "addedMatches": len(added_matches), "beforeRounds": len(old_rounds),
              "afterRounds": len(combined_rounds), "addedRounds": len(added_rounds),
              "addedPaifuMatches": added_paifu_matches, "publishedMatchesWithoutPaifu": unavailable,
              "preservedOmittedMatchUrls": omitted, "publishedMetadataChanges": metadata_changes,
              "repairedMatchesWithoutPaifu": repaired,
              "historicalRowsPreserved": True, "manifest": manifest,
              "addedMatchUrls": [row["matchUrl"] for row in added_matches]}
    print(json.dumps({k: result[k] for k in ("season", "publishedMatches", "beforeMatches", "afterMatches", "addedMatches", "addedRounds")}), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--verify-existing-rounds", action="store_true", help="Audit every stored match page for missing or corrected rounds")
    args = parser.parse_args()
    date.fromisoformat(args.date)
    if args.verify_existing_rounds:
        from verify_mleague_round_links import verify_existing
        return verify_existing(args.date, args.delay)
    audit_path = Path("data/mleague/ingest") / f"{args.date}.json"
    if audit_path.exists():
        raise ValueError(f"Audit already exists; use a new acquisition date: {audit_path}")
    root = Capture(Path("raw/sources/mleague") / "site" / args.date, None, args.date)
    home = root.get("index.html", BASE_URL + "/", args.delay)
    seasons = sorted(set(re.findall(r"/seasons/(\d{4}-\d{2})/?", home)), reverse=True)
    if not seasons:
        raise ValueError("No seasons discovered")
    root_manifest = root.finish()
    # Persist the original counts/row hashes before writing any normalized data.
    # An interrupted acquisition can then resume without losing its before/after audit.
    baseline_path = Path("data/mleague/ingest") / f"{args.date}.baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = {}
        for season in seasons:
            matches = read_jsonl(Path("data/mleague/matches") / f"{season}.jsonl")
            rounds = read_jsonl(Path("data/mleague/paifu") / f"{season}.jsonl")
            baseline[season] = {"matches": len(matches), "rounds": len(rounds),
                                "paifuMatches": len({canonical_url(row["matchUrl"]) for row in rounds}),
                                "matchUrls": [canonical_url(row["matchUrl"]) for row in matches],
                                "matchesHash": row_hash(matches), "roundsHash": row_hash(rounds)}
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    old_index = read_jsonl(Path("data/mleague/matches/all.jsonl"))
    results = [collect_season(season, args.date, args.delay) for season in seasons]
    for result in results:
        season = result["season"]
        initial = baseline[season]
        matches = read_jsonl(Path("data/mleague/matches") / f"{season}.jsonl")
        rounds = read_jsonl(Path("data/mleague/paifu") / f"{season}.jsonl")
        assert row_hash(matches[:initial["matches"]]) == initial["matchesHash"]
        assert row_hash(rounds[:initial["rounds"]]) == initial["roundsHash"]
        result["beforeMatches"] = initial["matches"]
        result["beforeRounds"] = initial["rounds"]
        result["addedMatches"] = len(matches) - initial["matches"]
        result["addedRounds"] = len(rounds) - initial["rounds"]
        result["addedPaifuMatches"] = len({canonical_url(row["matchUrl"]) for row in rounds}) - initial["paifuMatches"]
        result["addedMatchUrls"] = [row["matchUrl"] for row in matches if canonical_url(row["matchUrl"]) not in set(initial["matchUrls"])]
    all_matches = []
    for path in sorted(Path("data/mleague/matches").glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].jsonl"), reverse=True):
        all_matches.extend(read_jsonl(path))
    if duplicate_values([canonical_url(row["matchUrl"]) for row in all_matches]):
        raise ValueError("Duplicate matches in all-season index")
    if all_matches != old_index:
        write_jsonl(Path("data/mleague/matches/all.jsonl"), all_matches)
    totals = {field: sum(item[field] for item in results)
              for field in ("publishedMatches", "beforeMatches", "afterMatches", "addedMatches", "beforeRounds", "afterRounds", "addedRounds", "addedPaifuMatches")}
    audit = {"acquired": args.date, "source": BASE_URL + "/", "identity": "matchUrl with trailing slash removed",
             "scope": "All currently published season/month match lists; fetch matches with no existing paifu",
             "limitations": ["Existing match pages were not re-fetched, so historical paifu corrections and partial-round omissions are not checked.",
                             "Site-listed fixtures only; this is not a completeness check against the official M-League schedule."],
             "siteManifest": root_manifest, "baseline": baseline_path.as_posix(), "seasons": results, "totals": totals,
             "allIndexMatches": len(all_matches),
             "publishedMissingPaifuAfter": sum(len(item["publishedMatchesWithoutPaifu"]) for item in results),
             "historicalRowsPreserved": True, "allManifestsVerified": True}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists():
        raise ValueError(f"Audit already exists; do not overwrite: {audit_path}")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(totals), flush=True)
    print(f"Audit: {audit_path.as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
