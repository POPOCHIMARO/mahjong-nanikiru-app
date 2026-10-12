#!/usr/bin/env python3
"""Compare every published match's current round links with stored paifu.

Keep full dated page captures. Compare round/honba identity before comparing
payloads, so inserting a round on the site cannot shift the old record IDs.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from collect_mleague_paifu import decode_tenhou_link, extract_tenhou_json_links
from ingest_mleague_missing_paifu import Capture, canonical_url, read_jsonl, row_hash
from ingest_mleague_current_season import write_jsonl


def round_identity(paifu: dict) -> tuple[int, int]:
    log = paifu.get("log", [])
    if len(log) != 1:
        raise ValueError("Expected exactly one log per round link")
    round_info = log[0][0]
    return int(round_info[0]), int(round_info[1])


def verify_existing(acquired: str, delay: float, workers: int = 3) -> int:
    main_audit = Path("data/mleague/ingest") / f"{acquired}.json"
    prior = json.loads(main_audit.read_text(encoding="utf-8"))
    target = Path("data/mleague/ingest") / f"{acquired}.round-links.json"
    if target.exists():
        raise ValueError(f"Existing round-link audit is immutable: {target}")
    capture = Capture(Path("raw/sources/mleague/round-link-audit") / acquired, None, acquired)
    matches = read_jsonl(Path("data/mleague/matches/all.jsonl"))
    by_url = {}
    season_rounds = {}
    historical_hashes = {}
    for info in prior["seasons"]:
        season = info["season"]
        rows = read_jsonl(Path("data/mleague/paifu") / f"{season}.jsonl")
        season_rounds[season] = rows
        historical_hashes[season] = {"rows": len(rows), "hash": row_hash(rows)}
        for row in rows:
            by_url.setdefault(canonical_url(row["matchUrl"]), []).append(row)

    def compare(match: dict) -> dict:
        page = capture.get(f"{match['gameId']}.html", match["matchUrl"], delay)
        source = [decode_tenhou_link(link) for link in extract_tenhou_json_links(page)]
        existing = by_url.get(canonical_url(match["matchUrl"]), [])
        old_by_round = {round_identity(row["paifu"]): row for row in existing}
        site_by_round = {round_identity(paifu): (index, paifu) for index, paifu in enumerate(source)}
        if len(old_by_round) != len(existing) or len(site_by_round) != len(source):
            raise ValueError(f"Duplicate round/honba identity: {match['matchUrl']}")
        changes, additional = [], []
        links = extract_tenhou_json_links(page)
        next_index = max((row["roundIndex"] for row in existing), default=-1) + 1
        for identity, (site_index, paifu) in site_by_round.items():
            if set(paifu.get("name", [])) != set(match["players"]):
                raise ValueError(f"Player mismatch: {match['matchUrl']}")
            if identity in old_by_round:
                old = old_by_round[identity]
                if old["paifu"] != paifu:
                    changes.append({"round": list(identity), "storedRoundIndex": old["roundIndex"],
                                    "siteRoundIndex": site_index, "storedPaifuHash": row_hash([old["paifu"]]),
                                    "publishedPaifuHash": row_hash([paifu])})
            else:
                title = paifu.get("title", [])
                additional.append({**match, "roundIndex": next_index,
                                   "sourceRoundIndex": site_index,
                                   "roundName": title[1] if len(title) > 1 else f"round-{site_index + 1}",
                                   "tenhouUrl": links[site_index], "paifu": paifu})
                next_index += 1
        return {"matchUrl": match["matchUrl"], "season": match["season"], "gameId": match["gameId"],
                "storedRounds": len(existing), "publishedRounds": len(source),
                "missingStoredRoundIdentities": [list(key) for key in sorted(set(old_by_round) - set(site_by_round))],
                "changedRounds": changes, "additional": additional,
                "pageSha256": hashlib.sha256(page.encode("utf-8")).hexdigest()}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(compare, match): match for match in matches}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 50 == 0 or result["additional"] or result["changedRounds"]:
                print(f"Verified {len(results)}/{len(matches)} matches; {result['gameId']}: "
                      f"{len(result['additional'])} additional, {len(result['changedRounds'])} changed rounds", flush=True)
    results.sort(key=lambda item: item["matchUrl"])
    manifest = capture.finish()
    additional_by_season = {}
    for result in results:
        additional_by_season.setdefault(result["season"], []).extend(result.pop("additional"))
    for season, new_rows in additional_by_season.items():
        original = season_rounds[season]
        if new_rows:
            write_jsonl(Path("data/mleague/paifu") / f"{season}.jsonl", original + new_rows)
        updated = read_jsonl(Path("data/mleague/paifu") / f"{season}.jsonl")
        assert row_hash(updated[:len(original)]) == historical_hashes[season]["hash"]

    source_manifest = json.loads(Path(manifest["path"]).read_text(encoding="utf-8"))
    for entry in source_manifest["captures"]:
        assert hashlib.sha256((capture.directory / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    summary = {"verifiedMatches": len(results), "storedRoundsBefore": sum(len(rows) for rows in season_rounds.values()),
               "publishedRounds": sum(result["publishedRounds"] for result in results),
               "addedRounds": sum(len(rows) for rows in additional_by_season.values()),
               "changedRounds": sum(len(result["changedRounds"]) for result in results),
               "missingStoredRounds": sum(len(result["missingStoredRoundIdentities"]) for result in results),
               "matchesWithoutPublishedPaifu": sum(result["publishedRounds"] == 0 for result in results),
               "sourceBytes": sum((capture.directory / entry["file"]).stat().st_size for entry in source_manifest["captures"])}
    audit = {"acquired": acquired, "scope": "Every match in all.jsonl, including all previously acquired matches",
             "identity": "matchUrl; round wind/number and honba from paifu.log[0][0]",
             "workers": workers, "delaySecondsPerRequest": delay,
             "manifest": manifest, "summary": summary, "results": results,
             "historicalRowsPreserved": True, "allSourceHashesVerified": True,
             "additionalRoundsBySeason": {season: len(rows) for season, rows in additional_by_season.items()},
             "limits": "Only site-published rounds can be acquired; changed/omitted historical payloads are recorded without replacing stored data."}
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    prior["roundLinkAudit"] = target.as_posix()
    prior["scope"] = "All season/month match lists plus current round links from every previously acquired and newly listed match page"
    prior["limitations"] = ["Only site-published rounds can be acquired; changed historical payloads are preserved and logged as source revisions.",
                            "Site-listed fixtures only; this is not a completeness check against the official M-League schedule."]
    prior["roundLinkVerification"] = summary
    for info in prior["seasons"]:
        added = len(additional_by_season.get(info["season"], []))
        info["afterRounds"] += added
        info["addedRounds"] += added
    prior["totals"]["afterRounds"] += summary["addedRounds"]
    prior["totals"]["addedRounds"] += summary["addedRounds"]
    main_audit.write_text(json.dumps(prior, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary), flush=True)
    print(f"Audit: {target.as_posix()}", flush=True)
    write_corrections(acquired)
    return 0


def write_corrections(acquired: str) -> dict:
    """Create an analysis overlay without replacing the historical JSONL rows."""
    directory = Path("data/mleague/ingest")
    audit_path = directory / f"{acquired}.round-links.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    corrections_path = directory / f"{acquired}.paifu-corrections.jsonl"
    correction_audit_path = directory / f"{acquired}.paifu-corrections.audit.json"
    if corrections_path.exists() or correction_audit_path.exists():
        raise ValueError("Existing correction snapshot is immutable")
    changed_matches = [result for result in audit["results"] if result["changedRounds"]]
    stored_by_key = {}
    for season in {result["season"] for result in changed_matches}:
        for row in read_jsonl(Path("data/mleague/paifu") / f"{season}.jsonl"):
            stored_by_key[(row["gameId"], row["roundIndex"])] = row
    overlays, entries = [], []
    source_dir = Path(audit["manifest"]["path"]).parent
    for result in changed_matches:
        links = extract_tenhou_json_links((source_dir / f"{result['gameId']}.html").read_text(encoding="utf-8"))
        source = {round_identity(paifu): (link, paifu) for link, paifu in ((link, decode_tenhou_link(link)) for link in links)}
        for change in result["changedRounds"]:
            old = stored_by_key[(result["gameId"], change["storedRoundIndex"])]
            link, paifu = source[tuple(change["round"])]
            assert row_hash([old["paifu"]]) == change["storedPaifuHash"]
            assert row_hash([paifu]) == change["publishedPaifuHash"]
            new = {**old, "paifu": paifu, "tenhouUrl": link}
            overlays.append(new)
            old_log, new_log = old["paifu"]["log"][0], paifu["log"][0]
            entries.append({"gameId": old["gameId"], "roundIndex": old["roundIndex"], "matchUrl": old["matchUrl"],
                            "round": change["round"], "oldRowHash": row_hash([old]), "newRowHash": row_hash([new]),
                            "changedPaifuFields": [key for key in sorted(old["paifu"].keys() | paifu.keys()) if old["paifu"].get(key) != paifu.get(key)],
                            "changedLogIndices": [index for index in range(max(len(old_log), len(new_log)))
                                                  if (old_log[index] if index < len(old_log) else None) != (new_log[index] if index < len(new_log) else None)],
                            "uraIndicatorsChanged": old_log[3] != new_log[3],
                            "resultChanged": old_log[-1] != new_log[-1]})
    write_jsonl(corrections_path, overlays)
    summary = {"correctionRows": len(overlays), "correctedMatches": len(changed_matches),
               "uraIndicatorChanges": sum(entry["uraIndicatorsChanged"] for entry in entries),
               "resultChanges": sum(entry["resultChanged"] for entry in entries)}
    correction_audit = {"acquired": acquired, "sourceAudit": audit_path.as_posix(),
                        "snapshot": corrections_path.as_posix(), "hashFormat": "SHA256 of UTF-8 canonical JSON array containing one row (sorted keys, compact separators)",
                        "historicalRowsPreserved": True, "summary": summary, "entries": entries}
    correction_audit_path.write_text(json.dumps(correction_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    main_audit_path = directory / f"{acquired}.json"
    main_audit = json.loads(main_audit_path.read_text(encoding="utf-8"))
    main_audit["correctionsSnapshot"] = corrections_path.as_posix()
    main_audit["correctionsAudit"] = correction_audit_path.as_posix()
    main_audit_path.write_text(json.dumps(main_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary), flush=True)
    return summary
