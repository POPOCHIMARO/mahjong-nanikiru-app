"""既存Mリーグ観測データから暗槓の判断機会を集計する。アプリは変更しない。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration/.dependency-cache/mahjong-2.0.0-py3-none-any.whl"))
from tools.ev_policy_opponent import RoundFeatureState
from tools.ev_calibration_state import shanten


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def notation(counts):
    return "".join("".join(str(i + 1) * n for i, n in enumerate(counts[b:e])) + suit
                   for b, e, suit in [(0, 9, "m"), (9, 18, "p"), (18, 27, "s"), (27, 34, "z")]
                   if any(counts[b:e]))


def run(max_rounds=None, output=None):
    start = time.monotonic()
    folder = ROOT / "calibration/dataset-opponent"
    out = Path(output or Path(__file__).parent / "full")
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((folder / "extraction-summary.json").read_text(encoding="utf-8"))
    reservation = json.loads((ROOT / "calibration/future-evaluation-reservation.json").read_text(encoding="utf-8"))
    allowed = set(reservation["development"]["allowedSeasons"])
    assert set(summary["input"]["selectedSeasons"]) <= allowed
    hashes = {}
    for f in summary["generatedFiles"]["files"]:
        if f["path"] in {"teacher-windows.jsonl.gz", "public-events.jsonl.gz", "private-events.jsonl.gz"}:
            actual = digest(folder / f["path"])
            assert actual == f["sha256"], f["path"]
            hashes[f["path"]] = actual
    # 原牌譜も保存済みmanifestと照合し、レコード数と期間ラベルを残す。
    source_files = []
    for f in summary["input"]["files"]:
        path = ROOT / "麻雀強者の考え方" / f["path"]
        assert digest(path) == f["sha256"], str(path)
        with open(path, encoding="utf-8") as source:
            records = sum(bool(line.strip()) for line in source)
        source_files.append({**f, "records": records})

    public_iter = iter(read_rows(folder / "public-events.jsonl.gz"))
    private_iter = iter(itertools.groupby(read_rows(folder / "private-events.jsonl.gz"), key=lambda r: r["roundId"]))
    public = None
    totals = Counter()
    seasons = defaultdict(Counter)
    strata = defaultdict(Counter)
    sets = defaultdict(set)
    choice_by_episode = defaultdict(list)
    sample_checks = []
    opportunity_path = out / "opportunities.jsonl"
    app_path = out / "app-structural-hands.jsonl.gz"

    def bump(key, season, value=1):
        totals[key] += value
        seasons[season][key] += value

    with open(opportunity_path, "w", encoding="utf-8") as opportunities, gzip.open(app_path, "wt", encoding="utf-8") as app_hands:
        for rid, group in itertools.groupby(read_rows(folder / "teacher-windows.jsonl.gz"), key=lambda r: r["roundId"]):
            if max_rounds is not None and totals["rounds"] >= max_rounds:
                break
            windows = list(group)
            while public is None or public["roundId"] != rid:
                public = next(public_iter)
                prid, private_group = next(private_iter)
                private_values = list(private_group)
                assert prid == public["roundId"]
            private = {r["seat"]: r for r in private_values}
            assert set(private) == {0, 1, 2, 3}
            state = RoundFeatureState(public, private)
            season = windows[0]["source"]["season"]
            assert season in allowed
            bump("rounds", season)
            bump("seat_rounds", season, 4)
            sets["games"].add(":".join(rid.split(":")[:3]))
            round_observed_ankan = 0
            last_covered_raw = max(w["rawEventIndex"] for w in windows)
            # 最後の教師窓より後のイベントは、1イベント隣接していても含めない。
            covered_public_ankan = sum(e["type"] == "ankan" and e.get("rawEventIndex", 10**9) <= last_covered_raw for e in public["events"])
            for window in windows:
                bump("teacher_windows", season)
                actions = window.get("legalActions")
                if actions is None:
                    continue
                state.advance(window["publicEventCount"])
                seat = window["actorSeat"]
                phase = window["phase"]
                bump("self_windows", season)
                bump(phase, season)
                sets["observed_seat_rounds"].add((rid, seat))
                counts = [0] * 34
                for tile in state.hands[seat]:
                    counts[tile["tile34"]] += 1
                fixed = len(state.melds[seat])
                assert sum(counts) == 14 - 3 * fixed, window["windowId"]
                assert max(counts) <= 4
                own_riichi = seat in state.riichi
                opponent_riichi = bool(state.riichi - {seat})
                kans = sorted({a["tile34"] for a in actions if a["kind"] == "ankan"})
                obs = window["observation"]
                choice = (obs.get("action") or {}).get("kind", "unknown")
                if choice == "ankan":
                    assert obs["status"] == "exact" and obs["action"]["tile34"] in kans
                    round_observed_ankan += 1
                    bump("observed_ankan", season)
                tsumo = any(a["kind"] == "tsumo" for a in actions)
                is_live = phase == "self_action_after_live"
                plain = is_live and not fixed and not own_riichi
                if plain:
                    bump("plain_live_windows", season)
                before = None
                after_shanten = None
                base_shanten = None
                # 選択結果で絞らず、実ツモ前と選択前の手牌で2→1を判定する。
                if plain:
                    draw = state.draws[(seat, window["rawEventIndex"])]
                    before = counts.copy()
                    before[draw["tile34"]] -= 1
                    assert min(before) >= 0 and sum(before) == 13
                    base_shanten = shanten(tuple(before), 0)
                    if base_shanten == 2:
                        after_shanten = shanten(tuple(counts), 0)
                app_structure = plain and base_shanten == 2 and after_shanten == 1
                no_honor_candidate = False
                if app_structure:
                    bump("app_structural_windows", season)
                    if not opponent_riichi:
                        bump("app_no_opponent_riichi_windows", season)
                    honor_candidates = []
                    for t in range(27, 34):
                        if counts[t]:
                            work = counts.copy()
                            work[t] -= 1
                            if shanten(tuple(work), 0) == 1:
                                honor_candidates.append(t)
                    no_honor_candidate = not honor_candidates
                    if no_honor_candidate:
                        bump("app_no_honor_candidate_windows", season)
                    if any(n == 4 for n in counts):
                        bump("app_quad_windows", season)
                    app_record = {"windowId": window["windowId"], "counts": counts, "draw": draw["tile34"], "season": season,
                                  "legalAnkan": kans, "honorCandidates": honor_candidates, "opponentRiichi": opponent_riichi}
                    app_hands.write(json.dumps(app_record, ensure_ascii=False) + "\n")
                    # 14枚のシャンテンと全打牌の最小値が一致するか、固定標本で検査。
                    if len(sample_checks) < 40 or (kans and len(sample_checks) < 80):
                        minimum = 99
                        for t, n in enumerate(counts):
                            if n:
                                work = counts.copy()
                                work[t] -= 1
                                minimum = min(minimum, shanten(tuple(work), 0))
                        assert minimum == after_shanten
                        sample_checks.append({"windowId": window["windowId"], "minimumDiscardShanten": minimum})
                if not kans:
                    continue
                bump("ankan_windows", season)
                bump("ankan_observation_" + obs["status"], season)
                bump("ankan_win_action_" + window["winActionStatus"], season)
                bump("ankan_candidates", season, len(kans))
                sets["ankan_seat_rounds"].add((rid, seat))
                sets["ankan_rounds"].add(rid)
                sets["ankan_games"].add(":".join(rid.split(":")[:3]))
                if tsumo:
                    bump("ankan_with_tsumo", season)
                else:
                    bump("ankan_without_tsumo", season)
                if plain:
                    bump("plain_live_ankan", season)
                if app_structure:
                    bump("app_structural_ankan", season)
                    if not opponent_riichi:
                        bump("app_no_opponent_riichi_ankan", season)
                    if no_honor_candidate:
                        bump("app_no_honor_candidate_ankan", season)
                current_shanten = shanten(tuple(counts), fixed)
                segments = [phase, "own_riichi" if own_riichi else "not_own_riichi",
                            "opponent_riichi" if opponent_riichi else "no_opponent_riichi",
                            "no_fixed_meld" if fixed == 0 else "fixed_meld_present",
                            f"shanten_{current_shanten}"]
                if plain: segments.append("plain_live")
                if app_structure: segments.append("app_structural")
                if app_structure and no_honor_candidate: segments.append("app_no_honor_candidate")
                for label in segments:
                    strata[label]["windows"] += 1
                    strata[label][choice] += 1
                    strata[label]["tsumo_available"] += tsumo
                connections = {}
                for t in kans:
                    if t >= 27:
                        connections[str(t)] = "honor"
                    else:
                        b = t // 9 * 9
                        linked = any(counts[u] for u in range(max(b, t - 2), min(b + 9, t + 3)) if u != t)
                        connections[str(t)] = "connected_number" if linked else "unconnected_number"
                    episode = (rid, seat, t)
                    choice_by_episode[episode].append(choice == "ankan" and obs["action"]["tile34"] == t)
                    sets["opportunity_episodes"].add(episode)
                    if app_structure: sets["app_episodes"].add(episode)
                    strata["tile_" + connections[str(t)]]["candidate_windows"] += 1
                    strata["tile_" + connections[str(t)]]["chosen"] += choice == "ankan" and obs["action"]["tile34"] == t
                record = {"windowId": window["windowId"], "roundId": rid, "seat": seat, "source": window["source"],
                          "rawEventIndex": window["rawEventIndex"], "publicEventCount": window["publicEventCount"],
                          "phase": phase, "hand": notation(counts), "counts": counts,
                          "draw": state.draws.get((seat, window["rawEventIndex"])), "kans": kans, "connections": connections,
                          "choice": obs, "ownRiichi": own_riichi, "opponentRiichi": opponent_riichi,
                          "fixedMelds": fixed, "shanten": current_shanten, "tsumoAvailable": tsumo,
                          "turn": len(state.rivers[seat]) + 1, "appStructural": app_structure,
                          "appNoHonorCandidate": app_structure and no_honor_candidate}
                opportunities.write(json.dumps(record, ensure_ascii=False) + "\n")
            bump("covered_public_ankan", season, covered_public_ankan)
            if round_observed_ankan != covered_public_ankan:
                bump("public_ankan_count_mismatch_rounds", season)
            if totals["rounds"] % 1000 == 0:
                print(json.dumps({"rounds": totals["rounds"], "self": totals["self_windows"], "ankan": totals["ankan_windows"],
                                  "app": totals["app_structural_windows"], "elapsedSeconds": round(time.monotonic() - start, 1)}), flush=True)
    for name, values in sets.items():
        totals["unique_" + name] = len(values)
    episode_choices = Counter()
    for choices in choice_by_episode.values():
        episode_choices["episodes"] += 1
        episode_choices["first_chosen"] += choices[0]
        episode_choices["ever_chosen"] += any(choices)
        episode_choices["deferred_then_chosen"] += not choices[0] and any(choices)
        episode_choices["never_chosen_in_observed_prefix"] += not any(choices)
    result = {"schemaVersion": "ankan-opportunity-analysis/v1", "executedAt": datetime.now(timezone.utc).isoformat(),
              "maximumRounds": max_rounds, "totals": dict(totals), "seasons": dict(seasons), "strata": dict(strata),
              "episodeChoices": dict(episode_choices), "sourceFiles": source_files, "datasetHashes": hashes,
              "datasetTotals": summary["totals"], "datasetRejectionReasons": summary["rejectionReasons"],
              "discardShantenChecks": sample_checks, "analysisCodeSha256": digest(Path(__file__)),
              "runtimeSeconds": time.monotonic() - start}
    if max_rounds is None:
        assert totals["teacher_windows"] == 2085155
    assert totals["public_ankan_count_mismatch_rounds"] == 0
    assert totals["observed_ankan"] <= totals["ankan_windows"] <= totals["self_windows"]
    assert totals["app_no_honor_candidate_ankan"] <= totals["app_structural_ankan"] <= totals["ankan_windows"]
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": dict(totals), "seconds": result["runtimeSeconds"]}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.max_rounds, args.output)
