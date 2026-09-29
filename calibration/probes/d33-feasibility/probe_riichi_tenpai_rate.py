"""D.3.3設計用の読み取り専用probe：見えない牌から13枚を一様に引いたときのテンパイ率。

各判断時点（inference prefix）で、対象家から見えない牌のプールから13枚を一様に引き、
シャンテン0（テンパイ）になる割合を数える。
これは「一様13枚抽出を初期化や提案に使うと非効率」という事実だけを示す。
観測で条件付けた基準SMCの生存率の上限ではない（PHASE_D33_DESIGN.md 3節）。

実行（プロジェクト直下で）：
    python calibration/probes/d33-feasibility/probe_riichi_tenpai_rate.py
出力：同じフォルダの riichi-tenpai-rate.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# このファイルは <root>/calibration/probes/d33-feasibility/ にある。parents[3] がプロジェクト直下。
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.ev_calibration_state import shanten  # noqa: E402
from tools.ev_policy_opponent import RoundFeatureState  # noqa: E402

DATASET = ROOT / "calibration" / "dataset-opponent-v3"
INPUTS = ("inference-prefixes.jsonl.gz", "public-events.jsonl.gz", "private-events.jsonl.gz")
SAMPLE_PREFIXES = 200
HANDS_PER_PREFIX = 4000
SEED = 20260928


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def information_pool(public: dict, own_private: dict, seat: int, public_event_count: int) -> Counter:
    """公開履歴と対象家自身の私有履歴だけから、見えない牌の残数を数える（教師用の他家手牌を使わない経路）。"""
    draws = {
        int(event["rawEventIndex"]): int(event["tile"]["tile34"])
        for event in own_private["events"]
        if event["type"] == "draw_observation"
    }
    hand = Counter(int(tile["tile34"]) for tile in own_private["initialHand"])
    visible = Counter({int(public["initial"]["doraIndicator"]["tile34"]): 1})
    for event in public["events"][:public_event_count]:
        kind = event["type"]
        if kind == "draw" and int(event["seat"]) == seat:
            hand[draws[int(event["rawEventIndex"])]] += 1
        elif kind == "discard":
            tile34 = int(event["tile"]["tile34"])
            visible[tile34] += 1
            if int(event["seat"]) == seat:
                hand[tile34] -= 1
        elif kind not in {"draw", "response_resolution"}:
            # 入口条件（副露と槓なし）の外。ここに来たら対象の抽出が誤っている。
            raise ValueError(f"入口条件外のイベント: {kind}")
    seen = visible + hand
    return Counter({t: 4 - seen[t] for t in range(34) if 4 - seen[t] > 0})


def main() -> None:
    rng = random.Random(SEED)
    prefixes = []
    with gzip.open(DATASET / "inference-prefixes.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["developmentSplit"] == "train":
                prefixes.append(row)
    rng.shuffle(prefixes)
    prefixes = prefixes[:SAMPLE_PREFIXES]
    wanted = {row["roundId"] for row in prefixes}
    public, private = {}, defaultdict(dict)
    with gzip.open(DATASET / "public-events.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["roundId"] in wanted:
                public[row["roundId"]] = row
    with gzip.open(DATASET / "private-events.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["roundId"] in wanted:
                private[row["roundId"]][int(row["seat"])] = row

    decisions = []
    teacher_states = []
    for prefix in prefixes:
        seat = int(prefix["seat"])
        count = int(prefix["publicEventCount"])
        round_public = public[prefix["roundId"]]
        # 推定に使う経路：公開履歴と対象家の私有履歴だけでプールを作る。
        pool_counts = information_pool(round_public, private[prefix["roundId"]][seat], seat, count)
        state = RoundFeatureState(round_public, private[prefix["roundId"]])
        state.advance(count)
        riichi = sorted(state.riichi - {seat})
        if len(riichi) != 1:
            continue
        # 検査：実行器側（全家の私有履歴を持つ）で数えたプールと一致すること。
        seen = Counter(state.visible)
        for tile in state.hands[seat]:
            seen[int(tile["tile34"])] += 1
        assert pool_counts == Counter({t: 4 - seen[t] for t in range(34) if 4 - seen[t] > 0})
        pool = [t for t in range(34) for _ in range(pool_counts[t])]
        hits = 0
        for _ in range(HANDS_PER_PREFIX):
            hand = rng.sample(pool, 13)
            counts = Counter(hand)
            if shanten(tuple(counts[i] for i in range(34)), 0) == 0:
                hits += 1
        decisions.append({"decisionId": prefix["decisionId"], "tenpaiHits": hits})
        teacher_states.append((state, riichi[0]))

    # 参考：実際のリーチ者の手（教師情報）がテンパイか。上の抽出とは別の段で数え、推定には使わない。
    true_tenpai = 0
    for state, riichi_seat in teacher_states:
        actual = Counter(int(t["tile34"]) for t in state.hands[riichi_seat])
        true_tenpai += int(shanten(tuple(actual[i] for i in range(34)), 0) == 0)

    rates = sorted(row["tenpaiHits"] / HANDS_PER_PREFIX for row in decisions)
    n = len(rates)
    summary = {
        "seed": SEED,
        "inputSha256": {name: sha256_file(DATASET / name) for name in INPUTS},
        "prefixes": n,
        "handsPerPrefix": HANDS_PER_PREFIX,
        "meanTenpaiRate": sum(rates) / n,
        "median": rates[n // 2],
        "p90": rates[int(n * 0.9)],
        "max": rates[-1],
        "decisionsWithZeroHits": sum(row["tenpaiHits"] == 0 for row in decisions),
        "interpretation": "一様13枚抽出のテンパイ率。観測で条件付けた基準SMCの生存率の上限ではない",
        "actualRiichiHandTenpaiCheck": f"{true_tenpai}/{n}",
        "decisions": decisions,
    }
    out = Path(__file__).with_name("riichi-tenpai-rate.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "decisions"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
