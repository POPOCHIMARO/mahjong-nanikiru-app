"""D.2の依存選定用の小規模実験。本番の得点アダプターではない。

固定したwheelをインストールせずに読み込む。ダウンロードは行わず、
実験結果を標準出力へ返すため、同じwheelでオフライン再実行できる。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


WHEEL_SHA256 = "52f63a8dfda5a9e81542642ed315f0e1e8626c0d324c8d8eabb3418ae7d43870"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != WHEEL_SHA256:
        raise ValueError("選定対象と異なるwheel")
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read("mahjong-2.0.0.dist-info/METADATA").decode()
        license_text = archive.read("mahjong-2.0.0.dist-info/licenses/LICENSE.txt").decode()
        runtime_dependencies = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
        source_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in (
                "mahjong/hand_calculating/hand.py",
                "mahjong/hand_calculating/fu.py",
                "mahjong/hand_calculating/hand_config.py",
                "mahjong/hand_calculating/scores.py",
            )
        }
    if runtime_dependencies or "MIT License" not in license_text:
        raise ValueError("依存またはライセンスが選定条件と不一致")
    sys.path.insert(0, str(wheel))
    from mahjong.constants import AKA_DORAS, EAST, SOUTH
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.hand_calculating.scores import ScoresCalculator
    from mahjong.tile import TilesConverter

    def options():
        return OptionalRules(
            has_open_tanyao=True,
            has_aka_dora=True,
            has_double_yakuman=False,
            kazoe_limit=HandConfig.KAZOE_SANBAIMAN,
            kiriage=True,
            fu_for_open_pinfu=True,
            fu_for_pinfu_tsumo=False,
            limit_to_sextuple_yakuman=False,
        )

    checks = []

    def check(name, actual, expected):
        checks.append({"id": name, "actual": actual, "expected": expected, "pass": actual == expected})

    def hand(**suits):
        # 通常5を赤牌用IDへ割り当てないよう、変換側でも赤牌対応を指定する。
        return TilesConverter.string_to_136_array(has_aka_dora=True, **suits)

    check("normal_fives_not_red", len(set(hand(man="5", pin="5", sou="5")) & AKA_DORAS), 0)
    check("red_fives_preserved", len(set(hand(man="0", pin="0", sou="0")) & AKA_DORAS), 3)

    for han, fu, expected in ((4, 30, 8000), (3, 60, 8000), (13, 30, 24000)):
        config = HandConfig(player_wind=SOUTH, round_wind=EAST, options=options())
        score = ScoresCalculator.calculate_scores(han, fu, config, is_yakuman=False)
        check(f"score_{han}han_{fu}fu_child_ron", score["main"], expected)
    config = HandConfig(player_wind=SOUTH, round_wind=EAST, options=options())
    check("two_yakuman_score_multiplier", ScoresCalculator.calculate_scores(26, 0, config, True)["main"], 64000)

    # 連風牌が完成手全体で2枚なら必ず雀頭となり、同牌の刻子役は成立しない。
    # 場風だけを計算用コピーで省く。自風を残すため親子判定は変わらない。
    for wind, honors, expected_cost in ((EAST, "11", 2000), (SOUTH, "22", 1300)):
        tiles = hand(man="111234", pin="567", sou="789", honors=honors)
        win_tile = next(tile for tile in tiles if tile // 4 == 26)
        raw = HandCalculator.estimate_hand_value(
            tiles, win_tile,
            config=HandConfig(is_riichi=True, player_wind=wind, round_wind=wind, options=options()),
        )
        scoring_round_wind = None if sum(tile // 4 == wind for tile in tiles) == 2 else wind
        corrected = HandCalculator.estimate_hand_value(
            tiles, win_tile,
            config=HandConfig(is_riichi=True, player_wind=wind, round_wind=scoring_round_wind, options=options()),
        )
        check(f"double_wind_{wind}_raw", [raw.han, raw.fu], [1, 50])
        check(f"double_wind_{wind}_normalized", [corrected.han, corrected.fu, corrected.cost["main"]], [1, 40, expected_cost])

    tiles = hand(man="123", pin="456", sou="789", honors="11122")
    wind = EAST
    scoring_round_wind = None if sum(tile // 4 == wind for tile in tiles) == 2 else wind
    result = HandCalculator.estimate_hand_value(
        tiles, next(tile for tile in tiles if tile // 4 == 26),
        config=HandConfig(is_riichi=True, player_wind=wind, round_wind=scoring_round_wind, options=options()),
    )
    check("double_wind_triplet_preserved", [result.han, result.fu, result.cost["main"]], [3, 40, 7700])

    tiles = hand(man="1122", pin="3344", sou="5566", honors="11")
    result = HandCalculator.estimate_hand_value(
        tiles, tiles[-1], config=HandConfig(player_wind=SOUTH, round_wind=EAST, options=options()),
    )
    check("seven_pairs", [result.han, result.fu, result.cost["main"]], [2, 25, 1600])
    tiles = hand(man="123456", pin="789", sou="345", honors="11")
    result = HandCalculator.estimate_hand_value(
        tiles, next(tile for tile in tiles if tile // 4 == 22),
        config=HandConfig(player_wind=SOUTH, round_wind=EAST, options=options()),
    )
    check("no_yaku", result.error, "no_yaku")

    tiles = hand(man="19", pin="19", sou="19", honors="11234567")
    result = HandCalculator.estimate_hand_value(
        tiles, tiles[-7], config=HandConfig(player_wind=SOUTH, round_wind=EAST, options=options()),
    )
    check("kokushi_13_wait_single_yakuman", [result.han, result.cost["main"]], [13, 32000])

    # 一発と搶槓の役フラグを同時に与えられることを確認する。
    tiles = hand(man="123456", pin="789", sou="345", honors="11")
    result = HandCalculator.estimate_hand_value(
        tiles, next(tile for tile in tiles if tile // 4 == 22),
        config=HandConfig(is_riichi=True, is_ippatsu=True, is_chankan=True,
                          player_wind=SOUTH, round_wind=EAST, options=options()),
    )
    check("ippatsu_chankan", result.han, 3)
    check("loaded_from_verified_wheel", str(wheel) in sys.modules["mahjong"].__file__, True)
    output = {
        "schemaVersion": "ev-scoring-dependency-probe/v1",
        "status": "pass" if all(item["pass"] for item in checks) else "fail",
        "scope": "dependency_selection_only_not_D2_completion",
        "package": "mahjong", "version": "2.0.0", "wheelSha256": digest,
        "python": sys.version.split()[0], "runtimeDependencies": runtime_dependencies,
        "license": "MIT", "sourceSha256": source_hashes,
        "checks": checks,
        "unverified": ["complete_scoring_conformance", "transitions", "pao", "historical_rules", "replay", "throughput"],
    }
    if args.output:
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
