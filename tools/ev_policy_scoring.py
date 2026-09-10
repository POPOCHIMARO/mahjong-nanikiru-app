"""Mリーグ用の和了点計算と局末精算。

第三者ライブラリには役、符、本場なしの基本支払いだけを任せる。
本場、供託、責任払いはこのモジュールで一度だけ適用する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any, Literal, Sequence

from tools.ev_policy_state import TileInstance


RULE_PROFILE_ID = "mleague-round-v2"
SCORING_ADAPTER_ID = "mleague-mahjong-2.0.0-v1"
SCORING_PACKAGE = "mahjong"
SCORING_PACKAGE_VERSION = "2.0.0"

_RED_TILE34 = frozenset({4, 13, 22})
_RED_LIBRARY_ID = {4: 16, 13: 52, 22: 88}
_MELD_LENGTH = {"chi": 3, "pon": 3, "kan": 4, "shouminkan": 4}
_PAO_YAKU = frozenset({"Daisangen", "Daisuushi", "Suukantsu"})


class ScoringDependencyError(RuntimeError):
    """固定した得点計算依存を読み込めない。"""


class HandScoringError(ValueError):
    """和了入力が不正または得点を確定できない。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class MultiplePaoHonbaUnresolved(ValueError):
    """複数責任者への本場配分を公式根拠なしに推測できない。"""


@dataclass(frozen=True)
class MeldInput:
    """得点計算へ渡す副露または暗槓。牌は完成手内の物理IDで参照する。"""

    kind: Literal["chi", "pon", "kan", "shouminkan"]
    tile_ids: tuple[int, ...]
    opened: bool = True
    called_tile_id: int | None = None


@dataclass(frozen=True)
class HandFlags:
    is_tsumo: bool = False
    is_riichi: bool = False
    is_double_riichi: bool = False
    is_ippatsu: bool = False
    is_rinshan: bool = False
    is_chankan: bool = False
    is_haitei: bool = False
    is_houtei: bool = False
    is_tenhou: bool = False
    is_chiihou: bool = False


@dataclass(frozen=True)
class HandScoreRequest:
    """完成手。tilesは和了牌と副露の牌を含み、各物理牌を一度だけ持つ。"""

    tiles: tuple[TileInstance, ...]
    win_tile_id: int
    seat_wind_tile34: int
    round_wind_tile34: int
    melds: tuple[MeldInput, ...] = ()
    dora_indicators: tuple[TileInstance, ...] = ()
    ura_dora_indicators: tuple[TileInstance, ...] = ()
    flags: HandFlags = HandFlags()


@dataclass(frozen=True)
class YakuResult:
    name: str
    han: int
    is_yakuman: bool


@dataclass(frozen=True)
class FuResult:
    reason: str
    fu: int


@dataclass(frozen=True)
class BaseScore:
    """本場と供託を含まない、ライブラリで検証済みの基本支払い。"""

    han: int
    fu: int
    main: int
    additional: int
    total: int
    yaku_level: str
    is_tsumo: bool
    is_dealer: bool
    yakuman_multiplier: int = 0


@dataclass(frozen=True)
class HandScore:
    rule_profile_id: str
    adapter_id: str
    base: BaseScore
    yaku: tuple[YakuResult, ...]
    fu_details: tuple[FuResult, ...]
    is_open_hand: bool
    actual_round_wind_tile34: int
    scoring_round_wind_tile34: int | None
    double_wind_pair_normalized: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaoMeld:
    """責任発生の判定に必要な、成立済み面子の事実。"""

    kind: Literal["pon", "daiminkan", "ankan", "shouminkan"]
    tile34: int
    from_seat: int | None = None


@dataclass(frozen=True)
class PaoResponsibility:
    yaku: Literal["Daisangen", "Daisuushi", "Suukantsu"]
    liable_seat: int
    yakuman_multiplier: int = 1


@dataclass(frozen=True)
class Transfer:
    from_seat: int | None
    to_seat: int | None
    points: int
    reason: str


@dataclass(frozen=True)
class Settlement:
    deltas: tuple[int, int, int, int]
    kyoutaku_before: int
    kyoutaku_after: int
    transfers: tuple[Transfer, ...]

    def validate(self) -> None:
        if sum(self.deltas) + 1000 * (self.kyoutaku_after - self.kyoutaku_before) != 0:
            raise ValueError("点数と供託の保存則に違反")
        if any(item.points <= 0 for item in self.transfers):
            raise ValueError("精算移動は正の点数が必要")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _mahjong_api() -> dict[str, Any]:
    try:
        import mahjong
        from mahjong.constants import EAST, SOUTH, WEST, NORTH
        from mahjong.hand_calculating.hand import HandCalculator
        from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
        from mahjong.hand_calculating.scores import ScoresCalculator
        from mahjong.meld import Meld
    except ImportError as error:
        raise ScoringDependencyError(
            "mahjong==2.0.0が必要。calibration/requirements-scoring.txtから導入する"
        ) from error
    try:
        installed_version = package_version(SCORING_PACKAGE)
    except PackageNotFoundError as error:
        raise ScoringDependencyError("mahjongの配布メタデータを確認できない") from error
    if installed_version != SCORING_PACKAGE_VERSION:
        raise ScoringDependencyError(f"mahjongの版が不一致: {installed_version}")
    return {
        "EAST": EAST,
        "SOUTH": SOUTH,
        "WEST": WEST,
        "NORTH": NORTH,
        "HandCalculator": HandCalculator,
        "HandConfig": HandConfig,
        "OptionalRules": OptionalRules,
        "ScoresCalculator": ScoresCalculator,
        "Meld": Meld,
    }


def _options() -> Any:
    api = _mahjong_api()
    config = api["HandConfig"]
    return api["OptionalRules"](
        has_open_tanyao=True,
        has_aka_dora=True,
        has_double_yakuman=False,
        kazoe_limit=config.KAZOE_SANBAIMAN,
        kiriage=True,
        fu_for_open_pinfu=True,
        fu_for_pinfu_tsumo=False,
        renhou_as_yakuman=False,
        has_daisharin=False,
        has_daisharin_other_suits=False,
        has_sashikomi_yakuman=False,
        limit_to_sextuple_yakuman=False,
        paarenchan_needs_yaku=True,
        has_daichisei=False,
    )


def _validate_seat(seat: int, name: str) -> None:
    if seat not in range(4):
        raise ValueError(f"{name}は0〜3が必要")


def _library_tile_map(groups: Sequence[Sequence[TileInstance]]) -> dict[int, int]:
    physical: dict[int, TileInstance] = {}
    for group in groups:
        for tile in group:
            if tile.tile_id in physical and physical[tile.tile_id] != tile:
                raise HandScoringError("conflicting_tile_id")
            physical[tile.tile_id] = tile
    by_type: dict[int, list[TileInstance]] = {}
    for tile in physical.values():
        if tile.tile34 not in range(34):
            raise HandScoringError("tile34_out_of_range")
        if tile.is_red and tile.tile34 not in _RED_TILE34:
            raise HandScoringError("red_non_five")
        by_type.setdefault(tile.tile34, []).append(tile)
    result: dict[int, int] = {}
    for tile34, items in by_type.items():
        if len(items) > 4 or sum(tile.is_red for tile in items) > 1:
            raise HandScoringError("tile_count_over_four")
        available = list(range(tile34 * 4, tile34 * 4 + 4))
        red_id = _RED_LIBRARY_ID.get(tile34)
        red_tiles = [tile for tile in items if tile.is_red]
        # 通常5だけの部分集合でも、赤牌用IDを通常牌へ割り当てない。
        if red_id is not None:
            available.remove(red_id)
            if sum(not tile.is_red for tile in items) > len(available):
                raise HandScoringError("red_five_composition_invalid")
        if red_tiles:
            assert red_id is not None
            result[red_tiles[0].tile_id] = red_id
        for tile, library_id in zip(sorted((tile for tile in items if not tile.is_red), key=lambda x: x.tile_id), available):
            result[tile.tile_id] = library_id
    return result


def to_mahjong_136(tiles: Sequence[TileInstance]) -> tuple[int, ...]:
    """Dの物理牌をmahjong 2.0.0の136形式へ決定的に変換する。"""
    mapping = _library_tile_map((tiles,))
    return tuple(mapping[tile.tile_id] for tile in tiles)


def _validate_flags(flags: HandFlags) -> None:
    if flags.is_riichi and flags.is_double_riichi:
        raise HandScoringError("riichi_flags_conflict")
    if flags.is_ippatsu and not (flags.is_riichi or flags.is_double_riichi):
        raise HandScoringError("ippatsu_without_riichi")
    if flags.is_rinshan and not flags.is_tsumo:
        raise HandScoringError("rinshan_requires_tsumo")
    if flags.is_chankan and flags.is_tsumo:
        raise HandScoringError("chankan_requires_ron")
    if flags.is_haitei and not flags.is_tsumo:
        raise HandScoringError("haitei_requires_tsumo")
    if flags.is_houtei and flags.is_tsumo:
        raise HandScoringError("houtei_requires_ron")
    if flags.is_rinshan and flags.is_haitei:
        raise HandScoringError("rinshan_and_haitei_conflict")
    if flags.is_chankan and flags.is_houtei:
        raise HandScoringError("chankan_and_houtei_conflict")
    if flags.is_tenhou and (not flags.is_tsumo or flags.is_chiihou):
        raise HandScoringError("invalid_tenhou")
    if flags.is_chiihou and (not flags.is_tsumo or flags.is_tenhou):
        raise HandScoringError("invalid_chiihou")


def _validate_meld(meld: MeldInput, tiles_by_id: dict[int, TileInstance]) -> None:
    if meld.kind not in _MELD_LENGTH or len(meld.tile_ids) != _MELD_LENGTH[meld.kind]:
        raise HandScoringError("invalid_meld_length")
    if len(set(meld.tile_ids)) != len(meld.tile_ids) or any(tile_id not in tiles_by_id for tile_id in meld.tile_ids):
        raise HandScoringError("invalid_meld_tile_reference")
    kinds = sorted(tiles_by_id[tile_id].tile34 for tile_id in meld.tile_ids)
    if meld.kind == "chi":
        if kinds[0] >= 27 or kinds[0] // 9 != kinds[-1] // 9 or kinds != list(range(kinds[0], kinds[0] + 3)):
            raise HandScoringError("invalid_chi")
    elif len(set(kinds)) != 1:
        raise HandScoringError("invalid_pon_or_kan")
    if meld.opened and meld.called_tile_id not in meld.tile_ids:
        raise HandScoringError("open_meld_requires_called_tile")
    if not meld.opened and (meld.kind != "kan" or meld.called_tile_id is not None):
        raise HandScoringError("invalid_closed_meld")


def _base_score_from_cost(
    *, han: int, fu: int, cost: dict[str, Any], is_tsumo: bool, is_dealer: bool, yakuman_multiplier: int = 0
) -> BaseScore:
    if any(int(cost[key]) != 0 for key in ("main_bonus", "additional_bonus", "kyoutaku_bonus")):
        raise HandScoringError("library_bonus_must_be_zero")
    expected_total = int(cost["main"]) + (2 * int(cost["additional"]) if is_tsumo else 0)
    if int(cost["total"]) != expected_total:
        raise HandScoringError("library_total_mismatch")
    return BaseScore(
        han=han,
        fu=fu,
        main=int(cost["main"]),
        additional=int(cost["additional"]),
        total=expected_total,
        yaku_level=str(cost["yaku_level"]),
        is_tsumo=is_tsumo,
        is_dealer=is_dealer,
        yakuman_multiplier=yakuman_multiplier,
    )


def calculate_base_score(*, han: int, fu: int, is_tsumo: bool, is_dealer: bool, is_yakuman: bool = False) -> BaseScore:
    """公式得点表との固定例に使う、本場なしの基本支払いを返す。"""
    if han <= 0 or fu < 0 or (not is_yakuman and fu == 0):
        raise HandScoringError("invalid_han_fu")
    if is_yakuman and (han < 13 or han % 13 != 0):
        raise HandScoringError("invalid_yakuman_han")
    api = _mahjong_api()
    config = api["HandConfig"](
        is_tsumo=is_tsumo,
        player_wind=api["EAST"] if is_dealer else api["SOUTH"],
        tsumi_number=0,
        kyoutaku_number=0,
        options=_options(),
    )
    cost = api["ScoresCalculator"].calculate_scores(han, fu, config, is_yakuman)
    return _base_score_from_cost(
        han=han,
        fu=fu,
        cost=cost,
        is_tsumo=is_tsumo,
        is_dealer=is_dealer,
        yakuman_multiplier=han // 13 if is_yakuman else 0,
    )


def score_hand(request: HandScoreRequest) -> HandScore:
    """完成手をMリーグ設定で計算し、精算前の基本支払いを返す。"""
    _validate_flags(request.flags)
    if request.seat_wind_tile34 not in range(27, 31) or request.round_wind_tile34 not in range(27, 31):
        raise HandScoringError("invalid_wind")
    tiles_by_id = {tile.tile_id: tile for tile in request.tiles}
    if len(tiles_by_id) != len(request.tiles):
        raise HandScoringError("duplicate_hand_tile_id")
    if request.win_tile_id not in tiles_by_id:
        raise HandScoringError("winning_tile_missing")
    kan_count = sum(meld.kind in ("kan", "shouminkan") for meld in request.melds)
    if len(request.tiles) != 14 + kan_count:
        raise HandScoringError("invalid_complete_hand_size")
    meld_tile_ids = [tile_id for meld in request.melds for tile_id in meld.tile_ids]
    if len(set(meld_tile_ids)) != len(meld_tile_ids):
        raise HandScoringError("meld_tile_used_twice")
    if request.win_tile_id in meld_tile_ids:
        raise HandScoringError("winning_tile_in_meld")
    for meld in request.melds:
        _validate_meld(meld, tiles_by_id)
    dora_ids = [tile.tile_id for tile in request.dora_indicators]
    ura_ids = [tile.tile_id for tile in request.ura_dora_indicators]
    if len(set(dora_ids + ura_ids)) != len(dora_ids) + len(ura_ids):
        raise HandScoringError("indicator_tile_used_twice")
    if set(tiles_by_id).intersection(dora_ids + ura_ids):
        raise HandScoringError("indicator_tile_in_hand")
    if request.ura_dora_indicators and not (request.flags.is_riichi or request.flags.is_double_riichi):
        raise HandScoringError("ura_dora_without_riichi")
    mapping = _library_tile_map((request.tiles, request.dora_indicators, request.ura_dora_indicators))
    library_tiles = [mapping[tile.tile_id] for tile in request.tiles]
    api = _mahjong_api()
    library_melds = [
        api["Meld"](
            meld_type=getattr(api["Meld"], meld.kind.upper()),
            tiles=[mapping[tile_id] for tile_id in meld.tile_ids],
            opened=meld.opened,
            called_tile=mapping[meld.called_tile_id] if meld.called_tile_id is not None else None,
        )
        for meld in request.melds
    ]
    double_wind_pair = (
        request.seat_wind_tile34 == request.round_wind_tile34
        and sum(tile.tile34 == request.round_wind_tile34 for tile in request.tiles) == 2
    )
    scoring_round_wind = None if double_wind_pair else request.round_wind_tile34
    flags = request.flags
    config = api["HandConfig"](
        is_tsumo=flags.is_tsumo,
        is_riichi=flags.is_riichi,
        is_ippatsu=flags.is_ippatsu,
        is_rinshan=flags.is_rinshan,
        is_chankan=flags.is_chankan,
        is_haitei=flags.is_haitei,
        is_houtei=flags.is_houtei,
        is_daburu_riichi=flags.is_double_riichi,
        is_nagashi_mangan=False,
        is_tenhou=flags.is_tenhou,
        is_renhou=False,
        is_chiihou=flags.is_chiihou,
        is_open_riichi=False,
        player_wind=request.seat_wind_tile34,
        round_wind=scoring_round_wind,
        kyoutaku_number=0,
        tsumi_number=0,
        paarenchan=0,
        options=_options(),
    )
    response = api["HandCalculator"].estimate_hand_value(
        library_tiles,
        mapping[request.win_tile_id],
        melds=library_melds,
        dora_indicators=[mapping[tile.tile_id] for tile in request.dora_indicators],
        ura_dora_indicators=[mapping[tile.tile_id] for tile in request.ura_dora_indicators],
        config=config,
    )
    if response.error:
        raise HandScoringError(str(response.error))
    if response.cost is None or response.han is None or response.fu is None or response.yaku is None:
        raise HandScoringError("incomplete_library_response")
    yaku = tuple(
        YakuResult(
            name=str(item.name),
            han=int(item.han_open if response.is_open_hand else item.han_closed),
            is_yakuman=bool(item.is_yakuman),
        )
        for item in response.yaku
    )
    yakuman_multiplier = sum(item.han // 13 for item in yaku if item.is_yakuman)
    base = _base_score_from_cost(
        han=int(response.han), fu=int(response.fu), cost=response.cost,
        is_tsumo=flags.is_tsumo, is_dealer=request.seat_wind_tile34 == api["EAST"],
        yakuman_multiplier=yakuman_multiplier,
    )
    return HandScore(
        rule_profile_id=RULE_PROFILE_ID,
        adapter_id=SCORING_ADAPTER_ID,
        base=base,
        yaku=yaku,
        fu_details=tuple(
            FuResult(reason=str(item["reason"]), fu=int(item["fu"])) for item in (response.fu_details or [])
        ),
        is_open_hand=bool(response.is_open_hand),
        actual_round_wind_tile34=request.round_wind_tile34,
        scoring_round_wind_tile34=scoring_round_wind,
        double_wind_pair_normalized=double_wind_pair,
    )


def detect_pao(existing_melds: Sequence[PaoMeld], new_meld: PaoMeld) -> tuple[PaoResponsibility, ...]:
    """成立するポンまたは大明槓が新たに確定させた責任払いを返す。"""
    all_melds = tuple(existing_melds) + (new_meld,)
    for meld in all_melds:
        if meld.kind not in ("pon", "daiminkan", "ankan", "shouminkan") or meld.tile34 not in range(34):
            raise ValueError("不正な責任判定用面子")
        if meld.kind in ("pon", "daiminkan") and meld.from_seat not in range(4):
            raise ValueError("鳴き面子には放出者が必要")
        if meld.kind in ("ankan", "shouminkan") and meld.from_seat is not None:
            raise ValueError("暗槓と加槓に放出者は付けない")
    if len({meld.tile34 for meld in all_melds}) != len(all_melds):
        raise ValueError("同じ牌種の成立済み面子が重複")
    if new_meld.kind not in ("pon", "daiminkan"):
        return ()
    assert new_meld.from_seat is not None
    before_dragons = {meld.tile34 for meld in existing_melds if meld.tile34 in (31, 32, 33)}
    after_dragons = {meld.tile34 for meld in all_melds if meld.tile34 in (31, 32, 33)}
    before_winds = {meld.tile34 for meld in existing_melds if meld.tile34 in (27, 28, 29, 30)}
    after_winds = {meld.tile34 for meld in all_melds if meld.tile34 in (27, 28, 29, 30)}
    before_kans = sum(meld.kind != "pon" for meld in existing_melds)
    after_kans = sum(meld.kind != "pon" for meld in all_melds)
    responsibilities = []
    if len(before_dragons) < 3 and len(after_dragons) == 3:
        responsibilities.append(PaoResponsibility("Daisangen", new_meld.from_seat))
    if len(before_winds) < 4 and len(after_winds) == 4:
        responsibilities.append(PaoResponsibility("Daisuushi", new_meld.from_seat))
    if new_meld.kind == "daiminkan" and before_kans < 4 and after_kans == 4:
        responsibilities.append(PaoResponsibility("Suukantsu", new_meld.from_seat))
    return tuple(responsibilities)


def _normal_base_transfers(
    score: BaseScore, winner: int, dealer: int, loser: int | None, reason: str = "base"
) -> list[Transfer]:
    if score.is_tsumo:
        transfers = []
        for seat in range(4):
            if seat == winner:
                continue
            points = score.main if (winner == dealer or seat == dealer) else score.additional
            transfers.append(Transfer(seat, winner, points, reason))
        return transfers
    assert loser is not None
    return [Transfer(loser, winner, score.main, reason)]


def _validate_base_score(score: BaseScore) -> None:
    if score.han <= 0 or score.fu < 0 or score.main <= 0 or score.additional < 0:
        raise ValueError("基本点が不正")
    expected_total = score.main + (2 * score.additional if score.is_tsumo else 0)
    if score.total != expected_total:
        raise ValueError("基本点合計が不一致")
    if not score.is_tsumo and score.additional != 0:
        raise ValueError("ロンのadditionalは0が必要")
    if score.is_tsumo and score.is_dealer and score.main != score.additional:
        raise ValueError("親ツモの支払い額が不一致")
    if score.yakuman_multiplier < 0:
        raise ValueError("役満倍率が不正")


def settle_win(
    score: BaseScore,
    *,
    winner: int,
    dealer: int,
    loser: int | None = None,
    honba: int = 0,
    kyoutaku: int = 0,
    pao: Sequence[PaoResponsibility] = (),
) -> Settlement:
    """基本支払いへ本場、責任払い、供託を一度だけ適用する。"""
    _validate_base_score(score)
    _validate_seat(winner, "winner")
    _validate_seat(dealer, "dealer")
    if honba < 0 or kyoutaku < 0:
        raise ValueError("本場と供託は0以上が必要")
    if score.is_dealer != (winner == dealer):
        raise ValueError("得点の親子区分とwinnerが不一致")
    if score.is_tsumo:
        if loser is not None:
            raise ValueError("ツモ和了に放銃者は指定しない")
    else:
        if loser is None:
            raise ValueError("ロン和了には放銃者が必要")
        _validate_seat(loser, "loser")
        if loser == winner:
            raise ValueError("winnerとloserは別家")
    transfers: list[Transfer] = []
    if pao:
        if score.yakuman_multiplier <= 0:
            raise ValueError("責任払いは役満にだけ適用する")
        if any(item.yaku not in _PAO_YAKU or item.yakuman_multiplier != 1 for item in pao):
            raise ValueError("不正な責任払い成分")
        if len({item.yaku for item in pao}) != len(pao):
            raise ValueError("責任払い成分が重複")
        for item in pao:
            _validate_seat(item.liable_seat, "liable_seat")
            if item.liable_seat == winner:
                raise ValueError("和了者自身を責任者にできない")
        pao_units = sum(item.yakuman_multiplier for item in pao)
        if pao_units > score.yakuman_multiplier:
            raise ValueError("責任払い成分が和了役満数を超える")
        unit_score = calculate_base_score(
            han=13, fu=0, is_tsumo=score.is_tsumo, is_dealer=score.is_dealer, is_yakuman=True
        )
        if score.total != unit_score.total * score.yakuman_multiplier:
            raise ValueError("責任払い対象の基本点が役満倍率と不一致")
        non_pao_units = score.yakuman_multiplier - pao_units
        if non_pao_units:
            remainder = calculate_base_score(
                han=13 * non_pao_units,
                fu=0,
                is_tsumo=score.is_tsumo,
                is_dealer=score.is_dealer,
                is_yakuman=True,
            )
            transfers.extend(_normal_base_transfers(remainder, winner, dealer, loser, "non_pao_yakuman"))
        full_unit = 48000 if winner == dealer else 32000
        for item in pao:
            full_points = full_unit * item.yakuman_multiplier
            if score.is_tsumo or loser == item.liable_seat:
                transfers.append(Transfer(item.liable_seat, winner, full_points, f"pao:{item.yaku}"))
            else:
                assert loser is not None
                transfers.append(Transfer(item.liable_seat, winner, full_points // 2, f"pao:{item.yaku}"))
                transfers.append(Transfer(loser, winner, full_points // 2, f"pao_split:{item.yaku}"))
        liable_seats = {item.liable_seat for item in pao}
        if honba and len(liable_seats) > 1:
            raise MultiplePaoHonbaUnresolved("hold: multiple_pao_honba_unresolved")
        if honba:
            transfers.append(Transfer(next(iter(liable_seats)), winner, 300 * honba, "honba_pao"))
    else:
        transfers.extend(_normal_base_transfers(score, winner, dealer, loser))
        if honba:
            if score.is_tsumo:
                transfers.extend(Transfer(seat, winner, 100 * honba, "honba") for seat in range(4) if seat != winner)
            else:
                assert loser is not None
                transfers.append(Transfer(loser, winner, 300 * honba, "honba"))
    if kyoutaku:
        transfers.append(Transfer(None, winner, 1000 * kyoutaku, "kyoutaku"))
    deltas = [0, 0, 0, 0]
    for transfer in transfers:
        if transfer.from_seat is not None:
            deltas[transfer.from_seat] -= transfer.points
        if transfer.to_seat is not None:
            deltas[transfer.to_seat] += transfer.points
    settlement = Settlement(tuple(deltas), kyoutaku, 0, tuple(transfers))
    settlement.validate()
    return settlement
