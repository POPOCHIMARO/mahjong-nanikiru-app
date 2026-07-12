// engine.js — 麻雀なに切るアプリの中核ロジック
// 牌の表現・シャンテン計算・受け入れ計算・問題生成・押し引きEVモデルをまとめたファイル。
// ブラウザ（<script>読み込み）と Node.js（テスト用 require）の両方で使えるようにしている。
//
// 【牌のインデックス表現】
//   0〜 8: 萬子の1〜9（1m〜9m）
//   9〜17: 筒子の1〜9（1p〜9p）
//  18〜26: 索子の1〜9（1s〜9s）
//  27〜33: 字牌（東 南 西 北 白 發 中）
(function (global) {
  "use strict";

  // ---------------------------------------------------------------
  // 牌の基本情報
  // ---------------------------------------------------------------
  var HONOR_NAMES = ["東", "南", "西", "北", "白", "發", "中"];
  var SUIT_NAMES = ["萬", "筒", "索"];

  // 牌インデックス → 表示用の名前（例: "3萬", "白"）
  function tileName(t) {
    if (t >= 27) return HONOR_NAMES[t - 27];
    var num = (t % 9) + 1;
    return num + SUIT_NAMES[Math.floor(t / 9)];
  }

  // 牌インデックス → 短い記法（例: "3m", "7z"）。解説の受け入れ一覧などで使う
  function tileShort(t) {
    if (t >= 27) return HONOR_NAMES[t - 27];
    var suits = ["m", "p", "s"];
    return ((t % 9) + 1) + suits[Math.floor(t / 9)];
  }

  // 数牌かどうか
  function isNumber(t) { return t < 27; }
  // 字牌かどうか
  function isHonor(t) { return t >= 27; }
  // 数牌の数字（1〜9）。字牌は0を返す
  function numberOf(t) { return t < 27 ? (t % 9) + 1 : 0; }

  // 牌の配列 → 34種の枚数配列
  function toCounts(tiles) {
    var c = new Array(34).fill(0);
    for (var i = 0; i < tiles.length; i++) c[tiles[i]]++;
    return c;
  }

  // ---------------------------------------------------------------
  // シャンテン計算
  // 一般手（4面子1雀頭）・七対子・国士無双の3種の最小値を返す。
  // -1=和了形, 0=テンパイ, 1=イーシャンテン, ...
  // ---------------------------------------------------------------

  // 一般手のシャンテン。counts は34種の枚数配列
  function shantenRegular(counts) {
    var c = counts.slice();
    var best = 8;

    // 面子・搭子（部分ブロック）・雀頭の組み合わせを深さ優先で全探索する
    function walk(i, melds, partials, hasPair) {
      while (i < 34 && c[i] === 0) i++;
      if (i >= 34) {
        // ブロック数の上限は4（面子＋搭子）。超過分の搭子は数えない
        var p = partials;
        if (melds + p > 4) p = 4 - melds;
        var s = 8 - 2 * melds - p - (hasPair ? 1 : 0);
        if (s < best) best = s;
        return;
      }
      // 刻子として使う
      if (c[i] >= 3) {
        c[i] -= 3;
        walk(i, melds + 1, partials, hasPair);
        c[i] += 3;
      }
      // 順子として使う
      if (i < 27 && i % 9 <= 6 && c[i + 1] > 0 && c[i + 2] > 0) {
        c[i]--; c[i + 1]--; c[i + 2]--;
        walk(i, melds + 1, partials, hasPair);
        c[i]++; c[i + 1]++; c[i + 2]++;
      }
      // 対子として使う（雀頭 or 刻子候補の搭子）
      if (c[i] >= 2) {
        c[i] -= 2;
        if (!hasPair) walk(i, melds, partials, true);
        walk(i, melds, partials + 1, hasPair);
        c[i] += 2;
      }
      // 両面・辺張の搭子
      if (i < 27 && i % 9 <= 7 && c[i + 1] > 0) {
        c[i]--; c[i + 1]--;
        walk(i, melds, partials + 1, hasPair);
        c[i]++; c[i + 1]++;
      }
      // 嵌張の搭子
      if (i < 27 && i % 9 <= 6 && c[i + 2] > 0) {
        c[i]--; c[i + 2]--;
        walk(i, melds, partials + 1, hasPair);
        c[i]++; c[i + 2]++;
      }
      // この牌を浮き牌として飛ばす
      var saved = c[i];
      c[i] = 0;
      walk(i + 1, melds, partials, hasPair);
      c[i] = saved;
    }

    walk(0, 0, 0, false);
    return best;
  }

  // 七対子のシャンテン
  function shantenChiitoi(counts) {
    var pairs = 0, kinds = 0;
    for (var i = 0; i < 34; i++) {
      if (counts[i] >= 1) kinds++;
      if (counts[i] >= 2) pairs++;
    }
    return 6 - pairs + Math.max(0, 7 - kinds);
  }

  // 国士無双のシャンテン
  var YAOCHU = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
  function shantenKokushi(counts) {
    var kinds = 0, hasPair = false;
    for (var i = 0; i < YAOCHU.length; i++) {
      var t = YAOCHU[i];
      if (counts[t] >= 1) kinds++;
      if (counts[t] >= 2) hasPair = true;
    }
    return 13 - kinds - (hasPair ? 1 : 0);
  }

  // 3種の最小シャンテンを返す
  function shanten(counts) {
    return Math.min(shantenRegular(counts), shantenChiitoi(counts), shantenKokushi(counts));
  }

  // ---------------------------------------------------------------
  // 受け入れ計算
  // 13枚の手牌に対して「引くとシャンテンが進む牌」と残り枚数を数える。
  // visibleOutside: 手牌以外で見えている枚数（河・ドラ表示牌など）。省略可
  // ---------------------------------------------------------------
  function ukeire(counts13, visibleOutside) {
    var base = shanten(counts13);
    var tiles = [];
    var total = 0;
    for (var t = 0; t < 34; t++) {
      if (counts13[t] >= 4) continue;
      counts13[t]++;
      var s = shanten(counts13);
      counts13[t]--;
      if (s < base) {
        var seen = counts13[t] + (visibleOutside ? visibleOutside[t] : 0);
        var left = 4 - seen;
        if (left > 0) {
          tiles.push({ tile: t, count: left });
          total += left;
        }
      }
    }
    return { shanten: base, total: total, tiles: tiles };
  }

  // 14枚の手牌について、打牌候補ごとの（シャンテン, 受け入れ）を一覧にする
  function analyzeDiscards(counts14, visibleOutside) {
    var rows = [];
    var minShanten = 99;
    for (var d = 0; d < 34; d++) {
      if (counts14[d] === 0) continue;
      counts14[d]--;
      var u = ukeire(counts14, visibleOutside);
      counts14[d]++;
      rows.push({ discard: d, shanten: u.shanten, ukeire: u.total, tiles: u.tiles });
      if (u.shanten < minShanten) minShanten = u.shanten;
    }
    // シャンテンが進む打牌（=最小シャンテン維持）だけを受け入れ順に並べる
    var keep = rows.filter(function (r) { return r.shanten === minShanten; });
    keep.sort(function (a, b) { return b.ukeire - a.ukeire; });
    return { minShanten: minShanten, all: rows, keep: keep };
  }

  // ---------------------------------------------------------------
  // 乱数まわりのユーティリティ
  // ---------------------------------------------------------------
  function randInt(n) { return Math.floor(Math.random() * n); }
  function pick(arr) { return arr[randInt(arr.length)]; }

  // 136枚の山（34種×4枚）の枚数配列を作る
  function newWall() { return new Array(34).fill(4); }

  // 山から指定の牌を1枚引く。引けなければ false
  function drawTile(wall, t) {
    if (wall[t] <= 0) return false;
    wall[t]--;
    return true;
  }

  // 山から重み付きでランダムに1枚引く。weightFn(牌)→重み
  function drawWeighted(wall, weightFn) {
    var totalW = 0;
    var ws = new Array(34);
    for (var t = 0; t < 34; t++) {
      ws[t] = wall[t] > 0 ? weightFn(t) * wall[t] : 0;
      totalW += ws[t];
    }
    if (totalW <= 0) return -1;
    var r = Math.random() * totalW;
    for (t = 0; t < 34; t++) {
      r -= ws[t];
      if (r < 0) { wall[t]--; return t; }
    }
    return -1;
  }

  // ---------------------------------------------------------------
  // 手牌生成
  // ブロック（順子・搭子・対子）を核にして自然な14枚の手を作る。
  // 完全ランダムだと3〜4シャンテンばかりになるため、形を持たせている。
  // ---------------------------------------------------------------
  function buildStructuredHand(wall) {
    var hand = [];
    function takeRun() { // 順子
      var suit = randInt(3), start = randInt(7);
      var base = suit * 9 + start;
      if (wall[base] > 0 && wall[base + 1] > 0 && wall[base + 2] > 0) {
        wall[base]--; wall[base + 1]--; wall[base + 2]--;
        hand.push(base, base + 1, base + 2);
      }
    }
    function takePartial() { // 両面・嵌張の搭子
      var suit = randInt(3), start = randInt(7);
      var base = suit * 9 + start;
      var gap = Math.random() < 0.7 ? 1 : 2;
      if (base + gap < suit * 9 + 9 && wall[base] > 0 && wall[base + gap] > 0) {
        wall[base]--; wall[base + gap]--;
        hand.push(base, base + gap);
      }
    }
    function takePair() { // 対子
      var t = randInt(34);
      if (wall[t] >= 2) { wall[t] -= 2; hand.push(t, t); }
    }

    var runs = 1 + randInt(2);       // 順子1〜2組
    var partials = 1 + randInt(2);   // 搭子1〜2組
    var pairs = randInt(2);          // 対子0〜1組
    for (var i = 0; i < runs; i++) takeRun();
    for (i = 0; i < partials; i++) takePartial();
    for (i = 0; i < pairs; i++) takePair();

    // 残りは中張牌寄りのランダムで14枚まで埋める
    while (hand.length < 14) {
      var t = drawWeighted(wall, function (x) {
        if (isHonor(x)) return 0.6;
        var n = numberOf(x);
        return (n === 1 || n === 9) ? 0.8 : 1.4;
      });
      if (t < 0) break;
      hand.push(t);
    }
    hand.sort(function (a, b) { return a - b; });
    return hand;
  }

  // ---------------------------------------------------------------
  // 赤5（アカドラ）の割り当て
  // 各スート（萬子・筒子・索子）の5は4枚中1枚が赤ドラという前提で、
  // 手牌中のその5の枚数kに対しk/4の確率で「赤が手牌に入っている」とみなし、
  // 入っている場合は手牌中のk枚のうちどれか1枚をランダムに赤とする。
  // 山や河に赤が残っている場合の追跡はしない（見た目・打点への影響のみの近似）。
  // ---------------------------------------------------------------
  var FIVE_INDICES = [4, 13, 22]; // 5萬, 5筒, 5索
  function assignRedFives(hand) {
    var redAt = new Array(hand.length).fill(false);
    FIVE_INDICES.forEach(function (five) {
      var positions = [];
      for (var i = 0; i < hand.length; i++) if (hand[i] === five) positions.push(i);
      var k = positions.length;
      if (k === 0) return;
      if (Math.random() < k / 4) {
        redAt[positions[randInt(k)]] = true;
      }
    });
    return redAt;
  }

  // ---------------------------------------------------------------
  // 牌効率モードの問題生成
  // 条件: 1〜2シャンテン、最大受け入れの打牌が明確（同率2種以下・2位と3枚以上差）
  // 正解 = シャンテンを維持しつつ受け入れ枚数が最大の打牌（同数はすべて正解）
  // ---------------------------------------------------------------
  function generateEfficiencyProblem() {
    for (var attempt = 0; attempt < 400; attempt++) {
      var wall = newWall();
      var hand = buildStructuredHand(wall);
      if (hand.length !== 14) continue;
      var counts = toCounts(hand);
      var s = shanten(counts);
      if (s < 1 || s > 2) continue;

      var an = analyzeDiscards(counts, null);
      if (an.keep.length < 2) continue; // 候補が1つだけでは問題にならない
      var bestU = an.keep[0].ukeire;
      var bests = an.keep.filter(function (r) { return r.ukeire === bestU; });
      var second = an.keep.filter(function (r) { return r.ukeire < bestU; });
      if (bests.length > 2) continue;                       // 正解が多すぎる手は避ける
      if (second.length === 0) continue;                    // 全部同点なら出題しない
      if (bestU - second[0].ukeire < 3) continue;           // 僅差の問題は避ける

      return {
        hand: hand,
        shanten: s,
        bestDiscards: bests.map(function (r) { return r.discard; }),
        analysis: an.keep,
        redFlags: assignRedFives(hand),
      };
    }
    return null; // 400回試して見つからないことは実質ない
  }

  // ---------------------------------------------------------------
  // 危険牌の分類と放銃率
  // vault: data/mleague/danger/danger_summary.json のカテゴリ体系に対応。
  // 放銃率(%)は『科学する麻雀』系の統計に基づく一般的な近似値。
  // ---------------------------------------------------------------
  var DANGER_RATES = {
    genbutsu: { label: "現物", rate: 0.0 },
    honor_3_visible: { label: "字牌（3枚見え）", rate: 0.05 },
    honor_2_visible: { label: "字牌（2枚見え）", rate: 0.6 },
    honor_1_visible: { label: "字牌（1枚見え）", rate: 1.6 },
    honor_live: { label: "生牌の字牌", rate: 3.2 },
    suji_19: { label: "スジの1・9", rate: 2.2 },
    suji_28: { label: "スジの2・8", rate: 3.1 },
    suji_37: { label: "スジの3・7", rate: 3.8 },
    double_suji_middle: { label: "中スジの4・5・6", rate: 2.0 },
    one_chance: { label: "ワンチャンスの無スジ", rate: 3.0 },
    no_chance: { label: "ノーチャンスの無スジ", rate: 2.2 },
    non_suji_19: { label: "無スジの1・9", rate: 3.4 },
    non_suji_28: { label: "無スジの2・8", rate: 4.3 },
    non_suji_37: { label: "無スジの3・7", rate: 4.9 },
    non_suji_456: { label: "無スジの4・5・6", rate: 5.7 },
  };

  // 打牌 tile がリーチ者の河 river・見えている牌 visible からどのカテゴリかを判定
  function classifyDanger(tile, riverCounts, visibleCounts) {
    if (riverCounts[tile] > 0) return "genbutsu";

    if (isHonor(tile)) {
      var seen = visibleCounts[tile];
      if (seen >= 3) return "honor_3_visible";
      if (seen === 2) return "honor_2_visible";
      if (seen === 1) return "honor_1_visible";
      return "honor_live";
    }

    var n = numberOf(tile);
    var suitBase = Math.floor(tile / 9) * 9;

    // スジ判定: 1-3はn+3、7-9はn-3が河にあればスジ。4-6は両側必要（中スジ）
    var sujiLow = n >= 4 ? riverCounts[suitBase + (n - 3) - 1] > 0 : true;
    var sujiHigh = n <= 6 ? riverCounts[suitBase + (n + 3) - 1] > 0 : true;
    var isSuji = (n <= 3 && sujiHigh) || (n >= 7 && sujiLow) || (n >= 4 && n <= 6 && sujiLow && sujiHigh);
    if (isSuji) {
      if (n >= 4 && n <= 6) return "double_suji_middle";
      if (n === 1 || n === 9) return "suji_19";
      if (n === 2 || n === 8) return "suji_28";
      return "suji_37";
    }

    // 壁（ワンチャンス/ノーチャンス）判定:
    // この牌を両面で待つのに必要な隣接牌が4枚見え→ノーチャンス、3枚見え→ワンチャンス
    var blocks = []; // この牌をロン牌にする両面ターツの構成牌ペア
    if (n >= 3) blocks.push([suitBase + n - 3, suitBase + n - 2]); // (n-2,n-1)待ち
    if (n <= 7) blocks.push([suitBase + n, suitBase + n + 1]);     // (n+1,n+2)待ち
    if (blocks.length > 0) {
      var worst = 0; // 各ブロックの「最も見えている構成牌」の最小値を評価
      var minSeen = 4;
      for (var i = 0; i < blocks.length; i++) {
        var seenMax = Math.max(visibleCounts[blocks[i][0]], visibleCounts[blocks[i][1]]);
        if (seenMax < minSeen) minSeen = seenMax;
      }
      worst = minSeen;
      if (worst >= 4) return "no_chance";
      if (worst === 3) return "one_chance";
    }

    if (n === 1 || n === 9) return "non_suji_19";
    if (n === 2 || n === 8) return "non_suji_28";
    if (n === 3 || n === 7) return "non_suji_37";
    return "non_suji_456";
  }

  // ---------------------------------------------------------------
  // 押し引きEVモデル（簡易・局収支ベースの概算）
  // 「イーシャンテンから危険牌を押して和了に向かう」vs「現物を切ってベタオリ」
  // の局収支期待値を比較する。定数は下記を根拠にした概算:
  //  - 放銃率: DANGER_RATES（科学する麻雀系の統計値）
  //  - リーチ平均打点: 子5300 / 親7700
  //  - vault研究ノート「Mリーグ対リーチ押し率」: テンパイ88% / 1シャンテン78%
  // ---------------------------------------------------------------
  function evaluatePushFold(p) {
    // p: { turn, ukeire, dangerRate(%), ownValue, oppIsDealer }
    var remain = 18 - p.turn; // 残りツモ回数の目安

    // 押した場合の自分の和了率（イーシャンテン・巡目と受け入れで概算）
    var pWin = 0.018 * remain * (0.5 + p.ukeire / 48);
    pWin = Math.min(0.40, Math.max(0.03, pWin));

    // この牌の放銃率 + 押し続けた場合の追加放銃リスク
    var pNow = p.dangerRate / 100;
    var pFuture = Math.min(0.18, 0.023 * remain);
    var pDeal = pNow + (1 - pNow) * pFuture;

    // リーチの和了率（巡目が深いほど残り抽選が減る）
    var pOppWin = Math.min(0.52, 0.045 * remain);

    var dealLoss = p.oppIsDealer ? 7700 : 5300; // 放銃時の平均失点
    var tsumoPay = p.oppIsDealer ? 2300 : 1400; // 相手ツモ時の平均支払い

    // 押しEV = 和了収入 − 放銃失点 − (どちらも和了しない間の)ツモられ失点
    var winGain = p.ownValue + 1000; // 供託リーチ棒込み
    var evPush =
      pWin * winGain -
      pDeal * dealLoss -
      (1 - pWin - pDeal) * (pOppWin * 0.85) * 0.4 * tsumoPay;

    // オリEV = ツモられ失点のみ（放銃はほぼゼロ）+ テンパイ料などの機会損失
    var evFold = -(pOppWin * 0.4 * tsumoPay) - 300;

    return {
      pWin: pWin,
      pDeal: pDeal,
      pOppWin: pOppWin,
      dealLoss: dealLoss,
      evPush: Math.round(evPush),
      evFold: Math.round(evFold),
      answer: evPush > evFold ? "push" : "fold",
      diff: Math.round(evPush - evFold),
    };
  }

  // ---------------------------------------------------------------
  // 押し引きモードの問題生成
  // 他家リーチに対し、イーシャンテンを維持する打牌（危険牌）を押すか、
  // 現物を切ってオリるかを問う。正解はEVの高い方。
  // ---------------------------------------------------------------
  function generateRiver(wall, turn, ownCounts) {
    // リーチ者の河を巡目分だけ作る。序盤は字牌・端牌寄り、リーチ後はランダム
    var len = turn;
    var riichiIndex = 3 + randInt(Math.max(1, Math.min(4, len - 4))); // 4〜7巡目あたりで宣言
    var river = [];
    for (var i = 0; i < len; i++) {
      var early = i < riichiIndex;
      var t = drawWeighted(wall, function (x) {
        if (isHonor(x)) return early ? 5 : 0.7;
        var n = numberOf(x);
        if (n === 1 || n === 9) return early ? 3 : 1;
        if (n === 2 || n === 8) return early ? 1.2 : 1;
        return early ? 0.5 : 1.3;
      });
      if (t < 0) return null;
      river.push({ tile: t, riichi: i === riichiIndex });
    }
    return { tiles: river, riichiIndex: riichiIndex };
  }

  function generatePushFoldProblem() {
    // 押し/オリの正解が偏らないよう、先に目標の答えを決めて合致する局面を探す
    var target = Math.random() < 0.5 ? "push" : "fold";
    var fallback = null;

    for (var attempt = 0; attempt < 600; attempt++) {
      var wall = newWall();
      var hand = buildStructuredHand(wall);
      if (hand.length !== 14) continue;
      var counts = toCounts(hand);
      if (shanten(counts) !== 1) continue; // 全問イーシャンテンで出題

      var turn = 6 + randInt(7); // 6〜12巡目
      var riverData = generateRiver(wall, turn, counts);
      if (!riverData) continue;
      var riverCounts = toCounts(riverData.tiles.map(function (r) { return r.tile; }));

      // ドラ表示牌
      var doraIndicator = drawWeighted(wall, function () { return 1; });
      if (doraIndicator < 0) continue;
      var dora = isHonor(doraIndicator)
        ? (doraIndicator < 31 ? 27 + ((doraIndicator - 27 + 1) % 4) : 31 + ((doraIndicator - 31 + 1) % 3))
        : Math.floor(doraIndicator / 9) * 9 + (numberOf(doraIndicator) % 9);

      // 手牌の外で見えている牌 = 河 + ドラ表示牌（受け入れ計算用。手牌分は計算側で引かれる）
      var outside = new Array(34).fill(0);
      for (var t = 0; t < 34; t++) outside[t] = riverCounts[t];
      outside[doraIndicator]++;
      // 危険度判定用の「見えている牌」= 手牌 + 河 + ドラ表示牌
      var visible = new Array(34).fill(0);
      for (t = 0; t < 34; t++) visible[t] = counts[t] + outside[t];

      // イーシャンテン維持で受け入れ最大の打牌 = 勝負牌
      var an = analyzeDiscards(counts, outside);
      if (an.minShanten !== 1 || an.keep.length === 0) continue;
      var pushRow = an.keep[0];
      var pushTile = pushRow.discard;
      if (riverCounts[pushTile] > 0) continue; // 勝負牌が現物なら問題にならない

      // オリ候補 = 手牌の中の現物。無ければ出題しない
      var foldTile = -1;
      for (t = 0; t < 34; t++) {
        if (counts[t] > 0 && riverCounts[t] > 0) { foldTile = t; break; }
      }
      if (foldTile < 0) continue;

      var category = classifyDanger(pushTile, riverCounts, visible);
      var rate = DANGER_RATES[category].rate;
      if (rate <= 0.1) continue; // ほぼ安全な牌では押し引き問題にならない

      // 自分の打点期待: リーチ前提の概算（子4500+ドラ1500 / 親は1.5倍）
      // 赤5（アカドラ）も通常のドラと同枚数扱いで加算する
      var redFlags = assignRedFives(hand);
      var akaCount = redFlags.filter(function (r) { return r; }).length;
      var doraCount = counts[dora] + akaCount;
      var oppIsDealer = Math.random() < 0.3;
      var selfIsDealer = !oppIsDealer && Math.random() < 0.3;
      var ownValue = Math.min(12000, 4500 + 1500 * doraCount) * (selfIsDealer ? 1.5 : 1);

      var ev = evaluatePushFold({
        turn: turn,
        ukeire: pushRow.ukeire,
        dangerRate: rate,
        ownValue: ownValue,
        oppIsDealer: oppIsDealer,
      });

      var problem = {
        hand: hand,
        turn: turn,
        river: riverData.tiles,
        doraIndicator: doraIndicator,
        dora: dora,
        doraCount: doraCount,
        akaCount: akaCount,
        redFlags: redFlags,
        oppIsDealer: oppIsDealer,
        selfIsDealer: selfIsDealer,
        pushTile: pushTile,
        pushUkeire: pushRow.ukeire,
        pushUkeireTiles: pushRow.tiles,
        foldTile: foldTile,
        category: category,
        categoryLabel: DANGER_RATES[category].label,
        dangerRate: rate,
        ownValue: Math.round(ownValue),
        ev: ev,
        answer: ev.answer,
      };

      // EV差が小さい微妙な局面は出題しない（正解が議論にならないように）
      if (Math.abs(ev.diff) < 300) continue;
      if (ev.answer === target) return problem;
      if (!fallback) fallback = problem; // 目標の答えが見つからない場合の保険
    }
    return fallback;
  }

  // ---------------------------------------------------------------
  // 公開API
  // ---------------------------------------------------------------
  var Engine = {
    tileName: tileName,
    tileShort: tileShort,
    isHonor: isHonor,
    numberOf: numberOf,
    toCounts: toCounts,
    shanten: shanten,
    shantenRegular: shantenRegular,
    shantenChiitoi: shantenChiitoi,
    shantenKokushi: shantenKokushi,
    ukeire: ukeire,
    analyzeDiscards: analyzeDiscards,
    assignRedFives: assignRedFives,
    classifyDanger: classifyDanger,
    evaluatePushFold: evaluatePushFold,
    generateEfficiencyProblem: generateEfficiencyProblem,
    generatePushFoldProblem: generatePushFoldProblem,
    DANGER_RATES: DANGER_RATES,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = Engine; // Node.js（テスト）用
  } else {
    global.Engine = Engine; // ブラウザ用
  }
})(typeof window !== "undefined" ? window : globalThis);
