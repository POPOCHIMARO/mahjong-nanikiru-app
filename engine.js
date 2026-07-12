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

  // 一般手のシャンテン。counts は34種の枚数配列。
  // fixedMelds は暗槓などですでに完成している面子数（通常の門前手は0）。
  function shantenRegular(counts, fixedMelds) {
    fixedMelds = fixedMelds || 0;
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

    walk(0, fixedMelds, 0, false);
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

  // 3種の最小シャンテンを返す。
  // 固定面子がある手では七対子・国士にはならないため、一般手だけを計算する。
  function shanten(counts, fixedMelds) {
    fixedMelds = fixedMelds || 0;
    if (fixedMelds > 0) return shantenRegular(counts, fixedMelds);
    return Math.min(shantenRegular(counts), shantenChiitoi(counts), shantenKokushi(counts));
  }

  // ---------------------------------------------------------------
  // 受け入れ計算
  // 13枚相当の手（固定面子がある場合は残りの手牌）について、
  // 「引くとシャンテンが進む牌」と残り枚数を数える。
  // visibleOutside: 手牌以外で見えている枚数（河・ドラ表示牌・槓子など）。省略可
  // fixedMelds: 暗槓などですでに完成している面子数。省略時は0
  // ---------------------------------------------------------------
  function ukeire(counts13, visibleOutside, fixedMelds) {
    fixedMelds = fixedMelds || 0;
    var base = shanten(counts13, fixedMelds);
    var tiles = [];
    var total = 0;
    for (var t = 0; t < 34; t++) {
      if (counts13[t] >= 4) continue;
      counts13[t]++;
      var s = shanten(counts13, fixedMelds);
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

  // ---------------------------------------------------------------
  // 変化（改良ツモ）の評価
  // 打牌後の13枚について「シャンテンは進まないが、引いて最良の打牌をすると
  // 受け入れが2枚以上増えるツモ」の残り枚数を合計する。
  // 例: 5667p のような形は、4p/8p などを引くと受け入れが大きく伸びる。
  // この“隠れた価値”は1段階の受け入れ枚数には現れないため、別に数える。
  // ---------------------------------------------------------------
  // 内訳（どの牌のツモで・何枚残っているか）まで返す版。
  // 解説表示（どの牌を引けば伸びるかを見せる）と出題ガードの両方から使う。
  function improvementDetail(counts13, currentUkeire) {
    var base = shanten(counts13);
    var total = 0;
    var tiles = [];
    for (var t = 0; t < 34; t++) {
      if (counts13[t] >= 4) continue;
      counts13[t]++;
      if (shanten(counts13) >= base) {
        // シャンテンが進まないツモ。最良の応手で受け入れが2枚以上増えるか調べる
        var improved = false;
        for (var d = 0; d < 34 && !improved; d++) {
          if (counts13[d] === 0) continue;
          if (d === t) continue; // ツモ切りは元の形に戻るだけなので見ない
          counts13[d]--;
          // シャンテンが落ちる打牌は受け入れを数えるまでもなく対象外（高速化）
          if (shanten(counts13) === base) {
            var u = ukeire(counts13, null);
            if (u.total >= currentUkeire + 2) improved = true;
          }
          counts13[d]++;
        }
        if (improved) {
          var left = 4 - (counts13[t] - 1);
          if (left > 0) {
            tiles.push({ tile: t, count: left });
            total += left;
          }
        }
      }
      counts13[t]--;
    }
    return { total: total, tiles: tiles };
  }

  function improvementPotential(counts13, currentUkeire) {
    return improvementDetail(counts13, currentUkeire).total;
  }

  // 牌効率問題の出題可否チェック。
  // 受け入れ最大の打牌が、僅差（3枚以内）の対抗打牌に変化ポテンシャルで
  // 大きく劣る局面は「受け入れ枚数だけでは正解と言えない」ため出題しない。
  // 換算レートは 受け入れ1枚 ≒ 変化4枚 とする。
  // 例: 2m4m5m6m7m 5p6p6p7p 1s2s3s 北白 から 6p切り(受10) vs 北切り(受9) は、
  //     北切りの変化が大きく上回るため出題対象から外れる。
  function efficiencyAnswerIsSound(counts14, keepRows) {
    var bestU = keepRows[0].ukeire;

    // 同率正解（どれを選んでも正解扱い）の中で最大の変化ポテンシャルを基準にする
    var bestPot = -1;
    for (var i = 0; i < keepRows.length; i++) {
      var row = keepRows[i];
      if (row.ukeire !== bestU) break; // keepRows は受け入れ降順
      counts14[row.discard]--;
      var pot = improvementPotential(counts14, row.ukeire);
      counts14[row.discard]++;
      if (pot > bestPot) bestPot = pot;
    }

    for (i = 0; i < keepRows.length; i++) {
      row = keepRows[i];
      if (row.ukeire === bestU) continue;      // 同率はどちらも正解なので比較不要
      var gap = bestU - row.ukeire;
      if (gap > 3) break;                      // 大差の候補は受け入れ枚数で決着済み
      counts14[row.discard]--;
      var rivalPot = improvementPotential(counts14, row.ukeire);
      counts14[row.discard]++;
      if (rivalPot - bestPot > 4 * gap) return false;
    }
    return true;
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
  // 清一色モードの行動分析
  // 打牌と暗槓は手牌の状態が異なるため、それぞれ独立に受け入れを計算する。
  // ---------------------------------------------------------------
  function analyzeChinitsuActions(counts14, visibleOutside) {
    var discardAnalysis = analyzeDiscards(counts14, visibleOutside);
    var rows = discardAnalysis.all.map(function (r) {
      return {
        type: "discard",
        tile: r.discard,
        discard: r.discard, // 既存の表示・テストとの互換用
        shanten: r.shanten,
        ukeire: r.ukeire,
        tiles: r.tiles,
      };
    });
    var kanOptions = [];

    for (var k = 0; k < 9; k++) {
      if (counts14[k] !== 4) continue;
      kanOptions.push(k);

      // 暗槓した4枚を手牌から除き、固定面子1組として計算する。
      // 槓子の4枚はすでに見えているため、受け入れの残り枚数からも必ず引く。
      var afterKan = counts14.slice();
      afterKan[k] -= 4;
      var outsideAfterKan = visibleOutside ? visibleOutside.slice() : new Array(34).fill(0);
      outsideAfterKan[k] += 4;
      var u = ukeire(afterKan, outsideAfterKan, 1);
      rows.push({
        type: "ankan",
        tile: k,
        shanten: u.shanten,
        ukeire: u.total,
        tiles: u.tiles,
      });
    }

    var minShanten = 99;
    rows.forEach(function (r) {
      if (r.shanten < minShanten) minShanten = r.shanten;
    });
    var keep = rows.filter(function (r) { return r.shanten === minShanten; });
    keep.sort(function (a, b) {
      if (b.ukeire !== a.ukeire) return b.ukeire - a.ukeire;
      if (a.type !== b.type) return a.type === "discard" ? -1 : 1;
      return a.tile - b.tile;
    });
    return { minShanten: minShanten, all: rows, keep: keep, kanOptions: kanOptions };
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

  // 14枚の中から「ツモ前に指定シャンテンだった13枚」を復元する。
  // 戻り値の hand は、最初の13枚がツモ前の手牌、最後の1枚がツモ牌になる。
  function splitImprovingDraw(hand14, fromShanten) {
    var counts = toCounts(hand14);
    var drawCandidates = [];

    for (var i = 0; i < hand14.length; i++) {
      var t = hand14[i];
      counts[t]--;
      if (shanten(counts) === fromShanten) drawCandidates.push(i);
      counts[t]++;
    }
    if (drawCandidates.length === 0) return null;

    var drawIndex = pick(drawCandidates);
    var baseHand = hand14.slice();
    var drawnTile = baseHand.splice(drawIndex, 1)[0];
    return {
      baseHand: baseHand,
      drawnTile: drawnTile,
      hand: baseHand.concat([drawnTile]),
    };
  }

  // ---------------------------------------------------------------
  // 牌効率モードの問題生成
  // 条件: ツモ前は2シャンテン、打牌後は1シャンテン。
  // 正解 = 1シャンテンに進む打牌のうち、テンパイへの受け入れ枚数が最大の打牌。
  // 最大受け入れが同率の場合はすべて正解とする。
  // 通常問題（次点と3枚以上差）と高難度問題（次点と1〜2枚差）を半数ずつ出題する。
  // ---------------------------------------------------------------
  var EFFICIENCY_HARD_RATE = 0.5;

  function generateEfficiencyProblem(difficulty) {
    // テストでは難易度を固定できる。通常の画面からは未指定なので半数ずつ選ばれる。
    var targetDifficulty = difficulty === "hard" || difficulty === "standard"
      ? difficulty
      : (Math.random() < EFFICIENCY_HARD_RATE ? "hard" : "standard");

    for (var attempt = 0; attempt < 1200; attempt++) {
      var wall = newWall();
      var hand = buildStructuredHand(wall);
      if (hand.length !== 14) continue;
      var counts = toCounts(hand);
      var an = analyzeDiscards(counts, null);
      if (an.minShanten !== 1) continue;
      if (an.keep.length < 2) continue; // 候補が1つだけでは問題にならない
      var bestU = an.keep[0].ukeire;
      if (bestU <= 0) continue;
      var bests = an.keep.filter(function (r) { return r.ukeire === bestU; });
      var second = an.keep.filter(function (r) { return r.ukeire < bestU; });
      if (bests.length > 2) continue;                       // 正解が多すぎる手は避ける
      if (second.length === 0) continue;                    // 全部同点なら出題しない
      var ukeireGap = bestU - second[0].ukeire;
      if (targetDifficulty === "hard" && ukeireGap > 2) continue;
      if (targetDifficulty === "standard" && ukeireGap < 3) continue;

      // この14枚を作った実際のツモが、2シャンテンの13枚からの進展牌か確認する
      var split = splitImprovingDraw(hand, 2);
      if (!split) continue;

      // 受け入れ枚数の僅差だけでは決まらない局面（変化で逆転する形）は出題しない
      if (!efficiencyAnswerIsSound(counts, an.keep)) continue;

      // 解説表示用に、候補ごとの変化（好形へ伸びるツモ）の内訳を付与する。
      // 出題が確定した後の1回だけの計算なので、候補数（通常2〜3件）分のコストで済む。
      an.keep.forEach(function (r) {
        counts[r.discard]--;
        r.variation = improvementDetail(counts, r.ukeire);
        counts[r.discard]++;
      });

      return {
        hand: split.hand,
        baseHand: split.baseHand,
        drawnTile: split.drawnTile,
        fromShanten: 2,
        toShanten: 1,
        shanten: 1,
        difficulty: targetDifficulty,
        ukeireGap: ukeireGap,
        bestDiscards: bests.map(function (r) { return r.discard; }),
        analysis: an.keep,
        redFlags: assignRedFives(split.hand),
      };
    }
    return null; // 条件を満たす問題が見つからなかった場合は呼び出し側で再試行できる
  }

  // ---------------------------------------------------------------
  // 清一色（萬子）モードの問題生成
  // 条件: 萬子のみで、ツモ前は1シャンテン、打牌・暗槓後はテンパイ。
  // 正解 = テンパイに進む打牌・暗槓のうち、和了牌の受け入れ枚数が最大の行動。
  // 最大受け入れが同率の場合はすべて正解とする。
  // 清一色モードの制約を計算にも明示するため、受け入れは萬子のみを数える。
  // 清一色は受け入れ同率の打牌が出やすいため、出題の明確さ条件は
  // 牌効率モードより緩め（同率3種以下・2位と2枚以上差）にしている。
  // ---------------------------------------------------------------
  function generateChinitsuProblem() {
    // 萬子以外を「4枚全部見えている」扱いにして受け入れ計算から除外する
    var nonManzuOut = new Array(34).fill(4);
    for (var m = 0; m < 9; m++) nonManzuOut[m] = 0;

    for (var attempt = 0; attempt < 1000; attempt++) {
      // 萬子36枚（9種×4枚）だけの山から14枚引く
      var wall = new Array(34).fill(0);
      for (var t = 0; t < 9; t++) wall[t] = 4;
      var hand = [];
      for (var i = 0; i < 14; i++) {
        var d = drawWeighted(wall, function () { return 1; });
        if (d < 0) break;
        hand.push(d);
      }
      if (hand.length !== 14) continue;
      hand.sort(function (a, b) { return a - b; });
      var counts = toCounts(hand);
      var an = analyzeChinitsuActions(counts, nonManzuOut);
      if (an.minShanten !== 0 || an.keep.length < 2) continue;
      var bestU = an.keep[0].ukeire;
      if (bestU <= 0) continue; // 萬子の受け入れが無い形（純カラ）は出題しない
      var bests = an.keep.filter(function (r) { return r.ukeire === bestU; });
      var second = an.keep.filter(function (r) { return r.ukeire < bestU; });
      if (bests.length > 3) continue;             // 同率正解が多すぎる手は避ける
      if (second.length === 0) continue;          // 全部同点なら出題しない
      if (bestU - second[0].ukeire < 2) continue; // 僅差の問題は避ける

      // 表示するツモ牌を、1シャンテンの13枚から実際に引いた牌として確定する
      var split = splitImprovingDraw(hand, 1);
      if (!split) continue;

      return {
        hand: split.hand,
        baseHand: split.baseHand,
        drawnTile: split.drawnTile,
        fromShanten: 1,
        toShanten: 0,
        shanten: 0,
        bestActions: bests.map(function (r) { return { type: r.type, tile: r.tile }; }),
        // 打牌だけを参照する既存コードとの互換用。正解が暗槓だけなら空配列になる。
        bestDiscards: bests.filter(function (r) { return r.type === "discard"; }).map(function (r) { return r.tile; }),
        kanOptions: an.kanOptions,
        analysis: an.keep,
        redFlags: assignRedFives(split.hand),
      };
    }
    return null; // 1000回試して見つからないことは実質ない
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

    // 無理押しガードで候補がかなり絞られるため、試行上限は多めに取る
    for (var attempt = 0; attempt < 2000; attempt++) {
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

      // イーシャンテン維持打牌の候補全部に危険度を付ける
      var an = analyzeDiscards(counts, outside);
      if (an.minShanten !== 1 || an.keep.length === 0) continue;
      var keepRates = an.keep.map(function (row) {
        return DANGER_RATES[classifyDanger(row.discard, riverCounts, visible)].rate;
      });

      // 無理押しガード:
      // 手牌に「イーシャンテンを保ったまま切れるほぼ安全な牌」（現物・字牌・
      // 筋・ワンチャンスなど、Mリーグ危険度分類のsafe/guarded相当）が1枚でも
      // あるなら、それを切ればよいだけで危険牌との押し引き局面にならない。
      // 実際のMリーグの危険押し（moderate/high risk押し）に合わせ、
      // 手を保つ打牌がすべて放銃率3%以上の危険牌である牌姿だけを出題する。
      var PUSH_MIN_RATE = 3.0;
      var hasSafeEscape = keepRates.some(function (r) { return r < PUSH_MIN_RATE; });
      if (hasSafeEscape) continue;

      // 勝負牌 = 受け入れ最大の打牌。同数タイなら最も安全な牌を選ぶ
      // （同じ受け入れでより危険な牌を切らせるのは無理押しになるため）
      var pushIdx = 0;
      for (var k = 1; k < an.keep.length; k++) {
        if (an.keep[k].ukeire !== an.keep[0].ukeire) break;
        if (keepRates[k] < keepRates[pushIdx]) pushIdx = k;
      }
      var pushRow = an.keep[pushIdx];
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
        safestKeepRate: Math.min.apply(null, keepRates),
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
    improvementPotential: improvementPotential,
    improvementDetail: improvementDetail,
    efficiencyAnswerIsSound: efficiencyAnswerIsSound,
    analyzeDiscards: analyzeDiscards,
    analyzeChinitsuActions: analyzeChinitsuActions,
    assignRedFives: assignRedFives,
    classifyDanger: classifyDanger,
    evaluatePushFold: evaluatePushFold,
    generateEfficiencyProblem: generateEfficiencyProblem,
    EFFICIENCY_HARD_RATE: EFFICIENCY_HARD_RATE,
    generateChinitsuProblem: generateChinitsuProblem,
    generatePushFoldProblem: generatePushFoldProblem,
    DANGER_RATES: DANGER_RATES,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = Engine; // Node.js（テスト）用
  } else {
    global.Engine = Engine; // ブラウザ用
  }
})(typeof window !== "undefined" ? window : globalThis);
