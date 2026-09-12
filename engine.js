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
  // 変化の探索では同じ手を何度も評価する。上限付きキャッシュで再計算を省く。
  var shantenCache = new Map();
  function shanten(counts, fixedMelds) {
    fixedMelds = fixedMelds || 0;
    var key = fixedMelds + ":" + counts.join("");
    if (shantenCache.has(key)) return shantenCache.get(key);
    var result = fixedMelds > 0 ? shantenRegular(counts, fixedMelds)
      : Math.min(shantenRegular(counts), shantenChiitoi(counts), shantenKokushi(counts));
    if (shantenCache.size >= 20000) shantenCache.clear();
    shantenCache.set(key, result);
    return result;
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
  function improvementDetail(counts13, currentUkeire, visibleOutside) {
    var base = shanten(counts13);
    var outside = visibleOutside ? visibleOutside.slice() : new Array(34).fill(0);
    var total = 0;
    var tiles = [];
    var unseen = 136 - counts13.reduce(function (sum, count) { return sum + count; }, 0)
      - outside.reduce(function (sum, count) { return sum + count; }, 0);
    var continuation = 0;
    for (var t = 0; t < 34; t++) {
      var left = 4 - counts13[t] - outside[t];
      if (left <= 0) continue;
      counts13[t]++;
      if (shanten(counts13) >= base) {
        // シャンテンが進まないツモ。最良の応手で受け入れが2枚以上増えるか調べる
        var bestNextUkeire = currentUkeire; // ツモ切りでも元の受け入れは維持できる。
        for (var d = 0; d < 34; d++) {
          if (counts13[d] === 0) continue;
          if (d === t) continue; // ツモ切りは元の形に戻るだけなので見ない
          counts13[d]--;
          outside[d]++; // 改良ツモ後に切る牌も山へは戻らない。
          // シャンテンが落ちる打牌は受け入れを数えるまでもなく対象外（高速化）
          if (shanten(counts13) === base) {
            var u = ukeire(counts13, outside);
            bestNextUkeire = Math.max(bestNextUkeire, u.total);
          }
          counts13[d]++;
          outside[d]--;
        }
        continuation += left * bestNextUkeire;
        if (bestNextUkeire >= currentUkeire + 2) {
          if (left > 0) {
            tiles.push({ tile: t, count: left });
            total += left;
          }
        }
      }
      counts13[t]--;
    }
    // 次のツモで進む分 + 進まなかったツモごとに最善打牌した後の受け入れ。
    // 全候補で分母 unseen*(unseen-1) が共通なので、比較には分子だけを使う。
    return { total: total, tiles: tiles,
      twoDrawNumerator: currentUkeire * (unseen - 1) + continuation };
  }

  function improvementPotential(counts13, currentUkeire, visibleOutside) {
    return improvementDetail(counts13, currentUkeire, visibleOutside).total;
  }

  // 手牌に1枚だけある字牌が2種類以上あるかを調べる。
  // 孤立字牌どうしは受け入れも変化も同じになりやすく、何を切っても等価なため出題しない。
  function hasMultipleIsolatedHonors(counts) {
    var kinds = 0;
    for (var t = 27; t < 34; t++) {
      if (counts[t] === 1) kinds++;
    }
    return kinds >= 2;
  }

  // 同じ牌、または同じ色の前後2つ以内の数牌が無ければ孤立牌とする。
  // 字牌は順子を作れないので、手牌に1枚だけなら孤立牌になる。
  function isIsolatedTile(counts, tile) {
    if (counts[tile] !== 1) return false;
    if (isHonor(tile)) return true;

    var suitBase = Math.floor(tile / 9) * 9;
    var suitEnd = suitBase + 8;
    for (var other = Math.max(suitBase, tile - 2); other <= Math.min(suitEnd, tile + 2); other++) {
      if (other !== tile && counts[other] > 0) return false;
    }
    return true;
  }

  // 搭子2枚を完成させる受け入れ牌を返す。
  // 例: 23m は1m/4m、24m は3mを受け入れる。
  function partialWaits(a, b) {
    var waits = [];
    var gap = b - a;
    if (gap === 1) {
      if (a % 9 > 0) waits.push(a - 1);
      if (b % 9 < 8) waits.push(b + 1);
    } else if (gap === 2) {
      waits.push(a + 1);
    }
    return waits;
  }

  // 2つの搭子が同じ受け入れ牌を持つ「二度受け」があるかを調べる。
  // 完成済みの順子（例: 123m）は搭子として数えない。数えてしまうと、
  // 順子を含むだけのごく普通の手が二度受け扱いになってしまうため。
  // 判定は「受け入れ牌がすでに手牌にあるなら、その2枚は順子の一部」とみなす。
  // 例: 23m は 1m か 4m が手にあれば順子の一部。24m は 3m が手にあれば順子の一部。
  function hasDoubleAcceptance(counts) {
    var partials = [];
    for (var suit = 0; suit < 3; suit++) {
      var base = suit * 9;
      for (var a = base; a < base + 9; a++) {
        if (counts[a] === 0) continue;
        for (var gap = 1; gap <= 2; gap++) {
          var b = a + gap;
          if (b < base + 9 && counts[b] > 0) {
            var waits = partialWaits(a, b);
            var alreadyComplete = waits.some(function (w) { return counts[w] > 0; });
            if (alreadyComplete) continue; // 順子として埋まっている＝未完成の搭子ではない
            partials.push({ tiles: [a, b], waits: waits });
          }
        }
      }
    }

    for (var i = 0; i < partials.length; i++) {
      for (var j = i + 1; j < partials.length; j++) {
        var needed = {};
        partials[i].tiles.concat(partials[j].tiles).forEach(function (t) {
          needed[t] = (needed[t] || 0) + 1;
        });
        var canCoexist = Object.keys(needed).every(function (t) {
          return counts[Number(t)] >= needed[t];
        });
        if (!canCoexist) continue;

        var overlaps = partials[i].waits.some(function (wait) {
          return partials[j].waits.indexOf(wait) >= 0;
        });
        if (overlaps) return true;
      }
    }
    return false;
  }

  // 正解打牌と手牌全体から、学習者が見落としやすい形を罠型として記録する。
  function detectEfficiencyTraps(counts14, bestDiscard, discardRows) {
    var traps = [];
    var hasIsolatedMiddle = false;
    for (var t = 0; t < 27; t++) {
      var n = numberOf(t);
      if (n >= 3 && n <= 7 && isIsolatedTile(counts14, t)) {
        hasIsolatedMiddle = true;
        break;
      }
    }

    if (hasIsolatedMiddle && isIsolatedTile(counts14, bestDiscard)) {
      if (isHonor(bestDiscard)) {
        traps.push("isolated-honor");
      } else {
        var bestNumber = numberOf(bestDiscard);
        if (bestNumber <= 2 || bestNumber >= 8) traps.push("float-quality");
      }
    }

    var pairTiles = [];
    for (t = 0; t < 34; t++) if (counts14[t] === 2) pairTiles.push(t);
    if (pairTiles.length === 1) {
      var bestRow = discardRows.find(function (r) { return r.discard === bestDiscard; });
      var pairRow = discardRows.find(function (r) { return r.discard === pairTiles[0]; });
      if (bestRow && pairRow && pairRow.ukeire < bestRow.ukeire) traps.push("only-pair");
    }

    if (hasDoubleAcceptance(counts14)) traps.push("double-acceptance");
    return traps;
  }

  // 判定と解説で同じ変化を使う。各候補の計算は一度だけ行う。
  function efficiencyRowsWithVariation(counts14, keepRows) {
    return keepRows.map(function (row) {
      var after = counts14.slice();
      var outside = new Array(34).fill(0);
      after[row.discard]--;
      outside[row.discard]++;
      return Object.assign({}, row, { variation: improvementDetail(after, row.ukeire, outside) });
    }).sort(function (a, b) {
      return b.ukeire - a.ukeire || b.variation.total - a.variation.total;
    });
  }

  // 牌効率モードは打牌だけを比較するため、暗槓との比較が必要になる四枚使いと、
  // 数牌の形より先に字牌切りを選べる問題を出題対象から外す。
  function efficiencyCandidateSetIsAllowed(counts14, keepRows) {
    if (!keepRows.length || counts14.some(function (count) { return count === 4; })) return false;
    return keepRows.every(function (row) { return !isHonor(row.discard); });
  }

  function efficiencyRowsAreSound(counts14, rows) {
    if (!efficiencyCandidateSetIsAllowed(counts14, rows)) return false;
    var best = rows[0];
    // 変化の多さだけでは、元の受け入れが狭い手を過大評価してしまう。
    // 全ツモとその後の最善打牌を調べ、2ツモ以内に進む確率で逆転する手を除く。
    // 正解は受け入れ最大、同数時は変化最大のままで、逆転時は出題しない。
    return rows.every(function (row) {
      return row.variation.twoDrawNumerator <= best.variation.twoDrawNumerator;
    });
  }

  function efficiencyAnswerIsSound(counts14, keepRows) {
    if (!efficiencyCandidateSetIsAllowed(counts14, keepRows)) return false;
    return efficiencyRowsAreSound(counts14, efficiencyRowsWithVariation(counts14, keepRows));
  }

  // 14枚の手牌について、打牌候補ごとの（シャンテン, 受け入れ）を一覧にする
  function analyzeDiscards(counts14, visibleOutside) {
    var rows = [];
    var outside = visibleOutside ? visibleOutside.slice() : new Array(34).fill(0);
    var minShanten = 99;
    for (var d = 0; d < 34; d++) {
      if (counts14[d] === 0) continue;
      counts14[d]--;
      outside[d]++;
      var u = ukeire(counts14, outside);
      counts14[d]++;
      outside[d]--;
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
  // 最大受け入れが同率の場合は、変化（改良ツモ）の枚数が多い方だけを正解とする。
  // 受け入れが同じなら変化の多い形の方が期待値で上回るため。変化まで同数なら出題しない。
  // 通常問題（次点と3枚以上差）と高難度問題（次点と1〜2枚差）を半数ずつ出題する。
  // ---------------------------------------------------------------
  var EFFICIENCY_HARD_RATE = 0.5;
  var EFFICIENCY_TRAP_RATE = 0.5;
  // 字牌候補を除外した後は罠型の供給が減るため、有効問題を確保した後の
  // 追加探索を60回に抑え、罠型探しだけで待ち時間が伸びることを防ぐ。
  var EFFICIENCY_TRAP_SEARCH_LIMIT = 60;
  // 探索の絶対上限。条件を満たす手が見つからないまま無限に回り続けて
  // 画面が固まることを防ぐための保険で、通常はここまで到達しない。
  var EFFICIENCY_MAX_ATTEMPTS = 5000;

  function generateEfficiencyProblem(difficulty, diagnostics) {
    // テストでは難易度を固定できる。通常の画面からは未指定なので半数ずつ選ばれる。
    var targetDifficulty = difficulty === "hard" || difficulty === "standard"
      ? difficulty
      : (Math.random() < EFFICIENCY_HARD_RATE ? "hard" : "standard");
    // 半数は罠型を狙い、残り半数は通常形を狙う。見つからない場合は最初の有効問題を返す。
    var targetHasTrap = Math.random() < EFFICIENCY_TRAP_RATE;
    if (diagnostics) {
      diagnostics.targetDifficulty = targetDifficulty;
      diagnostics.targetHasTrap = targetHasTrap;
    }
    var fallback = null;
    var fallbackAttempt = -1;

    // 1200回を基本上限とするが、有効なフォールバックがまだ無い場合だけ探索を続ける。
    // 条件を満たさないまま上限に達した場合はnullを返し、画面で再試行を案内する。
    for (var attempt = 0; attempt < EFFICIENCY_MAX_ATTEMPTS && (attempt < 1200 || !fallback); attempt++) {
      // 有効問題を確保した後は追加探索を制限し、罠型が見つからない回でも待たせすぎない。
      if (fallback && attempt - fallbackAttempt >= EFFICIENCY_TRAP_SEARCH_LIMIT) {
        if (diagnostics) {
          diagnostics.attempts = attempt;
          diagnostics.usedFallback = true;
        }
        return fallback;
      }

      var wall = newWall();
      var hand = buildStructuredHand(wall);
      if (hand.length !== 14) continue;
      var counts = toCounts(hand);
      // 四枚使いは暗槓との比較が必要になるため、高価な打牌分析より前に除外する。
      if (counts.some(function (count) { return count === 4; })) continue;
      var an = analyzeDiscards(counts, null);
      if (an.minShanten !== 1) continue;
      if (an.keep.length < 2) continue; // 候補が1つだけでは問題にならない
      // 1シャンテンを維持する字牌切りが1種類でもあれば、問題全体を除外する。
      if (!efficiencyCandidateSetIsAllowed(counts, an.keep)) continue;
      var bestU = an.keep[0].ukeire;
      if (bestU <= 0) continue;
      var bests = an.keep.filter(function (r) { return r.ukeire === bestU; });
      var second = an.keep.filter(function (r) { return r.ukeire < bestU; });
      if (bests.length > 2) continue;                       // 正解が多すぎる手は避ける
      if (second.length === 0) continue;                    // 全部同点なら出題しない
      var ukeireGap = bestU - second[0].ukeire;
      if (targetDifficulty === "hard" && bests.length === 1 && ukeireGap > 2) continue;
      if (targetDifficulty === "standard" && (bests.length > 1 || ukeireGap < 3)) continue;

      // この14枚を作った実際のツモが、2シャンテンの13枚からの進展牌か確認する
      var split = splitImprovingDraw(hand, 2);
      if (!split) continue;

      an.keep = efficiencyRowsWithVariation(counts, an.keep);
      if (!efficiencyRowsAreSound(counts, an.keep)) continue;
      bests = an.keep.filter(function (r) { return r.ukeire === bestU; });
      if (bests.length > 1) {
        var maxPot = -1;
        bests.forEach(function (r) { if (r.variation.total > maxPot) maxPot = r.variation.total; });
        var tieWinners = bests.filter(function (r) { return r.variation.total === maxPot; });
        // 変化でしか差が付かない問題は受け入れ枚数差0の実質最難関なので、通常難度では出題しない
        if (tieWinners.length < bests.length && targetDifficulty === "standard") continue;
        bests = tieWinners;
      }
      // 変化まで同じ打牌が複数あれば、優劣を説明できないため出題しない。
      if (bests.length !== 1) continue;
      // 変化で不正解になる同数候補も「次点」に含める。
      ukeireGap = bestU - an.keep[1].ukeire;

      var traps = detectEfficiencyTraps(counts, bests[0].discard, an.all);
      var problem = {
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
        traps: traps,
        redFlags: assignRedFives(split.hand),
      };
      if ((traps.length > 0) === targetHasTrap) {
        if (diagnostics) {
          diagnostics.attempts = attempt + 1;
          diagnostics.usedFallback = false;
        }
        return problem;
      }
      if (!fallback) {
        fallback = problem;
        fallbackAttempt = attempt;
      }
    }
    // 狙った罠有無が見つからなくても、確保済みの有効問題を返して出題を止めない。
    if (diagnostics) {
      diagnostics.attempts = EFFICIENCY_MAX_ATTEMPTS;
      diagnostics.usedFallback = Boolean(fallback);
    }
    return fallback;
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
    return null; // 条件を緩めず、画面で再試行を案内する。
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
    suji_456: { label: "片スジの4・5・6", rate: 4.1 },
    double_suji_middle: { label: "中スジの4・5・6", rate: 2.0 },
    one_chance_19: { label: "ワンチャンスの1・9", rate: 3.0 },
    one_chance_28: { label: "ワンチャンスの2・8", rate: 3.0 },
    no_chance_19: { label: "ノーチャンスの1・9", rate: 2.2 },
    no_chance_28: { label: "ノーチャンスの2・8", rate: 2.2 },
    // 旧APIとの互換用。新しい分類結果では19/28別のカテゴリを返す。
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

    // 壁（ワンチャンス/ノーチャンス）はVault解析と同じく外側1/2/8/9だけを判定し、
    // スジより先に採用する。1/2は1つ内側、8/9も1つ内側の見え枚数を見る。
    var wallNeighbor = -1;
    if (n === 1 || n === 2) wallNeighbor = tile + 1;
    if (n === 8 || n === 9) wallNeighbor = tile - 1;
    if (wallNeighbor >= 0) {
      var wallSeen = visibleCounts[wallNeighbor];
      var wallBand = (n === 1 || n === 9) ? "19" : "28";
      if (wallSeen >= 4) return "no_chance_" + wallBand;
      if (wallSeen === 3) return "one_chance_" + wallBand;
    }

    // スジ判定: 1-3はn+3、7-9はn-3が河にあればスジ。4-6は両側必要（中スジ）
    var sujiLow = n >= 4 ? riverCounts[suitBase + (n - 3) - 1] > 0 : true;
    var sujiHigh = n <= 6 ? riverCounts[suitBase + (n + 3) - 1] > 0 : true;
    var isSuji = (n <= 3 && sujiHigh) || (n >= 7 && sujiLow) || (n >= 4 && n <= 6 && (sujiLow || sujiHigh));
    if (isSuji) {
      if (n >= 4 && n <= 6) return sujiLow && sujiHigh ? "double_suji_middle" : "suji_456";
      if (n === 1 || n === 9) return "suji_19";
      if (n === 2 || n === 8) return "suji_28";
      return "suji_37";
    }

    if (n === 1 || n === 9) return "non_suji_19";
    if (n === 2 || n === 8) return "non_suji_28";
    if (n === 3 || n === 7) return "non_suji_37";
    return "non_suji_456";
  }

  // ---------------------------------------------------------------
  // 押し引きEVモデル（簡易・局収支ベースの概算）
  // 打牌候補ごとに速度・打点・危険度を計算し、候補同士とベタオリを比較する。
  // 定数は統計値そのものではなく、下記の傾向を一つの尺度で比較するための近似:
  //  - 放銃率: DANGER_RATES（科学する麻雀系の統計値）
  //  - リーチ平均打点: 子5300 / 親7700
  //  - vault研究ノート「イーシャンテン危険牌押し条件」:
  //    受け入れ・良形率・ドラ・親子・巡目を分けて評価する必要がある
  // ---------------------------------------------------------------
  function evaluatePushFold(p) {
    // p: { turn, shanten(0|1), ukeire, dangerRate(%), ownValue, oppIsDealer, selfIsDealer }
    var remain = Math.max(1, 18 - p.turn); // 自分に残るツモ回数の目安
    var afterShanten = p.shanten === 0 ? 0 : 1; // 旧APIは1シャンテン扱い

    // 受け入れを約70枚の未知牌に対する抽選として扱う。
    // テンパイは次の有効牌が和了牌、1シャンテンは有効牌を引いた後に
    // もう一段階必要なので、同じ受け入れでも和了率を明確に分ける。
    var effectiveRate = Math.min(0.35, Math.max(0, p.ukeire) / 70);
    var pReachEffective = 1 - Math.pow(1 - effectiveRate, remain);
    var pHandWin;
    if (afterShanten === 0) {
      pHandWin = pReachEffective * 0.62;
      pHandWin = Math.min(0.58, Math.max(0.04, pHandWin));
    } else {
      // 対リーチの1シャンテンは、有効牌を引いてもそこから相手より先に
      // 和了する必要がある。研究ノートの「無筋1シャンテンは平均マイナス」
      // という傾向に合わせ、テンパイ後の先着率を保守的に置く。
      var winAfterAdvance = Math.min(0.24, 0.08 + 0.010 * remain);
      pHandWin = pReachEffective * winAfterAdvance;
      pHandWin = Math.min(0.34, Math.max(0.02, pHandWin));
    }
    // このモデルは手変わりを追わない。残り受け入れ0枚に和了率の下限を与えない。
    if (p.ukeire <= 0) pHandWin = 0;

    // 現在の打牌で放銃すれば和了機会は消えるので、現在牌が通る確率を和了率に掛ける。
    var pNow = p.dangerRate / 100;
    var pWin = (1 - pNow) * pHandWin;

    // 現在牌が通り、かつ先に和了しなかった場合の追加放銃リスク。
    // テンパイは手が完成しているため、1シャンテンより追加の押し回数を少なく見積もる。
    var pFuture = afterShanten === 0
      ? Math.min(0.14, 0.014 * remain)
      : Math.min(0.20, 0.022 * remain);
    var pDeal = pNow + (1 - pNow) * (1 - pHandWin) * pFuture;

    // リーチの和了率（巡目が深いほど残り抽選が減る）
    var pOppWin = Math.min(0.52, 0.045 * remain);

    var dealLoss = p.oppIsDealer ? 7700 : 5300; // 放銃時の平均失点
    // 子のツモに対する親の支払いは、他の子の2倍とする。
    var tsumoPay = p.oppIsDealer ? 2300 : (p.selfIsDealer ? 2800 : 1400);

    // 押しEV = 和了収入 − 放銃失点 − (どちらも和了しない間の)ツモられ失点
    var winGain = p.ownValue + 1000; // 供託リーチ棒込み
    var unresolved = Math.max(0, 1 - pWin - pDeal);
    var pTsumoPush = unresolved * (pOppWin * 0.85) * 0.4;
    var pTsumoFold = pOppWin * 0.4;
    var winIncome = pWin * winGain;
    var dealExpense = pDeal * dealLoss;
    var tsumoExpensePush = pTsumoPush * tsumoPay;
    var tsumoExpenseFold = pTsumoFold * tsumoPay;
    var evPush = winIncome - dealExpense - tsumoExpensePush;

    // オリEV = ツモられ失点のみ（放銃はほぼゼロ）+ テンパイ料などの機会損失
    var evFold = -tsumoExpenseFold - 300;
    var roundedPush = Math.round(evPush);
    var roundedFold = Math.round(evFold);

    return {
      pWin: pWin,
      pDeal: pDeal,
      pOppWin: pOppWin,
      shanten: afterShanten,
      dealLoss: dealLoss,
      tsumoPay: tsumoPay,
      pTsumoPush: pTsumoPush,
      pTsumoFold: pTsumoFold,
      breakdown: { winIncome: winIncome, dealExpense: dealExpense,
        tsumoExpensePush: tsumoExpensePush, tsumoExpenseFold: tsumoExpenseFold, foldCost: 300 },
      evPush: roundedPush,
      evFold: roundedFold,
      answer: roundedPush > roundedFold ? "push" : "fold",
      diff: roundedPush - roundedFold,
    };
  }

  // 指定の牌種を1枚切った後に残る赤5枚数を返す。
  // 同じ5が複数あれば通常牌を切って赤5を残すのが常に有利なので、その選択を採用する。
  function redCountAfterDiscard(hand, redFlags, discard) {
    var redCount = redFlags.filter(function (r) { return r; }).length;
    var copies = 0;
    var hasRed = false;
    for (var i = 0; i < hand.length; i++) {
      if (hand[i] !== discard) continue;
      copies++;
      if (redFlags[i]) hasRed = true;
    }
    if (hasRed && copies === 1) redCount--;
    return redCount;
  }

  function discardsRedFive(hand, redFlags, discard) {
    var before = redFlags.filter(function (r) { return r; }).length;
    return redCountAfterDiscard(hand, redFlags, discard) < before;
  }

  // 現在の簡易打点モデル。通常ドラと赤ドラはどちらも1枚分として加算し、
  // 赤5が通常ドラでもある場合は2枚分になる。親は子の1.5倍で評価する。
  function estimateOwnValue(countsAfterDiscard, dora, akaCount, selfIsDealer) {
    var doraCount = countsAfterDiscard[dora] + akaCount;
    var childValue = Math.min(12000, 4500 + 1500 * doraCount);
    return {
      doraCount: doraCount,
      ownValue: Math.round(childValue * (selfIsDealer ? 1.5 : 1)),
    };
  }

  // ツモ前1シャンテンの手から、ツモ後にテンパイまたは1シャンテンを保つ全打牌を
  // 同じEVモデルで比較する。テンパイ可能でも、安全な1シャンテン維持が高EVに
  // なる場合があるため、最小シャンテンの候補だけに先に絞らない。
  function analyzePushCandidates(p) {
    var counts = toCounts(p.hand);
    var discardAnalysis = analyzeDiscards(counts, p.outsideCounts);
    if (discardAnalysis.minShanten !== 0 && discardAnalysis.minShanten !== 1) return [];
    var candidates = discardAnalysis.all.filter(function (row) {
      return row.shanten === 0 || row.shanten === 1;
    }).map(function (row) {
      var category = classifyDanger(row.discard, p.riverCounts, p.visibleCounts);
      var dangerRate = DANGER_RATES[category].rate;
      var afterCounts = counts.slice();
      afterCounts[row.discard]--;
      var akaCount = redCountAfterDiscard(p.hand, p.redFlags, row.discard);
      var value = estimateOwnValue(afterCounts, p.dora, akaCount, p.selfIsDealer);
      var ev = evaluatePushFold({
        turn: p.turn,
        shanten: row.shanten,
        ukeire: row.ukeire,
        dangerRate: dangerRate,
        ownValue: value.ownValue,
        oppIsDealer: p.oppIsDealer,
        selfIsDealer: p.selfIsDealer,
      });
      return {
        discard: row.discard,
        shanten: row.shanten,
        ukeire: row.ukeire,
        tiles: row.tiles,
        category: category,
        categoryLabel: DANGER_RATES[category].label,
        dangerRate: dangerRate,
        discardsRed: discardsRedFive(p.hand, p.redFlags, row.discard),
        akaCount: akaCount,
        doraCount: value.doraCount,
        ownValue: value.ownValue,
        ev: ev,
      };
    });

    candidates.sort(function (a, b) {
      if (b.ev.evPush !== a.ev.evPush) return b.ev.evPush - a.ev.evPush;
      if (a.dangerRate !== b.dangerRate) return a.dangerRate - b.dangerRate;
      return a.discard - b.discard;
    });
    return candidates;
  }

  // ---------------------------------------------------------------
  // 押し引きモードの問題生成
  // 他家リーチに対し、テンパイまたはイーシャンテンを保つ打牌（危険牌）を押すか、
  // 現物を切ってオリるかを問う。正解はEVの高い方。
  // ---------------------------------------------------------------
  function generateRiver(wall, turn, ownCounts, requiredGenbutsu, forbiddenGenbutsu) {
    // リーチ者の河を巡目分だけ作る。序盤は字牌・端牌寄り、リーチ後はランダム
    var len = turn;
    var riichiIndex = 3 + randInt(Math.max(1, Math.min(4, len - 4))); // 4〜7巡目あたりで宣言
    var river = [];
    for (var i = 0; i < len; i++) {
      var early = i < riichiIndex;
      var t;
      if (i === 0 && requiredGenbutsu >= 0) {
        if (!drawTile(wall, requiredGenbutsu)) return null;
        t = requiredGenbutsu;
      } else {
        t = drawWeighted(wall, function (x) {
          // 攻撃候補そのものを現物にすると危険牌勝負にならない。
          if (forbiddenGenbutsu && forbiddenGenbutsu[x]) return 0;
          if (isHonor(x)) return early ? 5 : 0.7;
          var n = numberOf(x);
          if (n === 1 || n === 9) return early ? 3 : 1;
          if (n === 2 || n === 8) return early ? 1.2 : 1;
          return early ? 0.5 : 1.3;
        });
      }
      if (t < 0) return null;
      river.push({ tile: t, riichi: i === riichiIndex });
    }
    return { tiles: river, riichiIndex: riichiIndex };
  }

  function generatePushFoldProblem() {
    // 押し/オリの正解が偏らないよう、先に目標の答えを決めて合致する局面を探す
    var target = Math.random() < 0.5 ? "push" : "fold";
    var targetTransition = target === "push" && Math.random() < 0.5 ? 0 : null;
    var fallback = null;
    var fallbackAttempt = -1;
    var fallbackSearchLimit = 15;

    // 無理押しガードで候補がかなり絞られるため、試行上限は多めに取る
    for (var attempt = 0; attempt < 2000; attempt++) {
      // 答え比率の調整だけで画面を長時間待たせない。正解条件を満たす問題を
      // 1件確保した後は、反対側の答えを探す追加試行に上限を設ける。
      if (fallback && attempt - fallbackAttempt >= fallbackSearchLimit) return fallback;
      var wall = newWall();
      var hand = buildStructuredHand(wall);
      if (hand.length !== 14) continue;
      // 問題の起点はツモ前13枚の1シャンテン。最後の1枚を実際のツモ牌として並べ直す。
      // ツモ後の最善打牌は1シャンテン維持でもテンパイでもよい。
      var split = splitImprovingDraw(hand, 1);
      if (!split) continue;
      hand = split.hand;
      var counts = toCounts(hand);
      var drawnStateShanten = shanten(counts);
      // オリ問題は、テンパイ打牌が無い1→1の層から探す。河や全候補を作る前に
      // 安価なシャンテン判定で絞り、テンパイ局面を後段で大量に捨てない。
      if (target === "fold" && drawnStateShanten !== 1) continue;
      if (targetTransition === 0 && drawnStateShanten !== 0) continue;

      // 牌姿だけで、攻撃候補（0/1シャンテン）と最善進行から後退するオリ候補を分ける。
      // 河にはオリ候補を現物として1枚含め、それ以外の攻撃候補は現物にしない。
      var shapeAnalysis = analyzeDiscards(counts, null);
      if (shapeAnalysis.minShanten !== 0 && shapeAnalysis.minShanten !== 1) continue;
      var foldShapes = shapeAnalysis.all.filter(function (row) {
        return row.shanten > shapeAnalysis.minShanten && wall[row.discard] > 0;
      });
      if (foldShapes.length === 0) continue;
      var requiredFoldTile = pick(foldShapes).discard;
      var forbiddenGenbutsu = new Array(34).fill(false);
      shapeAnalysis.all.forEach(function (row) {
        if (row.shanten === 0 || row.shanten === 1) forbiddenGenbutsu[row.discard] = true;
      });

      // オリ有利は親リーチ・中終盤・1シャンテン維持に集中するため、
      // 目標回答ごとに該当層を多めに引く。正解そのものは後段の同じEV式で決める。
      var turn = target === "fold" ? 10 + randInt(3) : 6 + randInt(6); // fold: 10〜12 / push: 6〜11巡目
      var riverData = generateRiver(wall, turn, counts, requiredFoldTile, forbiddenGenbutsu);
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

      // 打牌後の形と、オリに使える現物を先に確認する。
      var an = analyzeDiscards(counts, outside);
      if (an.minShanten !== 0 && an.minShanten !== 1) continue;

      // オリ候補 = 現物を切ると最善の進行よりシャンテン数が後退する牌。
      // 1→テンパイ問題では1シャンテン戻し、1→1問題では2シャンテン戻しになる。
      var foldTile = -1;
      for (var a = 0; a < an.all.length; a++) {
        var foldRow = an.all[a];
        if (riverCounts[foldRow.discard] > 0 && foldRow.shanten > an.minShanten) {
          foldTile = foldRow.discard;
          break;
        }
      }
      if (foldTile < 0) continue;

      // 赤あり（各色1枚）の手牌を確定してから、全押し候補を候補固有の
      // 速度・打点・危険度で評価する。
      var redFlags = assignRedFives(hand);
      var oppIsDealer = target === "fold" ? true : Math.random() < 0.25;
      var selfIsDealer = !oppIsDealer && Math.random() < 0.3;
      var candidates = analyzePushCandidates({
        hand: hand,
        redFlags: redFlags,
        riverCounts: riverCounts,
        visibleCounts: visible,
        outsideCounts: outside,
        turn: turn,
        oppIsDealer: oppIsDealer,
        selfIsDealer: selfIsDealer,
        dora: dora,
      });
      if (candidates.length < 2) continue;

      var pushRow = candidates[0];
      var pushTile = pushRow.discard;
      var PUSH_MIN_RATE = 3.0;
      // 勝負牌と同じシャンテン数（またはそれ以上の進行）を安全に保てるなら二択にしない。
      // テンパイを崩す現物は撤退の選択肢なので、この検査からは除く。
      if (candidates.some(function (row) {
        return row.shanten <= pushRow.shanten && row.dangerRate < PUSH_MIN_RATE;
      })) continue;
      if (pushRow.ukeire <= 0) continue;
      if (riverCounts[pushTile] > 0) continue;

      // 候補同士が僅差なら「本当にこの牌が最大」と言い切れないため出題しない。
      var candidateEvGap = pushRow.ev.evPush - candidates[1].ev.evPush;
      if (candidateEvGap < 75) continue;

      var handAkaCount = redFlags.filter(function (r) { return r; }).length;
      var handDoraCount = counts[dora] + handAkaCount;
      var ev = pushRow.ev;

      var problem = {
        hand: hand,
        baseHand: split.baseHand,
        drawnTile: split.drawnTile,
        fromShanten: 1,
        toShanten: pushRow.shanten,
        handShanten: 1,
        turn: turn,
        river: riverData.tiles,
        doraIndicator: doraIndicator,
        dora: dora,
        handDoraCount: handDoraCount,
        handAkaCount: handAkaCount,
        // 旧表示との互換用。手牌にある枚数を指す。
        doraCount: handDoraCount,
        akaCount: handAkaCount,
        redFlags: redFlags,
        oppIsDealer: oppIsDealer,
        selfIsDealer: selfIsDealer,
        pushTile: pushTile,
        pushShanten: pushRow.shanten,
        pushUkeire: pushRow.ukeire,
        pushUkeireTiles: pushRow.tiles,
        pushDiscardsRed: pushRow.discardsRed,
        pushDoraCount: pushRow.doraCount,
        pushAkaCount: pushRow.akaCount,
        foldTile: foldTile,
        category: pushRow.category,
        categoryLabel: pushRow.categoryLabel,
        dangerRate: pushRow.dangerRate,
        safestKeepRate: Math.min.apply(null, candidates.filter(function (row) {
          return row.shanten <= pushRow.shanten;
        }).map(function (row) { return row.dangerRate; })),
        ownValue: pushRow.ownValue,
        candidateEvGap: candidateEvGap,
        candidateAnalysis: candidates,
        ev: ev,
        answer: ev.answer,
      };

      // EV差が小さい微妙な局面は出題しない（正解が議論にならないように）
      if (Math.abs(ev.diff) < 300) continue;
      if (ev.answer === target) return problem;
      if (!fallback) {
        fallback = problem; // 目標の答えが見つからない場合の保険
        fallbackAttempt = attempt;
      }
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
    hasMultipleIsolatedHonors: hasMultipleIsolatedHonors,
    isIsolatedTile: isIsolatedTile,
    hasDoubleAcceptance: hasDoubleAcceptance,
    detectEfficiencyTraps: detectEfficiencyTraps,
    efficiencyCandidateSetIsAllowed: efficiencyCandidateSetIsAllowed,
    efficiencyAnswerIsSound: efficiencyAnswerIsSound,
    analyzeDiscards: analyzeDiscards,
    analyzeChinitsuActions: analyzeChinitsuActions,
    assignRedFives: assignRedFives,
    classifyDanger: classifyDanger,
    evaluatePushFold: evaluatePushFold,
    redCountAfterDiscard: redCountAfterDiscard,
    estimateOwnValue: estimateOwnValue,
    analyzePushCandidates: analyzePushCandidates,
    generateEfficiencyProblem: generateEfficiencyProblem,
    EFFICIENCY_HARD_RATE: EFFICIENCY_HARD_RATE,
    EFFICIENCY_TRAP_RATE: EFFICIENCY_TRAP_RATE,
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
