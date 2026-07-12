// app.js — 画面の描画と回答処理
// エンジン（engine.js）が生成した問題を表示し、回答の正誤判定と解説を行う。
(function () {
  "use strict";

  var E = window.Engine;
  var app = document.getElementById("app");
  var scoreboard = document.getElementById("scoreboard");
  var tabEff = document.getElementById("tab-eff");
  var tabPush = document.getElementById("tab-push");
  var tabChin = document.getElementById("tab-chin");

  // 現在のモードと成績（成績は localStorage に保存して次回も引き継ぐ）
  var mode = "eff";
  var stats = loadStats();
  var current = null;   // 表示中の問題
  var answered = false; // 回答済みかどうか

  function loadStats() {
    try {
      var s = JSON.parse(localStorage.getItem("nanikiru-stats"));
      if (s && s.eff && s.push) {
        // 清一色モード追加前の保存データには chin が無いので補う
        if (!s.chin) s.chin = { ok: 0, total: 0 };
        return s;
      }
    } catch (e) { /* 壊れていたら初期化 */ }
    return { eff: { ok: 0, total: 0 }, push: { ok: 0, total: 0 }, chin: { ok: 0, total: 0 } };
  }
  function saveStats() {
    try { localStorage.setItem("nanikiru-stats", JSON.stringify(stats)); } catch (e) { /* 保存不可でも動作は続ける */ }
  }

  // ---------------------------------------------------------------
  // 牌の描画（自前SVG）
  // 実物の牌の絵柄に合わせて、筒子=円柄、索子=竹柄（1索は鳥）、
  // 萬子=漢数字＋萬、字牌=漢字一文字（白は枠のみ）をSVGで描く。
  // 座標系は牌面1枚 = viewBox 60×84。
  // ---------------------------------------------------------------
  var KANJI_NUM = ["一", "二", "三", "四", "五", "六", "七", "八", "九"];
  var SERIF = "Yu Mincho, Hiragino Mincho ProN, MS Mincho, serif";
  var COL = { red: "#c02c2c", green: "#2e7d32", blue: "#1e56b0", ink: "#233044", face: "#faf9f4" };

  // 筒子の円1個（外輪・白地・中心点の三重丸でコイン風にする）
  function coin(cx, cy, r, color) {
    return (
      '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="' + color + '"/>' +
      '<circle cx="' + cx + '" cy="' + cy + '" r="' + r * 0.6 + '" fill="' + COL.face + '"/>' +
      '<circle cx="' + cx + '" cy="' + cy + '" r="' + r * 0.28 + '" fill="' + color + '"/>'
    );
  }

  // 筒子: 数ごとの円の配置と配色（実物の牌のレイアウトに準拠）
  // isRed指定時は5筒のみ、全ての円を赤にした赤ドラ柄にする
  function pinzuSVG(n, isRed) {
    var R = COL.red, G = COL.green, B = COL.blue;
    if (isRed && n === 5) { R = COL.red; G = COL.red; B = COL.red; }
    if (n === 1) {
      // 1筒は大きな一輪（多重丸の飾り）
      return (
        '<circle cx="30" cy="42" r="21" fill="' + R + '"/>' +
        '<circle cx="30" cy="42" r="16" fill="' + COL.face + '"/>' +
        '<circle cx="30" cy="42" r="12" fill="' + G + '"/>' +
        '<circle cx="30" cy="42" r="7" fill="' + COL.face + '"/>' +
        '<circle cx="30" cy="42" r="3.5" fill="' + R + '"/>'
      );
    }
    var L = {
      2: [[30, 24, 12, B], [30, 60, 12, G]],
      3: [[15, 20, 10, B], [30, 42, 10, R], [45, 64, 10, G]],
      4: [[18, 24, 10, B], [42, 24, 10, G], [18, 60, 10, G], [42, 60, 10, B]],
      5: [[16, 22, 9, B], [44, 22, 9, G], [30, 42, 9, R], [16, 62, 9, G], [44, 62, 9, B]],
      6: [[18, 18, 9, G], [42, 18, 9, G], [18, 42, 9, R], [42, 42, 9, R], [18, 66, 9, R], [42, 66, 9, R]],
      7: [[13, 15, 8, G], [28, 20, 8, G], [43, 25, 8, G], [18, 48, 8, R], [42, 48, 8, R], [18, 69, 8, R], [42, 69, 8, R]],
      8: [[18, 13, 8, B], [42, 13, 8, B], [18, 32, 8, B], [42, 32, 8, B], [18, 52, 8, B], [42, 52, 8, B], [18, 71, 8, B], [42, 71, 8, B]],
      9: [[14, 18, 8, R], [30, 18, 8, G], [46, 18, 8, B], [14, 42, 8, R], [30, 42, 8, G], [46, 42, 8, B], [14, 66, 8, R], [30, 66, 8, G], [46, 66, 8, B]],
    };
    return L[n].map(function (c) { return coin(c[0], c[1], c[2], c[3]); }).join("");
  }

  // 索子の竹1本（丸角の棒に節の白線を2本入れる）
  // angle を指定すると竹の中心を軸に回転させる（8索の八の字配置用）
  function stick(cx, cy, h, color, angle) {
    var w = 7, y = cy - h / 2;
    var body =
      '<rect x="' + (cx - w / 2) + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="2.8" fill="' + color + '"/>' +
      '<rect x="' + (cx - w / 2) + '" y="' + (cy - h * 0.22) + '" width="' + w + '" height="1.8" fill="' + COL.face + '" opacity="0.8"/>' +
      '<rect x="' + (cx - w / 2) + '" y="' + (cy + h * 0.14) + '" width="' + w + '" height="1.8" fill="' + COL.face + '" opacity="0.8"/>';
    if (!angle) return body;
    return '<g transform="rotate(' + angle + " " + cx + " " + cy + ')">' + body + "</g>";
  }

  // 1索の鳥（実物のクジャクを簡略化したもの）
  function birdSVG() {
    var R = COL.red, G = COL.green, Y = "#d69e2e";
    return (
      // 尾羽3枚
      '<path d="M24 50 C 14 42, 9 32, 9 20" stroke="' + R + '" stroke-width="3.5" fill="none" stroke-linecap="round"/>' +
      '<path d="M26 48 C 20 38, 19 28, 22 17" stroke="' + Y + '" stroke-width="3.5" fill="none" stroke-linecap="round"/>' +
      '<path d="M29 47 C 28 37, 31 27, 36 20" stroke="' + G + '" stroke-width="3.5" fill="none" stroke-linecap="round"/>' +
      // 体と頭
      '<ellipse cx="29" cy="57" rx="11" ry="12" fill="' + G + '"/>' +
      '<circle cx="40" cy="46" r="6.5" fill="' + G + '"/>' +
      // くちばしと目
      '<polygon points="46,44 53,46.5 46,49" fill="' + Y + '"/>' +
      '<circle cx="42" cy="45" r="1.4" fill="' + COL.face + '"/>' +
      // 脚
      '<path d="M26 68 L 24 77 M32 68 L 34 77" stroke="' + Y + '" stroke-width="2" stroke-linecap="round"/>'
    );
  }

  // 索子: 数ごとの竹の配置。[cx, cy, 高さ, 色, 角度]（色省略時は緑、角度省略時は垂直）
  // 竹の長さは実物準拠で2種類に統一する:
  //   2段構成（2〜6索・8索の両端）= h31 / 3段構成（7索・9索）= h23 / 8索の中央（斜め）= h26
  // isRed指定時は5索のみ、全ての竹を赤にした赤ドラ柄にする
  function souzuSVG(n, isRed) {
    if (n === 1) return birdSVG();
    var R = COL.red;
    if (isRed && n === 5) {
      var Lred = [[10, 23.5, 31, R], [50, 23.5, 31, R], [30, 42, 31, R], [10, 60.5, 31, R], [50, 60.5, 31, R]];
      return Lred.map(function (s) { return stick(s[0], s[1], s[2], s[3]); }).join("");
    }
    var L = {
      2: [[30, 23.5, 31], [30, 60.5, 31]],
      3: [[30, 23.5, 31], [18, 60.5, 31], [42, 60.5, 31]],
      4: [[10, 23.5, 31], [50, 23.5, 31], [10, 60.5, 31], [50, 60.5, 31]],
      5: [[10, 23.5, 31], [50, 23.5, 31], [30, 42, 31, R], [10, 60.5, 31], [50, 60.5, 31]],
      6: [[10, 23.5, 31], [30, 23.5, 31], [50, 23.5, 31], [10, 60.5, 31], [30, 60.5, 31], [50, 60.5, 31]],
      7: [[30, 14, 23, R], [10, 42, 23], [30, 42, 23], [50, 42, 23], [10, 70, 23], [30, 70, 23], [50, 70, 23]],
      // 8索は実物準拠: 両端は垂直の長い竹、中央2本だけを傾けて組む。
      // 上段は中央2本が上端で合流するΛ形（枠付きでW形）、下段はその鏡像のV形（M形）
      8: [
        [10, 23.5, 31, null, 0], [24.5, 26, 26, null, 19], [35.5, 26, 26, null, -19], [50, 23.5, 31, null, 0],
        [10, 60.5, 31, null, 0], [24.5, 58, 26, null, -19], [35.5, 58, 26, null, 19], [50, 60.5, 31, null, 0],
      ],
      9: [[10, 14, 23], [30, 14, 23], [50, 14, 23], [10, 42, 23, R], [30, 42, 23, R], [50, 42, 23, R], [10, 70, 23], [30, 70, 23], [50, 70, 23]],
    };
    return L[n].map(function (s) { return stick(s[0], s[1], s[2], s[3] || COL.green, s[4]); }).join("");
  }

  // 萬子: 上に漢数字（黒）、下に「萬」（赤）
  // isRed指定時は5萬のみ、漢数字も赤にした赤ドラ柄にする（実物の赤五萬は全体が赤刷り）
  function manzuSVG(n, isRed) {
    var numColor = isRed && n === 5 ? COL.red : COL.ink;
    return (
      '<text x="30" y="34" text-anchor="middle" font-size="30" font-weight="bold" font-family="' + SERIF + '" fill="' + numColor + '">' + KANJI_NUM[n - 1] + "</text>" +
      '<text x="30" y="74" text-anchor="middle" font-size="30" font-weight="bold" font-family="' + SERIF + '" fill="' + COL.red + '">萬</text>'
    );
  }

  // 字牌: 東南西北=黒、發=緑、中=赤、白=青い枠のみ
  function honorSVG(t) {
    if (t === 31) { // 白
      return '<rect x="12" y="12" width="36" height="60" rx="4" fill="none" stroke="' + COL.blue + '" stroke-width="3"/>';
    }
    var color = t === 32 ? COL.green : t === 33 ? COL.red : COL.ink;
    return '<text x="30" y="58" text-anchor="middle" font-size="44" font-weight="bold" font-family="' + SERIF + '" fill="' + color + '">' + E.tileName(t) + "</text>";
  }

  function tileHTML(t, opts) {
    opts = opts || {};
    var cls = "tile";
    if (opts.mini) cls += " tile-mini";
    if (opts.extraClass) cls += " " + opts.extraClass;
    var body;
    var isRed = !!opts.red;
    if (t >= 27) {
      body = honorSVG(t);
    } else {
      var n = (t % 9) + 1;
      var suit = Math.floor(t / 9);
      body = suit === 0 ? manzuSVG(n, isRed) : suit === 1 ? pinzuSVG(n, isRed) : souzuSVG(n, isRed);
    }
    var label = E.tileName(t) + (isRed ? "（赤）" : "");
    return (
      '<span class="' + cls + '" role="img" aria-label="' + label + '">' +
      '<svg viewBox="0 0 60 84" aria-hidden="true">' + body + "</svg></span>"
    );
  }

  // 牌のリストをまとめて描画（解説用の小さい牌）
  // 要素は牌インデックスまたは {tile, count} オブジェクト（1萬=0 が falsy なので || では判定しない）
  function tileListHTML(tiles) {
    return tiles.map(function (x) {
      var t = typeof x === "object" ? x.tile : x;
      return tileHTML(t, { mini: true });
    }).join("");
  }

  // ---------------------------------------------------------------
  // 共通UI
  // ---------------------------------------------------------------
  function updateTabs() {
    var active = "flex-1 py-2.5 rounded-lg font-bold text-sm sm:text-base transition-colors bg-amber-400 text-emerald-950 shadow-lg";
    var inactive = "flex-1 py-2.5 rounded-lg font-bold text-sm sm:text-base transition-colors bg-emerald-800/60 text-emerald-100 hover:bg-emerald-700/60";
    tabEff.className = mode === "eff" ? active : inactive;
    tabChin.className = mode === "chin" ? active : inactive;
    tabPush.className = mode === "push" ? active : inactive;
  }

  var MODE_LABELS = { eff: "牌効率", chin: "清一色", push: "押し引き" };

  function updateScoreboard() {
    var s = stats[mode];
    var pct = s.total > 0 ? Math.round((s.ok / s.total) * 100) : 0;
    scoreboard.textContent =
      MODE_LABELS[mode] + "の成績: " + s.ok + " / " + s.total + " 問正解" +
      (s.total > 0 ? "（正解率 " + pct + "%）" : "");
  }

  function card(html, extra) {
    return '<section class="bg-emerald-900/70 border border-emerald-700/50 rounded-xl p-4 sm:p-5 shadow-xl ' + (extra || "") + '">' + html + "</section>";
  }

  function resultBanner(ok) {
    return ok
      ? '<div class="text-xl font-bold text-green-400 mb-2">○ 正解！</div>'
      : '<div class="text-xl font-bold text-red-400 mb-2">× 不正解…</div>';
  }

  function nextButtonHTML() {
    return '<div class="text-center mt-4"><button id="btn-next" class="px-8 py-2.5 bg-amber-400 hover:bg-amber-300 text-emerald-950 font-bold rounded-lg shadow-lg">次の問題へ</button></div>';
  }

  function record(ok) {
    stats[mode].total++;
    if (ok) stats[mode].ok++;
    saveStats();
    updateScoreboard();
  }

  // ---------------------------------------------------------------
  // 牌効率モード・清一色モード
  // 出題形式（14枚から1枚切る・受け入れ最大が正解）が同じなので描画を共用する。
  // ---------------------------------------------------------------
  function newEffProblem() {
    current = mode === "chin" ? E.generateChinitsuProblem() : E.generateEfficiencyProblem();
    answered = false;
    // 問題生成時に確定した「ツモ前13枚」と「実際のツモ牌」をそのまま表示する
    current.displayHand = current.baseHand.slice();
    current.displayTsumo = current.drawnTile;
    current.displayHandRed = current.redFlags.slice(0, 13);
    current.displayTsumoRed = current.redFlags[13];
    renderEff();
  }

  function shantenLabel(value) {
    return value === 0 ? "テンパイ" : value + "シャンテン";
  }

  function renderEff(clicked) {
    var p = current;
    var fromLabel = shantenLabel(p.fromShanten);
    var toLabel = shantenLabel(p.toShanten);
    var qHtml =
      '<div class="text-sm text-emerald-200/90 mb-3">' +
      "<span class='font-bold text-amber-300'>" + (mode === "chin" ? "清一色（萬子）の" : "") + fromLabel + "。</span> " +
      toLabel + "に進められるツモです。何を切る？（牌をタップ）" +
      "</div>" +
      '<div class="flex flex-wrap items-center gap-1" id="hand-area">' +
      p.displayHand.map(function (t, i) { return handTileBtn(t, "h" + i, clicked, p.displayHandRed[i]); }).join("") +
      '<span class="w-3"></span>' +
      handTileBtn(p.displayTsumo, "tsumo", clicked, p.displayTsumoRed) +
      "</div>";

    var html = card(qHtml);

    if (answered) {
      var ok = p.bestDiscards.indexOf(clicked) >= 0;
      var answerReason = p.toShanten === 0
        ? "テンパイに取り、和了牌の受け入れ枚数が最大になります。"
        : "1シャンテンに取り、テンパイへの受け入れ枚数が最大になります。";
      var expl =
        resultBanner(ok) +
        '<div class="text-sm mb-3">正解は <span class="font-bold text-amber-300">' +
        p.bestDiscards.map(E.tileName).join(" または ") +
        "切り</span>。" + answerReason + "</div>" +
        '<div class="overflow-x-auto"><table class="w-full text-sm">' +
        '<thead><tr class="text-emerald-300/80 text-left border-b border-emerald-700">' +
        '<th class="py-1.5 pr-2">打牌</th><th class="py-1.5 pr-2">受け入れ</th><th class="py-1.5">受け入れ牌</th></tr></thead><tbody>' +
        p.analysis.slice(0, 6).map(function (r) {
          var isBest = p.bestDiscards.indexOf(r.discard) >= 0;
          return '<tr class="border-b border-emerald-800/60 ' + (isBest ? "bg-emerald-800/40" : "") + '">' +
            '<td class="py-1.5 pr-2 whitespace-nowrap">' + tileHTML(r.discard, { mini: true }) +
            (isBest ? ' <span class="text-amber-300 font-bold">◎</span>' : "") + "</td>" +
            '<td class="py-1.5 pr-2 font-bold whitespace-nowrap">' + r.ukeire + "枚</td>" +
            '<td class="py-1.5"><div class="flex flex-wrap gap-0.5">' + tileListHTML(r.tiles) + "</div></td></tr>";
        }).join("") +
        "</tbody></table></div>" +
        '<div class="text-xs text-emerald-300/70 mt-3">※ ' + toLabel + "に進む打牌の中で受け入れ枚数を比較しています（上位6候補まで表示）。" +
        (mode === "chin" ? "清一色が崩れる萬子以外の受け入れは数えません。" : "") + "</div>" +
        nextButtonHTML();
      html += card(expl);
    }

    app.innerHTML = html;

    if (!answered) {
      app.querySelectorAll("button[data-tile]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          if (answered) return;
          answered = true;
          var t = Number(btn.getAttribute("data-tile"));
          record(current.bestDiscards.indexOf(t) >= 0);
          renderEff(t);
        });
      });
    } else {
      document.getElementById("btn-next").addEventListener("click", newEffProblem);
    }
  }

  // 手牌1枚分のボタン。回答後は正解◎・自分の選択に枠を付ける
  function handTileBtn(t, key, clicked, isRed) {
    var extra = "";
    if (answered) {
      var isBest = current.bestDiscards.indexOf(t) >= 0;
      if (isBest) extra = "tile-correct";
      else if (t === clicked) extra = "tile-wrong";
    }
    var inner = tileHTML(t, { extraClass: extra, red: isRed });
    return '<button class="tile-btn bg-transparent border-0 p-0" data-tile="' + t + '" data-key="' + key + '"' +
      (answered ? " disabled" : "") + ">" + inner + "</button>";
  }

  // ---------------------------------------------------------------
  // 押し引きモード
  // ---------------------------------------------------------------
  function newPushProblem() {
    current = E.generatePushFoldProblem();
    answered = false;
    renderPush(null);
  }

  function renderPush(chosen) {
    var p = current;

    // 場況カード
    var info =
      '<div class="flex flex-wrap gap-x-4 gap-y-1 text-sm">' +
      infoItem("巡目", p.turn + "巡目") +
      infoItem("自分", p.selfIsDealer ? "親" : "子") +
      infoItem("リーチ者", p.oppIsDealer ? "親" : "子") +
      infoItem("自分の手", "イーシャンテン") +
      '<span class="inline-flex items-center gap-1">' +
      '<span class="text-emerald-300/80">ドラ表示牌</span>' + tileHTML(p.doraIndicator, { mini: true }) +
      '<span class="text-emerald-300/80 ml-1">→ ドラ</span>' + tileHTML(p.dora, { mini: true }) +
      (p.doraCount > 0 ? '<span class="text-amber-300 font-bold ml-1">手牌に' + p.doraCount + "枚</span>" : "") +
      (p.akaCount > 0 ? '<span class="text-amber-300 ml-1">（うち赤5 ' + p.akaCount + "枚）</span>" : "") +
      "</span></div>";

    // リーチ者の河（6枚ずつ折り返し、宣言牌は横向き）
    var riverHtml =
      '<div class="text-xs text-emerald-300/80 mb-2">リーチ者の捨て牌（<span class="text-amber-300">横向き＝リーチ宣言牌</span>）</div>' +
      '<div class="flex flex-wrap gap-1 items-center">' +
      p.river.map(function (r) {
        return tileHTML(r.tile, { mini: true, extraClass: r.riichi ? "tile-riichi" : "" });
      }).join("") +
      "</div>";

    // 自分の手牌と設問
    var handHtml =
      '<div class="text-xs text-emerald-300/80 mb-2">自分の手牌（打点期待: 約' + p.ownValue.toLocaleString() + "点）</div>" +
      '<div class="flex flex-wrap gap-1 mb-4">' +
      p.hand.map(function (t, i) {
        var extra = "";
        if (t === p.pushTile) extra = "tile-focus";
        return tileHTML(t, { extraClass: extra, red: p.redFlags[i] });
      }).join("") +
      "</div>" +
      '<div class="text-sm mb-3">受け入れ最大の一打は <span class="text-amber-300 font-bold">' +
      E.tileName(p.pushTile) + "</span>（枠付きの牌・受け入れ" + p.pushUkeire + "枚）。ただしリーチに通っていません。どうする？</div>" +
      '<div class="grid grid-cols-1 sm:grid-cols-2 gap-2">' +
      choiceBtn("push", "押す", E.tileName(p.pushTile) + " を切って手を進める", chosen) +
      choiceBtn("fold", "オリる", "現物の " + E.tileName(p.foldTile) + " を切って撤退する", chosen) +
      "</div>";

    var html = card(info) + card(riverHtml) + card(handHtml);

    if (answered) {
      var ok = chosen === p.answer;
      var ev = p.ev;
      var expl =
        resultBanner(ok) +
        '<div class="text-sm mb-3">正解は <span class="font-bold text-amber-300">' +
        (p.answer === "push" ? "押す" : "オリる") + "</span>（期待値の差 約" + Math.abs(ev.diff).toLocaleString() + "点）。</div>" +
        '<div class="grid grid-cols-2 gap-2 text-center text-sm mb-3">' +
        evBox("押した場合のEV", ev.evPush, p.answer === "push") +
        evBox("オリた場合のEV", ev.evFold, p.answer === "fold") +
        "</div>" +
        '<ul class="text-sm space-y-1.5 text-emerald-100/90">' +
        li("勝負牌 " + E.tileName(p.pushTile) + " は「" + p.categoryLabel + "」— 放銃率 約" + p.dangerRate + "%") +
        li("押し切った場合の総放銃リスク: 約" + Math.round(ev.pDeal * 100) + "%（放銃時 平均 −" + ev.dealLoss.toLocaleString() + "点）") +
        li("押した場合の自分の和了率: 約" + Math.round(ev.pWin * 100) + "%（打点期待 約" + p.ownValue.toLocaleString() + "点 + 供託）") +
        li("リーチ者の和了率: 約" + Math.round(ev.pOppWin * 100) + "%") +
        "</ul>" +
        '<div class="text-xs text-emerald-300/70 mt-3 leading-relaxed">' +
        "※ 局収支ベースの概算モデルです。放銃率は統計にもとづく近似値、リーチ平均打点は子5,300点／親7,700点で計算。<br>" +
        "参考: Mリーグの牌譜集計では、他家リーチに対しイーシャンテンの選手が手を維持して押す割合は約78%（vault研究ノート「Mリーグ対リーチ押し率」より）。ただし安全牌での維持も含む数値です。" +
        "</div>" +
        nextButtonHTML();
      html += card(expl);
    }

    app.innerHTML = html;

    if (!answered) {
      app.querySelectorAll("button[data-choice]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          if (answered) return;
          answered = true;
          var c = btn.getAttribute("data-choice");
          record(c === current.answer);
          renderPush(c);
        });
      });
    } else {
      document.getElementById("btn-next").addEventListener("click", newPushProblem);
    }
  }

  function infoItem(label, value) {
    return '<span><span class="text-emerald-300/80">' + label + "</span> <span class='font-bold'>" + value + "</span></span>";
  }

  function li(text) {
    return '<li class="flex gap-2"><span class="text-amber-300">・</span><span>' + text + "</span></li>";
  }

  function evBox(label, value, isAnswer) {
    var color = isAnswer ? "border-amber-400 bg-amber-400/10" : "border-emerald-700 bg-emerald-900/40";
    return '<div class="border rounded-lg p-2.5 ' + color + '">' +
      '<div class="text-xs text-emerald-300/80">' + label + "</div>" +
      '<div class="text-lg font-bold ' + (value >= 0 ? "text-green-300" : "text-red-300") + '">' +
      (value >= 0 ? "+" : "") + value.toLocaleString() + "点</div>" +
      (isAnswer ? '<div class="text-[11px] text-amber-300 font-bold">こちらが有利</div>' : '<div class="text-[11px]">&nbsp;</div>') +
      "</div>";
  }

  function choiceBtn(kind, title, desc, chosen) {
    var base = "text-left rounded-lg px-4 py-3 border transition-colors ";
    var style;
    if (!answered) {
      style = kind === "push"
        ? "bg-red-900/50 border-red-500/60 hover:bg-red-800/60"
        : "bg-sky-900/50 border-sky-500/60 hover:bg-sky-800/60";
    } else {
      var isAnswer = current.answer === kind;
      var isChosen = chosen === kind;
      style = isAnswer ? "bg-green-900/50 border-green-400" :
        isChosen ? "bg-red-900/50 border-red-400 opacity-80" : "bg-emerald-900/40 border-emerald-700 opacity-60";
    }
    return '<button data-choice="' + kind + '" class="' + base + style + '"' + (answered ? " disabled" : "") + ">" +
      '<div class="font-bold text-base">' + title + "</div>" +
      '<div class="text-xs opacity-90 mt-0.5">' + desc + "</div></button>";
  }

  // ---------------------------------------------------------------
  // 起動・タブ切り替え
  // ---------------------------------------------------------------
  function switchMode(m) {
    mode = m;
    updateTabs();
    updateScoreboard();
    if (m === "push") newPushProblem(); else newEffProblem();
  }

  tabEff.addEventListener("click", function () { switchMode("eff"); });
  tabChin.addEventListener("click", function () { switchMode("chin"); });
  tabPush.addEventListener("click", function () { switchMode("push"); });

  // 牌デザインの確認用ギャラリー（index.html?gallery=1 で全34種を一覧表示）
  function renderGallery() {
    var rows = [
      { label: "萬子", from: 0, to: 8 },
      { label: "筒子", from: 9, to: 17 },
      { label: "索子", from: 18, to: 26 },
      { label: "字牌", from: 27, to: 33 },
    ];
    var akaRow =
      '<div class="mb-3"><div class="text-xs text-emerald-300/80 mb-1">赤5（アカドラ）</div>' +
      '<div class="flex flex-wrap gap-1">' +
      tileHTML(4, { red: true }) + tileHTML(13, { red: true }) + tileHTML(22, { red: true }) +
      '<span class="w-3"></span>' + tileHTML(4) + tileHTML(13) + tileHTML(22) +
      "</div></div>";
    app.innerHTML = card(
      '<div class="text-sm text-emerald-200/90 mb-3">牌デザイン一覧（確認用）</div>' +
      rows.map(function (r) {
        var tiles = [];
        for (var t = r.from; t <= r.to; t++) tiles.push(tileHTML(t));
        return '<div class="mb-3"><div class="text-xs text-emerald-300/80 mb-1">' + r.label + "</div>" +
          '<div class="flex flex-wrap gap-1">' + tiles.join("") + "</div></div>";
      }).join("") + akaRow
    );
  }

  if (window.location.search.indexOf("gallery") >= 0) {
    updateTabs();
    renderGallery();
  } else {
    switchMode("eff");
  }
})();
