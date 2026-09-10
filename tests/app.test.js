// app.test.js — 清一色の打牌・暗槓UIを外部ライブラリなしで確認する。
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const Engine = require("../engine.js");

function parseManzu(digits) {
  return Array.from(digits, (n) => Number(n) - 1);
}

function chinitsuProblem(digits) {
  const hand = parseManzu(digits);
  const counts = Engine.toCounts(hand);
  const outside = new Array(34).fill(4);
  for (let m = 0; m < 9; m++) outside[m] = 0;
  const actionAnalysis = Engine.analyzeChinitsuActions(counts, outside);
  const bestUkeire = actionAnalysis.keep[0].ukeire;
  const bestActions = actionAnalysis.keep
    .filter((row) => row.ukeire === bestUkeire)
    .map((row) => ({ type: row.type, tile: row.tile }));

  return {
    hand,
    baseHand: hand.slice(0, 13),
    drawnTile: hand[13],
    fromShanten: 1,
    toShanten: 0,
    shanten: 0,
    bestActions,
    bestDiscards: bestActions.filter((a) => a.type === "discard").map((a) => a.tile),
    kanOptions: actionAnalysis.kanOptions,
    analysis: actionAnalysis.keep,
    redFlags: new Array(14).fill(false),
  };
}

function efficiencyProblem() {
  const hand = parseManzu("11223344556678");
  return {
    hand,
    baseHand: hand.slice(0, 13),
    drawnTile: hand[13],
    fromShanten: 2,
    toShanten: 1,
    shanten: 1,
    bestDiscards: [7],
    analysis: [{ discard: 7, shanten: 1, ukeire: 4, tiles: [{ tile: 6, count: 4 }] }],
    traps: [],
    redFlags: new Array(14).fill(false),
  };
}

function makeElement(id) {
  return {
    id,
    className: "",
    textContent: "",
    innerHTML: "",
    listeners: {},
    addEventListener(type, handler) { this.listeners[type] = handler; },
    click() { this.listeners.click.call(this); },
    getAttribute(name) { return this.attributes ? this.attributes[name] : null; },
  };
}

function runAppWith(problem, tab = "tab-chin", options = {}) {
  const elements = {};
  const app = makeElement("app");
  app.actionButtons = [];
  app.querySelectorAll = function (selector) {
    const buttons = [];
    const buttonPattern = /<button\b([^>]*)>/g;
    let match;
    while ((match = buttonPattern.exec(this.innerHTML)) !== null) {
      const action = match[1].match(/data-action="([^"]+)"/);
      const tile = match[1].match(/data-tile="([^"]+)"/);
      const choice = match[1].match(/data-choice="([^"]+)"/);
      if (selector === "button[data-action]" ? (!action || !tile) : !choice) continue;
      const button = makeElement("action-button");
      button.attributes = choice ? { "data-choice": choice[1] }
        : { "data-action": action[1], "data-tile": tile[1] };
      buttons.push(button);
    }
    this.actionButtons = buttons;
    return buttons;
  };
  elements.app = app;

  const document = {
    getElementById(id) {
      if (!elements[id]) elements[id] = makeElement(id);
      return elements[id];
    },
  };
  const storage = Object.assign({}, options.storage);
  const localStorage = {
    getItem(key) { return storage[key] || null; },
    setItem(key, value) { storage[key] = value; },
  };
  const testEngine = Object.assign({}, Engine, {
    generateEfficiencyProblem: tab === "tab-eff" ? () => problem : efficiencyProblem,
    generateChinitsuProblem: () => problem,
    generatePushFoldProblem: () => problem,
  }, options.generators);
  const window = { Engine: testEngine, location: { search: "" } };
  const source = fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8");
  const timers = new Map();
  let timerId = 0;
  vm.runInNewContext(source, {
    window, document, localStorage, console,
    setTimeout: fn => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: id => timers.delete(id),
  });

  elements[tab].click();
  return { app, storage, elements, timers,
    runTimers() { const pending = Array.from(timers.values()); timers.clear(); pending.forEach(fn => fn()); } };
}

function findActionButton(app, type, tile) {
  return app.actionButtons.find((button) =>
    button.getAttribute("data-action") === type && Number(button.getAttribute("data-tile")) === tile
  );
}

{
  // 9m切り7枚・9m暗槓3枚のケース。暗槓を選ぶと不正解になる。
  const { app } = runAppWith(chinitsuProblem("11223345689999"));
  assert.strictEqual(app.actionButtons.filter((b) => b.getAttribute("data-action") === "discard").length, 14);
  const ankan9 = findActionButton(app, "ankan", 8);
  assert.ok(ankan9, "四枚ある9mの暗槓ボタンが表示される");
  ankan9.click();
  assert.ok(app.innerHTML.includes("× 不正解"), "9m切りより狭い9m暗槓は不正解");
  assert.ok(app.innerHTML.includes("9萬切り"), "正解行動を9m切りと表示する");
  assert.ok(app.innerHTML.includes("7枚") && app.innerHTML.includes("3枚"), "切りと暗槓の受け入れを別々に表示する");
}

{
  // 9m切りと9m暗槓がともに4枚のケース。暗槓を選んでも正解になる。
  const { app } = runAppWith(chinitsuProblem("13444555679999"));
  const ankan9 = findActionButton(app, "ankan", 8);
  assert.ok(ankan9, "同率ケースにも9m暗槓ボタンが表示される");
  ankan9.click();
  assert.ok(app.innerHTML.includes("○ 正解"), "受け入れ同数の暗槓は正解");
  assert.ok(app.innerHTML.includes("9萬切り または 9萬を暗槓"), "同率の2行動を両方正解として表示する");
}

console.log("ok - 清一色UIで打牌と暗槓を区別し、同率なら両方正解");

{
  // 牌効率モード: 受け入れ同数（8m切り・1m切りとも8枚）だが変化で8m切りだけが正解のケース。
  // 同数の候補を選んでも不正解になり、解説にタイブレークの理由が表示されることを確認する。
  const hand = parseManzu("11223344556678");
  const tieProblem = {
    hand,
    baseHand: hand.slice(0, 13),
    drawnTile: hand[13],
    fromShanten: 2,
    toShanten: 1,
    shanten: 1,
    difficulty: "hard",
    ukeireGap: 3,
    bestDiscards: [7],
    analysis: [
      { discard: 7, shanten: 1, ukeire: 8, tiles: [{ tile: 6, count: 4 }, { tile: 8, count: 4 }],
        variation: { total: 6, tiles: [{ tile: 0, count: 3 }, { tile: 1, count: 3 }] } },
      { discard: 0, shanten: 1, ukeire: 8, tiles: [{ tile: 6, count: 4 }, { tile: 8, count: 4 }],
        variation: { total: 2, tiles: [{ tile: 2, count: 2 }] } },
      { discard: 5, shanten: 1, ukeire: 5, tiles: [{ tile: 6, count: 4 }],
        variation: { total: 0, tiles: [] } },
    ],
    traps: [],
    redFlags: new Array(14).fill(false),
  };
  const { app } = runAppWith(tieProblem, "tab-eff");
  const tiedLoser = findActionButton(app, "discard", 0);
  assert.ok(tiedLoser, "受け入れ同数の1m切りボタンが表示される");
  tiedLoser.click();
  assert.ok(app.innerHTML.includes("× 不正解"), "受け入れ同数でも変化が少ない打牌は不正解");
  assert.ok(app.innerHTML.includes("8萬切り"), "変化最大の8m切りを正解と表示する");
  assert.ok(!app.innerHTML.includes("8萬切り または"), "同数の候補を複数正解として表示しない");
  assert.ok(
    app.innerHTML.includes("受け入れ枚数が同数の候補は、変化（好形へ伸びるツモ）の多い方が正解です。"),
    "タイブレークの理由を解説に表示する"
  );
  assert.ok(!app.innerHTML.includes("ポイント:"), "罠型が空なら罠の指摘を表示しない");
}

console.log("ok - 牌効率UIで受け入れ同数は変化タイブレークの結果と理由を表示");

{
  // 牌効率モードの罠型は、問題に付いた型だけを解説カードへ表示する。
  const trapProblem = efficiencyProblem();
  trapProblem.traps = ["isolated-honor", "float-quality", "only-pair", "double-acceptance"];
  const { app } = runAppWith(trapProblem, "tab-eff");
  findActionButton(app, "discard", 7).click();

  assert.ok(
    app.innerHTML.includes("孤立した字牌は受け入れが3枚しかありません。役牌期待で残すと手が狭くなります。"),
    "孤立字牌の指摘を表示する"
  );
  assert.ok(
    app.innerHTML.includes("孤立牌は3〜7が最も広く受け入れます。端寄りの牌から整理します。"),
    "浮き牌の質の指摘を表示する"
  );
  assert.ok(
    app.innerHTML.includes("手牌で唯一の対子は雀頭候補です。崩すと面子構成の自由度が下がります。"),
    "唯一の対子の指摘を表示する"
  );
  assert.ok(
    app.innerHTML.includes("2つの搭子が同じ牌を待っています（二度受け）。見た目のターツ数ほど受け入れは広くありません。"),
    "二度受けの指摘を表示する"
  );
}

console.log("ok - 牌効率UIで問題に付いた罠型の指摘を表示");

{
  // 押し引きモードは、受け入れ最大ではなく候補別EV最大とシャンテン遷移を表示する。
  const hand = parseManzu("11223344556678");
  const redFlags = new Array(14).fill(false);
  const pushProblem = {
    hand,
    baseHand: hand.slice(0, 13),
    drawnTile: hand[13],
    fromShanten: 1,
    toShanten: 0,
    pushShanten: 0,
    turn: 8,
    river: [{ tile: 7, riichi: true }],
    selfIsDealer: false,
    oppIsDealer: false,
    doraIndicator: 0,
    dora: 1,
    doraCount: 2,
    akaCount: 0,
    redFlags,
    ownValue: 6000,
    pushTile: 0,
    pushDiscardsRed: false,
    pushUkeire: 8,
    foldTile: 7,
  };
  const { app } = runAppWith(pushProblem, "tab-push");
  assert.ok(app.innerHTML.includes("押しEV最大の一打"), "押し引きUIは候補別EV最大と表示する");
  assert.ok(app.innerHTML.includes("ツモ前1シャンテン → テンパイ"), "ツモ前から打牌後へのシャンテン遷移を表示する");
  assert.ok(app.innerHTML.includes("最善打牌後の打点期待"), "打牌後に残る打点期待であることを表示する");
  assert.strictEqual((app.innerHTML.match(/tile-focus/g) || []).length, 1, "実際に切る物理牌1枚だけを枠表示する");
}

console.log("ok - 押し引きUIで候補別EV最大・シャンテン遷移・打牌後打点を表示");

{
  const hand = parseManzu("11223344556678");
  const ev = Engine.evaluatePushFold({ turn: 8, shanten: 0, ukeire: 8,
    dangerRate: 4.3, ownValue: 6000, oppIsDealer: false, selfIsDealer: true });
  const p = { hand, baseHand: hand.slice(0, 13), drawnTile: hand[13], redFlags: Array(14).fill(false),
    pushShanten: 0, turn: 8, river: [{ tile: 7, riichi: true }], selfIsDealer: true, oppIsDealer: false,
    doraIndicator: 0, dora: 1, doraCount: 2, akaCount: 0, ownValue: 6000, pushTile: 0,
    pushUkeire: 8, foldTile: 7, ev, answer: ev.answer, candidateAnalysis: [{}, {}],
    candidateEvGap: 100, categoryLabel: "無スジ", dangerRate: 4.3, pushDoraCount: 2 };
  const { app } = runAppWith(p, "tab-push");
  app.actionButtons.find(b => b.getAttribute("data-choice") === "push").click();
  for (const label of ["押した場合のEV", "オリた場合のEV", "和了収入", "放銃失点", "被ツモ失点", "撤退コスト"]) {
    assert.ok(app.innerHTML.includes(label), label + "を表示する");
  }
  assert.ok(app.innerHTML.includes(ev.evPush.toLocaleString() + "点"));
  assert.ok(app.innerHTML.includes(ev.evFold.toLocaleString() + "点"));
  assert.ok(app.innerHTML.includes("支払い平均 2,800点"));
  assert.ok(!app.innerHTML.includes("リーチ者の和了率:"), "条件付きの内部係数を実際の和了率として表示しない");

  // 各モードで生成失敗を表示し、同じモードの生成器で再試行する。
  for (const [tab, method, valid] of [
    ["tab-eff", "generateEfficiencyProblem", efficiencyProblem()],
    ["tab-chin", "generateChinitsuProblem", chinitsuProblem("11223345689999")],
    ["tab-push", "generatePushFoldProblem", p],
  ]) {
    let ready = false;
    const ui = runAppWith(valid, tab, { generators: { [method]: () => ready ? valid : null } });
    assert.ok(ui.app.innerHTML.includes("見つかりませんでした"));
    ready = true;
    ui.elements["btn-next"].click();
    assert.ok(!ui.app.innerHTML.includes("見つかりませんでした"));
    assert.ok(ui.app.actionButtons.length > 0);
  }
}

{
  let calls = 0;
  const ui = runAppWith(efficiencyProblem(), "tab-eff", { generators: {
    generateEfficiencyProblem: () => { calls++; return efficiencyProblem(); },
    generateChinitsuProblem: () => chinitsuProblem("11223345689999"),
  } });
  findActionButton(ui.app, "discard", 7).click();
  const before = calls;
  ui.runTimers();
  assert.strictEqual(calls, before + 1);
  ui.elements["btn-next"].click();
  assert.strictEqual(calls, before + 1, "次の問題は先読みを消費する");
  findActionButton(ui.app, "discard", 7).click();
  ui.elements["tab-chin"].click();
  ui.runTimers();
  assert.strictEqual(calls, before + 1, "モード変更で古い先読みを取り消す");
}

{
  const ui = runAppWith(efficiencyProblem(), "tab-eff", { storage: {
    "nanikiru-stats": JSON.stringify({ eff: { ok: "9", total: 2 }, push: { ok: 2, total: 3 } }),
  } });
  assert.ok(ui.elements.scoreboard.textContent.includes("0 / 0"), "不正な数値を初期化する");
  findActionButton(ui.app, "discard", 7).click();
  const saved = JSON.parse(ui.storage["nanikiru-stats"]);
  assert.deepStrictEqual(saved.eff, { ok: 1, total: 1 });
  assert.deepStrictEqual(saved.push, { ok: 2, total: 3 }, "正常な他モードの成績は保持する");
  assert.deepStrictEqual(saved.chin, { ok: 0, total: 0 }, "旧保存形式に清一色を補う");
}

console.log("ok - 押し引き回答とEV内訳、全モード再試行、先読み取消、成績復元");
