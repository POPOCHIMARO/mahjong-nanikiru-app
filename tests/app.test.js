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

function runAppWith(problem) {
  const elements = {};
  const app = makeElement("app");
  app.actionButtons = [];
  app.querySelectorAll = function (selector) {
    if (selector !== "button[data-action]") return [];
    const buttons = [];
    const buttonPattern = /<button\b([^>]*)>/g;
    let match;
    while ((match = buttonPattern.exec(this.innerHTML)) !== null) {
      const action = match[1].match(/data-action="([^"]+)"/);
      const tile = match[1].match(/data-tile="([^"]+)"/);
      if (!action || !tile) continue;
      const button = makeElement("action-button");
      button.attributes = { "data-action": action[1], "data-tile": tile[1] };
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
  const storage = {};
  const localStorage = {
    getItem(key) { return storage[key] || null; },
    setItem(key, value) { storage[key] = value; },
  };
  const testEngine = Object.assign({}, Engine, {
    generateEfficiencyProblem: efficiencyProblem,
    generateChinitsuProblem: () => problem,
  });
  const window = { Engine: testEngine, location: { search: "" } };
  const source = fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8");
  vm.runInNewContext(source, { window, document, localStorage, console });

  elements["tab-chin"].click();
  return { app, storage };
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
