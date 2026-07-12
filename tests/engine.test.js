// engine.test.js — エンジンの検証テスト（node tests/engine.test.js で実行）
// シャンテン計算・受け入れ計算・危険牌分類・問題生成を既知のケースで確認する。
"use strict";

const assert = require("assert");
const Engine = require("../engine.js");

// 短い記法から牌配列を作るヘルパー。例: parse("123m456p11z")
// m=萬子, p=筒子, s=索子, z=字牌(1東 2南 3西 4北 5白 6發 7中)
function parse(str) {
  const tiles = [];
  let nums = [];
  for (const ch of str) {
    if (ch >= "1" && ch <= "9") {
      nums.push(Number(ch));
    } else {
      const base = { m: 0, p: 9, s: 18, z: 27 }[ch];
      for (const n of nums) tiles.push(base + n - 1);
      nums = [];
    }
  }
  return tiles;
}

function shantenOf(str) {
  return Engine.shanten(Engine.toCounts(parse(str)));
}

let passed = 0;
function check(name, actual, expected) {
  assert.deepStrictEqual(actual, expected, `${name}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
  passed++;
  console.log(`ok - ${name}`);
}

// --- シャンテン計算 ---
check("和了形は-1", shantenOf("123m456m789m123p11s"), -1);
check("両面テンパイは0", shantenOf("123m456m789m12p11s"), 0);
check("イーシャンテン", shantenOf("123m456m78p12s115z"), 1); // 面子2+搭子2+雀頭+浮き牌の1シャンテン
check("九蓮宝燈13枚はテンパイ", shantenOf("1112345678999m"), 0);
check("国士無双13面待ちは0", shantenOf("19m19p19s1234567z"), 0);
check("七対子イーシャンテン", shantenOf("1133557799m224p"), 0); // 6対子=テンパイ
check("バラバラ手は七対子側で6", shantenOf("147m258p369s1234z"), 6);
check("対子4組の2シャンテン", shantenOf("1122m3344p567s19z"), 2);
check("完全イーシャンテン形", shantenOf("123m456m789m1245p"), 1);

// --- 受け入れ計算 ---
{
  // 123m456p789s45m11z → 3m/6m 待ちのテンパイ、受け入れ8枚
  const counts = Engine.toCounts(parse("123m45m456p789s11z"));
  const u = Engine.ukeire(counts, null);
  check("テンパイの受け入れ枚数", u.total, 7); // 3m残り3枚 + 6m残り4枚
  check(
    "受け入れ牌は3mと6m",
    u.tiles.map((x) => Engine.tileShort(x.tile)).sort(),
    ["3m", "6m"]
  );
}
{
  // 見えている牌を渡すと残り枚数が減る（3mが場に2枚見え → 8-2=6枚）
  const counts = Engine.toCounts(parse("123m45m456p789s11z"));
  const outside = new Array(34).fill(0);
  outside[2] = 2; // 3m を2枚見えに
  const u = Engine.ukeire(counts, outside);
  check("場に見えた分は受け入れから引く", u.total, 5); // 3m残り1枚 + 6m残り4枚
}

// --- 打牌分析 ---
{
  // 14枚: 123m456m789m1245p 11z? → 123m456m789m 1245p 1z のような手で確認
  const counts = Engine.toCounts(parse("123m456m789m12457p"));
  const an = Engine.analyzeDiscards(counts, null);
  check("打牌分析の最小シャンテンは1", an.minShanten, 1);
  // 7p切り（123m456m789m 12p 45p）が受け入れ最大級で残るはず
  const keeps = an.keep.map((r) => Engine.tileShort(r.discard));
  check("7p切りがシャンテン維持候補に含まれる", keeps.includes("7p"), true);
}

// --- 危険牌分類 ---
{
  const river = Engine.toCounts(parse("14m1z")); // リーチ者の河: 1m 4m 東
  const visible = river.slice();
  check("河にある牌は現物", Engine.classifyDanger(0, river, visible), "genbutsu"); // 1m
  check("4m切れの1mスジ…7mはスジ37", Engine.classifyDanger(6, river, visible), "suji_37"); // 7m (4m現物)
  check("無スジの5pはnon_suji_456", Engine.classifyDanger(13, river, visible), "non_suji_456"); // 5p
  check("無スジの2sはnon_suji_28", Engine.classifyDanger(19, river, visible), "non_suji_28"); // 2s
  const visible2 = visible.slice();
  visible2[28] = 2; // 南が2枚見え
  check("字牌2枚見え", Engine.classifyDanger(28, river, visible2), "honor_2_visible");
  check("生牌の字牌", Engine.classifyDanger(33, river, visible), "honor_live"); // 中
}
{
  // ワンチャンス: 8pが3枚見えていると9pの両面待ち(78p)が残り1組 → one_chance
  const river = new Array(34).fill(0);
  const visible = new Array(34).fill(0);
  visible[16] = 3; // 8p 3枚見え
  check("8p3枚見えの9pはワンチャンス", Engine.classifyDanger(17, river, visible), "one_chance");
  visible[16] = 4; // 8p 4枚見え
  check("8p4枚見えの9pはノーチャンス", Engine.classifyDanger(17, river, visible), "no_chance");
}

// --- 押し引きEVモデルの妥当性（傾向チェック） ---
{
  // 良形・高打点・序盤の無スジ → 押し有利
  const good = Engine.evaluatePushFold({ turn: 8, ukeire: 20, dangerRate: 5.7, ownValue: 7500, oppIsDealer: false });
  check("高打点・広い受けなら押し", good.answer, "push");
  // 愚形・安手・終盤の無スジ → オリ有利
  const bad = Engine.evaluatePushFold({ turn: 12, ukeire: 6, dangerRate: 5.7, ownValue: 3900, oppIsDealer: true });
  check("安手・狭い受け・親リーチならオリ", bad.answer, "fold");
  // 同条件なら危険度が低いほど押しEVが上がる（単調性）
  const safer = Engine.evaluatePushFold({ turn: 10, ukeire: 12, dangerRate: 2.0, ownValue: 5000, oppIsDealer: false });
  const risker = Engine.evaluatePushFold({ turn: 10, ukeire: 12, dangerRate: 5.7, ownValue: 5000, oppIsDealer: false });
  check("危険度が低いほど押しEVが高い", safer.evPush > risker.evPush, true);
}

// --- 赤5（アカドラ）の割り当て ---
{
  const origRandom = Math.random;
  // 5m×2, 5p×1, 5s×0 を含む手牌で、5m/5pは必ず赤とし5sは無しになるよう固定
  const hand = parse("55m5p678p11z234s");
  Math.random = () => 0; // k/4 判定は常に真（k>0なら)、位置選択は常に先頭を選ぶ
  const redAt = Engine.assignRedFives(hand);
  Math.random = origRandom;
  check("赤5フラグは手牌と同じ長さ", redAt.length, hand.length);
  const redCount = redAt.filter(Boolean).length;
  check("5m,5pにそれぞれ1枚ずつ赤が付く（5sは無いので付かない）", redCount, 2);
  passed++;
  console.log("ok - 赤5の割り当てが手牌の構成に応じて動く");
}
{
  // 5系の牌が無い手では赤5は一切付かない
  const hand = parse("123m456p789s11z22z");
  const redAt = Engine.assignRedFives(hand);
  check("5系の牌が無ければ赤5は0枚", redAt.filter(Boolean).length, 0);
}
{
  // 押し引き問題にも赤5情報が付与され、akaCountとredFlagsの整合性が取れている
  let found = false;
  for (let i = 0; i < 200 && !found; i++) {
    const p = Engine.generatePushFoldProblem();
    if (p.akaCount > 0) {
      found = true;
      const actual = p.redFlags.filter(Boolean).length;
      check("押し引き問題のakaCountはredFlagsの実数と一致", actual, p.akaCount);
    }
  }
  assert.ok(found, "200回中に赤5を含む問題が最低1回は出る（確率的におかしければ検出）");
  passed++;
  console.log("ok - 押し引き問題への赤5反映");
}

// --- 問題生成（形式チェックを複数回） ---
for (let i = 0; i < 20; i++) {
  const p = Engine.generateEfficiencyProblem();
  assert.ok(p, "牌効率問題が生成できる");
  assert.strictEqual(p.hand.length, 14, "手牌は14枚");
  assert.strictEqual(p.baseHand.length, 13, "ツモ前の手牌は13枚");
  assert.deepStrictEqual(p.hand, p.baseHand.concat([p.drawnTile]), "14枚目は実際のツモ牌");
  assert.strictEqual(p.fromShanten, 2, "ツモ前は2シャンテン");
  assert.strictEqual(p.toShanten, 1, "打牌後は1シャンテン");
  assert.strictEqual(Engine.shanten(Engine.toCounts(p.baseHand)), 2, "ツモ前13枚を再計算しても2シャンテン");
  assert.ok(p.bestDiscards.length >= 1 && p.bestDiscards.length <= 2, "正解打牌は1〜2種");
  assert.strictEqual(p.redFlags.length, 14, "赤5フラグは手牌と同じ14要素");

  const analysis = Engine.analyzeDiscards(Engine.toCounts(p.hand));
  assert.strictEqual(analysis.minShanten, 1, "このツモから1シャンテンに進める");
  const bestU = p.analysis[0].ukeire;
  assert.ok(bestU > 0, "テンパイへの最大受け入れは1枚以上");
  for (const row of p.analysis) {
    assert.strictEqual(row.shanten, 1, "候補打牌はすべて1シャンテンに進む");
    assert.ok(row.ukeire <= bestU, "受け入れ降順");
  }
  const expectedBest = p.analysis.filter((r) => r.ukeire === bestU).map((r) => r.discard).sort((a, b) => a - b);
  assert.deepStrictEqual([...p.bestDiscards].sort((a, b) => a - b), expectedBest, "受け入れ最大打牌だけが正解");
  assert.ok(!p.bestDiscards.includes(p.drawnTile), "ツモ切りでは2シャンテンに戻るため正解にならない");

  const second = p.analysis.filter((r) => r.ukeire < bestU);
  assert.ok(second.length > 0 && bestU - second[0].ukeire >= 3, "2位と3枚以上差");
}
passed++;
console.log("ok - 牌効率問題の生成（2シャンテン→1シャンテンを20回確認）");

for (let i = 0; i < 20; i++) {
  const p = Engine.generateChinitsuProblem();
  assert.ok(p, "清一色問題が生成できる");
  assert.strictEqual(p.hand.length, 14, "手牌は14枚");
  assert.strictEqual(p.baseHand.length, 13, "ツモ前の手牌は13枚");
  assert.deepStrictEqual(p.hand, p.baseHand.concat([p.drawnTile]), "14枚目は実際のツモ牌");
  assert.ok(p.hand.every((t) => t >= 0 && t <= 8), "すべて萬子");
  assert.strictEqual(p.fromShanten, 1, "ツモ前は1シャンテン");
  assert.strictEqual(p.toShanten, 0, "打牌後はテンパイ");
  assert.strictEqual(Engine.shanten(Engine.toCounts(p.baseHand)), 1, "ツモ前13枚を再計算しても1シャンテン");
  assert.ok(p.bestDiscards.length >= 1 && p.bestDiscards.length <= 3, "正解打牌は1〜3種");
  assert.strictEqual(p.redFlags.length, 14, "赤5フラグは手牌と同じ14要素");

  const nonManzuOut = new Array(34).fill(4);
  for (let m = 0; m < 9; m++) nonManzuOut[m] = 0;
  const analysis = Engine.analyzeDiscards(Engine.toCounts(p.hand), nonManzuOut);
  assert.strictEqual(analysis.minShanten, 0, "このツモからテンパイに進める");
  const bestU = p.analysis[0].ukeire;
  assert.ok(bestU > 0, "和了牌の最大受け入れは1枚以上");
  for (const row of p.analysis) {
    assert.strictEqual(row.shanten, 0, "候補打牌はすべてテンパイに進む");
    assert.ok(row.ukeire <= bestU, "受け入れ降順");
    // 清一色が崩れる萬子以外は和了牌の受け入れに数えない
    for (const x of row.tiles) assert.ok(x.tile <= 8, "受け入れ牌も萬子のみ");
  }
  const expectedBest = p.analysis.filter((r) => r.ukeire === bestU).map((r) => r.discard).sort((a, b) => a - b);
  assert.deepStrictEqual([...p.bestDiscards].sort((a, b) => a - b), expectedBest, "和了牌の受け入れ最大打牌だけが正解");
  assert.ok(!p.bestDiscards.includes(p.drawnTile), "ツモ切りでは1シャンテンに戻るため正解にならない");

  // 2位と2枚以上の差がある（出題の明確さ条件）
  const second = p.analysis.filter((r) => r.ukeire < bestU);
  assert.ok(second.length > 0 && bestU - second[0].ukeire >= 2, "2位と2枚以上差");
}
passed++;
console.log("ok - 清一色問題の生成（1シャンテン→テンパイを20回確認）");

for (let i = 0; i < 20; i++) {
  const p = Engine.generatePushFoldProblem();
  assert.ok(p, "押し引き問題が生成できる");
  assert.strictEqual(p.hand.length, 14, "手牌は14枚");
  assert.strictEqual(Engine.shanten(Engine.toCounts(p.hand)), 1, "全問イーシャンテン");
  const riverCounts = Engine.toCounts(p.river.map((r) => r.tile));
  assert.strictEqual(riverCounts[p.pushTile], 0, "勝負牌は現物ではない");
  assert.ok(riverCounts[p.foldTile] > 0, "オリ牌は現物");
  assert.ok(p.river.some((r) => r.riichi), "河にリーチ宣言牌がある");
  assert.ok(["push", "fold"].includes(p.answer), "答えはpush/fold");
  assert.ok(Math.abs(p.ev.diff) >= 300, "EV差300点以上の局面のみ出題");
}
passed++;
console.log("ok - 押し引き問題の生成（20回の形式チェック）");

console.log(`\nすべて成功 (${passed} 件)`);
