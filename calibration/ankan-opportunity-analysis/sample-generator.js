// 現在の出題器の構成を調べる補助調査。実際の対局の出現頻度は推定しない。
// 実行: node calibration/ankan-opportunity-analysis/sample-generator.js <出力先>
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const enginePath = path.resolve(__dirname, '../../engine.js');
const Engine = require(enginePath);
const seed = 20260911;
const perDifficulty = 100;
const outputPath = path.resolve(process.argv[2] || path.join(__dirname, 'generated-sample.json'));

// Mulberry32を使用し、問題生成器が使う乱数を一つの再現可能な列にする。
function mulberry32(initialSeed) {
  let state = initialSeed >>> 0;
  return function random() {
    let value = state += 0x6D2B79F5;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function compactHand(hand) {
  const counts = Engine.toCounts(hand);
  return ['m', 'p', 's', 'z'].map((suit, suitIndex) => {
    const size = suitIndex === 3 ? 7 : 9;
    let digits = '';
    for (let index = 0; index < size; index++) {
      digits += String(index + 1).repeat(counts[suitIndex * 9 + index]);
    }
    return digits ? digits + suit : '';
  }).join('');
}

const flagNames = [
  'anyQuad', 'correctDiscardFromQuad', 'honorDiscardCandidate',
  'correctHonorDiscard', 'isolatedHonor', 'anyTrap', 'isolatedHonorTrap',
  'quadOrHonorCandidate', 'quadAndHonorCandidate',
];

function summarize(records) {
  const counts = Object.fromEntries(flagNames.map(name => [name, 0]));
  const trapCounts = {};
  for (const record of records) {
    for (const name of flagNames) counts[name] += Number(record.flags[name]);
    for (const trap of record.traps) trapCounts[trap] = (trapCounts[trap] || 0) + 1;
  }
  return {
    n: records.length,
    counts,
    percentages: Object.fromEntries(flagNames.map(name => [name, counts[name] / records.length * 100])),
    trapCounts,
  };
}

const started = Date.now();
const originalRandom = Math.random;
const records = [];
const validation = {
  requested: perDifficulty * 2,
  generated: 0,
  generationFailures: 0,
  handCountViolations: 0,
  answerConditionViolations: 0,
};
Math.random = mulberry32(seed);
try {
  // standardを100問生成した後、同じ乱数列を継続してhardを100問生成する。
  for (const difficulty of ['standard', 'hard']) {
    for (let index = 0; index < perDifficulty; index++) {
      const problem = Engine.generateEfficiencyProblem(difficulty);
      if (!problem) {
        validation.generationFailures++;
        continue;
      }
      validation.generated++;
      const counts = Engine.toCounts(problem.hand);
      const handCountsAreValid = problem.hand.length === 14
        && problem.baseHand.length === 13
        && problem.hand.every((tile, handIndex) => tile === problem.baseHand.concat([problem.drawnTile])[handIndex])
        && counts.reduce((sum, count) => sum + count, 0) === 14
        && counts.every(count => count >= 0 && count <= 4);
      if (!handCountsAreValid) validation.handCountViolations++;
      const quadTiles = counts.flatMap((count, tile) => count === 4 ? [tile] : []);
      const honorCandidates = problem.analysis.filter(row => row.discard >= 27).map(row => row.discard);
      const isolatedHonors = counts.flatMap((count, tile) => tile >= 27 && count === 1 ? [tile] : []);
      const bestUkeire = problem.analysis[0].ukeire;
      const topRows = problem.analysis.filter(row => row.ukeire === bestUkeire);
      const maxVariation = Math.max(...topRows.map(row => row.variation.total));
      const expectedBest = topRows.filter(row => row.variation.total === maxVariation).map(row => row.discard);
      const second = problem.analysis.filter(row => !expectedBest.includes(row.discard));
      const actualGap = second.length ? bestUkeire - second[0].ukeire : null;
      const difficultyIsValid = difficulty === 'standard'
        ? topRows.length === expectedBest.length && actualGap >= 3
        : actualGap >= 0 && actualGap <= 2;
      const answerIsValid = problem.difficulty === difficulty
        && problem.bestDiscards.length === 1
        && JSON.stringify(problem.bestDiscards) === JSON.stringify(expectedBest)
        && problem.ukeireGap === actualGap
        && Engine.shanten(Engine.toCounts(problem.baseHand)) === 2
        && problem.analysis.every(row => row.shanten === 1 && row.ukeire <= bestUkeire)
        && problem.analysis.every(row => row.variation.twoDrawNumerator <= problem.analysis[0].variation.twoDrawNumerator)
        && Engine.efficiencyCandidateSetIsAllowed(counts, problem.analysis)
        && difficultyIsValid;
      if (!answerIsValid) validation.answerConditionViolations++;
      const flags = {
        anyQuad: quadTiles.length > 0,
        correctDiscardFromQuad: problem.bestDiscards.some(tile => quadTiles.includes(tile)),
        honorDiscardCandidate: honorCandidates.length > 0,
        correctHonorDiscard: problem.bestDiscards.some(tile => tile >= 27),
        isolatedHonor: isolatedHonors.length > 0,
        anyTrap: problem.traps.length > 0,
        isolatedHonorTrap: problem.traps.includes('isolated-honor'),
        quadOrHonorCandidate: quadTiles.length > 0 || honorCandidates.length > 0,
        quadAndHonorCandidate: quadTiles.length > 0 && honorCandidates.length > 0,
      };
      records.push({
        id: `${difficulty}-${String(index + 1).padStart(3, '0')}`,
        difficulty, handNotation: compactHand(problem.hand), hand: problem.hand,
        baseHand: problem.baseHand, drawnTile: problem.drawnTile,
        bestDiscards: problem.bestDiscards, ukeireGap: problem.ukeireGap,
        traps: problem.traps, quadTiles, honorCandidates, isolatedHonors, flags,
        analysis: problem.analysis.map(row => ({
          discard: row.discard, shanten: row.shanten, ukeire: row.ukeire,
          variationTotal: row.variation.total,
          twoDrawNumerator: row.variation.twoDrawNumerator,
        })),
      });
      if (records.length % 20 === 0) {
        process.stdout.write(`generated ${records.length}/${perDifficulty * 2}; elapsed ${((Date.now() - started) / 1000).toFixed(1)}s\n`);
      }
    }
  }
} finally {
  Math.random = originalRandom;
}

const byDifficulty = Object.fromEntries(['standard', 'hard'].map(difficulty => [
  difficulty, summarize(records.filter(record => record.difficulty === difficulty)),
]));
const aggregate = summarize(records);
const duplicateHandNotations = Object.entries(records.reduce((counts, record) => {
  counts[record.handNotation] = (counts[record.handNotation] || 0) + 1;
  return counts;
}, {})).filter(([, count]) => count > 1).map(([handNotation, count]) => ({ handNotation, count }));
const ukeireGapDistribution = records.reduce((counts, record) => {
  counts[record.ukeireGap] = (counts[record.ukeireGap] || 0) + 1;
  return counts;
}, {});
const reviewSelection = Object.fromEntries(['standard', 'hard'].map(difficulty => [
  difficulty,
  records.filter(record => record.difficulty === difficulty && (Number(record.id.slice(-3)) - 1) % 10 === 0)
    .map(record => record.id),
]));

const output = {
  metadata: {
    study: 'current efficiency generator sample; NOT real game frequency',
    engineSha256: crypto.createHash('sha256').update(fs.readFileSync(enginePath)).digest('hex'),
    seed, prng: 'Mulberry32', requestedPerDifficulty: perDifficulty,
    generationOrder: '100 standard, then 100 hard, one continuous PRNG stream',
    elapsedSeconds: (Date.now() - started) / 1000,
    nodeVersion: process.version,
    definitions: {
      anyQuad: 'Any tile with count=4 in the 14-tile hand. Proxy only; NOT verified legal ankan.',
      honorDiscardCandidate: 'At least one honor discard in problem.analysis, the shanten-preserving candidates.',
      isolatedHonor: 'An honor occurring exactly once in the 14-tile hand.',
      correctDiscardFromQuad: 'The generator-selected bestDiscards includes a tile present four times.',
      distribution: 'Artificial generator uses structured hands, difficulty filters and trap targeting. Not representative of M.League games.',
    },
    validation: 'Hand bounds, generation failures, answer conditions, difficulty totals, flag sums and union identity are asserted.',
  },
  validation, aggregate, byDifficulty, duplicateHandNotations, ukeireGapDistribution, reviewSelection, records,
};
fs.writeFileSync(outputPath, JSON.stringify(output, null, 2) + '\n');
// ファイルに保存した内容も読み返し、200問の個票が欠けていないことを確認する。
assert.equal(JSON.parse(fs.readFileSync(outputPath, 'utf8')).records.length, records.length);
assert.equal(validation.generationFailures, 0);
assert.equal(validation.handCountViolations, 0);
assert.equal(validation.answerConditionViolations, 0);
assert.equal(records.length, perDifficulty * 2);
assert.equal(new Set(records.map(record => record.id)).size, records.length);
for (const difficulty of ['standard', 'hard']) assert.equal(byDifficulty[difficulty].n, perDifficulty);
for (const name of flagNames) {
  assert.equal(aggregate.counts[name], byDifficulty.standard.counts[name] + byDifficulty.hard.counts[name]);
}
assert.equal(aggregate.counts.quadOrHonorCandidate,
  aggregate.counts.anyQuad + aggregate.counts.honorDiscardCandidate - aggregate.counts.quadAndHonorCandidate);
process.stdout.write(JSON.stringify({ outputPath, aggregate, byDifficulty }, null, 2) + '\n');
