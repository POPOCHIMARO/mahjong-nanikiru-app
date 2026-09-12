// 牌効率問題の生成待ち時間を、固定seedと難度別の同一件数で比較する。
// 実行例: node efficiency-generator-benchmark.js before benchmark-before.json
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { performance } = require('node:perf_hooks');

const enginePath = path.resolve(__dirname, '../../engine.js');
const Engine = require(enginePath);
const label = process.argv[2] || 'measurement';
const outputPath = path.resolve(process.argv[3] || `efficiency-benchmark-${label}.json`);
const seed = 20260911;
const perDifficulty = 20;

function mulberry32(initialSeed) {
  let state = initialSeed >>> 0;
  return function random() {
    let value = state += 0x6D2B79F5;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function percentile(sorted, fraction) {
  if (sorted.length === 0) return null;
  return sorted[Math.ceil(sorted.length * fraction) - 1];
}

function summarize(samples) {
  const successful = samples.filter(sample => sample.generated).map(sample => sample.elapsedMs).sort((a, b) => a - b);
  const middle = Math.floor(successful.length / 2);
  const median = successful.length === 0 ? null
    : successful.length % 2 === 1 ? successful[middle]
      : (successful[middle - 1] + successful[middle]) / 2;
  return {
    requested: samples.length,
    generated: successful.length,
    failures: samples.length - successful.length,
    medianMs: median,
    p95Ms: percentile(successful, 0.95),
    maxMs: successful.length ? successful[successful.length - 1] : null,
  };
}

const originalRandom = Math.random;
const samples = [];
const started = performance.now();
Math.random = mulberry32(seed);
try {
  for (const difficulty of ['standard', 'hard']) {
    for (let index = 0; index < perDifficulty; index++) {
      const itemStarted = performance.now();
      const diagnostics = {};
      const problem = Engine.generateEfficiencyProblem(difficulty, diagnostics);
      samples.push({
        difficulty,
        index: index + 1,
        generated: Boolean(problem),
        elapsedMs: performance.now() - itemStarted,
        traps: problem ? problem.traps : [],
        ...diagnostics,
      });
    }
  }
} finally {
  Math.random = originalRandom;
}

const result = {
  label,
  engineSha256: crypto.createHash('sha256').update(fs.readFileSync(enginePath)).digest('hex'),
  seed,
  prng: 'Mulberry32',
  requestedPerDifficulty: perDifficulty,
  generationOrder: '20 standard, then 20 hard, one continuous PRNG stream',
  nodeVersion: process.version,
  elapsedMs: performance.now() - started,
  byDifficulty: Object.fromEntries(['standard', 'hard'].map(difficulty => {
    const selected = samples.filter(sample => sample.difficulty === difficulty);
    return [difficulty, summarize(selected)];
  })),
  overall: summarize(samples),
  samples,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ outputPath, ...result.overall, byDifficulty: result.byDifficulty }, null, 2));
