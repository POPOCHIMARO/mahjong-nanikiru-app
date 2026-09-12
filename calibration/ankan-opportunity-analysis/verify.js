// 別実装でシャンテンを照合し、保存済み200問標本の集計を再計算する。
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const Engine = require('../../engine.js');
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const root = __dirname;
const sample = JSON.parse(fs.readFileSync(path.join(root, 'generated-sample.json'), 'utf8'));
assert.equal(sample.metadata.engineSha256, hash(fs.readFileSync(path.join(root, '../../engine.js'))));
assert.equal(sample.records.length, 200);
const counter = {};
for (const r of sample.records) {
  const c = Engine.toCounts(r.hand);
  const quad = c.some(n => n === 4);
  const honor = r.analysis.some(row => row.discard >= 27);
  const correctHonor = r.bestDiscards.some(t => t >= 27);
  assert.equal(r.flags.anyQuad, quad);
  assert.equal(r.flags.honorDiscardCandidate, honor);
  assert.equal(r.flags.correctHonorDiscard, correctHonor);
  for (const [key, flag] of Object.entries(r.flags)) counter[key] = (counter[key] || 0) + Number(flag);
}
assert.deepEqual(counter, sample.aggregate.counts);
const rows = zlib.gunzipSync(fs.readFileSync(path.join(root, 'full/app-structural-hands.jsonl.gz')))
  .toString('utf8').trim().split('\n').map(JSON.parse);
// ハッシュ順で固定標本を選び、通常手と四枚使いの両方を照合する。
const ordered = rows.map(r => ({r, h:hash(r.windowId)})).sort((a,b) => a.h.localeCompare(b.h));
const selected = ordered.slice(0, 100).map(x=>x.r);
selected.push(...ordered.filter(x=>x.r.legalAnkan.length).slice(0, 100).map(x=>x.r));
for (const r of selected) {
  const before = [...r.counts]; before[r.draw]--;
  assert.equal(Engine.shanten(before), 2, r.windowId);
  const a = Engine.analyzeDiscards([...r.counts]);
  assert.equal(a.minShanten, 1, r.windowId);
  const honors = a.keep.filter(row=>row.discard >= 27).map(row=>row.discard).sort((a,b)=>a-b);
  assert.deepEqual(honors, r.honorCandidates, r.windowId);
}
const result = {status:'pass', generatedRecords:sample.records.length, generatedRecomputed:counter,
  structuralRecords:rows.length, checkedDistinctHands:new Set(selected.map(r=>r.windowId)).size,
  checkedAnkanHands:selected.filter(r=>r.legalAnkan.length).length,
  selection:'SHA256(windowId) ascending: first 100 all, first 100 with legal ankan',
  independentOfPythonShanten:true, checkedWindowIds:[...new Set(selected.map(r=>r.windowId))]};
fs.writeFileSync(path.join(root, 'verification.json'), JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({...result,checkedWindowIds:undefined}));
