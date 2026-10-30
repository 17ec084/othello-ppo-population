const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('individual-monitor.js','utf8');
function state(steps){
  return {step:40,controller:{initial_learner:17,events:[{step:4,from:17,to:14}]},
    evaluation_history:steps.map(([step,rate,measured=step])=>({step,
      aggregate:{random:{win_rate:rate},position:{win_rate:.2}},
      individuals:[{id:17,evaluated_step:measured,random:{win_rate:rate},position:{win_rate:.2}}]}))};
}
function chart(s,choice='17'){
  const context={s,choice};vm.createContext(context);
  vm.runInContext(source+'\nindividualChartChoice=choice;result=individualChart(s);',context);
  return context.result;
}
// Regression: AI 17 learns before step 4, but old samples exist only at 0/25.
const gap=chart(state([[0,.4765625],[25,.5546875]]));
assert.equal((gap.match(/data-connection="known"/g)||[]).length,0);
assert.equal((gap.match(/data-measured-step=/g)||[]).length,4);
assert.ok(gap.includes('固定'));
// Values carried while the learner is elsewhere form a flat segment, not new dots.
const carried=chart(state([[0,.47],[25,.55],[26,.55,25],[27,.55,25]]));
assert.equal((carried.match(/data-connection="known"/g)||[]).length,4);
assert.equal((carried.match(/data-measured-step=/g)||[]).length,4);
// Equal values with proof of fixed weights can connect a longer interval.
const fixed=chart(state([[25,.55],[40,.55]]));
assert.equal((fixed.match(/data-connection="known"/g)||[]).length,2);
// Consecutive real measurements can show improvement during learning.
const consecutive=chart(state([[0,.47],[1,.51],[2,.55]]));
assert.equal((consecutive.match(/data-connection="known"/g)||[]).length,4);
// A group average also must not imply gradual improvement across a missing interval.
assert.equal((chart(state([[0,.47],[25,.55]]),'all').match(/data-connection="known"/g)||[]).length,0);
console.log('5 chart regression scenarios passed.');
