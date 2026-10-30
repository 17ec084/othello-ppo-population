let individualChartChoice='all';
function individualLearningSpans(s){
  const spans=[];let start=0,learner=s.controller.initial_learner;
  for(const event of s.controller.events){spans.push({start,end:event.step,learner});start=event.step;learner=event.to;}
  spans.push({start,end:s.step,learner});return spans;
}
function individualChartConnectionKnown(s,a,b,choice){
  if(b.step-a.step<=1)return true;
  if(choice==='all')return false;
  const changed=individualLearningSpans(s).some(span=>String(span.learner)===choice&&span.start<b.step&&span.end>a.step);
  return !changed&&Math.abs(a.value-b.value)<1e-12;
}
function individualChart(s){
  const history=s.evaluation_history||[],events=s.controller.events;
  const w=960,h=300,l=44,r=20,t=45,b=30,max=Math.max(1,s.step);
  const x=v=>l+(w-l-r)*v/max,y=v=>t+(h-t-b)*(1-v);
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="個体集団の固定AIへの勝率">`;
  for(const span of individualLearningSpans(s)){
    const width=x(span.end)-x(span.start),selected=individualChartChoice==='all'||String(span.learner)===individualChartChoice;
    const color=selected?`hsl(${span.learner*137.508%360} 48% 52%)`:'#738178';
    svg+=`<rect x="${x(span.start)}" y="${t}" width="${width}" height="${h-t-b}" fill="${color}" opacity="${selected ? .22 : .04}"><title>${individualChartChoice==='all'||selected?'AI '+span.learner+' 学習':'選択中のAIは固定'}：step${span.start}〜${span.end}</title></rect>`;
    if(width>40)svg+=`<text x="${(x(span.start)+x(span.end))/2}" y="${t+15}" fill="#e7efde" text-anchor="middle" font-size="11">${selected?'AI '+span.learner:'固定'}</text>`;
  }
  for(let p=0;p<=100;p+=25){const py=y(p/100);svg+=`<path d="M${l} ${py}H${w-r}" stroke="#365044"/><text x="3" y="${py+4}" fill="#a5bbaf" font-size="12">${p}%</text>`;}
  for(const [opponent,color] of [['random','#a0e9ad'],['position','#75b7ed']]){
    const points=history.map(e=>{
      const row=individualChartChoice==='all'?e.aggregate:e.individuals?.find(i=>String(i.id)===individualChartChoice);
      if(!row?.[opponent])return null;
      return {step:e.step,value:row[opponent].win_rate,measured:individualChartChoice==='all'||(row.evaluated_step??e.step)===e.step};
    }).filter(Boolean);
    for(let i=1;i<points.length;i++){
      const a=points[i-1],z=points[i];
      if(individualChartConnectionKnown(s,a,z,individualChartChoice))svg+=`<path data-series="${opponent}" data-connection="known" fill="none" stroke="${color}" stroke-width="2.5" ${opponent==='position'?'stroke-dasharray="6 5"':''} d="M${x(a.step)} ${y(a.value)}L${x(z.step)} ${y(z.value)}"/>`;
    }
    for(const point of points.filter(p=>p.measured))svg+=`<circle data-series="${opponent}" data-measured-step="${point.step}" cx="${x(point.step)}" cy="${y(point.value)}" r="4" fill="${color}"><title>step${point.step}で測定：${opponent==='random'?'ランダムAI':'位置評価AI'} ${(point.value*100).toFixed(1)}%</title></circle>`;
  }
  for(const e of events)svg+=`<path d="M${x(e.step)} ${t}V${h-b}" stroke="#dce9e0" stroke-dasharray="3 5"><title>step${e.step}：AI ${e.from}→AI ${e.to}、巻き戻しなし</title></path>`;
  return svg+`<text x="${l}" y="${h-4}" fill="#a5bbaf" font-size="12">0 step</text><text x="${w-r}" text-anchor="end" y="${h-4}" fill="#a5bbaf" font-size="12">${max} step</text></svg>`;
}

function renderIndividual(s){
  const c=s.controller,u=s.unfreezing,b=c.best,current=c.current,last=s.latest_training,thresholdMode=s.selection_mode==='win_rate_threshold';
  $('status').textContent=s.status==='running'?'実行中':s.status==='stopped'?'停止済み':s.status==='completed'?'完了':s.status==='waiting_storage'?'空き容量待ち':'要確認';
  const signed=s.reward.mode==='signed_win',ppo=s.algorithm==='ppo';
  $('experiment-title').textContent=`${ppo?'PPO＋価値関数':'従来法'}：${s.model_count}体から1体ずつ学習。`;
  $('intro').textContent=thresholdMode?`学習者はAI ${c.learner}。他の${s.model_count-1}体それぞれと${s.games_per_opponent}局ずつ対戦し、合計の勝率が${Number((s.win_threshold*100).toFixed(2))}%を超えるまで学習します。他の個体の重みは固定です。`:`学習者はAI ${c.learner}。他の${s.model_count-1}体を1局ごとに無作為に相手にします。他の個体の重みは全て固定。共有・異端・平坦さ報酬はありません。`;
  $('progress').textContent=`${s.step} step · ${s.updates?'上限 '+s.updates:'上限なし'}`;
  $('bar').max=s.updates||Math.max(1,s.step+1);$('bar').value=s.step;
  const phase=s.status==='stopped'?'保存して停止済み':s.status==='failed'?'学習エラー':s.phase==='waiting_storage'?'保存用の空き容量を待っています':s.phase==='waiting_pair'?'もう一方の比較条件の処理待ち':s.phase==='competition_play'?'学習者と他の個体が対局中':s.phase==='competition_update'?'学習者の重みを更新中':s.phase==='evaluation_record'?'他の個体への勝率を評価中':s.phase==='evaluation_fixed'?'固定AIへの勝率を評価中':s.phase==='saving'?'保存中':'実行中';
  $('phase').textContent=`${phase} · 経過 ${Math.floor(s.elapsed_seconds/60)}分 ${Math.floor(s.elapsed_seconds%60)}秒`;
  $('reward-label').textContent='学習の報酬';$('reward').textContent=signed?'勝ち +1 ／ 負け −1 ／ 引分 0':'自分の石数 − 相手の石数';
  $('reward-detail').textContent=`選ばれた1体だけが${signed?'勝敗':'終局石差'}で学びます。共有学習・異端学習・平坦さ報酬なし。`;
  if($('active-scope-label'))$('active-scope-label').textContent='現在の学習範囲 · 全個体で共通の位置';
  $('active').textContent=`${u.percent.toFixed(1)}% を開放`;
  $('active-count').textContent=`全${s.model_count}体で共通の開放位置 · 各個体 ${count(u.active_parameters)} / ${count(u.total_parameters)} 個 · 更新するのはAI ${c.learner}だけ${ppo?'。追加の価値予測層 '+s.ppo.critic_parameters+'個は最初から学習':''}`;
  $('rule').textContent=`最高成績を${c.patience}step更新できなければ、重みを戻さず他の個体へ交代します。交代ごとに全個体を0.1ポイント追加開放（最大100%）。`;
  $('condition').textContent=`学習者 AI ${c.learner} ／ 他の${s.model_count-1}体は全重み固定 · 第${c.phase}区間 · 交代${c.switches}回`;
  $('leaders').innerHTML=b?`<div class="metric">区間最高成績 ${pct(b.score)}</div><p>記録step ${b.step} ／ ${count(b.games)}局：${b.wins}勝・${b.draws}分・${b.losses}敗</p>`:'新しい学習者の初回評価待ち';
  if(current)$('leaders').innerHTML+=`<p>直近評価step${current.step}：${pct(current.score)} ／ 勝率 ${pct(current.win_rate)}</p>`;
  $('leaders').innerHTML+='<small>成績＝（勝ち＋引き分け×0.5）÷対局数。他の個体から1局ごとに無作為抽出して評価します。交代時に最高記録の比較をリセットします。</small>';
  $('batch-definition').textContent=`1step＝学習者の${s.batch_games}局${ppo?'を最大'+s.ppo.epochs+'周学習（直近の重み更新は'+(last?.update?.optimizer_steps??'—')+'回）':'と1回の重み更新'}。先後半々、相手は毎局再抽選。${s.record_every}stepごとに別の${s.record_evaluation_games}局で記録を比較します。`;
  $('plateau').innerHTML=b?`<h3>最高記録から ${s.step-b.step} / ${c.patience} step</h3><p>最後の最高更新：step${b.step} ／ 交代期限：step${b.step+c.patience}</p><small>期限でも先に評価し、最高を更新すれば続行。交代時も全個体の現在の重みと学習履歴を保持します。</small>`:'';
  if(thresholdMode){
    if($('experiment-eyebrow'))$('experiment-eyebrow').textContent='OTHELLO / PPO / FULL-PARAMETER LEARNING';
    $('active-scope-label').textContent='学習者の重み';
    $('active').textContent='最初から100%を学習';
    $('active-count').textContent=`AI ${c.learner}の全${count(u.total_parameters)}個を学習。他の${s.model_count-1}体の重みは固定。${ppo?'価値予測層も学習します。':''}`;
    $('rule').textContent=`合計${count(s.batch_games)}局の勝率が${Number((s.win_threshold*100).toFixed(2))}%を超えたら、現在の学習者を除く${s.model_count-1}体から無作為に選び直します。`;
    $('leaders').innerHTML=current?`<div class="metric">直近の勝率 ${pct(current.win_rate)}</div><p>step${current.step} · AI ${current.learner}：${current.wins}勝・${current.draws}分・${current.losses}敗 ／ ${count(current.games)}局</p>`:'現在の学習者の最初の対局結果を待っています。';
    if(last?.switch)$('leaders').innerHTML+=`<p>直前はAI ${last.learner}が${pct(last.wins/last.games)}で達成し、AI ${c.learner}へ交代しました。</p>`;
    $('leaders').innerHTML+='<small>勝率＝勝ち数÷全対局数。引き分けは勝ちに数えません。各回の対局中は全モデルの重みを固定します。</small>';
    $('batch-definition').textContent=`1step＝他の${s.model_count-1}体それぞれと${s.games_per_opponent}局（先後半々）、合計${count(s.batch_games)}局。未達ならこの対局から${ppo?'PPOで最大'+s.ppo.epochs+'周':'1回'}学習します。達成した回は追加更新せず、判定時の重みを保持して交代します。`;
    $('plateau').innerHTML=`<p>交代に必要な勝ち数：${count(Math.floor(s.win_threshold*s.batch_games)+1)}勝以上。学習step数による強制交代・重みの巻き戻し・追加開放はありません。</p>`;
    const per=last?.per_opponent;
    $('winrate-opponents').hidden=!per;
    if(per)$('winrate-opponents').innerHTML=`<h2>直近step${last.step}：AI ${last.learner}の相手別成績</h2><p>各相手${s.games_per_opponent}局 · 学習更新前の成績</p><table><tr><th>相手</th><th>勝率</th><th>勝／分／負</th><th>局数</th></tr>`+per.map(r=>`<tr><td>AI ${r.id}</td><td>${pct(r.win_rate)}</td><td>${r.wins}／${r.draws}／${r.losses}</td><td>${r.games}</td></tr>`).join('')+'</table>';
  }else if($('winrate-opponents'))$('winrate-opponents').hidden=true;
  $('events').textContent=c.switches?'':'初回学習者は無作為に選んだAI '+c.initial_learner+'です。';
  for(const id of ['population-details','sharing-details','entropy-details'])$(id).hidden=true;
  $('individual-details').hidden=false;
  $('individual-opponents').textContent=last?`直近step${last.step}の学習者 AI ${last.learner}の相手：`+last.opponent_counts.map((n,i)=>n?`AI ${i}×${n}局`:null).filter(Boolean).join('、'):'最初の学習対局を待っています。';
  $('individual-members').innerHTML='<table><tr><th>個体</th><th>状態</th><th>対ランダム</th><th>対位置評価</th><th>学習step</th><th>相手として参加</th></tr>'+Array.from({length:s.model_count},(_,i)=>{
    const row=s.evaluation?.individuals?.find(x=>x.id===i);
    return `<tr><td>AI ${i}</td><td>${c.learner===i?'学習中':'固定'}</td><td>${row?pct(row.random.win_rate):'—'}</td><td>${row?pct(row.position.win_rate):'—'}</td><td>${count(s.learner_steps[i])}</td><td>${count(s.opponent_games[i])}局</td></tr>`;
  }).join('')+'</table>';
  $('label-A').closest('.card').hidden=false;$('label-B').closest('.card').hidden=true;
  $('label-A').textContent=`学習者 AI ${c.learner} の固定AI評価`;
  $('scores-A').innerHTML=scoreTable(s.evaluation?.individuals?.find(i=>i.id===c.learner));
  $('metrics-A').textContent=`学習累計 ${s.learner_steps[c.learner]}step ／ ${count(s.learner_steps[c.learner]*s.batch_games)}局`;
  $('games').textContent=thresholdMode?`対局累計 ${count(s.training_games)}局 · 学習した回数 ${count(s.learner_steps.reduce((a,b)=>a+b,0))}回 · 交代${c.switches}回 · 更新数の上限なし`:`学習 ${count(s.training_games)}局 · 記録の評価 ${count(s.record_games)}局 · 共有学習なし · ロールバックなし`;
  const graphRow=s.evaluation?.individuals?.find(i=>String(i.id)===individualChartChoice);
  $('evaluation-step').textContent=!s.evaluation?'初期評価中':individualChartChoice==='all'?`評価状態：step${s.evaluation.step} ／ 各個体・各相手${s.evaluation.individuals[0].random.games}局（未更新個体は直近値）`:`AI ${individualChartChoice}の最終実測：step${graphRow?.evaluated_step??s.evaluation.step} ／ 各相手${graphRow?.random.games??s.eval_games_per_individual}局`;

  if(!$('individual-chart-choice')){
    const select=document.createElement('select');select.id='individual-chart-choice';select.setAttribute('aria-label','勝率グラフの個体');
    select.innerHTML='<option value="all">全個体の平均</option>'+Array.from({length:s.model_count},(_,i)=>`<option value="${i}">AI ${i}</option>`).join('');
    select.onchange=()=>{individualChartChoice=select.value;renderIndividual(s);};
    $('chart').before(select);
  }
  $('individual-chart-choice').onchange=()=>{individualChartChoice=$('individual-chart-choice').value;renderIndividual(s);};
  $('chart').innerHTML=individualChart(s);
  $('timeline-legend').textContent='緑：ランダムAI ／ 青破線：位置評価AI · ●：実測値 · '+(individualChartChoice==='all'?'背景：学習担当の個体':'色付き背景：選択したAIの学習期間 ／ 薄い背景：固定期間')+' · 縦線：学習者交代';
  const chartHistory=s.evaluation_history||[];
  const chartHasGaps=chartHistory.some((row,i)=>i>0&&row.step-chartHistory[i-1].step>1);
  $('chart-note').textContent='点は実測値です。学習者を毎step評価し、重みを変更していない個体は直近の評価を保持します。'+(chartHasGaps?'間隔が空いた区間の中間値は未記録のため、斜めの線で補間しません。':'step0から評価を記録しています。')+'固定AI評価は交代判定に使いません。';
  $('timeline-heading').textContent=thresholdMode?'学習者の交代履歴':'学習者の交代と追加開放の履歴';
  $('timeline').innerHTML=c.events.length?'<table><tr><th>step</th><th>学習者の交代</th><th>開放率</th><th>重み</th></tr>'+c.events.slice(-12).reverse().map(e=>`<tr><td>${e.step}</td><td>AI ${e.from} → AI ${e.to}</td><td>${e.percent_after.toFixed(1)}%</td><td>${e.reason==='win_rate_threshold'?'勝率'+pct(e.result.win_rate)+'で交代':'現在値を保持'}</td></tr>`).join('')+'</table>':'まだ交代していません。';
  latestGame=s.last_games.transformer;if(latestGame&&!replayLocked)followLatest();
  $('updated').textContent=`5秒ごとに確認 ／ 最終更新 ${Math.floor(s.heartbeat_age_seconds||0)}秒前 ／ ${s.run_dir}`;
  $('device').textContent=s.device;
  $('error').textContent=s.error||(s.storage?'空き容量待ち：最後の完全な保存データと現在の重みを保持し、5秒ごとに再試行しています。':'')||(s.stale?'学習の進捗が60秒以上更新されていません。':'');
}
