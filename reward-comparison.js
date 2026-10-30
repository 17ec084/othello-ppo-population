let comparisonReward='margin';
let comparisonIndividual=null;
function renderRewardComparison(s){
  const panel=$('reward-comparison'),c=s.comparison;
  if(!panel)return;
  panel.hidden=!c;if(!c)return;
  const paired=c.mode==='algorithm_comparison',entries=Object.entries(c.runs);
  const selected=entries.find(([key,r])=>paired?r.algorithm===s.algorithm:key===s.reward.mode)?.[0]||entries[0][0];
  comparisonReward=selected;
  panel.querySelector('h2').textContent=paired?'従来法とPPO＋価値関数の比較':'石差と勝敗報酬の比較';
  $('reward-view-label').textContent='詳しく見る条件 ';
  $('reward-view').innerHTML=entries.map(([key,r])=>`<option value="${key}">${r.label||key}</option>`).join('');
  $('reward-view').value=selected;
  $('reward-comparison-settings').textContent=`同じ初期重み・初期学習者・乱数の種から開始${c.initial_weights_verified?'（重みの一致を確認済み）':''}。固定AI評価は1体・各相手${c.eval_games}局。${paired?'報酬は両方とも終局石差。1つの実行プロセスで両条件を1stepずつ進めます。':''}交代・追加開放は各条件の実績に応じて独立に進みます。`;
  if(comparisonIndividual===null)comparisonIndividual=paired?String(entries[0][1].controller?.initial_learner??11):'all';
  const scope=$('comparison-individual');
  scope.innerHTML='<option value="all">全個体の平均</option>'+Array.from({length:c.models},(_,i)=>`<option value="${i}">AI ${i}</option>`).join('');
  scope.value=comparisonIndividual;
  scope.onchange=()=>{comparisonIndividual=scope.value;renderRewardComparison(s);};
  const rowFor=e=>!e?null:comparisonIndividual==='all'?e.aggregate:e.individuals?.find(i=>String(i.id)===comparisonIndividual);
  $('reward-comparison-table').innerHTML='<table><tr><th>条件</th><th>状況</th><th>学習step</th><th>学習者</th><th>方策の開放率</th><th>対ランダム</th><th>対位置評価</th></tr>'+entries.map(([mode,r])=>{
    const e=rowFor(r.matched_evaluation);
    return `<tr><td>${r.label||mode}</td><td>${({running:'実行中',stopped:'停止済み',initializing:'準備中',waiting_storage:'空き容量待ち',failed:'要確認',completed:'完了'})[r.status]||r.status}</td><td>${r.step}</td><td>${r.controller?'AI '+r.controller.learner:'—'}</td><td>${r.unfreezing?r.unfreezing.percent.toFixed(1)+'%':'—'}</td><td>${e?pct(e.random.win_rate):'評価待ち'}</td><td>${e?pct(e.position.win_rate):'評価待ち'}</td></tr>`;
  }).join('')+'</table>';
  $('reward-comparison-note').textContent=c.matched_step===null?'両条件の初回評価を待っています。':`勝率欄とグラフは両条件が完了した同じ評価stepまでを比較します（最新：step${c.matched_step}）。表示対象：${comparisonIndividual==='all'?'全'+c.models+'体平均':'AI '+comparisonIndividual}。平均では1体の改善が薄まるため、個体別でも比較できます。下の画面は選択中の条件の詳細です。`;
  $('comparison-legend').textContent=`緑：${entries[0][1].label} ／ 青：${entries[1][1].label} · 実線：ランダムAI ／ 破線：位置評価AI`;
  $('reward-view').onchange=()=>{comparisonReward=$('reward-view').value;refresh();};
  const w=960,h=260,l=44,r=20,t=25,b=30,end=c.matched_step??0;
  const max=Math.max(1,end),x=v=>l+(w-l-r)*v/max;
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="条件別の固定AI勝率比較">`;
  for(let p=0;p<=100;p+=25){const y=t+(h-t-b)*(1-p/100);svg+=`<path d="M${l} ${y}H${w-r}" stroke="#365044"/><text x="3" y="${y+4}" fill="#a5bbaf" font-size="12">${p}%</text>`;}
  for(let i=0;i<entries.length;i++)for(const opponent of ['random','position']){
    const color=i===0?'#a0e9ad':'#75b7ed';
    const points=entries[i][1].evaluation_history.filter(e=>e.step<=end).map(e=>{const row=rowFor(e);return row?[x(e.step),t+(h-t-b)*(1-row[opponent].win_rate)]:null;}).filter(Boolean);
    if(points.length)svg+=`<polyline fill="none" stroke="${color}" stroke-width="2.5" ${opponent==='position'?'stroke-dasharray="6 5"':''} points="${points.map(p=>p.join(',')).join(' ')}"/>`+points.map(([px,py])=>`<circle cx="${px}" cy="${py}" r="3" fill="${color}"/>`).join('');
  }
  $('reward-comparison-chart').innerHTML=svg+`<text x="${l}" y="${h-4}" fill="#a5bbaf" font-size="12">0 step</text><text x="${w-r}" text-anchor="end" y="${h-4}" fill="#a5bbaf" font-size="12">${end} step</text></svg>`;
}
