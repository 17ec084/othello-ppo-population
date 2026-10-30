const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = n => (n*100).toFixed(1)+'%';
const count = n => Number(n).toLocaleString();
let refreshing=false, replay=null, replayLocked=false, latestGame=null, displayedRun=null;

function scoreTable(scores){
  if(!scores)return '<p>最初の評価を待っています。</p>';
  let html='<table><tr><th>相手</th><th>勝率</th><th>平均石差</th><th>勝 / 分 / 負</th></tr>';
  for(const [key,name] of [['random','ランダム'],['position','位置評価'],['initial','学習前モデル']]){
    const e=scores[key];if(!e)continue;
    html+=`<tr><td>${name}</td><td>${pct(e.win_rate)}</td><td>${e.mean_margin_diagnostic_only.toFixed(1)}</td><td>${e.wins} / ${e.draws} / ${e.losses}</td></tr>`;
  }
  return html+'</table>';
}


function renderRecord(s,a){
  const b=a.best,c=a.current,q=s.plateau;
  $('rule').textContent=`最高成績を更新した重みと学習状態を保存。最後の最高更新から${a.patience}step記録更新がなければ、その時点へ戻して交代。B→Aのときだけ0.1ポイント追加開放。`;
  $('condition').textContent=`学習担当 ${a.learner}群 ／ ${a.frozen_opponent}群の全重みを固定 · 第${a.phase}区間 · 交代${a.switches}回`;
  $('leaders').innerHTML=b?`<div class="metric">区間最高成績 ${pct(b.score)}</div><p>記録step ${b.step} ／ ${count(b.games)}局：${b.wins}勝・${b.draws}分・${b.losses}敗</p>`:'<div class="metric">新しい相手群で初回評価待ち</div>';
  if(c)$('leaders').innerHTML+=`<p>直近評価：step${c.step} · 成績 ${pct(c.score)} ／ 勝率 ${pct(c.win_rate)}（学習担当側）</p>`;
  $('leaders').innerHTML+='<small>成績＝（勝ち＋引き分け×0.5）÷対局数。相手群を区間中ずっと固定して比較します。評価値の最高記録による運用ルールです。</small>';
  $('events').innerHTML=a.events.length?'<table><tr><th>交代step</th><th>交代</th><th>復元したstep</th><th>最高成績</th><th>開放率</th></tr>'+a.events.slice(-8).reverse().map(e=>`<tr><td>${e.step}</td><td>${e.from}→${e.to}</td><td>${e.restored_step}</td><td>${pct(e.best.score)}</td><td>${e.percent_after.toFixed(1)}%</td></tr>`).join('')+'</table>':'A群から学習を開始しています。';
  $('plateau').innerHTML=q.initialized?`<h3>最高記録から ${q.unchanged_updates} / ${q.patience} step</h3><p>最後の最高更新：step${q.t} ／ 期限：step${q.deadline_step}</p><small>期限のstepでも先に評価し、最高を更新すれば学習を続けます。更新がなければ最高時点へ戻して交代します。</small>`:'';
}

function eventName(e){return e.reason==='record_plateau'?`${e.from}群をstep${e.restored_step}へ復元・${e.to}群へ交代`: e.type==='shared_learning'?`共有学習 ${e.group}群 ${e.students}体`: e.type==='alien_learning'?'異端学習 '+Object.entries(e.rounds).map(([g,n])=>g+'群 '+n+'回').join('・'):e.type==='mutation'?'突然変異':e.type==='opponent_copy'?`${e.source}→${e.learner} 全重みコピー`:`学習 ${e.from}→${e.to}`;}
function eventReason(e){return e.reason==='record_plateau'?`${e.patience}step最高記録の更新なし`: e.type==='shared_learning'?'別評価の上位個体から打ち方を学習': e.type==='alien_learning'?'群内の多様性を初期値まで回復':e.type==='mutation'?'旧ルール':e.type==='opponent_copy'?'劣勢p≤0.05':e.reason==='plateau_50'?'50更新停滞':e.reason==='plateau_timeout'?'旧ルール・停滞復元':'優勢p≤0.05';}
function chart(history,events=[],currentStep=0){
  const w=960,h=290,l=44,r=20,t=48,b=30,max=Math.max(1,currentStep,...history.map(x=>x.step));
  const x=step=>l+(w-l-r)*step/max;
  let s=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="固定AIへの勝率の推移">`;
  let start=0,learner='A';
  const band=(end)=>{const width=x(end)-x(start);if(width<=0)return;
    s+=`<rect x="${x(start)}" y="${t}" width="${width}" height="${h-t-b}" fill="${learner==='A'?'#3d9861':'#3d70ae'}" opacity="0.22"><title>${learner}学習：更新${start}〜${end}</title></rect>`;
    if(width>42)s+=`<text x="${(x(start)+x(end))/2}" y="${t+15}" text-anchor="middle" fill="${learner==='A'?'#c0f2cc':'#bbd9ff'}" font-size="11">${learner}競争学習</text>`;};
  for(const e of events.filter(e=>e.type==='learner_switch')){const end=e.timing==='before_update'?Math.max(0,e.step-1):e.step;band(end);start=end;learner=e.to;}
  band(max);
  for(let p=0;p<=100;p+=25){const y=t+(h-t-b)*(1-p/100);s+=`<path d="M${l} ${y}H${w-r}" stroke="#365044"/><text x="3" y="${y+4}" fill="#a5bbaf" font-size="12">${p}%</text>`;}
  for(const [agent,color] of [['A','#a0e9ad'],['B','#75b7ed']])for(const opponent of ['random','position']){
    const points=history.filter(e=>e.families.transformer?.[agent]?.[opponent]).map(e=>[l+(w-l-r)*e.step/max,t+(h-t-b)*(1-e.families.transformer[agent][opponent].win_rate)]);
    if(points.length){s+=`<polyline fill="none" stroke="${color}" stroke-width="2.5" ${opponent==='position'?'stroke-dasharray="6 5"':''} points="${points.map(p=>p.join(',')).join(' ')}"/>`;for(const [x,y] of points)s+=`<circle cx="${x}" cy="${y}" r="3" fill="${color}"/>`;}
  }
  for(const e of events){
    const px=x(e.step),color=e.type==='shared_learning'?'#f3d87c':e.type==='alien_learning'?'#f5a2bf':e.type==='mutation'?'#ffbd69':e.type==='opponent_copy'?'#e6a1f5':'#dce9e0';
    const label=`更新${e.step}：${eventName(e)}（${eventReason(e)}${e.timing==='before_update'?'・学習前':''}）`;
    s+=`<g aria-label="${esc(label)}"><title>${esc(label)}</title>`;
    if(e.type!=='shared_learning')s+=`<path d="M${px} ${t}V${h-b}" stroke="${color}" stroke-width="1" stroke-dasharray="3 5" opacity="0.65"/>`;
    if(e.type==='shared_learning')s+=`<circle cx="${px}" cy="${t-40}" r="2" fill="${color}"/>`;
    else if(e.type==='alien_learning')s+=`<rect x="${px-3}" y="${t-32}" width="6" height="6" fill="${color}"/>`;
    else if(e.type==='mutation')s+=`<path d="M${px} ${t-11}l-5 -8h10z" fill="${color}"/>`;
    else if(e.type==='opponent_copy')s+=`<path d="M${px} ${t-23}l5 6 -5 6 -5 -6z" fill="${color}"/>`;
    else s+=`<circle cx="${px}" cy="${t-17}" r="4" fill="${color}"/>`;
    s+='</g>';
  }
  return s+`<text x="${l}" y="${h-4}" fill="#a5bbaf" font-size="12">0 更新</text><text x="${w-r}" text-anchor="end" y="${h-4}" fill="#a5bbaf" font-size="12">${max} 更新</text></svg>`;
}

function drawBoard(index){
  if(!replay)return;
  let cells=Array(64).fill(0);cells[28]=cells[35]=1;cells[27]=cells[36]=-1;
  for(const m of replay.moves.slice(0,index)){
    if(m.action===null)continue;
    const color=m.color===0?1:-1,idx=m.action;cells[idx]=color;
    for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++){
      if(!dr&&!dc)continue;
      let r=Math.floor(idx/8)+dr,c=idx%8+dc,ray=[];
      while(r>=0&&r<8&&c>=0&&c<8&&cells[8*r+c]===-color){ray.push(8*r+c);r+=dr;c+=dc;}
      if(ray.length&&r>=0&&r<8&&c>=0&&c<8&&cells[8*r+c]===color)for(const j of ray)cells[j]=color;
    }
  }
  $('board').innerHTML=cells.map(c=>'<div class="cell">'+(c?'<div class="stone '+(c===1?'black':'white')+'"></div>':'')+'</div>').join('');
  $('move').textContent=`${index} / ${replay.moves.length} 手（パスを含む）`;
  $('game-title').textContent=`${replay.pair===undefined?'更新':'競争step'}${replay.step}の対局${replay.pair===undefined?'':'（代表：a'+replay.pair+'–b'+(replay.b_index??replay.pair)+'）'} · 黒A ${replay.black_stones}石 ／ 白B ${replay.white_stones}石`;
  if(replay.learner_id!==undefined)$('game-title').textContent=`step${replay.step}の代表対局 · 黒AI ${replay.black_id} ${replay.black_stones}石 ／ 白AI ${replay.white_id} ${replay.white_stones}石`;
}
function selectMove(index){replayLocked=true;drawBoard(index);}
function followLatest(){if(!latestGame)return;replayLocked=false;replay=latestGame;$('moves').max=replay.moves.length;$('moves').value=replay.moves.length;drawBoard(replay.moves.length);}

async function refresh(){
  if(refreshing)return;refreshing=true;
  try{
    const response=await fetch('/api/state?reward='+encodeURIComponent(comparisonReward),{signal:AbortSignal.timeout(8000)});
    if(!response.ok)throw Error('HTTP '+response.status);
    const s=await response.json();
    renderRewardComparison(s);
    if(displayedRun!==s.run_dir){
      displayedRun=s.run_dir;replay=null;latestGame=null;replayLocked=false;
      $('board').innerHTML='';$('move').textContent='';$('moves').value=0;$('moves').max=0;
      $('game-title').textContent='新しい実験の最初の対局を待っています。';
      $('individual-chart-choice')?.remove();individualChartChoice='all';
    }
    if(s.mode==='individual_pool'){renderIndividual(s);return;}
    if($('individual-details'))$('individual-details').hidden=true;
    if($('active-scope-label'))$('active-scope-label').textContent='現在の学習範囲 · AとBで共通の位置';
    $('label-A').closest('.card').hidden=false;$('label-B').closest('.card').hidden=false;
    $('individual-chart-choice')?.remove();
    const population=Boolean(s.population),entropyMode=Boolean(s.entropy_reward),recordMode=s.selection_mode==='record_score';
    const alienEnabled=s.alien_learning_enabled!==false;
    if($('experiment-title'))$('experiment-title').textContent=population?`${s.population}対${s.population}、競争・共有${alienEnabled?'・異端':''}学習。`:entropyMode?'2体のTransformerを、石差と探索報酬で育てる。':'2体のTransformerを、石差で育てる。';
    if($('intro')&&population)$('intro').textContent=`A群${s.population}体とB群${s.population}体を、毎step無作為に組み替えて対戦。担当群を石差で学習させ、仲間から打ち方を学ぶ共有学習を追加。その後、両群それぞれの初期多様性を異端学習で維持します。`;
    if($('intro')&&entropyMode)$('intro').textContent='AもBも1体。担当の1体だけが、石差と着手確率の広がりを報酬にして学習します。同じモデルで多数の対局を並行し、経験を集めます。';
    $('status').textContent=({running:s.stale?'更新待ち':'実行中',completed:'完了',stopped:'停止済み',failed:'学習停止・エラー',interrupted:'中断'})[s.status]||s.status;
    const stepUnit=population?'競争step':'更新';
    $('progress').textContent=s.updates?`${s.step} / ${s.updates} ${stepUnit}`:`${s.step} ${stepUnit} · 上限なし`;
    $('bar').hidden=!s.updates;if(s.updates){$('bar').max=s.updates;$('bar').value=s.step;}
    const phase=s.status==='failed'?'学習プロセスが終了':s.status==='stopped'?'保存して停止済み':s.phase==='sharing_evaluation'?'共有学習の教師を別対局で評価中':s.phase==='sharing_A'||s.phase==='sharing_B'?`${s.phase.slice(-1)}群が仲間から共有学習中`:s.phase.startsWith('alien_')?`${s.phase.slice(-1)}群の異端学習中`:s.phase==='diversity_measure'?'方策多様性を参考測定中':s.phase==='record_rollback'?'最高記録の重みと学習状態を復元中':s.phase==='competition_play'?`${s.population}組の競争対局中`:s.phase==='competition_update'?'担当群の方策を更新中':s.phase==='saving'?'学習状態を保存中':s.phase==='mutation'?'起点の重みから突然変異を作成中':s.phase.startsWith('selfplay')?'AI同士の対戦中':s.phase.startsWith('update')?'学習担当の重みを更新中':s.phase==='evaluation_record'?'最高記録を比較する対局中':s.phase==='evaluation_switch'?'交代判定用の対戦中':s.phase.startsWith('evaluation')?'固定AIとの評価中':s.status==='completed'?'実験完了':'準備中';
    $('phase').textContent=`${phase} · 経過 ${Math.floor(s.elapsed_seconds/60)}分 ${Math.floor(s.elapsed_seconds%60)}秒`;
    $('device').textContent=s.device;
    const margin=s.reward.mode==='margin';
    $('reward').textContent=margin?'自分の石数 − 相手の石数':'勝ち1 ／ 引分・負け0';
    if(entropyMode){
      $('reward-label').textContent='石差と探索の報酬';
      $('reward').textContent=`石差 ＋ ${s.entropy_reward.coefficient} × 平坦さの合計`;
      $('reward-detail').textContent='各手の平坦さは0〜1。合法手が1つだけなら追加報酬は0。交代判定のp値には勝敗だけを使います。';
      $('batch-definition').textContent=`1更新＝学習${s.batch_games}局＋独立の判定${s.test_batch_games}局。学習するモデルは1体だけで、対局を並行します。更新数の上限はありません。`;
    }
    const u=s.unfreezing?.transformer;
    if(u){
      $('active').textContent=Number(u.percent).toFixed(1)+'% を開放';
      $('active-count').textContent=`各${population?'個体':'モデル'} ${count(u.active_parameters)} / ${count(u.total_parameters)} 個 · ${population?'競争・共有学習は担当群のみ、異端学習は両群が対象':'更新するのは学習担当だけ'}`;
      const a=s.alternating;
      if(a){
        if(recordMode){renderRecord(s,a);}else{
        const test=a.test;
        const modern=Boolean(a.loss_test);
        $('rule').textContent=modern?`優勢p ≤ ${test.alpha}で交代。p値が${s.plateau.patience}更新改善しなければ、重みを保って交代。劣勢p ≤ ${test.alpha}なら固定相手の全重みをコピーして同じ側の学習を継続。B→Aのときだけ0.1ポイント追加開放。`:`旧ルール：優勢p ≤ ${test.alpha}または停滞期限で交代。突然変異と起点への復元を使用。`;
        $('condition').textContent=population?`競争学習担当 ${a.learner}群 ／ ${a.frozen_opponent}群は競争・共有学習を停止（異端学習では両群を更新） · 第${a.phase}区間 · 交代${a.switches}回`:`学習担当 ${a.learner} ／ ${a.frozen_opponent}の重みは全て固定 · 第${a.phase}区間 · 交代${a.switches}回`;
        $('leaders').innerHTML=`<div class="metric">優勢 p = ${test.p_value.toPrecision(7)}</div>${modern?`<p class="metric" style="font-size:24px">劣勢 p = ${a.loss_test.p_value.toPrecision(7)}</p>`:''}<p>累積p値・過去最小 ／ 判定開始から ${test.games}局：${test.wins}勝・${test.draws}分・${test.losses}敗（学習担当側）</p><small>${modern?'優勢と劣勢を別々の片側検定で判定。交代・コピー時に両方をリセット。各方向・判定区間ごとの水準で、実験全体の誤判定率5%を保証するものではありません。':'旧ルール：突然変異ではp値をリセットせず、交代時にリセット。'}</small>`;
        if(population)$('leaders').innerHTML+=`<p>判定対象：群全体の勝敗の符号平均（勝ち+1・分け0・負け−1）。石差は競争学習の報酬です。評価${test.rounds}回${test.score===null?'':' ／ 引き分け半勝の群スコア '+pct(test.score)}。</p>`;
        $('events').innerHTML=a.events.length?'<table><tr><th>更新</th><th>交代</th><th>理由</th><th>優勢p値</th><th>開放割合</th></tr>'+a.events.slice(-8).reverse().map(e=>`<tr><td>${e.step}</td><td>${e.from} → ${e.to}</td><td>${eventReason({...e,type:'learner_switch'})}</td><td>${e.test.p_value.toPrecision(5)}</td><td>${e.percent_after.toFixed(1)}%${e.released?'（＋0.1）':''}</td></tr>`).join('')+'</table>':'Aの学習から開始しています。最初の交代を待っています。';
        const q=s.plateau;
        if($('plateau'))$('plateau').innerHTML=q?.initialized?`<h3>p値の停滞${modern?'':'と突然変異（旧ルール）'}</h3><p>起点 t = <b>${q.t}</b> ／ 起点p = ${q.anchor_p.toPrecision(7)} ／ 起点から ${q.unchanged_updates} 更新経過</p><p>${modern?'重みを保ったまま交代する予定':'復元・交代の期限'}：${q.deadline_step}更新目${modern?'の学習前':''}</p>${modern?`<small>突然変異・巻き戻しは行いません。コピー累計${a.copies}回。p値が下がれば起点tを更新します。</small>`:`<p>突然変異累計 ${q.total_mutations}回。標準偏差${q.sigma}。</p>`}`:'';
        }
      }else{
        $('rule').textContent='旧方式の実験記録を表示中';$('condition').textContent='';$('leaders').innerHTML='';$('events').innerHTML='';
      }
    }
    const training=s.latest_training.transformer;
    if($('population-details'))$('population-details').hidden=!population;
    if(population){
      $('batch-definition').textContent=`1競争step＝${s.population}組の先後交換対局＋担当群の共有学習＋必要な異端学習＋独立の判定対局。共有・異端学習の回数は競争step数に含めません。更新数の上限はありません。`;
      const processing=s.phase.startsWith('competition_')?s.step+1:s.competition_step;
      const pairing=training?.pairing;
      const perGame=Array.isArray(pairing?.[0]);
      const pairingDetail=perGame?` 1局ごとに組み替え。直近stepのa0の相手：${Array.from({length:s.population},(_,j)=>'b'+j+'×'+pairing[0].filter(k=>k===j).length+'局').join('、')}。`:pairing?` 直近の組み合わせ例：${pairing.slice(0,5).map((j,i)=>'a'+i+'–b'+j).join('、')}。`:'';
      $('population-phase').textContent=`完了した競争step：${s.step} ／ 処理中：${processing} ／ ${alienEnabled?'各群の初期多様性を維持':'異端学習なし・多様性は参考測定のみ'}。背景色は競争学習の担当を示します。`+pairingDetail;
      for(const agent of ['A','B']){
        const d=s.diversity[agent];
        $('diversity-'+agent).innerHTML=alienEnabled?`<h3>${agent}群</h3><p>多様性 ${d.value.toPrecision(7)} ／ 初期目標 ${d.target.toPrecision(7)} nats · ${d.met?'目標達成':'異端学習が必要'}</p><p>異端学習 累計${count(d.total_alien_updates)}回 ／ 個体別の異端度 ${Math.min(...d.individual_novelty).toPrecision(4)}〜${Math.max(...d.individual_novelty).toPrecision(4)}</p>`:`<h3>${agent}群</h3><p>多様性 ${d.value.toPrecision(7)} ／ 初期参考値 ${d.target.toPrecision(7)} nats</p><p>異端学習は無効 · 多様性が下がっても補正しません。</p>`;
      }
    }
    if(recordMode){
      $('intro').textContent=`A群${s.population}体とB群${s.population}体。${s.matchmaking==='per_game_permutation'?'1局ごとに':'毎step'}相手をランダムに組み替え、担当群だけが競争・共有${alienEnabled?'・異端':''}学習を行います。${alienEnabled?'':'異端学習は無効です。'}最高記録を保存し、200step更新がなければ復元して交代します。`;
      $('reward-label').textContent='競争学習の報酬';
      $('reward-detail').textContent='自分の終局石差を使います。共有学習は仲間の打ち方を学びます。'+(alienEnabled?'異端学習は群内の方策の違いを保ちます。':'異端報酬・平坦さ報酬は使いません。');
      $('active-count').textContent=`各個体 ${count(u.active_parameters)} / ${count(u.total_parameters)} 個 · 競争・共有${alienEnabled?'・異端':''}学習は担当群だけ`;
      $('batch-definition').textContent=`1step＝競争${s.batch_games}局＋共有${s.sharing_rounds_per_step}回${alienEnabled?'＋必要な異端学習':''}。${s.record_every}stepごとに全組み合わせ${s.record_evaluation_games}局で最高記録を比較。更新数の上限はありません。`;
      $('diversity-detail').textContent=`固定の共通盤面で${s.population}体の方策のJSダイバージェンスを測ります。`+(alienEnabled?'担当群だけを初期の多様性へ戻し、相手群の重みは固定します。個々の着手分布を平坦にする報酬は使いません。':'多様性は表示用の測定だけで、報酬・重み更新・交代条件には使いません。');
      if($('diversity-heading'))$('diversity-heading').textContent=alienEnabled?'群内の方策多様性と異端学習':'群内の方策多様性（測定のみ）';
      if($('learning-pipeline'))$('learning-pipeline').textContent=alienEnabled?'競争学習 → 担当群内の共有学習 → 異端学習 → 独立の交代判定。':'競争学習 → 担当群内の共有学習 → 多様性の参考測定 → 独立の交代判定。異端学習なし。';
      $('sharing-detail').textContent=`別評価で自分より成績の良い仲間を最大2体選び、対局の盤面を群内で共有します。各stepで共有学習${s.sharing_rounds_per_step}回、各回で個体ごとに${s.sharing_positions}盤面。重みの異なる仲間の行動を、そのまま自分の強化学習データとして使う方式ではありません。`;
      $('timeline-heading').textContent=`最高記録への復元・交代・共有${alienEnabled?'・異端':''}学習の履歴`;
      $('timeline-legend').innerHTML='線：緑 A ／ 青 B · 実線 ランダムAI ／ 破線 位置評価AI<br>背景：緑 A群学習 ／ 青 B群学習<br>目印：● 最高記録へ復元して交代 ／ ● 共有学習'+(alienEnabled?' ／ ■ 異端学習':'');
      $('chart-note').textContent='固定AIへの診断勝率です。交代の最高記録は、別途行うA・B間の評価成績（引き分け半勝）で比較します。';
    }
    if($('sharing-details'))$('sharing-details').hidden=!s.sharing;
    if(s.sharing)for(const agent of ['A','B']){
      const d=s.sharing[agent],v=d.last,r=d.teacher_evaluation;
      $('sharing-'+agent).innerHTML=`<h3>${agent}群 · 共有学習 ${count(d.updates)}回</h3><p>仲間から学んだ盤面数：1体あたり平均 ${d.mean_boards_per_individual.toFixed(1)}盤面</p>`+(v?`<p>直近は${v.students}体が各${v.positions_per_student}盤面を学習 ／ 教師との隔たり ${v.kl_before.toPrecision(5)} → ${v.kl_after.toPrecision(5)}（小さいほど近い）</p>`:'<p>最初の共有学習を待っています。</p>')+(r?`<small>教師評価：競争step${r.step}、各個体${r.games_per_individual}局 ／ 成績 ${pct(Math.min(...r.scores))}〜${pct(Math.max(...r.scores))}（引分半勝）</small>`:'');
    }
    if($('entropy-details'))$('entropy-details').hidden=!entropyMode;
    if(entropyMode){
      $('timeline-heading').textContent='交代・コピーの履歴';
      $('timeline-legend').innerHTML='線：緑 A ／ 青 B · 実線 ランダムAI ／ 破線 位置評価AI<br>背景：緑 A学習 ／ 青 B学習 · 目印：● 交代 ／ ◆ コピー';
      $('chart-note').textContent='学習には終局石差とエントロピー報酬を使います。このグラフは固定AIへの勝率で、引き分けを0.5勝に換算しません。';
      $('entropy-definition').textContent=`終局石差に、各手の平坦さ×${s.entropy_reward.coefficient}の合計を加えます。平坦にする学習も担当中の1体だけに行います。`;
      for(const agent of ['A','B']){
        const e=s.entropy[agent],u=training?.agents?.[agent];
        $('entropy-'+agent).innerHTML=`<h3>${agent} · 平坦さ ${pct(e.normalized_entropy)}</h3><p>最大着手確率の平均 ${pct(e.mean_max_probability)} ／ 同じ盤面で一様なら ${pct(e.mean_uniform_probability)}</p>`+(u?`<p>直近の追加報酬：1局平均 ＋${u.mean_entropy_bonus.toFixed(2)} ／ 石差との合計 ${u.mean_augmented_reward.toFixed(2)}</p>`:'');
      }
    }
    for(const agent of ['A','B']){
      if($('label-'+agent))$('label-'+agent).textContent=population?`Transformer ${agent}群 · ${s.population}体`:`Transformer ${agent}`;
      $('scores-'+agent).innerHTML=scoreTable(s.evaluation?.families.transformer?.[agent]);
      $('metrics-'+agent).textContent=(s.alternating?`${s.alternating.learner===agent?'競争学習を担当':population?'競争・共有学習は停止、異端学習の対象':'全重みを固定中'} · `:'')+(training?`直近${training.games}局：${training[agent==='A'?'a_wins':'b_wins']}勝 / 引分${training.draws} · 平均報酬 ${training[agent==='A'?'a_mean_reward':'b_mean_reward'].toFixed(2)}`:'最初の学習を待っています。');
      if(recordMode&&s.alternating.learner!==agent)$('metrics-'+agent).textContent=$('metrics-'+agent).textContent.replace('競争・共有学習は停止、異端学習の対象','全学習を停止・全重み固定');
      if(s.experience)$('metrics-'+agent).textContent+=` ／ 競争学習に使った対局：各個体${count(s.experience[agent].training_games_per_individual)}局`;
    }
    $('games').textContent=`学習 ${count(s.training_games.transformer||0)}局 · ${recordMode?'最高記録の評価':'交代判定の評価'} ${count(s.test_games||0)}局${s.teacher_games===undefined?'':' · 教師選び '+count(s.teacher_games)+'局'} · 先後を半数ずつ交換`;
    $('evaluation-step').textContent=s.evaluation?`評価時点：${population?'競争step':'更新'}${s.evaluation.step} ／ 各相手${s.evaluation.families.transformer.A.random.games}局${population?'（群全体）':''}`:'初期評価中';
    const timeline=s.timeline_events||[];
    $('chart').innerHTML=chart(s.evaluation_history||[],timeline,s.step);
    $('timeline').innerHTML=timeline.length?'<table><tr><th>更新</th><th>出来事</th><th>理由</th></tr>'+timeline.slice(-12).reverse().map(e=>`<tr><td>${e.step}${e.timing==='before_update'?'（学習前）':''}</td><td>${eventName(e)}</td><td>${eventReason(e)}</td></tr>`).join('')+'</table>':'交代・コピーなどが起きた更新をここに表示します。新ルールでは突然変異を行いません。';
    latestGame=s.last_games.transformer;
    if(latestGame&&!replayLocked)followLatest();
    $('updated').textContent=`5秒ごとに確認 ／ 学習の最終更新 ${Math.floor(s.heartbeat_age_seconds||0)}秒前 ／ ${s.run_dir}`;
    $('error').textContent=s.error||(s.stale?'学習の進捗が60秒以上更新されていません。':'');
  }catch(e){$('status').textContent='接続エラー';$('error').textContent='進捗を取得できません。'+e.message;}
  finally{refreshing=false;}
}
refresh();setInterval(refresh,5000);
