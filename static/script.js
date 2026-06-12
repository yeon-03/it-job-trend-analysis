let drawn = {};

function setActive(el, panelId) {
  document.querySelectorAll('.card').forEach(function(c) { c.classList.remove('active'); });
  el.classList.add('active');
  document.querySelectorAll('.panel').forEach(function(p){ p.style.display = "none"; })

  const target = document.getElementById(panelId);
  if(target) target.style.display = "flex";

  if (panelId === 'panel-tech') {
    document.getElementById('trend-tabs').style.display = 'flex';
    if (!drawn.tech) { drawTechChart(); drawn.tech = true; }   // 기본 탭 차트
  } else {
    document.getElementById('trend-tabs').style.display = 'none';
  }

  if(panelId === 'panel-period') {
    document.getElementById('period-tabs').style.display = 'flex';
    if(!drawn.period) { drawperiodChart('백엔드'); drawn.period = true; }
  } else {
    document.getElementById('period-tabs').style.display = 'none';
  }

  if(panelId === 'panel-company' && !drawn.company) { drawCompany(); drawn.company = true; }
  if(panelId === "panel-main" && !drawn.main) { drawMain(); drawn.main = true; }

  if (panelId === 'panel-chat' && !drawn.chat) {
  addMessage('안녕하세요! IT 채용 트렌드 상담 챗봇입니다. 어떤 직무를 준비 중이신가요?', 'bot');
  drawn.chat = true;
}
}

let mainLoaded = false;

async function drawMain() {
  const [summary, jobGrowth, jobSize] = await Promise.all([
    fetch('/api/trend-summary').then(r => r.json()),
    fetch('/api/job-growth').then(r => r.json()),
    fetch('/api/job-by-size').then(r => r.json())
  ]);

  const t = summary.rising_tech[0];
  document.getElementById('ic-tech').textContent = t['기술'] + ' +' + t['증가율(%)'] + '%';

  const j = [...jobGrowth].sort((a,b) => b['증감률(%)'] - a['증감률(%)'])[0];
  document.getElementById('ic-job').textContent = j['분야'] + ' +' + j['증감률(%)'] + '%';

  const big = {}, start = {};
  jobSize.forEach(r => {
    if (r['회사규모'] === '대기업')        big[r['직무카테고리']]   = r['ratio'];
    if (r['회사규모'] === '스타트업/중소') start[r['직무카테고리']] = r['ratio'];
  });
  const cats = Object.keys(big);

  const common = cats.sort((a,b) => (big[b]+start[b]) - (big[a]+start[a]))[0];
  document.getElementById('ic-common').textContent =
    common + ' (대 ' + big[common] + '% / 스 ' + start[common] + '%)';

  const diff = cats.sort((a,b) => Math.abs(big[b]-start[b]) - Math.abs(big[a]-start[a]))[0];
  const gap = (big[diff] - start[diff]).toFixed(1);
  document.getElementById('ic-diff').textContent =
    diff + ' (' + (gap > 0 ? '대기업 +' : '스타트업 +') + Math.abs(gap) + '%p)';
}

function switchtrendTab(el, paneId) {
  document.querySelectorAll('.nav-tab').forEach(function(t) { t.classList.remove('active'); });
  el.classList.add('active');

  document.querySelectorAll('.chart-pane').forEach(function(p) { p.style.display = 'none'; });
  document.getElementById(paneId).style.display = 'flex';

  if(paneId === "tab-tech" && !drawn.tech) { drawTechChart(); drawn.tech = true; }
  if(paneId === "tab-job" && !drawn.job) { drawJobChart(); drawn.job = true; }
  
}

function switchPeriodtab(el, paneId){
  document.querySelectorAll('#period-tabs .nav-tab').forEach(function(t) {
    t.classList.remove('active');
  });
  el.classList.add('active');
  drawperiodChart(paneId);
}

async function drawJobChart() {
  const res = await fetch('/api/job-monthly');
  const data = await res.json();
  const jobs = ['백엔드', '프론트엔드', 'AI_ML', 'DevOps', '데이터', '모바일'];

  const first = data[0];
  const last  = data[data.length - 1];

  // 방식 A: 첫 달 대비 끝 달 증감률 + 끝 달 공고 수
  const stats = jobs.map(function(job) {
    const f = first[job], l = last[job];
    const growth = f ? ((l - f) / f * 100) : 0;
    return { job: job, growth: Math.round(growth * 10) / 10, count: l };
  });

  // 증감률 높은 순으로 정렬 (차트·카드 공용)
  const byGrowth = [...stats].sort(function(a, b) { return b.growth - a.growth; });

  // --- 차트 (가로 막대) ---
  new Chart(document.getElementById('jobChart'), {
    type: 'bar',
    data: {
      labels: byGrowth.map(d => d.job),
      datasets: [{
        label: '공고 수 증감률(%)',
        data: byGrowth.map(d => d.growth),
        backgroundColor: byGrowth.map(d => d.growth >= 0 ? '#22c55e' : '#ef4444')
      }]
    },
    options: {
      indexAxis: 'y',
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#e2e8f0' } } },
      scales: {
        x: { ticks: { color: '#94a3b8', callback: function(v){ return v + '%'; } } },
        y: { ticks: { color: '#94a3b8' } }
      }
    }
  });

  const up = byGrowth.filter(d => d.growth > 0);
  document.getElementById('jobUpList').innerHTML = up.map(function(d) {
    return '<div class="reco-item"><span class="tech">' + d.job +
           '</span><span class="pct pct-up">+' + d.growth + '%</span></div>';
  }).join('');

  // 규모가 큰 분야: 끝 달 공고 수 많은 순 상위 4
  const byCount = [...stats].sort(function(a, b) { return b.count - a.count; }).slice(0, 4);
  document.getElementById('jobBaseList').innerHTML = byCount.map(function(d) {
    return '<div class="reco-item"><span class="tech">' + d.job +
           '</span><span class="pct pct-base">' + d.count + '건</span></div>';
  }).join('');
}

let techData = [];   // tech-by-size 전체 보관 (카테고리 전환에 재사용)
let jobSizeChart = null, techSizeChart = null;

const SIZE_BIG = '대기업';
const SIZE_START = '스타트업/중소';

// --- 회사 규모 패널 첫 진입 시 ---
async function drawCompany() {
  // 직무 데이터
  const jobRes = await fetch('/api/job-by-size');
  const jobRows = await jobRes.json();
  drawJobBySize(jobRows);

  // 기술 데이터 (전체 보관 후 첫 카테고리 그리기)
  const techRes = await fetch('/api/tech-by-size');
  techData = await techRes.json();
  drawTechBySize('언어');
}

// --- 직무별 ---
function drawJobBySize(rows) {
  const cats = [...new Set(rows.map(r => r['직무카테고리']))];
  const big = {}, start = {};
  rows.forEach(r => {
    if (r['회사규모'] === SIZE_BIG)   big[r['직무카테고리']]   = r['ratio'];
    if (r['회사규모'] === SIZE_START) start[r['직무카테고리']] = r['ratio'];
  });

  if (jobSizeChart) jobSizeChart.destroy();
  jobSizeChart = new Chart(document.getElementById('jobSizeChart'), {
    type: 'bar',
    data: {
      labels: cats,
      datasets: [
        { label: '대기업',   data: cats.map(c => big[c]   || 0), backgroundColor: '#3b82f6' },
        { label: '스타트업', data: cats.map(c => start[c] || 0), backgroundColor: '#14b8a6' }
      ]
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#e2e8f0' } } },
      scales: {
        x: { ticks: { color: '#94a3b8' } },
        y: { ticks: { color: '#94a3b8', callback: v => v + '%' } }
      }
    }
  });

  // 차이 / 공통 카드
  const stats = cats.map(c => ({
    name: c, big: big[c] || 0, start: start[c] || 0,
    diff: (big[c] || 0) - (start[c] || 0)
  }));
  renderDiff(stats, 'jobDiffList');
  renderCommon(stats, 'jobCommonList');
}

// --- 기술별 (카테고리 선택) ---
function drawTechBySize(category) {
  const rows = techData.filter(r => r['카테고리'] === category);
  const techs = [...new Set(rows.map(r => r['기술']))];
  const big = {}, start = {};
  rows.forEach(r => {
    if (r['회사규모'] === SIZE_BIG)   big[r['기술']]   = r['비율(%)'];
    if (r['회사규모'] === SIZE_START) start[r['기술']] = r['비율(%)'];
  });

  if (techSizeChart) techSizeChart.destroy();
  techSizeChart = new Chart(document.getElementById('techSizeChart'), {
    type: 'bar',
    data: {
      labels: techs,
      datasets: [
        { label: '대기업',   data: techs.map(t => big[t]   || 0), backgroundColor: '#3b82f6' },
        { label: '스타트업', data: techs.map(t => start[t] || 0), backgroundColor: '#14b8a6' }
      ]
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#e2e8f0' } } },
      scales: {
        x: { ticks: { color: '#94a3b8' } },
        y: { ticks: { color: '#94a3b8', callback: v => v + '%' } }
      }
    }
  });

  const stats = techs.map(t => ({
    name: t, big: big[t] || 0, start: start[t] || 0,
    diff: (big[t] || 0) - (start[t] || 0)
  }));
  renderDiff(stats, 'techDiffList');
  renderCommon(stats, 'techCommonList');
}

// --- 카드 렌더 (직무·기술 공용) ---
function renderDiff(stats, elId) {
  const sorted = [...stats].sort((a, b) => b.diff - a.diff);
  const bigTop   = sorted.filter(s => s.diff > 0).slice(0, 3);   // 대기업 우세
  const startTop = sorted.filter(s => s.diff < 0).slice(-3).reverse(); // 스타트업 우세
  let html = '';
  bigTop.forEach(s => {
    html += '<div class="reco-item"><span class="tech">' + s.name +
            '</span><span class="pct" style="color:#60a5fa">대기업 +' + s.diff.toFixed(1) + '%p</span></div>';
  });
  startTop.forEach(s => {
    html += '<div class="reco-item"><span class="tech">' + s.name +
            '</span><span class="pct" style="color:#2dd4bf">스타트업 +' + Math.abs(s.diff).toFixed(1) + '%p</span></div>';
  });
  document.getElementById(elId).innerHTML = html;
}

function renderCommon(stats, elId) {
  // 양쪽 평균이 높은 순 = 공통으로 많이 뽑는 것
  const sorted = [...stats].sort((a, b) => (b.big + b.start) - (a.big + a.start)).slice(0, 4);
  document.getElementById(elId).innerHTML = sorted.map(s =>
    '<div class="reco-item"><span class="tech">' + s.name +
    '</span><span class="pct pct-base">대 ' + s.big + '% / 스 ' + s.start + '%</span></div>'
  ).join('');
}

// --- 탭 전환 ---
function switchCompanyTab(el, paneId) {
  document.querySelectorAll('#company-tabs .nav-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('cpane-job').style.display  = (paneId === 'cpane-job')  ? 'flex' : 'none';
  document.getElementById('cpane-tech').style.display = (paneId === 'cpane-tech') ? 'flex' : 'none';
}

function switchTechCat(el, category) {
  document.querySelectorAll('#tech-cat-tabs .nav-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  drawTechBySize(category);
}

let periodchart = null;

async function drawperiodChart(job) {
  const [monthlyRes, summaryRes] = await Promise.all([
    fetch('/api/job-monthly'),
    fetch('/api/trend-summary')
  ]);
  const monthly = await monthlyRes.json();
  const summary = await summaryRes.json();
                
  const peak = summary.job_peak[job];          
  const labels = monthly.map(d => d.month);
  const values = monthly.map(d => d[job]);

  if (periodchart) periodchart.destroy();

  periodchart = new Chart(document.getElementById('periodChart'), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: job + ' 월별 공고 수',
        data: values,
        backgroundColor: labels.map(m => m === peak ? '#f59e0b' : '#3b82f6')
      }]
    },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }, 
        title: {
        display: true,
        text: job + ' 월별 채용 공고 수',
        color: '#e2e8f0',
        font: { size: 15 }
        }},
      scales: {
        x: { ticks: { color: '#94a3b8' } },
        y: { ticks: { color: '#94a3b8' }, title: { display: true, text: '공고 수', color: '#94a3b8' } }
      }
    }
  });
}

async function drawTechChart() {
  const res = await fetch('/api/trend-summary');
  const summary = await res.json();

  const rising  = summary.rising_tech;
  const falling = summary.falling_tech;
  const all = [...rising, ...falling];

  new Chart(document.getElementById('techChart'), {
    type: 'bar',
    data: {
      labels: all.map(d => d['기술']),
      datasets: [{
        label: '공고 비중 증감률(%)',
        data: all.map(d => d['증가율(%)']),
        backgroundColor: all.map(d => d['증가율(%)'] >= 0 ? '#22c55e' : '#ef4444')
      }]
    },
    options: {
      indexAxis: 'y',                  // 가로 막대
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#e2e8f0' } } },
      scales: {
        x: { ticks: { color: '#94a3b8', callback: function(v){ return v + '%'; } } },
        y: { ticks: { color: '#94a3b8' } }
      }
    }
  });

  // 추천 카드
  document.getElementById('risingList').innerHTML = rising.map(function(d) {
    return '<div class="reco-item"><span class="tech">' + d['기술'] +
           '</span><span class="pct pct-up">+' + d['증가율(%)'] + '%</span></div>';
  }).join('');

  document.getElementById('baseList').innerHTML = summary.top_tech.slice(0, 6).map(function(d) {
    return '<div class="reco-item"><span class="tech">' + d['기술'] +
           '</span><span class="pct pct-base">' + d['전체평균(%)'] + '%</span></div>';
  }).join('');
}


let selectedJob = '';


document.querySelectorAll('.job-chip').forEach(function(chip) {
  chip.addEventListener('click', function() {
    document.querySelectorAll('.job-chip').forEach(c => c.classList.remove('on'));
    chip.classList.add('on');
    selectedJob = chip.dataset.v;
  });
});

document.querySelectorAll('.topic-chip').forEach(function(chip) {
  chip.addEventListener('click', function() {
    if (!selectedJob) {
      addMessage('먼저 직무를 선택해주세요.', 'bot');
      return;
    }
    const question = selectedJob + ' ' + chip.dataset.q; 
    document.getElementById('chatInput').value = question;
    sendChat();
  });
});


function addMessage(text, sender) {
  const box = document.getElementById('chatMessages');
  const msg = document.createElement('div');
  msg.className = 'chat-msg ' + (sender === 'user' ? 'chat-msg-user' : 'chat-msg-bot');
  msg.textContent = text;
  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;   // 항상 최신 메시지로 스크롤
}

// 질문 전송
async function sendChat() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;                   // 빈 입력 무시

  addMessage(text, 'user');            // 내 질문 화면에 표시
  input.value = '';

  const sendBtn = document.getElementById('chatSend');
  sendBtn.disabled = true;             // 응답 대기 중 버튼 잠금
  addMessage('답변을 생성하고 있습니다...', 'bot');
  const loadingMsg = document.getElementById('chatMessages').lastChild;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    loadingMsg.innerHTML = marked.parse(data.answer);   // 로딩 문구를 실제 답변으로 교체
  } catch (e) {
    loadingMsg.textContent = '오류가 발생했습니다. 서버 상태를 확인해주세요.';
  } finally {
    sendBtn.disabled = false;
  }
}

// 버튼 클릭 + 엔터키 연결
document.getElementById('chatSend').addEventListener('click', sendChat);
document.getElementById('chatInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') sendChat();
});

// 언어 칩 토글
document.querySelectorAll('#rm-stack .rm-chip').forEach(function(chip) {
  chip.addEventListener('click', function() {
    chip.classList.toggle('on');
  });
});

document.getElementById('rm-generate').addEventListener('click', async function() {
  // 선택된 칩(on 클래스) 모으기
  const stack = Array.from(document.querySelectorAll('#rm-stack .rm-chip.on'))
                     .map(c => c.dataset.v);
  
  if (stack.length === 0) {
    document.getElementById('rm-result').innerHTML =
      '<p style="color:#ef4444; font-size:13px;">언어를 하나 이상 선택해주세요.</p>';
    return;
  }

  const body = {
    job: document.getElementById('rm-job').value,
    company: document.getElementById('rm-company').value,
    experience: document.getElementById('rm-exp').value,
    current_stack: stack,
    goal_period: document.getElementById('rm-period').value
  };

  const box = document.getElementById('rm-result');
  const btn = document.getElementById('rm-generate');
  box.innerHTML = '로드맵을 생성하고 있습니다... (모델 추론에 시간이 걸릴 수 있습니다)';
  btn.disabled = true;

  try {
    const res = await fetch('/api/roadmap', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    box.innerHTML =
      '<h3 style="color:#e2e8f0; margin-bottom:8px;">' + data.title + '</h3>' +
      '<img src="' + data.image_url + '?t=' + Date.now() + '" style="width:100%; max-width:840px; border-radius:12px;">' +
      '<p style="color:#94a3b8; font-size:13px; margin-top:8px;">' + (data.market_insight || '') + '</p>';
  } catch (e) {
    box.innerHTML = '오류가 발생했습니다. 서버 로그를 확인해주세요.';
  } finally {
    btn.disabled = false;
  }
});

document.getElementById('mainCard').click();