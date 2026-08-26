/* ============================================================
   结果页逻辑：加载任务 → 按模式渲染
   布局：关键指标总览 + 模块预览卡片 + 点击弹出详情模态框
   ============================================================ */

const app = document.getElementById('app');
const taskId = (() => {
  const pathId = location.pathname.split('/').pop();
  const queryId = new URLSearchParams(location.search).get('id');
  return queryId || (pathId !== 'result' ? pathId : '') || '';
})();

const MODE_LABEL = { outside: '巢外视频', inside: '巢内视频', multi: '双路同步' };

// ---------- 工具 ----------
function esc(s) {
  if (s === null || s === undefined) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function num(v, digits = 1) {
  if (v === null || v === undefined || isNaN(v)) return '-';
  return Number(v).toFixed(digits);
}

const STATUS_PILL = {
  normal: '<span class="status-pill normal">正常</span>',
  warning: '<span class="status-pill warning">需复核</span>',
  unknown: '<span class="status-pill unknown">未知</span>',
};

// ---------- 可折叠卡片组件 ----------
function foldCard(title, summary, body, open = false) {
  return `
    <div class="fold-card${open ? ' open' : ''}" onclick="this.classList.toggle('open')">
      <div class="fold-head">
        <span class="fold-title">${title}</span>
        ${summary ? `<span class="fold-summary">${summary}</span>` : ''}
        <span class="fold-arrow">▼</span>
      </div>
      <div class="fold-body">${body}</div>
    </div>`;
}

// ---------- 全局模态框 ----------
let _modalContent = {};
function showModal(id) {
  const data = _modalContent[id];
  if (!data) return;
  const el = document.getElementById('modal');
  el.querySelector('.modal-title').innerHTML = data.title;
  el.querySelector('.modal-body').innerHTML = data.body;
  el.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeModal() {
  const el = document.getElementById('modal');
  el.classList.remove('open');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

// ---------- 任务选择列表 ----------
async function renderTaskSelector() {
  let tasks = [];
  try {
    const resp = await fetch('/api/tasks');
    const data = await resp.json();
    tasks = (data.tasks || []).filter((t) => t.status === 'done' || t.status === 'failed');
  } catch (e) {
    app.innerHTML = '<div class="empty-tip">无法连接服务，请确认服务已启动</div>';
    return;
  }

  if (!tasks.length) {
    app.innerHTML = `
      <div class="empty-tip">
        <p>暂无分析任务</p>
        <p style="margin-top:12px;font-size:14px">
          <a href="/" style="color:var(--leaf);font-weight:600">返回首页上传视频</a>
        </p>
      </div>`;
    return;
  }

  const statusMap = { done: '已完成', failed: '失败' };
  const modeMap = { outside: '巢外', inside: '巢内', multi: '双路同步' };
  const fileNames = (t) => (t.files || []).map((f) => f.name).join('、') || '-';

  app.innerHTML = `
    <div class="page-header">
      <h1>选择分析任务</h1>
      <p>请选择要查看结果的任务</p>
    </div>
    <div class="card" style="padding:0">
      ${tasks.map((t) => `
        <a href="/result?id=${t.id}" class="task-selector-item" style="
          display:flex; justify-content:space-between; align-items:center;
          padding:14px 20px; border-bottom:1px solid var(--line); text-decoration:none; color:inherit;
        " onmouseover="this.style.background='#f7f8f6'" onmouseout="this.style.background=''">
          <div>
            <div style="font-weight:600">${esc(fileNames(t))}</div>
            <div style="font-size:13px;color:var(--text-muted);margin-top:4px">
              #${t.id} · ${modeMap[t.mode] || t.mode} · ${t.created_at}
            </div>
          </div>
          <span class="badge ${t.status}">${statusMap[t.status] || t.status}</span>
        </a>
      `).join('')}
    </div>
    <div style="margin-top:16px;text-align:center">
      <a href="/" style="color:var(--primary-dark);font-weight:600;font-size:14px">← 返回首页上传新视频</a>
    </div>`;
}

// ---------- 加载 ----------
async function load() {
  if (!taskId) {
    await renderTaskSelector();
    return;
  }
  let task;
  try {
    const resp = await fetch(`/api/tasks/${taskId}`);
    task = await resp.json();
    if (resp.status === 404) throw new Error(task.error || '任务不存在');
  } catch (e) {
    app.innerHTML = `<div class="empty-tip">${esc(e.message)}</div>`;
    return;
  }

  if (task.status !== 'done') {
    renderProgress(task);
    setTimeout(load, 2000);
    return;
  }
  renderResult(task);
}

// ---------- 未完成状态 ----------
function renderProgress(task) {
  const statusText = task.status === 'failed' ? '分析失败' : task.message || '分析中';
  const bar = task.status === 'failed'
    ? '<div class="progress-wrap"><div class="progress-bar failed" style="width:100%"></div></div>'
    : `<div class="progress-wrap"><div class="progress-bar" style="width:${task.progress || 0}%"></div></div>`;
  app.innerHTML = `
    <div class="page-header">
      <h1>任务 #${task.id}</h1>
      <p>${MODE_LABEL[task.mode] || task.mode} · 创建于 ${task.created_at}</p>
    </div>
    <div class="card">
      <h2>${statusText}</h2>
      ${bar}
      <div class="progress-text" style="margin-top:8px">${task.progress || 0}%</div>
      ${task.error ? `<div class="alert-item danger" style="margin-top:12px">${esc(task.error)}</div>` : ''}
    </div>`;
}

// ---------- 结果渲染 ----------
function renderResult(task) {
  _modalContent = {};
  const r = task.result;
  let body = '';

  if (r.mode === 'multi') {
    body = renderMulti(r);
  } else if (r.mode === 'inside') {
    body = renderInside(r);
  } else {
    body = renderOutside(r);
  }

  app.innerHTML = `
    <div class="page-header">
      <h1>${esc(r.mode_label || '')}结果</h1>
      <p>任务 #${task.id} · 创建于 ${task.created_at} · 完成于 ${task.finished_at || ''}</p>
    </div>
    ${renderOverview(r)}
    <div class="detail-grid">${body}</div>
    ${renderDownloads(task.id, r)}
    <div class="modal-overlay" id="modal" onclick="if(event.target===this)closeModal()">
      <div class="modal-panel">
        <div class="modal-head">
          <span class="modal-title"></span>
          <span class="modal-close" onclick="closeModal()">✕</span>
        </div>
        <div class="modal-body"></div>
      </div>
    </div>
  `;
}

// ---------- 健康总览（始终展开） ----------
function renderOverview(r) {
  const alerts = collectAlerts(r);
  const warnCount = alerts.filter((a) => a.level === 'warning' || a.level === 'danger').length;
  const level = warnCount > 0 ? 'warning' : 'ok';
  const levelText = warnCount > 0 ? `${warnCount} 项需关注` : '整体状态良好';
  const s = r.summary || {};

  const kpis = [
    ['总帧数', s.total_frames ?? '-'],
    ['帧率', num(s.fps) + ' fps'],
    ['总轨迹数', s.total_tracks ?? '-'],
    ['处理耗时', num(s.processing_time_s) + ' s'],
  ];
  if (r.mode === 'outside' || (r.mode === 'multi' && r.outside)) {
    const b = (r.mode === 'multi' ? r.outside.summary && r.outside.summary.behavior : s.behavior) || {};
    kpis.push(['进巢', b.entering ?? '-'], ['出巢', b.exiting ?? '-'], ['采粉', b.foraging ?? '-']);
  }
  if (r.mode === 'inside' || (r.mode === 'multi' && r.inside)) {
    const metrics = (r.mode === 'multi' ? r.inside.metrics : r.metrics) || [];
    const active = metrics.find((m) => m.name && m.name.includes('活跃'));
    if (active) kpis.push(['活跃趋势', fmtVal('trend', (active.values || {}).trend)]);
  }

  return `
    <div class="overview" style="background:#fff">
      <div class="ov-dot ${level}"></div>
      <div>
        <div class="ov-title">健康评估总览</div>
        <div class="ov-desc">基于行为量化指标的蜂群健康状态快速评估</div>
      </div>
      <div class="ov-alerts" style="color:${level === 'ok' ? 'var(--ok)' : 'var(--warn)'}">${levelText}</div>
    </div>
    <div class="kpi-grid">
      ${kpis.slice(0, 8).map(([k, v]) => `<div class="kpi-item"><div class="kpi-value">${v}</div><div class="kpi-label">${k}</div></div>`).join('')}
    </div>
    ${alerts.length ? `<div class="card" style="padding:14px 18px;margin-top:16px"><h2>预警</h2>${alerts.map(renderAlert).join('')}</div>` : ''}
  `;
}

function collectAlerts(r) {
  const list = [];
  if (r.mode === 'multi') {
    for (const part of [r.outside, r.inside]) {
      for (const a of collectAlerts(part)) list.push(a);
    }
    return list;
  }
  if (Array.isArray(r.alerts)) for (const a of r.alerts) list.push({ level: a.level || 'info', text: a.text });
  if (Array.isArray(r.anomalies)) for (const a of r.anomalies) list.push({ level: a.severity || 'info', text: a.detail });
  for (const m of r.metrics || []) {
    if (m.status === 'warning') list.push({ level: 'warning', text: `${m.name}：${m.description}` });
  }
  return list;
}

function renderAlert(a) {
  const map = { danger: '严重', warning: '需复核', info: '提示' };
  return `<div class="alert-item ${a.level}"><b>[${map[a.level] || a.level}]</b> ${esc(a.text)}</div>`;
}

// ---------- 详情卡片（预览 + 点击弹出模态框） ----------
function detailCard(id, icon, title, preview, body) {
  _modalContent[id] = { title: `${icon} ${title}`, body };
  return `
    <div class="detail-card" onclick="showModal('${id}')">
      <div class="detail-card-icon">${icon}</div>
      <div class="detail-card-title">${title}</div>
      <div class="detail-card-preview">${preview}</div>
      <div class="detail-card-arrow">查看详情 →</div>
    </div>`;
}

// ---------- 巢外 ----------
function renderOutside(r) {
  const s = r.summary || {};
  const b = s.behavior || {};
  const q = r.track_quality || {};
  const p = r.pollen_analysis || {};
  const pollenRatio = num(p.incoming_with_pollen_ratio * 100, 1);
  const pollenCount = p.pollen_carrying_tracks ?? '-';
  let html = '';

  if (r.annotated_video) {
    html += detailCard('video', '🎬', '标注视频',
      '查看分析标注视频',
      `<video class="annotated-video" src="${r.annotated_video}" controls preload="metadata" style="width:100%;max-height:480px;border-radius:8px;background:#000"></video>`);
  }

  html += detailCard('track', '📊', '轨迹质量评估',
    `平均轨迹长度 ${num(q.continuity && q.continuity.mean_track_length, 0)} 帧`,
    renderTrackQuality(q));

  html += detailCard('pollen', '🌼', '花粉采集评估',
    `携粉进巢 ${pollenRatio}% · 携粉个体 ${pollenCount}`,
    `<div class="m-values">
      <div class="kv"><span>携粉进巢占比</span><span>${pollenRatio}%</span></div>
      <div class="kv"><span>携粉个体数</span><span>${pollenCount}</span></div>
    </div>
    <div style="margin-top:10px;font-size:13px;color:var(--text-muted)">${esc(p.assessment || '')}</div>`);

  const behaviorPreview = [
    ['进巢', b.entering], ['出巢', b.exiting], ['采粉', b.foraging],
    ['停歇', b.resting], ['徘徊', b.wandering], ['移动', b.moving],
  ].map(([k, v]) => `<span style="margin:0 6px"><b>${k}</b> ${v ?? '-'}</span>`).join('');

  html += detailCard('behavior', '🐝', '巢口行为分布', behaviorPreview,
    `<div class="m-values">${[
      ['进巢', b.entering], ['出巢', b.exiting], ['采粉', b.foraging],
      ['停歇', b.resting], ['徘徊', b.wandering], ['移动', b.moving],
    ].map(([k, v]) => `<div class="kv"><span>${k}</span><span>${v === undefined ? '-' : v}</span></div>`).join('')}</div>`);

  html += detailCard('info', 'ℹ️', '视频与处理信息',
    `${s.total_frames ?? '-'} 帧 · ${num(s.fps)} fps · ${num(s.processing_time_s)}s`,
    `<div class="m-values" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:4px 20px">
      <div class="kv"><span>总帧数</span><span>${s.total_frames ?? '-'}</span></div>
      <div class="kv"><span>帧率</span><span>${num(s.fps)} fps</span></div>
      <div class="kv"><span>总轨迹数</span><span>${s.total_tracks ?? '-'}</span></div>
      <div class="kv"><span>处理耗时</span><span>${num(s.processing_time_s)} s</span></div>
    </div>`);

  if (r.report_html) {
    html += detailCard('report', '📄', '详细分析报告',
      '点击查看完整分析报告',
      `<iframe class="report-frame" src="${r.report_html}"></iframe>`);
  }

  return html;
}

// 轨迹质量板块
function renderTrackQuality(q) {
  if (!q || q.implemented === false) {
    const c = q.continuity || {};
    const st = q.stability || {};
    return `
      <div class="alert-item info" style="margin-bottom:12px">
        轨迹质量模块<b>尚未开发</b>，以下为占位数据。
      </div>
      <div class="m-values">
        <div class="kv"><span>平均轨迹长度（帧）</span><span>${num(c.mean_track_length, 0)}</span></div>
        <div class="kv"><span>平均存活时长（秒）</span><span>${num(c.mean_track_lifetime_s)}</span></div>
        <div class="kv"><span>轨迹断裂率</span><span>${num(c.track_break_rate * 100)}%</span></div>
        <div class="kv"><span>平均速度标准差</span><span>${num(st.mean_speed_std)}</span></div>
        <div class="kv"><span>方向变化率</span><span>${num(st.direction_change_rate * 100)}%</span></div>
        <div class="kv"><span>轨迹点缺失率</span><span>${num(st.missing_frame_ratio * 100)}%</span></div>
      </div>`;
  }
  return '<div class="m-values"><div class="kv"><span>轨迹质量</span><span>-</span></div></div>';
}

// ---------- 巢内 ----------
function renderInside(r) {
  const metrics = r.metrics || [];
  const grid = (metrics.find((m) => m.name.includes('高密度')) || {}).values || {};
  const stationary = metrics.find((x) => x.name.includes('静止'));
  const stationaryCount = (stationary && stationary.values && stationary.values.candidate_tracks || []).length;
  const s = r.summary || {};
  let html = '';

  if (r.annotated_video) {
    html += detailCard('video', '🎬', '标注视频',
      '查看分析标注视频',
      `<video class="annotated-video" src="${r.annotated_video}" controls preload="metadata" style="width:100%;max-height:480px;border-radius:8px;background:#000"></video>`);
  }

  html += detailCard('metrics', '📋', '个体姿态与行为指标',
    `${metrics.length} 项指标`,
    `<div class="metric-grid">${metrics.map(renderMetric).join('')}</div>`);

  if (grid.grid) {
    html += detailCard('heatmap', '🔥', '高密度聚集热力图',
      `峰值网格占比 ${num(grid.peak_cell_share * 100, 1)}%`,
      `<div class="heatmap-wrap">
        <canvas id="densityMap" width="360" height="360"></canvas>
        <div class="m-values" style="flex:1;min-width:220px">
          <div class="kv"><span>峰值网格占比</span><span>${num(grid.peak_cell_share * 100, 1)}%</span></div>
          <div class="kv"><span>峰值网格位置</span><span>${(grid.peak_grid_cell || []).join(', ')}</span></div>
          <div style="margin-top:8px;font-size:12px;color:var(--text-muted)">颜色越亮表示该区域蜜蜂聚集密度越高。</div>
        </div>
      </div>`);
  }

  if (stationaryCount) {
    html += detailCard('stationary', '⚠️', '静止候选',
      `${stationaryCount} 个候选个体`,
      renderStationary(metrics));
  }

  html += detailCard('info', 'ℹ️', '视频与处理信息',
    `${s.total_frames ?? '-'} 帧 · ${num(s.fps)} fps · ${num(s.processing_time_s)}s`,
    `<div class="m-values" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:4px 20px">
      <div class="kv"><span>总帧数</span><span>${s.total_frames ?? '-'}</span></div>
      <div class="kv"><span>帧率</span><span>${num(s.fps)} fps</span></div>
      <div class="kv"><span>总轨迹数</span><span>${s.total_tracks ?? '-'}</span></div>
      <div class="kv"><span>处理耗时</span><span>${num(s.processing_time_s)} s</span></div>
    </div>`);

  if (r.report_html) {
    html += detailCard('report', '📄', '详细分析报告',
      '点击查看完整分析报告',
      `<iframe class="report-frame" src="${r.report_html}"></iframe>`);
  }

  return html;
}

function renderMetric(m) {
  const pill = STATUS_PILL[m.status] || STATUS_PILL.unknown;
  const values = m.values || {};
  const rows = Object.entries(values)
    .filter(([k]) => k !== 'grid' && k !== 'candidate_tracks')
    .map(([k, v]) => `<div class="kv"><span>${fmtKey(k)}</span><span>${fmtVal(k, v)}</span></div>`)
    .join('');
  return `
    <div class="metric-card">
      <div class="m-head">
        <span class="m-name">${esc(m.name)}</span>
        ${pill}
      </div>
      <div class="m-desc">${esc(m.description || '')}</div>
      <div class="m-values">${rows}</div>
      ${m.candidates ? `<div style="margin-top:8px;font-size:12px;color:var(--warn)">候选个体：${m.candidates} 个，建议人工复核</div>` : ''}
    </div>`;
}

function fmtKey(k) {
  const map = {
    mean_orientation_degrees: '平均朝向（度）', motion_alignment: '运动一致性',
    candidate_ratio: '异常姿态占比', median_body_aspect_ratio: '姿态长宽比中位数',
    mean_active_tracks: '平均活跃轨迹数', mean_speed: '平均速度',
    trend: '活跃趋势', peak_cell_share: '峰值网格占比',
    peak_grid_cell: '峰值网格位置', threshold_frames: '静止判定阈值（帧）',
  };
  return map[k] || k;
}

function fmtVal(k, v) {
  if (k === 'motion_alignment' || k === 'candidate_ratio' || k === 'peak_cell_share') return num(v * 100, 1) + '%';
  if (k === 'trend') return { up: '上升', down: '下降', stable: '稳定' }[v] || v;
  if (Array.isArray(v)) return v.join(', ');
  return num(v);
}

// 静止候选表格
function renderStationary(metrics) {
  const m = metrics.find((x) => x.name.includes('静止'));
  const tracks = (m && m.values && m.values.candidate_tracks) || [];
  if (!tracks.length) return '';
  const rows = tracks.map((t) => `
    <tr><td>#${t.track_id}</td><td>${t.stationary_frames ?? '-'}</td><td>${t.start_frame ?? '-'} – ${t.end_frame ?? '-'}</td></tr>`).join('');
  return `
    <p style="font-size:12px;color:var(--text-muted);margin-bottom:8px">
      长时间静止（阈值 ${(m.values && m.values.threshold_frames) || 500} 帧）的个体轨迹，疑似病弱或掉落。
    </p>
    <table class="data-table">
      <thead><tr><th>轨迹 ID</th><th>静止帧数</th><th>起始帧 – 结束帧</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ---------- 多模态 ----------
function renderMulti(r) {
  const outsidePart = { ...r.outside, alerts: [], anomalies: [] };
  const insidePart = { ...r.inside, alerts: [], anomalies: [] };
  return `
    <div style="grid-column:1/-1;margin-bottom:8px"><h2 style="font-size:16px;color:var(--brown)">巢外通道</h2></div>
    ${renderOutside(outsidePart)}
    <div style="grid-column:1/-1;margin:12px 0 8px"><h2 style="font-size:16px;color:var(--brown)">巢内通道</h2></div>
    ${renderInside(insidePart)}
    ${r.report_html ? detailCard('report', '📄', '详细分析报告',
      '点击查看完整分析报告',
      `<iframe class="report-frame" src="${r.report_html}"></iframe>`) : ''}
  `;
}

// ---------- 下载 ----------
function renderDownloads(taskId, r) {
  const videos = [];
  if (r.annotated_video) videos.push({ href: r.annotated_video, label: '下载标注视频' });
  if (r.mode === 'multi') {
    if (r.outside && r.outside.annotated_video) videos.push({ href: r.outside.annotated_video, label: '下载巢外标注视频' });
    if (r.inside && r.inside.annotated_video) videos.push({ href: r.inside.annotated_video, label: '下载巢内标注视频' });
  }
  return `
    <section class="card">
      <h2>下载</h2>
      <div class="action-row">
        ${r.report_html ? `<a class="btn btn-outline" href="${r.report_html}" target="_blank">打开分析报告</a>` : ''}
        ${videos.map((v) => `<a class="btn btn-outline" href="${v.href}" download>${v.label}</a>`).join('')}
      </div>
    </section>`;
}

// ---------- 启动 ----------
load();