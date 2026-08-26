/* ============================================================
   首页逻辑：模式选择、拖拽上传、任务列表轮询
   ============================================================ */

const MODE_META = {
  outside: { label: '巢外视频', needsFiles: 1 },
  inside: { label: '巢内视频', needsFiles: 1 },
  multi: { label: '双路同步', needsFiles: 2 },
};

// ---------- 状态 ----------
let currentMode = 'outside';
let selectedFiles = {}; // { 上传位index: File }
let refreshTimer = null;

// ---------- DOM ----------
const modeGrid = document.getElementById('modeGrid');
const uploadGrid = document.getElementById('uploadGrid');
const btnAnalyze = document.getElementById('btnAnalyze');
const btnClear = document.getElementById('btnClear');
const taskList = document.getElementById('taskList');

// ---------- 模式选择 ----------
modeGrid.addEventListener('click', (e) => {
  const opt = e.target.closest('.mode-option');
  if (!opt) return;
  currentMode = opt.dataset.mode;
  modeGrid.querySelectorAll('.mode-option').forEach((o) =>
    o.classList.toggle('active', o === opt)
  );
  selectedFiles = {};
  rebuildUploadGrid();
});

// ---------- 上传区 ----------
function rebuildUploadGrid() {
  uploadGrid.classList.toggle('single', currentMode !== 'multi');
  uploadGrid.innerHTML = '';
  const need = MODE_META[currentMode].needsFiles;

  for (let i = 0; i < need; i++) {
    const label = currentMode === 'multi' ? (i === 0 ? '巢外视频' : '巢内视频') : '上传视频';
    const dz = document.createElement('div');
    dz.className = 'dropzone';
    dz.dataset.index = i;
    dz.innerHTML = `
      <div class="dz-label">${label}：拖拽到此处，或<b>点击选择</b></div>
      <div class="dz-hint">支持 mp4 / avi / mov</div>
      <input type="file" accept="video/*,.mp4,.avi,.mov">
    `;
    attachZone(dz, i);
    uploadGrid.appendChild(dz);
  }
  updateAnalyzeBtn();
}

function attachZone(dz, index) {
  const input = dz.querySelector('input[type="file"]');
  dz.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    if (input.files.length) setFile(index, input.files[0]);
    input.value = '';
  });
  ['dragover', 'dragenter'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('dragover'); })
  );
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('dragover'); })
  );
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(index, f);
  });
}

function setFile(index, file) {
  selectedFiles[index] = file;
  const dz = uploadGrid.children[index];
  if (!dz) return;
  dz.querySelector('.dz-label').style.display = 'none';
  dz.querySelector('.dz-hint').style.display = 'none';
  const input = dz.querySelector('input[type="file"]');
  if (input) input.remove();

  const chip = document.createElement('div');
  chip.className = 'file-chip';
  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  chip.innerHTML = `&#127916; ${esc(file.name)} <span class="chip-size">${sizeMB}MB</span>
    <span class="remove" title="移除">&#10005;</span>`;
  chip.querySelector('.remove').addEventListener('click', (e) => {
    e.stopPropagation();
    delete selectedFiles[index];
    rebuildUploadGrid();
  });
  dz.appendChild(chip);
  updateAnalyzeBtn();
}

function updateAnalyzeBtn() {
  const need = MODE_META[currentMode].needsFiles;
  const filled = Object.keys(selectedFiles).length;
  btnAnalyze.disabled = filled < need;
  btnAnalyze.textContent = filled >= need ? '开始分析' : `还需上传 ${need - filled} 个文件`;
}

btnClear.addEventListener('click', () => {
  selectedFiles = {};
  rebuildUploadGrid();
});

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ---------- 创建任务 ----------
btnAnalyze.addEventListener('click', async () => {
  btnAnalyze.disabled = true;
  const fd = new FormData();
  fd.append('mode', currentMode);
  for (const f of Object.values(selectedFiles)) fd.append('files', f);
  try {
    const resp = await fetch('/api/tasks', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || '创建任务失败');
    selectedFiles = {};
    rebuildUploadGrid();
    refreshTasks(true);
  } catch (err) {
    alert('上传失败：' + err.message);
    updateAnalyzeBtn();
  }
});

// ---------- 任务列表 ----------
async function refreshTasks(force) {
  let tasks = [];
  try {
    const resp = await fetch('/api/tasks');
    tasks = (await resp.json()).tasks || [];
  } catch (e) {
    if (force) alert('无法连接分析服务，请确认服务已启动');
    return;
  }

  if (!tasks.length) {
    taskList.innerHTML = '<div class="empty-tip">暂无任务</div>';
    return;
  }

  taskList.innerHTML = tasks.map(renderTask).join('');

  const busy = tasks.some((t) => t.status === 'pending' || t.status === 'running');
  if (busy) scheduleRefresh();
}

function renderTask(t) {
  const modeName = (MODE_META[t.mode] || {}).label || t.mode;
  const fileNames = (t.files || []).map((f) => f.name).join('、') || '-';
  const statusMap = { pending: '排队中', running: t.message || '分析中', done: '已完成', failed: '失败' };
  const statusText = statusMap[t.status] || t.status;

  let barHtml = '';
  let actionHtml = '';
  if (t.status === 'done') {
    barHtml = `<div class="progress-wrap"><div class="progress-bar done" style="width:100%"></div></div>`;
    actionHtml = `<a class="link-btn" href="/result?id=${t.id}">查看结果 &rsaquo;</a>`;
  } else if (t.status === 'failed') {
    barHtml = `<div class="progress-wrap"><div class="progress-bar failed" style="width:100%"></div></div>`;
    actionHtml = `<span class="link-btn" style="cursor:not-allowed;color:var(--text-muted)">失败</span>`;
  } else {
    const pct = t.progress || 0;
    barHtml = `
      <div class="progress-wrap"><div class="progress-bar" style="width:${pct}%"></div></div>
      <div class="progress-text">${pct}%</div>`;
    actionHtml = `<span class="link-btn" style="cursor:default;color:var(--text-muted)">分析中</span>`;
  }

  return `
    <div class="task-item">
      <div class="t-info">
        <div class="t-name">${esc(fileNames)}</div>
        <div class="t-meta">#${t.id} · ${modeName} · ${t.created_at}</div>
      </div>
      <div class="t-progress">${barHtml}</div>
      <div class="t-status"><span class="badge ${t.status}">${statusText}</span></div>
      <div class="t-action">${actionHtml}</div>
    </div>`;
}

function scheduleRefresh() {
  if (refreshTimer) clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => refreshTasks(), 2000);
}

// ---------- 初始化 ----------
rebuildUploadGrid();
refreshTasks();
