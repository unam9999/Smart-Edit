/* ═══════════════════════════════════════════════════════════
   VisionSort — Application Logic
   ═══════════════════════════════════════════════════════════ */

const API = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : 'https://web-production-849f70.up.railway.app';
let selectedFiles = [];
let sessionId = null;
let classificationData = null;

const CAT = {
  blurred: { dot: '#60a5fa', emoji: '◻' },
  people: { dot: '#f472b6', emoji: '◻' },
  animals: { dot: '#34d399', emoji: '◻' },
  aesthetic: { dot: '#a78bfa', emoji: '◻' },
  uncategorized: { dot: '#fb923c', emoji: '◻' },
  unlabelled: { dot: '#666666', emoji: '◻' },
};

const PRESETS = {
  blurred: [{ v: 'none', l: 'None' }],
  people: [{ v: 'none', l: 'None' }, { v: 'portraits', l: 'Portraits' }, { v: 'landscapes', l: 'Landscapes' }],
  animals: [{ v: 'none', l: 'None' }, { v: 'landscapes', l: 'Landscapes' }],
  aesthetic: [{ v: 'none', l: 'None' }, { v: 'aesthetic', l: 'Aesthetic' }, { v: 'landscapes', l: 'Landscapes' }],
  uncategorized: [{ v: 'none', l: 'None' }, { v: 'portraits', l: 'Portraits' }, { v: 'landscapes', l: 'Landscapes' }, { v: 'aesthetic', l: 'Aesthetic' }],
  unlabelled: [{ v: 'none', l: 'None' }],
};

// ── DOM ──
const $ = id => document.getElementById(id);
const dropZone = $('drop-zone');
const fileInput = $('file-input');
const fileListSec = $('file-list-sec');
const fileChips = $('file-chips');
const fileCount = $('file-count');
const actionRow = $('action-row');
const actionCount = $('action-count');
const btnClassify = $('btn-classify');
const bcIcon = $('bc-icon');
const bcText = $('bc-text');
const btnClear = $('btn-clear');
const progSec = $('progress-sec');
const progFill = $('prog-fill');
const progLabel = $('prog-label');
const phase1 = $('phase1');
const phase2 = $('phase2');
const ph1Btn = $('ph1-btn');
const ph2Btn = $('ph2-btn');
const presetSec = $('preset-sec');
const folderCards = $('folder-cards');
const btnProcess = $('btn-process');
const bpIcon = $('bp-icon');
const bpText = $('bp-text');
const btnBack = $('btn-back');
const resultsSec = $('results-sec');
const summaryCards = $('summary-cards');
const resultsTbody = $('results-tbody');
const toast = $('toast');

// ── Toast ──
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── Phase switching ──
function setPhase(n) {
  if (n === 1) {
    phase1.style.display = 'flex';
    phase1.style.flexDirection = 'column';
    phase1.style.gap = '0px';
    phase2.style.display = 'none';
    ph1Btn.classList.add('active');
    ph2Btn.classList.remove('active');
  } else {
    phase1.style.display = 'none';
    phase2.style.display = 'block';
    ph2Btn.classList.add('active');
    ph1Btn.classList.remove('active');
  }
}

// ── File handling ──
function addFiles(list) {
  const names = new Set(selectedFiles.map(f => f.name));
  for (const f of list) {
    if (!f.type.startsWith('image/')) { showToast(`Skipped "${f.name}"`); continue; }
    if (names.has(f.name)) continue;
    if (selectedFiles.length >= 100) { showToast('Max 100 images'); break; }
    selectedFiles.push(f);
    names.add(f.name);
  }
  renderFiles();
}

function removeFile(name) {
  selectedFiles = selectedFiles.filter(f => f.name !== name);
  renderFiles();
}

function renderFiles() {
  fileChips.innerHTML = '';
  selectedFiles.forEach(f => {
    const c = document.createElement('div');
    c.className = 'chip';
    c.innerHTML = `<span class="chip-name" title="${f.name}">${f.name}</span><span class="chip-rm" role="button" tabindex="0">✕</span>`;
    c.querySelector('.chip-rm').onclick = () => removeFile(f.name);
    fileChips.appendChild(c);
  });
  const n = selectedFiles.length;
  fileCount.textContent = n;
  actionCount.textContent = n;
  fileListSec.style.display = n ? 'block' : 'none';
  actionRow.style.display = n ? 'flex' : 'none';
}

// ── Drop zone events ──
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); addFiles([...e.dataTransfer.files]); });
dropZone.addEventListener('click', e => { if (!e.target.closest('label')) fileInput.click(); });
fileInput.addEventListener('change', () => { addFiles([...fileInput.files]); fileInput.value = ''; });

// ── Clear — does NOT destroy session ──
btnClear.addEventListener('click', () => {
  selectedFiles = [];
  renderFiles();
  // Note: sessionId is intentionally NOT cleared here
});

// ── Classify ──
btnClassify.addEventListener('click', async () => {
  if (!selectedFiles.length) return;
  btnClassify.disabled = true;
  bcIcon.innerHTML = '<span class="spinner"></span>';
  bcText.textContent = 'PROCESSING';
  progSec.style.display = 'block';
  progFill.style.width = '0%';
  progLabel.textContent = 'Uploading images...';

  let prog = 0;
  const timer = setInterval(() => {
    prog = Math.min(prog + Math.random() * 9, 88);
    progFill.style.width = prog + '%';
  }, 280);

  try {
    const fd = new FormData();
    selectedFiles.forEach(f => fd.append('files', f));
    progLabel.textContent = `Running EfficientNet-B0 on ${selectedFiles.length} image(s)...`;

    const res = await fetch(`${API}/api/classify/batch`, { method: 'POST', body: fd });
    clearInterval(timer);
    if (!res.ok) throw new Error(`Server ${res.status}: ${await res.text()}`);

    progFill.style.width = '100%';
    progLabel.textContent = 'Classification complete.';
    const data = await res.json();
    classificationData = data;
    sessionId = data.session_id;

    setTimeout(() => {
      progSec.style.display = 'none';
      showPhase2(data);
    }, 400);
  } catch (e) {
    clearInterval(timer);
    progSec.style.display = 'none';
    showToast('Error: ' + e.message);
  } finally {
    btnClassify.disabled = false;
    bcIcon.textContent = '→';
    bcText.textContent = 'CLASSIFY';
  }
});

// ── Phase 2 ──
function showPhase2(data) {
  setPhase(2);
  renderSummaryAndTable(data);
  renderFolderCards(data.summary);
  presetSec.style.display = 'block';
  resultsSec.style.display = 'block';
}

function renderFolderCards(summary) {
  folderCards.innerHTML = '';
  const folders = Object.entries(summary).filter(([, n]) => n > 0);
  if (!folders.length) {
    folderCards.innerHTML = '<p style="color:var(--muted)">No images classified.</p>';
    return;
  }
  folders.forEach(([cat, count]) => {
    const meta = CAT[cat] || { dot: '#666', emoji: '◻' };
    const opts = (PRESETS[cat] || [{ v: 'none', l: 'None' }])
      .map(p => `<option value="${p.v}">${p.l}</option>`).join('');
    const card = document.createElement('div');
    card.className = 'folder-card';
    card.innerHTML = `
      <div class="fc-top">
        <div class="fc-dot" style="background:${meta.dot}"></div>
        <div>
          <div class="fc-name">${cat}</div>
          <div class="fc-count">${count} image${count > 1 ? 's' : ''}</div>
        </div>
      </div>
      <select class="preset-select" data-folder="${cat}">${opts}</select>`;
    folderCards.appendChild(card);
  });
}

function renderSummaryAndTable(data) {
  summaryCards.innerHTML = '';
  Object.entries(data.summary || {}).forEach(([cat, n]) => {
    if (!n) return;
    const meta = CAT[cat] || { dot: '#666' };
    const c = document.createElement('div');
    c.className = 's-card';
    c.innerHTML = `
      <div class="fc-dot" style="background:${meta.dot};width:8px;height:8px;border-radius:50%"></div>
      <div class="s-count" style="color:${meta.dot}">${n}</div>
      <div class="s-label">${cat}</div>`;
    summaryCards.appendChild(c);
  });

  resultsTbody.innerHTML = '';
  (data.results || []).forEach((r, i) => {
    const cat = r.category || 'unlabelled';
    const pct = Math.round((r.confidence || 0) * 100);
    const meta = CAT[cat] || { dot: '#666' };
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i + 1}</td>
      <td class="td-fn" title="${r.filename}">${r.filename}</td>
      <td><span class="pill pill-${cat}">${cat}</span></td>
      <td><div class="conf-bar"><div class="conf-track"><div class="conf-fill" style="width:${pct}%;background:${meta.dot}"></div></div><span class="conf-val">${pct}%</span></div></td>
      <td style="color:var(--muted);max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${r.label}">${r.label}</td>`;
    resultsTbody.appendChild(tr);
  });
}

// ── Back button — preserves session ──
btnBack.addEventListener('click', () => setPhase(1));

// ── Process with session validation ──
btnProcess.addEventListener('click', async () => {
  if (!sessionId) {
    showToast('No active session — please re-classify');
    setPhase(1);
    return;
  }

  // Validate session is still alive
  try {
    const check = await fetch(`${API}/api/session/${sessionId}/status`);
    const status = await check.json();
    if (!status.alive) {
      showToast('Session expired — please re-upload and classify');
      sessionId = null;
      setPhase(1);
      return;
    }
  } catch (e) {
    // If status check fails, proceed anyway — the process endpoint will catch it
  }

  const selects = folderCards.querySelectorAll('.preset-select');
  const presets = [...selects].map(s => ({ folder: s.dataset.folder, preset: s.value }));

  btnProcess.disabled = true;
  bpIcon.innerHTML = '<span class="spinner"></span>';
  bpText.textContent = 'PROCESSING...';

  try {
    const res = await fetch(`${API}/api/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, presets }),
    });
    if (!res.ok) throw new Error(`Server ${res.status}: ${await res.text()}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'VisionSort_processed.zip';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    sessionId = null; // Only clear after successful download
    showToast('Download complete');
  } catch (e) {
    showToast('Error: ' + e.message);
  } finally {
    btnProcess.disabled = false;
    bpIcon.textContent = '↓';
    bpText.textContent = 'DOWNLOAD ZIP';
  }
});

// ── Scroll-to-app from hero CTA ──
document.querySelector('.hero-cta')?.addEventListener('click', () => {
  document.getElementById('app-section')?.scrollIntoView({ behavior: 'smooth' });
});
document.querySelector('.nav-cta')?.addEventListener('click', () => {
  document.getElementById('app-section')?.scrollIntoView({ behavior: 'smooth' });
});

// ═══════════════════════════════════════════════════════════
//  SCROLLYTELLING — Intersection Observer Animations
// ═══════════════════════════════════════════════════════════

function initScrollAnimations() {
  // Chaos tiles — scatter with random rotation
  const chaosTiles = document.querySelectorAll('.chaos-tile');
  chaosTiles.forEach(tile => {
    tile.style.setProperty('--rot', (Math.random() * 30 - 15) + 'deg');
    tile.style.setProperty('--tx', (Math.random() * 20 - 10) + 'px');
    tile.style.setProperty('--ty', (Math.random() * 20 - 10) + 'px');
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const tiles = entry.target.querySelectorAll('.chaos-tile, .clarity-tile');
        tiles.forEach((tile, i) => {
          setTimeout(() => tile.classList.add('visible'), i * 60);
        });
      }
    });
  }, { threshold: 0.3 });

  const chaosSection = document.getElementById('chaos');
  const claritySection = document.getElementById('clarity');
  if (chaosSection) observer.observe(chaosSection);
  if (claritySection) observer.observe(claritySection);
}

document.addEventListener('DOMContentLoaded', initScrollAnimations);
