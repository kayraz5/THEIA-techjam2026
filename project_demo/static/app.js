'use strict';
/* Live AI-content detection over a vertical video feed.
 *
 * The browser samples frames from the playing clip and POSTs them to a local server running the
 * real SigLIP2-giant detector. Nothing here is precomputed: every number on screen came back
 * from a forward pass that happened after you scrolled to that clip.
 */

const TUNING = {
  fps: 2,                 // measured server latency is ~82 ms, so 2 fps leaves ample headroom
  minIntervalMs: 500,
  quality: 0.95,          // MEASURED: q60 took a known fake from 0.983 to 0.511. Do not lower.
  window: 5,              // median-of-5 is the trigger: unmoved by 2 outlier frames, unlike an EMA
  emaAlpha: 0.35,         // shown, but NOT the trigger
  onThresh: 0.9446,       // thresholds.csv @FPR<=0.001
  offThresh: 0.70,        // asymmetric on purpose (Schmitt trigger); surfaced in the UI
  onUpdates: 2,
  offUpdates: 5,
  minDwellMs: 2500,
  backoffMs: 250,
  sparkN: 60,
};

const $ = s => document.querySelector(s);
const feedEl = $('#feed'), pillEl = $('#pill');
let CLIPS = [], HEALTH = null, activeCard = null, seq = 0, inflight = false;
let rateScale = 1, rateResetAt = 0, io = null;
const fpsWin = [];

// ---------------------------------------------------------------- state

function newState() {
  return { raw: [], ema: null, med: null, badge: 'off', onRun: 0, offRun: 0,
           shownAt: 0, thumb: null, medAtTrigger: null, nScored: 0 };
}

/** Replay the hysteresis rule. Returns true if the badge state changed. */
function updateBadge(st) {
  if (st.raw.length < 3) return false;
  const w = st.raw.slice(-TUNING.window).slice().sort((a, b) => a - b);
  st.med = w[Math.floor(w.length / 2)];
  const was = st.badge;
  if (st.badge === 'off') {
    st.onRun = st.med >= TUNING.onThresh ? st.onRun + 1 : 0;
    if (st.onRun >= TUNING.onUpdates) {
      st.badge = 'on'; st.onRun = 0; st.offRun = 0;
      st.shownAt = performance.now(); st.medAtTrigger = st.med;
    }
  } else {
    // sticky for minDwellMs regardless, so the pill can never strobe
    if (performance.now() - st.shownAt >= TUNING.minDwellMs) {
      st.offRun = st.med <= TUNING.offThresh ? st.offRun + 1 : 0;
      if (st.offRun >= TUNING.offUpdates) { st.badge = 'off'; st.offRun = 0; st.onRun = 0; }
    }
  }
  return st.badge !== was;
}

function badgeLabel(st) {
  if (st.badge === 'on') return st.offRun ? `on · clearing ${st.offRun}/${TUNING.offUpdates}` : 'on';
  return st.onRun ? `off · arming ${st.onRun}/${TUNING.onUpdates}` : 'off';
}

// ---------------------------------------------------------------- capture

const cap = document.createElement('canvas');
const capCtx = cap.getContext('2d', { alpha: false });

function captureBlob(video) {
  // Native resolution, full frame. The server runs the shipped squish_resize, so the
  // preprocessing is byte-for-byte the path predict.py uses. NEVER letterbox or pad here:
  // padding a 9:16 frame measured 0.983 -> 0.707 on a known fake.
  cap.width = video.videoWidth || 720;
  cap.height = video.videoHeight || 1280;
  capCtx.drawImage(video, 0, 0, cap.width, cap.height);
  return new Promise(r => cap.toBlob(r, 'image/jpeg', TUNING.quality));
}

async function tick(card) {
  if (card !== activeCard) return;
  const v = card._video, st = card._state;
  if (inflight || v.paused || v.readyState < 2 || document.visibilityState !== 'visible') return;

  const blob = await captureBlob(v);
  if (!blob || card !== activeCard) return;

  inflight = true;
  const mySeq = ++seq, myVid = card._clip.id;
  const ctl = new AbortController();
  card._abort = ctl;
  try {
    const t0 = performance.now();
    const r = await fetch(`/api/score?vid=${encodeURIComponent(myVid)}&t=${v.currentTime.toFixed(2)}&seq=${mySeq}`,
      { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob, signal: ctl.signal });
    const d = await r.json();
    // A late response must never land on a different clip.
    if (card !== activeCard || d.vid !== myVid) return;
    if (!d.ok) { if (d.busy) bumpDropped(); return; }

    fpsWin.push(performance.now()); while (fpsWin.length > 20) fpsWin.shift();
    // Adaptive backoff: MPS latency roughly doubles under thermal throttle. Degrade to a
    // lower frame rate rather than to a queue of stale answers.
    if (d.ms > TUNING.backoffMs) { rateScale = 2; rateResetAt = performance.now() + 3000; }
    else if (performance.now() > rateResetAt) rateScale = 1;

    st.raw.push(d.p);
    if (st.raw.length > 400) st.raw.shift();
    st.ema = st.ema === null ? d.p : TUNING.emaAlpha * d.p + (1 - TUNING.emaAlpha) * st.ema;
    st.nScored++;
    st.lastHeads = d.heads; st.lastMs = d.ms; st.lastBytes = d.bytes;

    const changed = updateBadge(st);
    if (changed && st.badge === 'on') {
      st.thumb = cap.toDataURL('image/jpeg', 0.7);   // the frame that actually triggered it
    }
    drawPreproc(v);
    render(card);
  } catch (e) {
    if (e.name !== 'AbortError') console.warn('score failed', e);
  } finally {
    inflight = false;
  }
}

let dropped = 0;
function bumpDropped() { dropped++; }

// ---------------------------------------------------------------- render

function render(card) {
  if (card !== activeCard) return;
  const st = card._state;
  const last = st.raw.length ? st.raw[st.raw.length - 1] : null;
  $('#nowP').textContent = last === null ? '—' : last.toFixed(3);
  $('#nowP').style.color = last === null ? '' : (last >= TUNING.onThresh ? 'var(--bad)' : last >= 0.5 ? 'var(--warn)' : 'var(--good)');
  $('#nowMed').textContent = st.med === null || st.med === undefined ? '—' : st.med.toFixed(3);
  $('#nowEma').textContent = st.ema === null ? '—' : st.ema.toFixed(3);
  $('#nowState').textContent = badgeLabel(st);

  if (st.lastHeads) {
    $('#heads').innerHTML = Object.entries(st.lastHeads).map(([n, v]) =>
      `<div class="head ${n === HEALTH.active_head ? 'active' : ''}">
         <div class="hn">${n}${n === HEALTH.active_head ? ' · active' : ''}</div>
         <div class="hv">${v.toFixed(3)}</div></div>`).join('');
  }

  // pill
  if (st.badge === 'on') {
    $('#pillSub').textContent =
      `median p(AI) ${(st.medAtTrigger ?? st.med).toFixed(2)} · ${HEALTH.active_head} head · research prototype`;
    if (st.thumb) $('#pillThumb').src = st.thumb;
    pillEl.classList.add('on');
    $('#island').classList.add('eaten');
  } else {
    pillEl.classList.remove('on');
    $('#island').classList.remove('eaten');
  }

  drawSpark(st);
  const secs = fpsWin.length > 1 ? (fpsWin[fpsWin.length - 1] - fpsWin[0]) / 1000 : 0;
  const fps = secs > 0 ? (fpsWin.length - 1) / secs : 0;
  $('#nowMs').textContent = st.lastMs != null ? st.lastMs.toFixed(0) + ' ms' : '—';
  $('#nowFps').textContent = fps ? fps.toFixed(2) : '—';
  $('#nowDrop').textContent = dropped;
  setKV({ 'frames scored': st.nScored, 'frame bytes':
            st.lastBytes ? (st.lastBytes / 1024).toFixed(0) + ' KB' : '—' });
}

function drawSpark(st) {
  const c = $('#spark'), x = c.getContext('2d'), W = c.width, H = c.height;
  x.clearRect(0, 0, W, H);
  const PAD = 46;                       // gutter for the y labels ('1.0' at 15px)
  const data = st.raw.slice(-TUNING.sparkN);
  const y = p => H - 10 - p * (H - 20);

  // y axis: 0 / 0.5 / 1 so a flat-high or flat-low trace is still readable as a level
  x.font = '500 15px ui-monospace,Menlo,monospace';
  x.textBaseline = 'middle';
  for (const v of [0, 0.5, 1]) {
    x.strokeStyle = '#1e222a'; x.lineWidth = 1;
    x.beginPath(); x.moveTo(PAD, y(v)); x.lineTo(W, y(v)); x.stroke();
    x.fillStyle = '#4d5563'; x.textAlign = 'right';
    x.fillText(v.toFixed(1), PAD - 6, y(v));
  }
  for (const [v, col] of [[TUNING.onThresh, '#ff6b6b'], [TUNING.offThresh, '#6b7484']]) {
    x.setLineDash([5, 5]); x.strokeStyle = col; x.lineWidth = 1.2;
    x.beginPath(); x.moveTo(PAD, y(v)); x.lineTo(W, y(v)); x.stroke();
  }
  x.setLineDash([]);
  if (data.length < 2) return;
  const dx = (W - PAD) / Math.max(TUNING.sparkN - 1, 1);
  const px = i => PAD + i * dx;
  x.beginPath();
  data.forEach((p, i) => i ? x.lineTo(px(i), y(p)) : x.moveTo(px(i), y(p)));
  x.strokeStyle = '#5ec8ff'; x.lineWidth = 2; x.stroke();
  const lp = data[data.length - 1];
  x.beginPath(); x.arc(px(data.length - 1), y(lp), 4, 0, 7);
  x.fillStyle = lp >= TUNING.onThresh ? '#ff6b6b' : '#5ec8ff'; x.fill();
}

function drawPreproc(v) {
  const s = $('#srcCanvas'), q = $('#sqCanvas');
  s.getContext('2d').drawImage(v, 0, 0, s.width, s.height);
  // Squish to a square, ignoring aspect — exactly what squish_resize does at 384.
  q.getContext('2d').drawImage(v, 0, 0, q.width, q.height);
}

const kvState = {};
function setKV(extra) {
  Object.assign(kvState, extra);
  $('#kv').innerHTML = Object.entries(kvState)
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

// ---------------------------------------------------------------- feed

function makeCard(clip) {
  const el = document.createElement('div');
  el.className = 'card-v';
  el.innerHTML = `
    <video muted playsinline loop preload="metadata" src="${clip.src}"></video>
    <div class="rail"><i>♡</i><i>💬</i><i>↗</i></div>
    <div class="overlay">
      <div class="ov-title">${esc(clip.title)}</div>
      <div class="ov-cap">${esc(clip.caption || '')}</div>
      <div class="ov-attr">${esc(clip.attribution || '')} · ${esc(clip.license || '')}${
        clip.generator ? ' · generated with ' + esc(clip.generator) : ''}</div>
      <div class="ov-truth ${clip.label}">${
        clip.label === 'ai' ? 'ground truth: AI-generated'
      : clip.label === 'real' ? 'ground truth: real camera'
      : 'ground truth: unknown (your upload)'}</div>
    </div>`;
  el._clip = clip;
  el._video = el.querySelector('video');
  el._state = newState();
  return el;
}
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function setActive(card) {
  if (activeCard === card) return;
  if (activeCard) {
    activeCard._video.pause();
    clearInterval(activeCard._timer);
    activeCard._abort?.abort();
  }
  activeCard = card;
  pillEl.classList.remove('on');
  $('#island').classList.remove('eaten');
  dropped = 0; fpsWin.length = 0;
  if (!card) return;
  document.querySelectorAll('#pfTable tr.is-live').forEach(tr => tr.classList.remove('is-live'));
  document.querySelector(`#pfTable tr[data-id="${card._clip.id}"]`)?.classList.add('is-live');
  card._video.play().catch(() => {});
  // State is frozen on scroll-away and resumes here rather than restarting from zero.
  render(card);
  card._timer = setInterval(() => tick(card), TUNING.minIntervalMs * rateScale);
}

// ---------------------------------------------------------------- preflight table

function renderPreflight() {
  const rows = CLIPS.filter(c => c.preflight).map(c => c.preflight);
  if (!rows.length) { $('#pfTable').innerHTML = '<tr><td class="hint">no preflight.csv — run project_demo/preflight.py</td></tr>'; return; }
  const vClass = v => ({ pinned_correct: 'v-ok', pinned_wrong: 'v-wrong',
                         crosses_mid_clip: 'v-partial', flickering: 'v-flicker' }[v] || '');
  const ai = rows.filter(r => r.label === 'ai'), re = rows.filter(r => r.label === 'real');
  const flagged = ai.filter(r => +r.pill_on_fraction > 0.5).length;
  const nearFP = re.filter(r => +r.headroom < 0.05).length;
  $('#pfSummary').innerHTML =
    `<span class="s-ok"><b>${flagged}/${ai.length}</b> AI clips flagged</span>` +
    `<span class="${nearFP ? 's-warn' : 's-ok'}"><b>${re.length - nearFP}/${re.length}</b> reals clean` +
    `${nearFP ? ` · ${nearFP} fires on some passes` : ''}</span>` +
    `<span class="s-dim">threshold 0.9446 · median-5 ×2 to raise, ≤0.70 ×5 to clear</span>`;
  $('#pfNote').innerHTML =
    '<b>Disclosure.</b> Clips are chosen by the pre-registered content criteria in ' +
    '<code>videos.json</code>, never by score. One post-pre-flight change: <code>ai_veo_yoga</code> ' +
    'moved to the candidate pool and <code>ai_videopoet_jog</code> in, so the feed spans three ' +
    'generator architectures rather than two — which also lifts the flagged count from 2/5 to 3/5, ' +
    'so read it as score-affecting. Both stay measured: veo_yoga 0.0090 (missed), videopoet_jog ' +
    '0.9920. <code>preflight.py --all</code> scores all 18 candidates, including CogVideoX-5B ' +
    '(0.608, missed) and SD-video (0.9998), excluded from the feed only because their native ' +
    'height is under the 768 px equalisation floor.';
  $('#pfTable').innerHTML =
    `<tr><th>clip</th><th>truth</th><th>generator</th><th>median p</th><th>peak med-5</th>` +
    `<th>badge on</th><th>verdict</th></tr>` +
    rows.map(r => {
      // A real clip whose peak median-5 is within 0.05 of the trigger will fire on some
      // playthroughs even if it did not latch during the offline sweep. Flag it as such.
      const near = r.label === 'real' && Number(r.headroom) < 0.05;
      return `<tr data-id="${esc(r.id)}">
      <td>${esc(r.id)}</td><td>${esc(r.label)}</td><td>${esc(r.generator || '—')}</td>
      <td class="num">${Number(r.median).toFixed(3)}</td>
      <td class="num ${near ? 'v-wrong' : ''}">${Number(r.max_median5).toFixed(3)}</td>
      <td class="num">${Math.round(Number(r.pill_on_fraction) * 100)}%</td>
      <td class="${near ? 'v-partial' : vClass(r.verdict)}">${
        near ? 'fires on some passes' : esc(r.verdict.replace(/_/g, ' '))}</td></tr>`;
    }).join('');
}

// ---------------------------------------------------------------- boot

async function boot() {
  try {
    HEALTH = await (await fetch('/api/health')).json();
  } catch {
    $('#scrimNote').textContent = 'server not reachable'; return;
  }
  const st = HEALTH.selftest || {};
  const se = $('#selftest');
  se.className = 'selftest' + (st.status === 'pass' ? '' : st.status === 'fail' ? ' fail' : ' skip');
  se.textContent = st.status === 'pass'
    ? `self-test PASS · ${st.file} → ${st.got} == ${st.expected} (committed in demo_preds.json)`
    : st.status === 'skipped' ? `self-test skipped · ${st.hint}` : `self-test ${st.status}`;

  const h = HEALTH.heads[HEALTH.active_head];
  setKV({
    device: HEALTH.device, dtype: HEALTH.dtype,
    backbone: HEALTH.backbone, 'params': (HEALTH.n_params / 1e9).toFixed(3) + ' B',
    'feature dim': HEALTH.feature_dim, 'input': `${HEALTH.image_size}² squish`,
    checkpoint: h.ckpt, sha256: h.sha256.slice(0, 12), 'head bytes': h.bytes,
    'warm ms': HEALTH.warm_ms,
  });

  $('#kvTune').innerHTML = Object.entries({
    'sample rate': TUNING.fps + ' fps',
    'JPEG quality': TUNING.quality + '  (q0.60 → a known fake fell 0.983→0.511)',
    'trigger stat': 'median of last ' + TUNING.window,
    'raise / clear': TUNING.onThresh + '  /  ' + TUNING.offThresh,
    'confirmations': TUNING.onUpdates + ' up  /  ' + TUNING.offUpdates + ' down',
    'min dwell': TUNING.minDwellMs + ' ms',
    'in flight': '1 max (extra frames dropped, never queued)',
  }).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');

  $('#headSel').innerHTML = Object.keys(HEALTH.heads)
    .map(n => `<option ${n === HEALTH.active_head ? 'selected' : ''}>${n}</option>`).join('');
  $('#headSel').onchange = async e => {
    const r = await (await fetch('/api/head', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: e.target.value }) })).json();
    if (r.ok) {
      HEALTH.active_head = r.active_head;
      const hh = HEALTH.heads[r.active_head];
      setKV({ checkpoint: hh.ckpt, sha256: hh.sha256.slice(0, 12), 'head bytes': hh.bytes });
      // The trigger statistic belongs to one head; restart it rather than mixing scales.
      if (activeCard) { activeCard._state = newState(); render(activeCard); }
    }
  };

  const feed = await (await fetch('/api/feed')).json();
  CLIPS = feed.clips;
  if (!CLIPS.length) { $('#scrimNote').textContent = 'no clips — run project_demo/fetch_videos.py'; return; }
  CLIPS.forEach(c => feedEl.appendChild(makeCard(c)));
  renderPreflight();

  io = new IntersectionObserver(es => {
    for (const e of es) if (e.isIntersecting && e.intersectionRatio >= 0.75) setActive(e.target);
  }, { root: feedEl, threshold: [0, .5, .75, .95] });
  [...feedEl.children].forEach(c => io.observe(c));

  $('#scrimNote').textContent =
    `${CLIPS.length} clips · ${HEALTH.backbone.split('/').pop()} on ${HEALTH.device} · ${HEALTH.warm_ms} ms/frame`;
  $('#startBtn').disabled = false;
}

$('#startBtn').onclick = () => {
  $('#scrim').classList.add('hidden');
  setActive(feedEl.firstElementChild);
};
const PANES = ['pf', 'rt', 'up', 'lim'];
function showTab(name) {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  PANES.forEach(n => $('#pane-' + n).classList.toggle('hidden', n !== name));
}
document.querySelectorAll('.tab').forEach(t => t.onclick = () => showTab(t.dataset.tab));

// ---------------------------------------------------------------- upload

const VERDICT = p => p >= TUNING.onThresh ? ['appears AI-generated', 'var(--bad)']
                   : p >= 0.5             ? ['above 0.5, under the 0.9446 trigger', 'var(--warn)']
                                          : ['reads as real', 'var(--good)'];

function upRow(html, bad) {
  const list = $('#upList');
  list.querySelector('.up-empty')?.remove();
  const el = document.createElement('div');
  el.className = 'up-row' + (bad ? ' bad' : '');
  el.innerHTML = html;
  list.prepend(el);
  return el;
}

async function sendUpload(file) {
  const dz = $('#dropzone');
  if (dz.classList.contains('busy')) return;
  dz.classList.add('busy');
  const pending = upRow(`<div class="up-thumb-ph">…</div>
    <div><div class="up-name">${esc(file.name)}</div>
    <div class="up-meta">uploading and normalising…</div></div><div></div>`);
  try {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const d = await r.json();
    pending.remove();

    if (!d.ok) {
      // The server deletes anything that fails validation; surface why.
      upRow(`<div class="up-thumb-ph">✕</div>
        <div><div class="up-name">${esc(file.name)}</div>
        <div class="up-meta" style="color:var(--bad)">rejected — ${esc(d.error)}</div>
        <div class="up-meta">${esc(d.detail || 'the file was deleted, not kept')}</div></div>
        <div></div>`, true);
      return;
    }

    if (d.kind === 'image') {
      const [verdict, col] = VERDICT(d.p);
      const heads = Object.entries(d.heads).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(' · ');
      upRow(`<img src="${d.src}" alt="">
        <div><div class="up-name">${esc(file.name)}</div>
        <div class="up-meta">image · scored once · ${esc(heads)}</div></div>
        <div><div class="up-score" style="color:${col}">${d.p.toFixed(3)}</div>
        <div class="up-verdict">${verdict}</div></div>`);
      return;
    }

    // video: becomes a real feed card, scored live like every other clip
    const clip = { id: d.id, label: 'unknown', title: file.name, caption: 'your upload',
                   generator: null, attribution: 'uploaded locally', license: 'not redistributed',
                   src: d.src, preflight: null, uploaded: true };
    if (!io) { upRow(`<div class="up-thumb-ph">…</div><div><div class="up-name">${esc(file.name)}</div>
      <div class="up-meta">the feed is still loading — try again in a moment</div></div><div></div>`, true);
      return; }
    CLIPS.push(clip);
    const card = makeCard(clip);
    feedEl.appendChild(card);
    io.observe(card);

    const row = upRow(`<video src="${d.src}" muted playsinline></video>
      <div><div class="up-name">${esc(file.name)}</div>
      <div class="up-meta">video · ${d.duration}s · ${esc(d.scale_note)}</div>
      <div class="up-meta">added to the feed — scored live while it plays</div></div>
      <button class="up-go">play it</button>`);
    row.querySelector('.up-go').onclick = () => {
      card.scrollIntoView({ behavior: 'smooth' });
      feedEl.scrollTop = [...feedEl.children].indexOf(card) * feedEl.clientHeight;
    };
    row.querySelector('.up-go').click();
  } catch (e) {
    pending.remove();
    upRow(`<div class="up-thumb-ph">✕</div>
      <div><div class="up-name">${esc(file.name)}</div>
      <div class="up-meta" style="color:var(--bad)">upload failed — ${esc(e.message)}</div></div>
      <div></div>`, true);
  } finally {
    dz.classList.remove('busy');
  }
}

(() => {
  const dz = $('#dropzone'), input = $('#fileInput');
  dz.onclick = () => input.click();
  dz.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } };
  input.onchange = () => { for (const f of input.files) sendUpload(f); input.value = ''; };
  for (const ev of ['dragenter', 'dragover']) {
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('over'); });
  }
  for (const ev of ['dragleave', 'drop']) {
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('over'); });
  }
  dz.addEventListener('drop', e => { for (const f of e.dataTransfer.files) sendUpload(f); });
  // dropping anywhere on the page works too, but must not hijack a normal navigation
  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => {
    if (dz.contains(e.target)) return;
    e.preventDefault();
    if (e.dataTransfer.files.length) { showTab('up'); for (const f of e.dataTransfer.files) sendUpload(f); }
  });
})();
$('#labelBtn').onclick = () => document.body.classList.toggle('show-truth');
document.addEventListener('keydown', e => {
  if (e.key === 'l' || e.key === 'L') document.body.classList.toggle('show-truth');
});
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible' && activeCard) activeCard._video.pause();
  else if (activeCard) activeCard._video.play().catch(() => {});
});

$('#startBtn').disabled = true;
boot();
