const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

let dashboard = null;
let lastPrediction = null;

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (value, decimals = 2) => value === null || value === undefined || value === '' || Number.isNaN(Number(value)) ? '—' : Number(value).toFixed(decimals);
const fmtDate = (value) => value ? new Date(value).toLocaleString() : '—';
const titleCase = (value) => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, (m) => m.toUpperCase());
const toast = (message) => { const node = $('#toast'); node.textContent = message; node.classList.add('show'); clearTimeout(window.__toast); window.__toast = setTimeout(() => node.classList.remove('show'), 2800); };

async function api(url, options = {}) {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || 'Request failed');
  return body;
}

function bandClass(label) {
  if ((label || '').startsWith('High')) return 'high';
  if ((label || '').startsWith('Moderate')) return 'moderate';
  if ((label || '').startsWith('Needs')) return 'support';
  if ((label || '').startsWith('Low')) return 'low';
  return 'neutral';
}
function bandBadge(label) { return `<span class="badge ${bandClass(label)}">${esc(label || '—')}</span>`; }

function showView(name) {
  $$('.view').forEach((view) => view.classList.remove('active'));
  const target = $(`#${name}View`);
  if (!target) return;
  target.classList.add('active');
  $$('.nav').forEach((nav) => nav.classList.toggle('active', nav.dataset.view === name));
  const titles = {
    overview: ['Overview', 'Current application activity, training-data evidence, and model evaluation.'],
    predict: ['Predict final marks', 'Estimate final marks from one input profile and store the real prediction request.'],
    scenario: ['Scenario lab', 'Run what-if comparisons without writing to prediction history.'],
    students: ['Saved students', 'Persist named student prediction snapshots created by actual application actions.'],
    analytics: ['Academic analytics', 'Explore observed dataset relationships, validation rows, and real runtime activity.'],
    history: ['Prediction history', 'Every successful prediction request saved by the application.'],
    data: ['Training data', 'The exact CSV records used by the current training pipeline.'],
    model: ['Model lab', 'Candidate model comparison, explainability, versioning, and holdout evidence.']
  };
  $('#pageTitle').textContent = titles[name][0];
  $('#pageSubtitle').textContent = titles[name][1];
  $('#crumb').textContent = titles[name][0];
  if (name === 'overview') loadDashboard();
  if (name === 'predict') loadDashboard();
  if (name === 'students') loadStudents();
  if (name === 'analytics') loadAnalytics();
  if (name === 'history') loadHistory();
  if (name === 'data') loadData();
  if (name === 'model') loadDashboard();
}

$$('[data-view]').forEach((element) => element.addEventListener('click', () => showView(element.dataset.view)));

async function loadHealth() {
  try {
    const result = await api('/health');
    $('#statusText').textContent = 'System ready';
    $('#statusDot').className = 'status-dot ok';
    $('#healthText').textContent = `${result.model} · ${result.dataset_rows} training records · ${result.model_version || 'version unavailable'}`;
    $('#footerModel').textContent = `Model: ${result.model}${result.model_version ? ` · ${result.model_version}` : ''}`;
  } catch {
    $('#statusText').textContent = 'System unavailable';
    $('#statusDot').className = 'status-dot bad';
    $('#healthText').textContent = 'Health check failed';
  }
}

function renderOverview(d) {
  const cards = [
    ['Prediction requests', d.total_predictions, 'Actual successful requests stored in SQLite'],
    ['Saved students', d.saved_students, 'Named records currently stored'],
    ['Training records', d.dataset.rows, 'Rows currently present in CSV'],
    ['Selected model', d.model.name || '—', `${d.model.cv_folds}-fold validation`]
  ];
  $('#overviewStats').innerHTML = cards.map(([label, value, meta]) => `<div class="stat-card"><div class="stat-label">${esc(label)}</div><div class="stat-value stat-value-sm">${esc(value)}</div><div class="stat-meta">${esc(meta)}</div></div>`).join('');

  const m = d.model;
  const s = m.selected_metrics || {};
  $('#modelOverviewCard').innerHTML = `<div class="model-overview">
    <div class="model-overview-title"><div><div class="dataset-tag">${esc(m.version || 'version unavailable')}</div><div class="selected-name">${esc(m.name || '—')}</div><div class="small-note">${esc(m.family || 'Estimator')} · trained ${esc(fmtDate(m.trained_at))}</div></div><button class="link-button" data-view="model" onclick="showView('model')">Details</button></div>
    <div class="metric-row four"><div class="metric-box"><b>${fmt(s.mae, 3)}</b><span>Holdout MAE</span></div><div class="metric-box"><b>${fmt(s.rmse, 3)}</b><span>Holdout RMSE</span></div><div class="metric-box"><b>${fmt(s.r2, 3)}</b><span>Holdout R²</span></div><div class="metric-box"><b>${fmt(s.cv_mae, 3)}</b><span>CV MAE</span></div></div>
    <div class="model-note">Selection: ${esc(m.selection_method || '—')} Holdout: ${m.holdout_size} rows. Training data hash: <code>${esc((m.training_data_hash || '').slice(0, 16))}</code>…</div>
  </div>`;

  $('#dataQuality').innerHTML = [
    ['Rows', d.dataset.rows],
    ['Fields', d.dataset.columns.length],
    ['Missing values', d.dataset.missing_values],
    ['Duplicate rows', d.dataset.duplicates]
  ].map(([label, value]) => `<div class="quality-item"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join('');
  $('#dataQualityNote').textContent = `Observed final-mark average: ${fmt(d.dataset.mean_final_marks)}%. Training data and runtime history are separate sources.`;

  $('#activitySummary').innerHTML = [
    ['Predictions', d.total_predictions],
    ['Average predicted mark', d.avg_prediction === null ? '—' : `${fmt(d.avg_prediction)}%`],
    ['Predicted range', d.min_prediction === null ? '—' : `${fmt(d.min_prediction)}–${fmt(d.max_prediction)}%`]
  ].map(([label, value]) => `<div class="activity-item"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join('');

  renderCountBars('#bandChart', d.performance_bands || {}, 'No prediction history yet.');
  renderCountBars('#supportChart', d.support_levels || {}, 'No support-level history yet.');

  const rows = d.recent_predictions || [];
  $('#recentPredictions').innerHTML = rows.length ? `<div class="table-wrap"><table><thead><tr><th>#</th><th>Time</th><th>Student ID</th><th>Predicted</th><th>Band</th><th>Support</th><th>Model</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${row.id}</td><td>${fmtDate(row.created_at)}</td><td>${esc(row.student_code || 'Ad hoc')}</td><td><b>${fmt(row.predicted_marks)}%</b></td><td>${bandBadge(row.performance_band)}</td><td>${esc(row.risk_level || '—')}</td><td>${esc(row.model_name)} ${row.model_version ? `<span class="tiny-muted">${esc(row.model_version)}</span>` : ''}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty-inline">No prediction requests have been recorded yet. The section will populate only after a real prediction is submitted.</div>';

  $('#overviewModelContext').innerHTML = `<div class="context-grid">
    <div class="context-box"><b>${esc(m.name || '—')}</b><span>Selected model</span></div>
    <div class="context-box"><b>${m.dataset_rows}</b><span>Training rows</span></div>
    <div class="context-box"><b>${m.holdout_size}</b><span>Holdout rows</span></div>
    <div class="context-box"><b>${m.cv_folds}-fold</b><span>Cross-validation</span></div>
  </div>
  <div class="model-note">R² is not prediction accuracy. The current 20-record dataset supports an academic demonstration and model comparison, not a claim of institutional or real-world accuracy.</div>`;

  updateTrainingRanges(d.dataset.feature_ranges || {});
}

function renderCountBars(selector, counts, emptyText) {
  const entries = Object.entries(counts);
  const total = entries.reduce((sum, [, value]) => sum + Number(value), 0);
  if (!total) { $(selector).innerHTML = `<div class="empty-inline">${esc(emptyText)}</div>`; return; }
  $(selector).innerHTML = entries.map(([label, count]) => {
    const pct = Math.round((Number(count) / total) * 100);
    return `<div class="band-row"><div class="band-label">${esc(label)}</div><div class="band-bar"><div class="band-fill" style="width:${pct}%"></div></div><div class="band-count">${count}</div></div>`;
  }).join('');
}

function updateTrainingRanges(summary) {
  const map = [['study_hours', '#rangeStudy'], ['attendance', '#rangeAttendance'], ['previous_marks', '#rangePrevious']];
  map.forEach(([key, selector]) => {
    const item = summary[key];
    if (item) $(selector).textContent = `${fmt(item.min, 2)}–${fmt(item.max, 2)}`;
  });
}

function renderPrediction(p) {
  const warning = p.training_range_notes?.length ? `<div class="scope-warning"><b>Training-range warning</b><ul class="caution-list">${p.training_range_notes.map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>` : '';
  const interval = (p.prediction_low !== null && p.prediction_high !== null) ? `${fmt(p.prediction_low)}–${fmt(p.prediction_high)}%` : 'Not available';
  const recommendations = (p.recommendations || []).map((x) => `<li>${esc(x)}</li>`).join('');
  const drivers = p.explainability?.features || [];
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(Number(d.contribution ?? d.importance ?? 0))), 1);
  const driverRows = drivers.map((d) => {
    const value = d.contribution ?? d.importance ?? 0;
    const width = Math.max(5, Math.round(Math.abs(Number(value)) / maxAbs * 100));
    const label = d.contribution !== undefined ? `${d.direction === 'negative' ? '−' : '+'}${fmt(Math.abs(value), 2)}` : fmt(Number(value) * 100, 1) + '%';
    return `<div class="driver-row"><div class="driver-head"><span>${esc(titleCase(d.feature))}</span><b>${esc(label)}</b></div><div class="driver-bar"><span class="driver-fill ${d.direction === 'negative' ? 'negative' : ''}" style="width:${width}%"></span></div><div class="driver-meta">Input: ${fmt(d.input_value)}${d.contribution !== undefined ? ` · coefficient: ${fmt(d.coefficient, 4)}` : ''}</div></div>`;
  }).join('');

  $('#predictionResult').innerHTML = `<div class="result-main">
    <div class="result-header"><div><div class="result-label">${p.prediction_id ? `Prediction #${p.prediction_id}` : 'Simulation'}</div><div class="result-mark">${fmt(p.predicted_marks)}<span>%</span></div><div class="result-label">Estimated final marks</div></div><div class="result-badges">${bandBadge(p.performance_band)}<span class="badge neutral">${esc(p.risk_level || '—')}</span></div></div>
    <div class="result-meta three"><div class="meta-box"><b>${esc(p.model_name)}</b><span>Model</span></div><div class="meta-box"><b>${esc(p.model_version || '—')}</b><span>Model version</span></div><div class="meta-box"><b>${esc(p.raw_predicted_marks)}</b><span>Raw model output</span></div></div>
    <div class="result-meta three"><div class="meta-box"><b>${fmt(p.model_mae, 3)}</b><span>Holdout MAE</span></div><div class="meta-box"><b>${fmt(p.model_rmse, 3)}</b><span>Holdout RMSE</span></div><div class="meta-box"><b>${esc(interval)}</b><span>RMSE reference range</span></div></div>
    ${warning}
    <div class="result-block"><h3>Feature explanation</h3><div class="small-note">${esc(p.explainability?.note || 'No explanation available.')}</div><div class="driver-list">${driverRows || '<div class="empty-inline">No feature explanation is available for this model.</div>'}</div></div>
    <div class="result-block"><h3>Support guidance</h3><div class="support-score"><div><b>${p.risk_score ?? '—'}/100</b><span>Data-derived support index</span></div><div class="small-note">This is a heuristic comparison with training-data averages. It is not a probability.</div></div><ul>${recommendations || '<li>No additional guidance was generated.</li>'}</ul></div>
    <div class="result-block"><h3>Input context</h3><ul>${(p.guidance || []).map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>
    ${p.prediction_id ? `<div class="result-block"><h3>Student record</h3><div class="small-note">This prediction is already recorded in history. Save it as a named student snapshot without creating another prediction.</div><button class="button secondary" id="openSaveModal">Save as student</button></div>` : '<div class="result-block"><h3>Simulation only</h3><div class="small-note">This result was not written to prediction history.</div></div>'}
  </div>`;
  $('#openSaveModal')?.addEventListener('click', openSaveModal);
}

$('#predictionForm').onsubmit = async (event) => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.target).entries());
  const payload = Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Number(value)]));
  try {
    const result = await api('/api/predict', { method: 'POST', body: JSON.stringify(payload) });
    lastPrediction = result.prediction;
    renderPrediction(lastPrediction);
    await loadDashboard();
    toast(`Prediction #${lastPrediction.prediction_id} recorded`);
  } catch (error) { toast(error.message); }
};

function formPayload(form) {
  const values = Object.fromEntries(new FormData(form).entries());
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Number(value)]));
}

$('#copyBaselineToScenario').onclick = () => {
  const baseline = formPayload($('#baselineForm'));
  Object.entries(baseline).forEach(([key, value]) => { $('#scenarioForm').elements[key].value = value; });
  toast('Baseline copied to alternative scenario');
};

$('#compareScenario').onclick = async () => {
  if (!$('#baselineForm').reportValidity() || !$('#scenarioForm').reportValidity()) return;
  const button = $('#compareScenario');
  button.disabled = true;
  button.textContent = 'Comparing…';
  try {
    const [baseline, scenario] = await Promise.all([
      api('/api/simulate', { method: 'POST', body: JSON.stringify(formPayload($('#baselineForm'))) }),
      api('/api/simulate', { method: 'POST', body: JSON.stringify(formPayload($('#scenarioForm'))) })
    ]);
    const a = baseline.simulation;
    const b = scenario.simulation;
    const delta = Number(b.predicted_marks) - Number(a.predicted_marks);
    const direction = delta > 0 ? 'increase' : delta < 0 ? 'decrease' : 'no change';
    $('#scenarioResult').className = 'scenario-result';
    $('#scenarioResult').innerHTML = `<div class="scenario-grid"><div class="scenario-card"><span>Baseline</span><b>${fmt(a.predicted_marks)}%</b>${bandBadge(a.performance_band)}</div><div class="scenario-arrow">→</div><div class="scenario-card"><span>Alternative</span><b>${fmt(b.predicted_marks)}%</b>${bandBadge(b.performance_band)}</div></div><div class="scenario-delta"><b>${delta >= 0 ? '+' : ''}${fmt(delta)} marks</b><span>Estimated ${direction} from baseline to alternative. This is a model simulation, not an observed outcome.</span></div>`;
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.textContent = 'Compare scenarios'; }
};

function closeSaveModal() { $('#studentSaveModal').classList.add('hidden'); $('#studentSaveModal').setAttribute('aria-hidden', 'true'); }
function openSaveModal() {
  if (!lastPrediction) return;
  $('#studentSaveModal').classList.remove('hidden');
  $('#studentSaveModal').setAttribute('aria-hidden', 'false');
  $('#studentSaveModal input[name="name"]').focus();
}
$('#closeModal').onclick = closeSaveModal;
$('#cancelModal').onclick = closeSaveModal;
$('#studentSaveModal').addEventListener('click', (event) => { if (event.target.id === 'studentSaveModal') closeSaveModal(); });

$('#studentSaveForm').onsubmit = async (event) => {
  event.preventDefault();
  if (!lastPrediction) return;
  const identity = Object.fromEntries(new FormData(event.target).entries());
  try {
    await api('/api/students', { method: 'POST', body: JSON.stringify({ name: identity.name, student_code: identity.student_code, prediction_id: lastPrediction.prediction_id }) });
    closeSaveModal();
    event.target.reset();
    await loadDashboard();
    toast('Student record saved without creating another prediction');
  } catch (error) { toast(error.message); }
};

$('#studentCreateForm').onsubmit = async (event) => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.target).entries());
  const payload = { name: values.name, student_code: values.student_code, study_hours: Number(values.study_hours), attendance: Number(values.attendance), previous_marks: Number(values.previous_marks) };
  const button = event.target.querySelector('button[type="submit"]');
  button.disabled = true; button.textContent = 'Calculating…';
  try {
    const response = await api('/api/students', { method: 'POST', body: JSON.stringify(payload) });
    const student = response.student;
    $('#studentCreateResult').innerHTML = `<div class="record-confirm"><div class="confirm-icon">✓</div><div><h3>${esc(student.name)} saved</h3><p>${student.student_code ? `Student ID: ${esc(student.student_code)} · ` : ''}Prediction: <b>${fmt(student.predicted_marks)}%</b></p><div>${bandBadge(student.performance_band)} <span class="badge neutral">${esc(student.risk_level)}</span></div></div></div>`;
    event.target.reset();
    await Promise.all([loadStudents(), loadDashboard(), loadHistory()]);
    toast('Student and one prediction saved');
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.textContent = 'Predict & save student'; }
};

async function loadStudents() {
  try {
    const q = $('#studentSearch').value.trim();
    const response = await api(`/api/students${q ? `?q=${encodeURIComponent(q)}` : ''}`);
    const rows = response.students || [];
    $('#studentCount').textContent = `${rows.length} record${rows.length === 1 ? '' : 's'}`;
    $('#studentsTable').innerHTML = rows.length ? `<div class="table-wrap"><table><thead><tr><th>Name</th><th>Student ID</th><th>Study</th><th>Attendance</th><th>Previous</th><th>Predicted</th><th>Band</th><th>Model version</th><th>Added</th><th></th></tr></thead><tbody>${rows.map((row) => `<tr><td><b>${esc(row.name)}</b></td><td>${esc(row.student_code || '—')}</td><td>${fmt(row.study_hours)}h</td><td>${fmt(row.attendance)}%</td><td>${fmt(row.previous_marks)}%</td><td><b>${fmt(row.predicted_marks)}%</b></td><td>${bandBadge(row.performance_band)}</td><td>${esc(row.model_version || '—')}</td><td>${fmtDate(row.created_at)}</td><td><button class="button secondary" onclick="removeStudent(${row.id})">Delete</button></td></tr>`).join('')}</tbody></table></div>` : '<div class="empty-inline">No saved student records match the current search.</div>';
  } catch (error) { toast(error.message); }
}
window.removeStudent = async (id) => {
  if (!confirm('Delete this saved student record? This does not delete the prediction history record.')) return;
  try { await api(`/api/students/${id}`, { method: 'DELETE' }); await Promise.all([loadStudents(), loadDashboard()]); toast('Student record deleted'); }
  catch (error) { toast(error.message); }
};
$('#studentSearch').addEventListener('input', loadStudents);

async function loadHistory() {
  try {
    const q = $('#historySearch').value.trim();
    const response = await api(`/api/history${q ? `?q=${encodeURIComponent(q)}` : ''}`);
    const rows = response.predictions || [];
    $('#historyCount').textContent = `${rows.length} displayed`;
    $('#historyTable').innerHTML = rows.length ? `<div class="table-wrap"><table><thead><tr><th>#</th><th>Timestamp</th><th>Student ID</th><th>Study</th><th>Attendance</th><th>Previous</th><th>Predicted</th><th>Band</th><th>Support</th><th>Model</th><th>Version</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${row.id}</td><td>${fmtDate(row.created_at)}</td><td>${esc(row.student_code || 'Ad hoc')}</td><td>${fmt(row.study_hours)}h</td><td>${fmt(row.attendance)}%</td><td>${fmt(row.previous_marks)}%</td><td><b>${fmt(row.predicted_marks)}%</b></td><td>${bandBadge(row.performance_band)}</td><td>${esc(row.risk_level || '—')}</td><td>${esc(row.model_name || '—')}</td><td>${esc(row.model_version || '—')}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty-inline">No prediction history matches the current search.</div>';
  } catch (error) { toast(error.message); }
}
$('#historySearch').addEventListener('input', loadHistory);

function renderCorrelationChart(correlations) {
  const entries = Object.entries(correlations || {});
  if (!entries.length) { $('#correlationChart').innerHTML = '<div class="empty-inline">No correlation data available.</div>'; return; }
  const max = Math.max(...entries.map(([, v]) => Math.abs(Number(v))), 0.001);
  $('#correlationChart').innerHTML = entries.map(([feature, value]) => {
    const pct = Math.round(Math.abs(Number(value)) / max * 100);
    const sign = Number(value) >= 0 ? 'positive' : 'negative';
    return `<div class="corr-row"><div class="corr-label">${esc(titleCase(feature))}</div><div class="corr-track"><span class="corr-fill ${sign}" style="width:${pct}%"></span></div><div class="corr-value">${fmt(value, 3)}</div></div>`;
  }).join('') + '<div class="chart-footnote">Correlation describes association in the supplied dataset; it does not prove causation.</div>';
}

function renderDistribution(values) {
  const data = (values || []).map(Number).filter(Number.isFinite);
  if (!data.length) { $('#targetDistribution').innerHTML = '<div class="empty-inline">No values available.</div>'; return; }
  const min = Math.min(...data), max = Math.max(...data);
  const bins = 6;
  const width = (max - min || 1) / bins;
  const counts = Array.from({ length: bins }, () => 0);
  data.forEach((value) => { const idx = Math.min(bins - 1, Math.floor((value - min) / width)); counts[idx] += 1; });
  const maxCount = Math.max(...counts, 1);
  $('#targetDistribution').innerHTML = `<div class="histogram">${counts.map((count, i) => `<div class="hist-col"><div class="hist-count">${count}</div><div class="hist-bar"><span style="height:${Math.round(count/maxCount*100)}%"></span></div><small>${fmt(min + i*width, 0)}–${fmt(i === bins-1 ? max : min + (i+1)*width, 0)}</small></div>`).join('')}</div>`;
}

function renderHoldout(rows) {
  if (!rows.length) { $('#holdoutChart').innerHTML = '<div class="empty-inline">No holdout detail is available.</div>'; return; }
  const max = Math.max(...rows.flatMap((r) => [Number(r.actual_final_marks), Number(r.predicted_final_marks)]), 100);
  $('#holdoutChart').innerHTML = `<div class="holdout-list">${rows.map((r, i) => {
    const actual = Number(r.actual_final_marks), pred = Number(r.predicted_final_marks);
    const a = Math.max(4, Math.round(actual / max * 100));
    const p = Math.max(4, Math.round(pred / max * 100));
    return `<div class="holdout-row"><div class="holdout-label">Row ${r.source_row}</div><div class="holdout-bars"><div class="mini-bar"><span class="actual" style="width:${a}%"></span></div><div class="mini-bar"><span class="predicted" style="width:${p}%"></span></div></div><div class="holdout-values"><b>${fmt(actual)}%</b><span>actual</span><b>${fmt(pred)}%</b><span>predicted</span><em>${Number(r.residual) >= 0 ? '+' : ''}${fmt(r.residual, 2)}</em></div></div>`;
  }).join('')}</div><div class="legend"><span><i class="legend-dot actual"></i>Actual</span><span><i class="legend-dot predicted"></i>Predicted</span></div>`;
}

async function loadAnalytics() {
  try {
    const response = await api('/api/analytics');
    renderCorrelationChart(response.dataset.correlations_with_target);
    renderDistribution(response.dataset.distribution.final_marks);
    const s = response.dataset.target_summary;
    $('#targetStats').innerHTML = [['Mean', `${fmt(s.mean)}%`], ['Median', `${fmt(s.median)}%`], ['Std. dev.', fmt(s.std)], ['IQR', `${fmt(s.q1)}–${fmt(s.q3)}`]].map(([label, value]) => `<div class="mini-stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join('');
    renderHoldout(response.evaluation.holdout_rows || []);
    const runtime = response.runtime;
    $('#runtimeAnalytics').innerHTML = runtime.total_predictions ? `<div class="runtime-grid"><div class="runtime-stat"><b>${runtime.total_predictions}</b><span>Prediction requests</span></div><div class="runtime-stat"><b>${runtime.saved_students}</b><span>Saved students</span></div><div class="runtime-stat"><b>${fmt(dashboard?.avg_prediction)}%</b><span>Average predicted mark</span></div></div><div class="small-note">Runtime analytics contain only actual saved prediction history.</div>` : '<div class="empty-inline">No runtime prediction records exist yet. This is intentionally empty until a real prediction is made.</div>';
    const missing = response.advanced_fields_not_present || [];
    $('#dataReadiness').innerHTML = `<div class="readiness-grid"><div class="readiness-good"><b>Available now</b><p>${response.dataset.columns.map(esc).join(' · ')}</p></div><div class="readiness-muted"><b>Not present in current CSV</b><p>${missing.length ? missing.map(esc).join(' · ') : 'None'}</p></div></div><div class="small-note">No subject-level, semester-level, assignment, or internal-assessment results are synthesized. Add a validated dataset with these fields before enabling such analytics.</div>`;
  } catch (error) { toast(error.message); }
}

async function loadData() {
  try {
    const response = await api('/api/data');
    const s = response.summary;
    $('#dataSummary').innerHTML = [
      ['Rows', response.rows.length, 'Records in CSV'],
      ['Columns', response.columns.length, 'Fields used by pipeline'],
      ['Missing values', response.missing_values, 'Validation result'],
      ['Training hash', (response.training_data_hash || '').slice(0, 12), 'SHA-256 prefix']
    ].map(([label, value, meta]) => `<div class="stat-card"><div class="stat-label">${esc(label)}</div><div class="stat-value stat-value-sm code-value">${esc(value)}</div><div class="stat-meta">${esc(meta)}</div></div>`).join('');

    $('#featureSummary').innerHTML = `<div class="table-wrap"><table class="feature-table"><thead><tr><th>Field</th><th>Minimum</th><th>Maximum</th><th>Mean</th><th>Median</th><th>Std. dev.</th></tr></thead><tbody>${Object.entries(s).map(([name, values]) => `<tr><td>${esc(titleCase(name))}</td><td>${fmt(values.min)}</td><td>${fmt(values.max)}</td><td>${fmt(values.mean)}</td><td>${fmt(values.median)}</td><td>${fmt(values.std)}</td></tr>`).join('')}</tbody></table></div>`;
    $('#datasetNotes').innerHTML = `<ul class="notes-list"><li>Rows shown are exactly the records currently present in <code>data/student_data.csv</code>.</li><li>Training data and prediction history are separate sources.</li><li>Current file size: ${response.rows.length} records. No external or fabricated student records are added.</li><li>Duplicate rows detected: ${response.duplicate_rows}. Duplicates removed during training: ${response.duplicates_removed}.</li><li>Training data SHA-256 prefix: <code>${esc((response.training_data_hash || '').slice(0, 16))}</code></li></ul>`;
    $('#datasetTable').innerHTML = `<div class="table-wrap"><table><thead><tr>${response.columns.map((col) => `<th>${esc(col)}</th>`).join('')}</tr></thead><tbody>${response.rows.map((row) => `<tr>${response.columns.map((col) => `<td>${fmt(row[col])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  } catch (error) { toast(error.message); }
}

function renderModel(d) {
  const rows = d.model.all_models || [];
  const best = d.model.name;
  $('#modelTable').innerHTML = `<div class="table-wrap"><table><thead><tr><th>Model</th><th>Holdout MAE</th><th>Holdout RMSE</th><th>Holdout R²</th><th>CV MAE</th><th>CV R²</th></tr></thead><tbody>${rows.map((row) => `<tr class="${row.name === best ? 'selected-row' : ''}"><td><b>${esc(row.name)}</b>${row.name === best ? ' <span class="badge neutral">selected</span>' : ''}</td><td>${fmt(row.mae, 3)}</td><td>${fmt(row.rmse, 3)}</td><td>${fmt(row.r2, 3)}</td><td>${fmt(row.cv_mae, 3)}</td><td>${fmt(row.cv_r2, 3)}</td></tr>`).join('')}</tbody></table></div>`;
  const s = d.model.selected_metrics || {};
  $('#selectedModel').innerHTML = `<div class="selected-model"><div><div class="dataset-tag">Model version ${esc(d.model.version || '—')}</div><div class="selected-name">${esc(d.model.name || '—')}</div><div class="small-note">${esc(d.model.family || 'Estimator')} · training hash <code>${esc((d.model.training_data_hash || '').slice(0, 16))}</code>…</div></div><div class="metric-row four"><div class="metric-box"><b>${fmt(s.mae, 3)}</b><span>Holdout MAE</span></div><div class="metric-box"><b>${fmt(s.rmse, 3)}</b><span>Holdout RMSE</span></div><div class="metric-box"><b>${fmt(s.r2, 3)}</b><span>Holdout R²</span></div><div class="metric-box"><b>${fmt(s.cv_mae, 3)}</b><span>CV MAE</span></div></div><div class="model-note">Trained: ${esc(fmtDate(d.model.trained_at))}<br>Holdout rows: ${d.model.holdout_size} · CV folds: ${d.model.cv_folds}<br>${esc(d.model.selection_method || '')}</div></div>`;

  const exp = d.model.explainability || {};
  const features = exp.features || [];
  $('#modelDrivers').innerHTML = features.length ? `<div class="small-note">${esc(exp.note || '')}</div><div class="driver-list">${features.map((item) => {
    const value = item.contribution ?? item.importance ?? 0;
    return `<div class="driver-row"><div class="driver-head"><span>${esc(titleCase(item.feature))}</span><b>${item.contribution !== undefined ? fmt(item.coefficient, 4) : fmt(item.importance, 4)}</b></div><div class="driver-meta">Reference input at training mean: ${fmt(item.input_value)} · ${item.contribution !== undefined ? `coefficient contribution at mean: ${fmt(item.contribution, 3)}` : 'relative feature importance'}</div></div>`;
  }).join('')}</div>` : '<div class="empty-inline">No feature-explanation metadata available.</div>';

  const holdout = d.model.holdout_rows || [];
  $('#holdoutTable').innerHTML = holdout.length ? `<div class="table-wrap"><table><thead><tr><th>CSV row</th><th>Actual</th><th>Predicted</th><th>Residual</th><th>Attendance</th><th>Study hours</th></tr></thead><tbody>${holdout.map((row) => `<tr><td>${row.source_row}</td><td>${fmt(row.actual_final_marks)}%</td><td>${fmt(row.predicted_final_marks)}%</td><td>${Number(row.residual) >= 0 ? '+' : ''}${fmt(row.residual, 3)}</td><td>${fmt(row.attendance)}%</td><td>${fmt(row.study_hours)}h</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty-inline">No holdout details available.</div>';
  $('#modelLimitations').innerHTML = `<ul class="caution-list">${(d.model.limitations || []).map((item) => `<li>${esc(item)}</li>`).join('')}</ul>`;
}

async function loadDashboard() {
  try {
    dashboard = await api('/api/dashboard');
    renderOverview(dashboard);
    renderModel(dashboard);
    if ($('.view.active')?.id === 'analyticsView') loadAnalytics();
  } catch (error) { toast(error.message); }
}

$('#retrainBtn').onclick = async () => {
  const button = $('#retrainBtn'); button.disabled = true; button.textContent = 'Retraining…';
  try {
    const result = await api('/api/retrain', { method: 'POST' });
    await Promise.all([loadHealth(), loadDashboard(), loadAnalytics()]);
    toast(`Model retrained: ${result.model.model_name}`);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.textContent = 'Retrain from CSV'; }
};

$('#refreshBtn').onclick = async () => {
  await loadHealth();
  await loadDashboard();
  const active = $('.view.active')?.id?.replace('View', '');
  if (active === 'history') await loadHistory();
  if (active === 'students') await loadStudents();
  if (active === 'data') await loadData();
  if (active === 'analytics') await loadAnalytics();
  toast('Data refreshed');
};

loadHealth();
loadDashboard();
