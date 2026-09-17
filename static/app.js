const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

let dashboard = null;
let lastPrediction = null;

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (value, decimals = 2) => value === null || value === undefined || value === '' ? '—' : Number(value).toFixed(decimals);
const fmtDate = (value) => value ? new Date(value).toLocaleString() : '—';
const toast = (message) => { const node = $('#toast'); node.textContent = message; node.classList.add('show'); clearTimeout(window.__toast); window.__toast = setTimeout(() => node.classList.remove('show'), 2600); };

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

function bandBadge(label) {
  return `<span class="badge ${bandClass(label)}">${esc(label || '—')}</span>`;
}

function showView(name) {
  $$('.view').forEach((view) => view.classList.remove('active'));
  const target = $(`#${name}View`);
  if (!target) return;
  target.classList.add('active');
  $$('.nav').forEach((nav) => nav.classList.toggle('active', nav.dataset.view === name));

  const titles = {
    overview: ['Overview', 'Current application activity, supplied training data, and model evaluation.'],
    predict: ['Predict final marks', 'Use the trained model to estimate final marks for one input profile.'],
    students: ['Saved students', 'Student records contain the prediction that was calculated at the time of saving.'],
    history: ['Prediction history', 'A record of every successful prediction request made in this application.'],
    data: ['Training data', 'The CSV records used to train and evaluate the current model.'],
    model: ['Model evaluation', 'Candidate model comparison, validation metrics, and limitations.']
  };
  $('#pageTitle').textContent = titles[name][0];
  $('#pageSubtitle').textContent = titles[name][1];
  $('#crumb').textContent = titles[name][0];

  if (name === 'overview') loadDashboard();
  if (name === 'students') loadStudents();
  if (name === 'history') loadHistory();
  if (name === 'data') loadData();
  if (name === 'model') loadDashboard();
}

async function loadHealth() {
  try {
    const result = await api('/health');
    $('#statusText').textContent = 'System ready';
    $('#statusDot').className = 'status-dot ok';
    $('#healthText').textContent = `${result.model} · ${result.dataset_rows} training records`;
    $('#footerModel').textContent = `Model: ${result.model}`;
  } catch {
    $('#statusText').textContent = 'System unavailable';
    $('#statusDot').className = 'status-dot bad';
    $('#healthText').textContent = 'Health check failed';
  }
}

function renderStats(d) {
  const cards = [
    ['Total predictions', d.total_predictions, 'Count of successful prediction requests'],
    ['Saved students', d.saved_students, 'Explicitly saved student records'],
    ['Training records', d.dataset.rows, 'Rows currently used from the CSV'],
    ['Last prediction', d.last_prediction ? fmtDate(d.last_prediction.created_at) : '—', d.last_prediction ? `Prediction #${d.total_predictions}` : 'No prediction has been made']
  ];
  $('#overviewStats').innerHTML = cards.map(([label, value, meta]) => `<div class="stat-card"><div class="stat-label">${esc(label)}</div><div class="stat-value">${esc(value)}</div><div class="stat-meta">${esc(meta)}</div></div>`).join('');

  $('#activitySummary').innerHTML = [
    ['Predictions', d.total_predictions],
    ['Average predicted mark', d.avg_prediction === null ? '—' : `${fmt(d.avg_prediction)}%`],
    ['Predicted range', d.min_prediction === null ? '—' : `${fmt(d.min_prediction)}–${fmt(d.max_prediction)}%`]
  ].map(([label, value]) => `<div class="activity-item"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join('');

  const bands = d.performance_bands || {};
  const total = Object.values(bands).reduce((sum, value) => sum + Number(value), 0);
  $('#bandChart').innerHTML = Object.entries(bands).map(([label, count]) => {
    const percent = total ? Math.round((Number(count) / total) * 100) : 0;
    return `<div class="band-row"><div class="band-label">${esc(label)}</div><div class="band-bar"><div class="band-fill" style="width:${percent}%"></div></div><div class="band-count">${count}</div></div>`;
  }).join('');
  if (!total) $('#bandChart').innerHTML += '<div class="small-note">No prediction history yet. Run a prediction to create activity.</div>';

  $('#dataQuality').innerHTML = [
    ['Rows', d.dataset.rows],
    ['Fields', d.dataset.columns.length],
    ['Missing values', d.dataset.missing_values],
    ['Duplicate rows', d.dataset.duplicates]
  ].map(([label, value]) => `<div class="quality-item"><b>${esc(value)}</b><span>${esc(label)}</span></div>`).join('');
  $('#dataQualityNote').textContent = `Target mean final mark in training data: ${fmt(d.dataset.mean_final_marks)}%. Training data and prediction history are separate sources.`;
}

function renderRecent(d) {
  const rows = d.recent_predictions || [];
  if (!rows.length) {
    $('#recentPredictions').innerHTML = '<div class="small-note">No prediction requests have been recorded yet.</div>';
    return;
  }
  $('#recentPredictions').innerHTML = `<div class="table-wrap"><table><thead><tr><th>#</th><th>Time</th><th>Student ID</th><th>Predicted</th><th>Band</th><th>Model</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${row.id}</td><td>${fmtDate(row.created_at)}</td><td>${esc(row.student_code || 'Ad hoc prediction')}</td><td><b>${fmt(row.predicted_marks)}%</b></td><td>${bandBadge(row.performance_band)}</td><td>${esc(row.model_name)}</td></tr>`).join('')}</tbody></table></div>`;
}

function renderOverviewModelContext(d) {
  const m = d.model;
  const s = m.selected_metrics || {};
  $('#overviewModelContext').innerHTML = `<div class="context-grid">
    <div class="context-box"><b>${esc(m.name || '—')}</b><span>Selected model</span></div>
    <div class="context-box"><b>${fmt(s.mae, 3)}</b><span>Holdout MAE · ${m.holdout_size} rows</span></div>
    <div class="context-box"><b>${fmt(s.r2, 3)}</b><span>Holdout R² · ${m.holdout_size} rows</span></div>
    <div class="context-box"><b>${m.cv_folds}-fold</b><span>Cross-validation</span></div>
  </div>
  <div class="model-note">Evaluation metrics come from the ${d.dataset.rows}-row supplied dataset. They should not be presented as evidence of real-world accuracy on a larger student population.</div>`;
}

function renderModel(d) {
  const rows = d.model.all_models || [];
  $('#modelTable').innerHTML = `<div class="table-wrap"><table><thead><tr><th>Model</th><th>Holdout MAE</th><th>Holdout RMSE</th><th>Holdout R²</th><th>${d.model.cv_folds}-fold CV MAE</th><th>CV R²</th></tr></thead><tbody>${rows.map((row) => `<tr><td><b>${esc(row.name)}</b></td><td>${fmt(row.mae, 3)}</td><td>${fmt(row.rmse, 3)}</td><td>${fmt(row.r2, 3)}</td><td>${fmt(row.cv_mae, 3)}</td><td>${fmt(row.cv_r2, 3)}</td></tr>`).join('')}</tbody></table></div>`;

  const s = d.model.selected_metrics || {};
  $('#selectedModel').innerHTML = `<div class="selected-model">
    <div><div class="dataset-tag">Selected using cross-validation MAE</div><div class="selected-name">${esc(d.model.name || '—')}</div></div>
    <div class="metric-row"><div class="metric-box"><b>${fmt(s.mae, 3)}</b><span>Holdout MAE</span></div><div class="metric-box"><b>${fmt(s.rmse, 3)}</b><span>Holdout RMSE</span></div><div class="metric-box"><b>${fmt(s.r2, 3)}</b><span>Holdout R²</span></div></div>
    <div class="model-note">Trained: ${fmtDate(d.model.trained_at)}<br>Holdout rows: ${d.model.holdout_size} · CV folds: ${d.model.cv_folds}<br>${esc(d.model.selection_method || '')}</div>
  </div>`;

  $('#modelLimitations').innerHTML = `<ul class="caution-list">${(d.model.limitations || []).map((item) => `<li>${esc(item)}</li>`).join('')}</ul>`;
}

async function loadDashboard() {
  try {
    dashboard = await api('/api/dashboard');
    renderStats(dashboard);
    renderRecent(dashboard);
    renderOverviewModelContext(dashboard);
    renderModel(dashboard);
    updateTrainingRanges(dashboard.dataset.feature_ranges || {});
  } catch (error) {
    toast(error.message);
  }
}

function updateTrainingRanges(summary) {
  const map = [['study_hours', '#rangeStudy'], ['attendance', '#rangeAttendance'], ['previous_marks', '#rangePrevious']];
  map.forEach(([key, selector]) => {
    const item = summary[key];
    if (item) $(selector).textContent = `${fmt(item.min, 2)}–${fmt(item.max, 2)}`;
  });
}

function renderPrediction(p) {
  const warning = p.training_range_notes.length ? `<div class="scope-warning"><b>Training-range warning</b><ul class="caution-list">${p.training_range_notes.map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>` : '';
  const interval = (p.prediction_low !== null && p.prediction_high !== null) ? `${fmt(p.prediction_low)}–${fmt(p.prediction_high)}%` : 'Not available';
  const recommendations = (p.recommendations || []).map((x) => `<li>${esc(x)}</li>`).join('');
  $('#predictionResult').innerHTML = `<div class="result-main">
    <div class="result-header"><div><div class="result-label">Prediction #${p.prediction_id}</div><div class="result-mark">${fmt(p.predicted_marks)}<span>%</span></div><div class="result-label">Estimated final marks</div></div><div class="result-badges">${bandBadge(p.performance_band)}<span class="badge neutral">${esc(p.risk_level || '—')}</span></div></div>
    <div class="result-meta"><div class="meta-box"><b>${esc(p.model_name)}</b><span>Model used</span></div><div class="meta-box"><b>${fmt(p.model_mae, 3)}</b><span>Holdout MAE</span></div><div class="meta-box"><b>${fmt(p.model_rmse, 3)}</b><span>Holdout RMSE</span></div></div>
    <div class="result-meta"><div class="meta-box"><b>${esc(interval)}</b><span>RMSE-based error band</span></div><div class="meta-box"><b>${p.risk_score ?? '—'}/100</b><span>Heuristic support indicator</span></div><div class="meta-box"><b>${esc(p.model_name)}</b><span>Prediction source</span></div></div>
    ${warning}
    <div class="result-block"><h3>Recommendations</h3><ul>${recommendations || '<li>No additional recommendation generated.</li>'}</ul></div>
    <div class="result-block"><h3>Input guidance</h3><ul>${p.guidance.map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>
    <div class="result-block"><h3>Student record</h3><div class="small-note">This prediction is already recorded in prediction history. Save it as a student profile to link it to a name or student ID.</div><button class="button secondary" id="openSaveModal">Save as student</button></div>
  </div>`;
  $('#openSaveModal').onclick = openSaveModal;
}

$('#predictionForm').onsubmit = async (event) => {
  event.preventDefault();
  const formData = Object.fromEntries(new FormData(event.target).entries());
  const payload = Object.fromEntries(Object.entries(formData).map(([key, value]) => [key, Number(value)]));
  try {
    const result = await api('/api/predict', { method: 'POST', body: JSON.stringify(payload) });
    lastPrediction = result.prediction;
    renderPrediction(lastPrediction);
    await loadDashboard();
    toast(`Prediction #${lastPrediction.prediction_id} recorded`);
  } catch (error) {
    toast(error.message);
  }
};

function closeSaveModal() {
  $('#studentSaveModal').classList.add('hidden');
  $('#studentSaveModal').setAttribute('aria-hidden', 'true');
}
function openSaveModal() {
  if (!lastPrediction) return;
  $('#studentSaveModal').classList.remove('hidden');
  $('#studentSaveModal').setAttribute('aria-hidden', 'false');
  $('#studentSaveForm').querySelector('input').focus();
}
$('#closeModal').onclick = closeSaveModal;
$('#cancelModal').onclick = closeSaveModal;
$('#studentSaveModal').addEventListener('click', (event) => { if (event.target.id === 'studentSaveModal') closeSaveModal(); });

$('#studentSaveForm').onsubmit = async (event) => {
  event.preventDefault();
  if (!lastPrediction) return;
  const identity = Object.fromEntries(new FormData(event.target).entries());
  try {
    await api('/api/students', {
      method: 'POST',
      body: JSON.stringify({ name: identity.name, student_code: identity.student_code, prediction_id: lastPrediction.prediction_id })
    });
    closeSaveModal();
    event.target.reset();
    await loadDashboard();
    toast('Student record saved');
  } catch (error) {
    toast(error.message);
  }
};

$('#studentCreateForm').onsubmit = async (event) => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.target).entries());
  const payload = {
    name: values.name,
    student_code: values.student_code,
    study_hours: Number(values.study_hours),
    attendance: Number(values.attendance),
    previous_marks: Number(values.previous_marks)
  };
  const button = event.target.querySelector('button[type="submit"]');
  button.disabled = true;
  button.textContent = 'Calculating…';
  try {
    const response = await api('/api/students', { method: 'POST', body: JSON.stringify(payload) });
    const student = response.student;
    $('#studentCreateResult').innerHTML = `<div class="record-confirm"><div class="confirm-icon">✓</div><div><h3>${esc(student.name)} saved</h3><p>${student.student_code ? `Student ID: ${esc(student.student_code)} · ` : ''}Prediction: <b>${fmt(student.predicted_marks)}%</b></p><div>${bandBadge(student.performance_band)} <span class="badge neutral">${esc(student.risk_level)}</span></div></div></div>`;
    event.target.reset();
    await loadStudents();
    await loadDashboard();
    await loadHistory();
    toast('Student and prediction saved');
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Predict & save student';
  }
};

async function loadStudents() {
  try {
    const q = $('#studentSearch').value.trim();
    const response = await api(`/api/students${q ? `?q=${encodeURIComponent(q)}` : ''}`);
    const rows = response.students || [];
    $('#studentCount').textContent = `${rows.length} record${rows.length === 1 ? '' : 's'}`;
    $('#studentsTable').innerHTML = rows.length ? `<div class="table-wrap"><table><thead><tr><th>Name</th><th>Student ID</th><th>Study hours</th><th>Attendance</th><th>Previous marks</th><th>Predicted</th><th>Band</th><th>Added</th><th></th></tr></thead><tbody>${rows.map((row) => `<tr><td><b>${esc(row.name)}</b></td><td>${esc(row.student_code || '—')}</td><td>${fmt(row.study_hours)}h</td><td>${fmt(row.attendance)}%</td><td>${fmt(row.previous_marks)}%</td><td><b>${fmt(row.predicted_marks)}%</b></td><td>${bandBadge(row.performance_band)}</td><td>${fmtDate(row.created_at)}</td><td><button class="button secondary" onclick="removeStudent(${row.id})">Delete</button></td></tr>`).join('')}</tbody></table></div>` : '<div class="small-note">No saved student records match the current search.</div>';
  } catch (error) {
    toast(error.message);
  }
}
window.removeStudent = async (id) => {
  if (!confirm('Delete this saved student record?')) return;
  try {
    await api(`/api/students/${id}`, { method: 'DELETE' });
    await loadStudents();
    await loadDashboard();
    toast('Student record deleted');
  } catch (error) {
    toast(error.message);
  }
};
$('#studentSearch').addEventListener('input', loadStudents);

async function loadHistory() {
  try {
    const q = $('#historySearch').value.trim();
    const response = await api(`/api/history${q ? `?q=${encodeURIComponent(q)}` : ''}`);
    const rows = response.predictions || [];
    $('#historyCount').textContent = `${rows.length} displayed`;
    $('#historyTable').innerHTML = rows.length ? `<div class="table-wrap"><table><thead><tr><th>#</th><th>Timestamp</th><th>Student ID</th><th>Study</th><th>Attendance</th><th>Previous</th><th>Predicted</th><th>Band</th><th>Model</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${row.id}</td><td>${fmtDate(row.created_at)}</td><td>${esc(row.student_code || 'Ad hoc')}</td><td>${fmt(row.study_hours)}h</td><td>${fmt(row.attendance)}%</td><td>${fmt(row.previous_marks)}%</td><td><b>${fmt(row.predicted_marks)}%</b></td><td>${bandBadge(row.performance_band)}</td><td>${esc(row.model_name)}</td></tr>`).join('')}</tbody></table></div>` : '<div class="small-note">No prediction history matches the current search.</div>';
  } catch (error) {
    toast(error.message);
  }
}
$('#historySearch').addEventListener('input', loadHistory);

async function loadData() {
  try {
    const response = await api('/api/data');
    const s = response.summary;
    const cards = [
      ['Rows', response.rows.length, 'Records in CSV'],
      ['Columns', response.columns.length, 'Input + target fields'],
      ['Missing values', response.missing_values, 'Dataset validation'],
      ['Final-mark average', `${fmt(s.final_marks.mean)}%`, 'Across training records']
    ];
    $('#dataSummary').innerHTML = cards.map(([label, value, meta]) => `<div class="stat-card"><div class="stat-label">${label}</div><div class="stat-value">${esc(value)}</div><div class="stat-meta">${esc(meta)}</div></div>`).join('');

    $('#featureSummary').innerHTML = `<div class="table-wrap"><table class="feature-table"><thead><tr><th>Field</th><th>Minimum</th><th>Maximum</th><th>Average</th></tr></thead><tbody>${Object.entries(s).map(([name, values]) => `<tr><td>${esc(name.replaceAll('_', ' '))}</td><td>${fmt(values.min)}</td><td>${fmt(values.max)}</td><td>${fmt(values.mean)}</td></tr>`).join('')}</tbody></table></div>`;
    $('#datasetNotes').innerHTML = `<ul class="notes-list"><li>Rows shown are the records currently present in <code>data/student_data.csv</code>.</li><li>No prediction-history values are mixed into model training statistics.</li><li>The current supplied file contains ${response.rows.length} records. More diverse data is needed before claiming general real-world accuracy.</li><li>Duplicate rows currently present: ${response.duplicate_rows}; rows removed during training: ${response.duplicates_removed}.</li></ul>`;
    $('#datasetTable').innerHTML = `<div class="table-wrap"><table><thead><tr>${response.columns.map((col) => `<th>${esc(col)}</th>`).join('')}</tr></thead><tbody>${response.rows.map((row) => `<tr>${response.columns.map((col) => `<td>${fmt(row[col])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  } catch (error) {
    toast(error.message);
  }
}

$('#retrainBtn').onclick = async () => {
  const button = $('#retrainBtn');
  button.disabled = true;
  button.textContent = 'Retraining…';
  try {
    await api('/api/retrain', { method: 'POST' });
    await loadDashboard();
    await loadHealth();
    toast('Model retrained from the current CSV dataset');
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Retrain from CSV';
  }
};

$('#refreshBtn').onclick = async () => {
  await loadHealth();
  await loadDashboard();
  const active = $('.view.active')?.id?.replace('View', '');
  if (active === 'history') await loadHistory();
  if (active === 'students') await loadStudents();
  if (active === 'data') await loadData();
  toast('Data refreshed');
};

$$('[data-view]').forEach((element) => element.addEventListener('click', () => showView(element.dataset.view)));

loadHealth();
loadDashboard();
