import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const FEATURES = ['Time', ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), 'Amount'];
const NAV = [['Overview', '⌂'], ['Try a transaction', '◎'], ['Batch file', '⇧'], ['Model results', '▥'], ['How it works', 'ⓘ']];

async function get(path) {
  const response = await fetch(API + path);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

function parseCsv(text) {
  const rows = []; let row = []; let field = ''; let quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"' && !field) quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (quoted) throw new Error('CSV contains an unclosed quoted field.');
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function saveCsv(name, body) {
  const url = URL.createObjectURL(new Blob([body], { type: 'text/csv' }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  URL.revokeObjectURL(url);
}

function App() {
  const [page, setPage] = useState('Overview');
  const [status, setStatus] = useState(null); const [info, setInfo] = useState(null);
  const [metrics, setMetrics] = useState(null); const [plots, setPlots] = useState([]);
  const [samples, setSamples] = useState(null); const [apiError, setApiError] = useState('');
  const [form, setForm] = useState(Object.fromEntries(FEATURES.map(name => [name, ''])));
  const [sampleName, setSampleName] = useState(''); const [prediction, setPrediction] = useState(null);
  const [predictionError, setPredictionError] = useState(''); const [busy, setBusy] = useState(false);
  const [batchBusy, setBatchBusy] = useState(false); const [batchError, setBatchError] = useState('');
  const [rowErrors, setRowErrors] = useState([]); const [batchRows, setBatchRows] = useState([]);

  async function refresh() {
    setApiError('');
    try {
      const readiness = await get('/setup-status'); setStatus(readiness);
      if (!readiness.model_available) { setInfo(null); setSamples(null); setMetrics(null); setPlots([]); return; }
      const responses = await Promise.allSettled([get('/model-info'), get('/metrics'), get('/visualizations'), get('/demo-samples')]);
      if (responses[0].status === 'fulfilled') setInfo(responses[0].value);
      else setApiError(responses[0].reason.message);
      if (responses[1].status === 'fulfilled') setMetrics(responses[1].value);
      if (responses[2].status === 'fulfilled') setPlots(responses[2].value.items || []);
      if (responses[3].status === 'fulfilled') setSamples(responses[3].value);
    } catch (error) { setStatus(null); setApiError(error.message); }
  }
  useEffect(() => { refresh(); }, []);

  const comparison = useMemo(() => info?.validation_comparison || [], [info]);
  const distribution = useMemo(() => info ? [
    { name: 'Legitimate', value: info.class_counts?.['0'] || 0, color: '#538ff0' },
    { name: 'Fraud', value: info.class_counts?.['1'] || 0, color: '#f37480' }
  ] : [], [info]);
  const modelReady = Boolean(status?.model_available && info);

  function loadSample(kind) {
    if (!samples?.[kind]) return;
    setForm(Object.fromEntries(FEATURES.map(name => [name, String(samples[kind][name])] )));
    setSampleName(kind === 'fraud' ? 'Fraud example' : 'Legitimate example');
    setPrediction(null); setPredictionError('');
  }

  async function submitPrediction(event) {
    event.preventDefault(); setBusy(true); setPrediction(null); setPredictionError('');
    try {
      const features = {};
      for (const name of FEATURES) {
        if (form[name] === '') throw new Error(`Enter ${name}, or load one of the real dataset examples.`);
        const value = Number(form[name]);
        if (!Number.isFinite(value)) throw new Error(`${name} must be a finite number.`);
        if (['Time', 'Amount'].includes(name) && value < 0) throw new Error(`${name} cannot be negative.`);
        features[name] = value;
      }
      const response = await fetch(API + '/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features }) });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || 'Prediction request failed.');
      setPrediction(body);
    } catch (error) { setPredictionError(error.message); }
    finally { setBusy(false); }
  }

  async function submitBatch(event) {
    const file = event.target.files?.[0]; if (!file) return;
    setBatchError(''); setRowErrors([]); setBatchRows([]); setBatchBusy(true);
    try {
      if (!file.name.toLowerCase().endsWith('.csv')) throw new Error('Choose a CSV file.');
      if (file.size > 5 * 1024 * 1024) throw new Error('The file is over 5 MB. Split it into smaller CSV files.');
      const parsed = parseCsv(await file.text()).filter((row, index, all) => !(index === all.length - 1 && row.length === 1 && row[0] === ''));
      if (parsed.length < 2) throw new Error('The CSV needs a header row and at least one transaction.');
      const headers = parsed[0].map(value => value.trim());
      const missing = FEATURES.filter(name => !headers.includes(name));
      const extra = headers.filter(name => !FEATURES.includes(name));
      const duplicate = [...new Set(headers.filter((name, index) => headers.indexOf(name) !== index))];
      if (missing.length || extra.length || duplicate.length) {
        const issues = [];
        if (missing.length) issues.push('missing: ' + missing.join(', '));
        if (extra.length) issues.push('unexpected: ' + extra.join(', '));
        if (duplicate.length) issues.push('duplicated: ' + duplicate.join(', '));
        throw new Error('Headers do not match (' + issues.join('; ') + '). Download the template and use its column names.');
      }
      const data = parsed.slice(1);
      if (data.length > 500) throw new Error('The API accepts at most 500 transaction rows per upload.');
      const transactions = []; const problems = [];
      data.forEach((cells, index) => {
        const line = index + 2;
        if (cells.length !== headers.length) { problems.push(`Row ${line}: expected ${headers.length} values; found ${cells.length}.`); return; }
        const item = {};
        headers.forEach((name, column) => {
          const value = cells[column].trim() === '' ? NaN : Number(cells[column]);
          if (!Number.isFinite(value)) problems.push(`Row ${line}, ${name}: enter a finite numeric value.`);
          else if (['Time', 'Amount'].includes(name) && value < 0) problems.push(`Row ${line}, ${name}: value cannot be negative.`);
          item[name] = value;
        });
        transactions.push(item);
      });
      if (problems.length) { setRowErrors(problems); throw new Error('No rows were sent. Fix the listed issues and upload the file again.'); }
      const response = await fetch(API + '/predict/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ transactions }) });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || 'Batch prediction failed.');
      setBatchRows(body.predictions || []);
    } catch (error) { setBatchError(error.message); }
    finally { setBatchBusy(false); event.target.value = ''; }
  }

  function downloadTemplate() { saveCsv('fraud_prediction_template.csv', FEATURES.join(',') + '\r\n'); }
  function downloadResults() {
    const lines = batchRows.map(row => [row.row, row.predicted_class, row.fraud_probability, row.threshold, '"' + row.label.replaceAll('"', '""') + '"'].join(','));
    saveCsv('fraud_predictions.csv', ['row,predicted_class,fraud_probability,threshold,label', ...lines].join('\r\n'));
  }
  const navigate = name => setPage(name);

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">S</span><span><b>SENTINEL</b><small>FRAUD DETECTION LAB</small></span></div>
      <p className="nav-label">PROJECT</p><nav>{NAV.map(([name, icon]) => <button key={name} className={'nav ' + (page === name ? 'active' : '')} aria-current={page === name ? 'page' : undefined} onClick={() => navigate(name)}><span aria-hidden="true">{icon}</span>{name}</button>)}</nav>
      <div className="sidebar-note"><i className={'status-dot ' + (status ? 'good' : 'bad')}></i><span>{status ? 'API connected' : 'API not connected'}</span><small>Historical-data academic prototype</small></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><p className="eyebrow">INTELLIGENT FINANCIAL FRAUD DETECTION</p><h1>{page}</h1></div><div className="top-actions"><span className={'status-pill ' + (modelReady ? 'ready' : '')}><i className="status-dot"></i>{modelReady ? 'MODEL READY' : 'SETUP NEEDED'}</span><button className="icon-button" onClick={refresh} title="Refresh connection and model status" aria-label="Refresh status">↻</button></div></header>
      {apiError && <div className="alert error" role="alert"><b>Can’t load project status.</b><span>{apiError}</span><code>py -m uvicorn backend.main:app --reload</code></div>}
      {page === 'Overview' && <Overview status={status} info={info} metrics={metrics} comparison={comparison} distribution={distribution} go={navigate} />}
      {page === 'Try a transaction' && <section className="page">
        <PageIntro eyebrow="GUIDED DEMO" title="Try a transaction">Start with a real dataset example. This model scores anonymized dataset features, not card numbers.</PageIntro>
        <div className="notice"><b>What are these values?</b><p><strong>Time</strong> is seconds from the dataset’s start. <strong>Amount</strong> is the transaction amount. <strong>V1–V28</strong> are anonymized PCA components; they have no readable card or merchant meaning.</p></div>
        {!modelReady && <SetupNotice status={status} />}
        <div className="panel sample-panel"><div className="sample-heading"><span className="step-number">01</span><div><h3>Choose a real example</h3><p>Examples below are actual labeled rows saved during training.</p></div></div>
          <div className="sample-buttons"><button className="sample-card" disabled={!samples} onClick={() => loadSample('legitimate')}><span className="sample-symbol legit">✓</span><span><b>Legitimate example</b><small>Load a real Class 0 row</small></span><span className="arrow">→</span></button>
            <button className="sample-card" disabled={!samples} onClick={() => loadSample('fraud')}><span className="sample-symbol fraud">!</span><span><b>Fraud example</b><small>Load a real Class 1 row</small></span><span className="arrow">→</span></button></div>
          {!samples && <p className="small-note">Real examples are available after a model has been trained from the Kaggle CSV.</p>}
          {sampleName && <p className="selected-sample">Selected: {sampleName}. Run it as-is, or edit fields below.</p>}
        </div>
        <details className="panel advanced"><summary><span><b>Advanced: edit feature values</b><small>Only use values in the dataset’s 30-feature format.</small></span><span className="summary-chevron">⌄</span></summary>
          <form onSubmit={submitPrediction}><div className="feature-explainer"><b>These are not ordinary card details.</b><p>The PCA features V1–V28 are anonymized. A card number, merchant, or location cannot be entered instead.</p></div>
            <div className="field-grid">{FEATURES.map(name => <label key={name}><span>{name}{name === 'Time' ? ' · seconds' : name === 'Amount' ? ' · amount' : ''}</span><input aria-label={name + (name === 'Time' ? ', seconds from dataset start' : name === 'Amount' ? ', transaction amount' : ', anonymized PCA feature')} value={form[name]} onChange={event => { setForm({ ...form, [name]: event.target.value }); setSampleName('Custom values'); setPrediction(null); }} type="number" step="any" min={['Time', 'Amount'].includes(name) ? '0' : undefined} placeholder="Required" /></label>)}</div>
            <div className="form-actions"><button className="button primary" disabled={!modelReady || busy}>{busy ? 'Scoring…' : 'Run prediction'}</button><span className="small-note">All 30 numeric values are required.</span></div>
            {predictionError && <div className="alert error" role="alert">{predictionError}</div>}
          </form>
        </details>
        {prediction && <PredictionResult result={prediction} />}
        <p className="disclaimer">This is a model risk estimate for a historical academic dataset, not a definitive fraud determination.</p>
      </section>}
      {page === 'Batch file' && <section className="page"><PageIntro eyebrow="BATCH SCORING" title="Score a CSV file">Upload up to 500 transactions using the same feature names as the training dataset.</PageIntro>
        <div className="notice"><b>Download the blank template first</b><p>It contains the exact 30 required columns: Time, V1–V28, and Amount. Fill one numeric transaction per row. Do not include Class.</p><button className="button secondary" onClick={downloadTemplate}>Download CSV template</button></div>
        {!modelReady && <SetupNotice status={status} />}
        <div className="panel upload-panel"><label className="upload-zone"><span className="upload-icon">↑</span><b>{batchBusy ? 'Checking file and scoring rows…' : 'Choose a CSV file'}</b><small>Maximum 5 MB · up to 500 rows</small><input type="file" accept=".csv,text/csv" onChange={submitBatch} disabled={!modelReady || batchBusy} /></label>
          {batchError && <div className="alert error" role="alert">{batchError}</div>}
          {rowErrors.length > 0 && <div className="row-errors"><b>Rows to fix</b><ul>{rowErrors.slice(0, 20).map((message, index) => <li key={index}>{message}</li>)}</ul>{rowErrors.length > 20 && <p>{rowErrors.length - 20} more issues not shown.</p>}</div>}
          {batchRows.length > 0 && <div className="results-table"><div className="table-heading"><div><h3>Prediction results</h3><p>{batchRows.length} rows scored by the model.</p></div><button className="button secondary" onClick={downloadResults}>Download results CSV</button></div><div className="table-scroll"><table><thead><tr><th>CSV row</th><th>Model result</th><th>Fraud probability</th><th>Threshold</th></tr></thead><tbody>{batchRows.map(row => <tr key={row.row}><td>{row.row}</td><td><span className={'label-chip ' + (row.predicted_class ? 'flagged' : 'clear')}>{row.label}</span></td><td>{(row.fraud_probability * 100).toFixed(2)}%</td><td>{row.threshold.toFixed(3)}</td></tr>)}</tbody></table></div></div>}
        </div>
      </section>}
      {page === 'Model results' && <ModelResults status={status} metrics={metrics} plots={plots} comparison={comparison} />}
      {page === 'How it works' && <HowItWorks go={navigate} />}
      <footer><span>Sentinel · academic portfolio prototype</span><span>Historical data only · not for banking decisions</span></footer>
    </main>
  </div>;
}

function PageIntro({ eyebrow, title, children }) { return <div className="page-intro"><p className="eyebrow cyan">{eyebrow}</p><h2>{title}</h2><p>{children}</p></div>; }
function SetupNotice({ status }) {
  const message = !status ? 'The website cannot reach the backend API.' : !status.dataset_available ? 'The real Kaggle CSV is missing from the project data folder.' : !status.model_available ? 'The dataset is present, but model training has not completed.' : 'The model is ready. Refresh project status.';
  const command = !status ? 'py -m uvicorn backend.main:app --reload' : status.dataset_available && !status.model_available ? 'py -m src.train' : '';
  const note = !status ? 'Run from F:\\Financial-Fraud-Detection-Project in a terminal.' : !status.dataset_available ? 'Download the Kaggle CSV and place it at F:\\Financial-Fraud-Detection-Project\\data\\creditcard.csv.' : 'Run from the project root after confirming data\\creditcard.csv is present.';
  return <div className="setup-notice panel"><span className="setup-icon">i</span><div><h3>Setup needed</h3><p>{message}</p>{command && <code>{command}</code>}<small>{note}</small></div></div>;
}

function Overview({ status, info, metrics, comparison, distribution, go }) {
  const ready = Boolean(info && status?.model_available);
  return <section className="page">
    <div className="hero"><div className="hero-orb orb-one"></div><div className="hero-orb orb-two"></div><div className="hero-copy"><span className="hero-kicker">A GUIDED PROJECT DEMO</span><h2>Learn how a model<br />spots transaction risk.</h2><p>Train on labeled historical transactions, then score examples in the same dataset format.</p><button className="button primary hero-button" onClick={() => go('Try a transaction')} disabled={!ready}>{ready ? 'Try a real example' : 'Complete setup to try'} <span>→</span></button></div><div className="hero-orbit"><i></i><b>S</b><span>•</span></div></div>
    <SectionHeading eyebrow="BEFORE YOUR FIRST DEMO" title="Getting started" caption={ready ? 'The model is ready to demonstrate' : 'Follow the steps to enable predictions'} />
    <div className="setup-grid"><StatusStep number="01" title="Dataset" ok={Boolean(status?.dataset_available)} detail={status?.dataset_available ? 'Kaggle CSV found in data folder' : 'Place creditcard.csv in the data folder'} command={status?.dataset_available ? '' : 'data\creditcard.csv'} /><StatusStep number="02" title="Trained model" ok={Boolean(status?.model_available)} detail={status?.model_available ? 'Saved model and threshold found' : 'Train using the real dataset'} command={status?.model_available ? '' : 'py -m src.train'} /><StatusStep number="03" title="API service" ok={Boolean(status)} detail={status ? 'Backend is responding' : 'Start the API in a terminal'} command={status ? '' : 'py -m uvicorn backend.main:app --reload'} /><StatusStep number="04" title="Website" ok detail="You are viewing the running website" /></div>
    {!ready && <div className="setup-instructions panel"><h3>What to do next</h3><p>Download the Kaggle dataset, extract <code>creditcard.csv</code> into the project’s <code>data</code> folder, then train from the project root. This model uses historical dataset rows; it does not connect to a bank.</p><div className="command-row"><code>py -m src.train</code><span>Run in F:\Financial-Fraud-Detection-Project</span></div></div>}
    <SectionHeading eyebrow="REAL PROJECT DATA" title="Dataset snapshot" caption="Counts from the Kaggle source file" />
    {info ? <div className="stat-grid"><Stat label="Dataset transactions" value={info.dataset_rows?.toLocaleString()} sub="All rows in the source dataset" /><Stat label="Legitimate examples" value={info.class_counts?.['0']?.toLocaleString()} sub="Class 0 · source dataset" /><Stat label="Fraud examples" value={info.class_counts?.['1']?.toLocaleString()} sub={`${Number(info.fraud_percentage).toFixed(3)}% of source rows`} tone="red" /><Stat label="Selected model" value={info.model_name} sub="Selected by validation PR-AUC" /></div> : <EmptyState title="Dataset summary appears after training" detail="No dataset counts or model scores are made up." />}
    <SectionHeading eyebrow="HOW WELL DID IT PERFORM?" title="Held-out test results" caption="Final metrics on data held aside until evaluation" />
    {metrics ? <div className="metric-grid"><Metric label="Precision" value={metrics.precision} hint="Of the transactions flagged in the test set, how many were actually fraud?" /><Metric label="Recall" value={metrics.recall} hint="Of all fraud cases in the test set, how many did the model flag?" /><Metric label="F1 score" value={metrics.f1} hint="A combined score balancing precision and recall." /><Metric label="PR-AUC" value={metrics.pr_auc} hint="Summarizes precision and recall across thresholds; useful when fraud is rare." /></div> : <EmptyState title="Test metrics are not available" detail="Train and evaluate the model before presenting its performance." />}
    {info && <div className="two-column"><div className="panel chart-panel"><PanelTitle title="Class distribution" sub="Source dataset · actual labels" tag="DATASET" /><div className="chart"><ResponsiveContainer><PieChart><Pie data={distribution} dataKey="value" nameKey="name" innerRadius={64} outerRadius={94} paddingAngle={3}>{distribution.map(item => <Cell key={item.name} fill={item.color} />)}</Pie><Tooltip formatter={value => Number(value).toLocaleString()} /><Legend /></PieChart></ResponsiveContainer></div></div><div className="panel chart-panel"><PanelTitle title="Confusion matrix" sub="Model outcomes on held-out test rows" tag="TEST SET" />{metrics?.confusion_matrix ? <ConfusionMatrix values={metrics.confusion_matrix} /> : <EmptyState title="Test matrix unavailable" detail="Train and evaluate the model to generate it." />}</div></div>}
    <div className="reassurance"><b>What this demo can and can’t do</b><p>It scores rows represented by anonymized dataset features. It cannot read a card number, access bank activity, or prove a transaction is fraudulent.</p></div>
  </section>;
}

function SectionHeading({ eyebrow, title, caption }) { return <div className="section-heading"><div><p className="eyebrow cyan">{eyebrow}</p><h3>{title}</h3></div><span className="section-caption">{caption}</span></div>; }
function StatusStep({ number, title, ok, detail, command }) { return <div className={'status-step panel ' + (ok ? 'step-ok' : '')}><div className="step-top"><span>{number}</span><i className={'status-dot ' + (ok ? 'good' : 'bad')}>{ok ? '✓' : '!'}</i></div><h4>{title}</h4><p>{detail}</p>{command && <code>{command}</code>}</div>; }
function Stat({ label, value, sub, tone }) { return <div className="stat-card panel"><span>{label}</span><b className={tone || ''}>{value ?? '—'}</b><small>{sub}</small></div>; }
function Metric({ label, value, hint }) { return <div className="metric-card panel"><div><span>{label}</span><button className="info-tip" title={hint} aria-label={`${label}: ${hint}`}>?</button></div><b>{Number(value).toFixed(3)}</b><small>{hint}</small></div>; }
function EmptyState({ title, detail }) { return <div className="empty-state panel"><b>{title}</b><span>{detail}</span></div>; }
function PanelTitle({ title, sub, tag }) { return <div className="panel-title"><div><h3>{title}</h3><p>{sub}</p></div>{tag && <span>{tag}</span>}</div>; }

function ConfusionMatrix({ values }) { return <div className="matrix-wrap"><p className="matrix-axis">Actual transaction class ↓ &nbsp; · &nbsp; Predicted class →</p><div className="matrix"><span></span><b>Legitimate</b><b>Fraud</b><strong>Legitimate</strong><span className="matrix-cell tn"><b>{values.tn.toLocaleString()}</b><small>correctly cleared</small></span><span className="matrix-cell fp"><b>{values.fp.toLocaleString()}</b><small>false alert</small></span><strong>Fraud</strong><span className="matrix-cell fn"><b>{values.fn.toLocaleString()}</b><small>missed fraud</small></span><span className="matrix-cell tp"><b>{values.tp.toLocaleString()}</b><small>correctly flagged</small></span></div></div>; }

function PredictionResult({ result }) { const percent = Math.max(0, Math.min(100, result.fraud_probability * 100)); return <div className={'prediction-result panel ' + (result.predicted_class ? 'result-risk' : 'result-clear')} role="status"><div className="result-copy"><span className="eyebrow">MODEL ESTIMATE</span><h3>{result.label}</h3><p>{result.explanation}</p><p className="threshold-note">{percent.toFixed(2)}% estimated fraud probability · decision threshold {(result.threshold * 100).toFixed(2)}%</p></div><div className="probability"><b>{percent.toFixed(1)}<small>%</small></b><span>FRAUD PROBABILITY</span><div className="prob-track"><i style={{ width: percent + '%' }} /></div></div></div>; }

function ModelResults({ status, metrics, plots, comparison }) {
  return <section className="page"><PageIntro eyebrow="MODEL EVALUATION" title="Results from the real dataset">The model comparison uses validation rows. Final metrics below use a separate held-out test set.</PageIntro>
    {!status?.model_available && <SetupNotice status={status} />}
    {metrics ? <><div className="metric-grid"><Metric label="Precision" value={metrics.precision} hint="When the model flags a test transaction, this is the share that was fraud." /><Metric label="Recall" value={metrics.recall} hint="The share of all test fraud cases the model caught." /><Metric label="F1 score" value={metrics.f1} hint="Balances precision and recall at the selected threshold." /><Metric label="PR-AUC" value={metrics.pr_auc} hint="Precision-recall area; useful when fraud examples are rare." /></div><div className="two-column"><div className="panel chart-panel"><PanelTitle title="Test confusion matrix" sub="Actual versus predicted classes" tag="HELD-OUT TEST" /><ConfusionMatrix values={metrics.confusion_matrix} /></div><div className="panel chart-panel"><PanelTitle title="Additional test rates" sub="Calculated from final test predictions" /><div className="rate-list"><Rate label="Specificity" value={metrics.specificity} detail="Legitimate transactions correctly left unflagged." /><Rate label="False positive rate" value={metrics.false_positive_rate} detail="Legitimate transactions incorrectly flagged." /><Rate label="False negative rate" value={metrics.false_negative_rate} detail="Fraud transactions the model missed." /></div></div></div></> : <EmptyState title="No evaluation report is available" detail="Train and evaluate the model on the real Kaggle file to create the report." />}
    {comparison.length > 0 && <div className="panel comparison-panel"><PanelTitle title="Candidate model comparison" sub="Validation metrics · default 0.50 threshold" tag="VALIDATION" /><div className="table-scroll"><table><thead><tr>{['Model', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'TN', 'FP', 'FN', 'TP'].map(name => <th key={name}>{name}</th>)}</tr></thead><tbody>{comparison.map(model => <tr key={model.model}>{[model.model, model.precision, model.recall, model.f1, model.roc_auc, model.pr_auc, model.tn, model.fp, model.fn, model.tp].map((value, index) => <td key={index}>{index === 0 ? value : index <= 6 ? Number(value).toFixed(3) : Number(value).toLocaleString()}</td>)}</tr>)}</tbody></table></div></div>}
    {plots.length > 0 && <><SectionHeading eyebrow="GENERATED REPORTS" title="Evaluation charts" caption="Plots served by the API" /><div className="plot-grid">{plots.map(plot => <figure className="panel plot-card" key={plot.url}><figcaption>{plot.name.replaceAll('_', ' ')}</figcaption><img src={API + plot.url} alt={plot.name.replaceAll('_', ' ')} /></figure>)}</div></>}
    <div className="definition-grid"><Definition title="Precision" text="How often a fraud alert was correct. Lower precision means more legitimate purchases might be reviewed." /><Definition title="Recall" text="How many known fraud cases were caught. Lower recall means more fraud could be missed." /><Definition title="PR-AUC" text="A summary of precision and recall over possible thresholds. It is more informative than accuracy when fraud is rare." /><Definition title="False positive and false negative" text="A false positive is a legitimate transaction flagged for review. A false negative is fraud the model did not flag." /></div>
  </section>;
}
function Rate({ label, value, detail }) { return <div className="rate-row"><div><b>{label}</b><small>{detail}</small></div><strong>{(Number(value) * 100).toFixed(2)}%</strong></div>; }
function Definition({ title, text }) { return <details className="definition panel"><summary>{title}<span>+</span></summary><p>{text}</p></details>; }

function HowItWorks({ go }) { return <section className="page"><PageIntro eyebrow="PROJECT GUIDE" title="How this demo works">A short walkthrough of the dataset, model, and prediction.</PageIntro><div className="flow">{[['01', 'Historical dataset', 'Kaggle rows are labeled legitimate (Class 0) or fraud (Class 1). The source covers roughly two days.'], ['02', 'Prepare and learn', 'The training code validates data, splits train, validation, and test partitions, and fits preprocessing on training rows only.'], ['03', 'Compare and set a threshold', 'Models are compared on validation data. The held-out test set is reserved for final evaluation.'], ['04', 'Score a feature row', 'The API returns a fraud probability, then applies the saved threshold to label the row.']].map(([number, title, text]) => <div className="flow-step panel" key={number}><span>{number}</span><div><h3>{title}</h3><p>{text}</p></div></div>)}</div><div className="notice"><b>Why are V1–V28 hard to understand?</b><p>They are anonymized PCA components. Their original meanings are not provided, so they cannot be read like card number, shop name, or location.</p></div><div className="notice warning"><b>What does “Potential fraud” mean?</b><p>The estimated probability met or exceeded the threshold selected on validation data. This is a risk signal, not proof of criminal activity.</p></div><div className="limitations"><h3>Limitations</h3><ul><li>The dataset covers about two days; performance may change over time.</li><li>A random test split does not establish performance on future transactions.</li><li>This is a learning and portfolio prototype, not a banking system.</li><li>Predictions use the dataset’s numeric features, not ordinary card details.</li></ul></div><button className="button primary" onClick={() => go('Try a transaction')}>Go to transaction demo →</button></section>; }

function createRootElement() { createRoot(document.getElementById('root')).render(<App />); }
createRootElement();
