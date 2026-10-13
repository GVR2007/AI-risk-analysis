/* Risk Desk — front-end logic (no framework). */
(() => {
  'use strict';

  const BOOT = JSON.parse(document.getElementById('boot').textContent);
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const css = (name, el = document.documentElement) => getComputedStyle(el).getPropertyValue(name).trim();
  const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(+v)) ? '—'
    : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const SVGNS = 'http://www.w3.org/2000/svg';
  const TAU = Math.PI * 2;

  /* ------------------------------------------------------------------ guilloche line-work
     Security-print rosettes and bands, generated from simple trigonometric curves.
     The hero rosette's lobe counts follow the project: 10 regression algorithms (outer ring),
     10 classifiers (middle ring) and 4 customer segments (inner star). */
  const poly = (pts) => `M${pts.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join('L')}Z`;

  function ringCurves(count, steps, radius) {
    const out = [];
    for (let i = 0; i < count; i += 1) {
      const pts = [];
      for (let s = 0; s < steps; s += 1) {
        const t = (TAU * s) / steps;
        const r = radius(t, i);
        pts.push([r * Math.cos(t), r * Math.sin(t)]);
      }
      out.push(poly(pts));
    }
    return out;
  }
  function hypotrochoid(R, r, d, rot, steps) {
    const pts = [];
    const turns = r / gcd(R, r);
    for (let s = 0; s <= steps; s += 1) {
      const t = (TAU * turns * s) / steps;
      const x = (R - r) * Math.cos(t) + d * Math.cos(((R - r) / r) * t);
      const y = (R - r) * Math.sin(t) - d * Math.sin(((R - r) / r) * t);
      pts.push([x * Math.cos(rot) - y * Math.sin(rot), x * Math.sin(rot) + y * Math.cos(rot)]);
    }
    return poly(pts);
  }
  function gcd(a, b) { return b ? gcd(b, a % b) : a; }

  function drawRosette(svg, kind) {
    const layers = [];
    if (kind === 'mark') {
      ringCurves(6, 160, (t, i) => 10.5 + 4.2 * Math.sin(5 * t + (i * TAU) / 30)).forEach((d) => layers.push(['', d]));
      svg.innerHTML = layers.map(([, d]) => `<path transform="translate(16 16)" d="${d}"/>`).join('');
      return;
    }
    // outer woven band: 10 lobes
    ringCurves(28, 720, (t, i) => 248 + 24 * Math.sin(10 * t + (i * TAU) / 28 / 10 * 3) + 6 * Math.sin(40 * t))
      .forEach((d) => layers.push(['g3', d]));
    // middle band: 10 lobes, counter-phase, green ink
    ringCurves(22, 600, (t, i) => 176 + 17 * Math.sin(10 * t + Math.PI + (i * TAU) / 22 / 10 * 1.6)
      + 9 * Math.sin(20 * t - (i * TAU) / 22 / 20 * 3))
      .forEach((d) => layers.push(['g2', d]));
    // inner star: 4-fold spirographs
    for (let i = 0; i < 18; i += 1) layers.push(['g1', hypotrochoid(120, 30, 44, (i * TAU) / 72, 900)]);
    // centre ring
    ringCurves(10, 360, (t, i) => 30 + 6 * Math.sin(8 * t + (i * TAU) / 80)).forEach((d) => layers.push(['g1', d]));

    svg.classList.add('rosette');
    svg.innerHTML = layers.map(([cls, d], i) => `<path class="${cls}" d="${d}" pathLength="1" style="--i:${i}"/>`).join('');
    if (!reduceMotion) {
      svg.classList.add('draw');
    }
  }

  function bandSVG() {
    const W = 400; const H = 64; const out = [];
    for (let i = 0; i < 12; i += 1) {
      const pts = [];
      for (let x = 0; x <= W; x += 2) {
        const env = 0.55 + 0.45 * Math.cos((TAU * x) / W * 2);
        const y = H / 2 + 15 * env * Math.sin((TAU * x) / 100 + (i * Math.PI) / 12) + 4 * Math.sin((TAU * x) / 33.3 - (i * Math.PI) / 24);
        pts.push(`${x},${y.toFixed(2)}`);
      }
      out.push(`<path d="M${pts.join('L')}"/>`);
    }
    return `<div class="v-band-art" aria-hidden="true"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${out.join('')}</svg></div>`;
  }

  /* ------------------------------------------------------------------ form specs */
  // [key, label, type, options]  type: num | money | ratio | select | yesno | date
  const SPECS = {
    loan: [
      { title: 'Applicant', fields: [
        ['Age', 'Age', 'num', { min: 18, max: 80 }],
        ['EducationLevel', 'Education', 'select'],
        ['EmploymentStatus', 'Employment', 'select'],
        ['Experience', 'Work experience', 'num', { hint: 'Years' }],
        ['JobTenure', 'Years in current job', 'num'],
        ['MaritalStatus', 'Marital status', 'select'],
        ['NumberOfDependents', 'Dependents', 'num', { min: 0, max: 10 }],
        ['HomeOwnershipStatus', 'Home ownership', 'select'],
      ] },
      { title: 'Income and assets', note: 'Amounts in the dataset currency', fields: [
        ['AnnualIncome', 'Annual income', 'money'],
        ['MonthlyDebtPayments', 'Monthly debt payments', 'money'],
        ['SavingsAccountBalance', 'Savings balance', 'money'],
        ['CheckingAccountBalance', 'Current-account balance', 'money'],
        ['TotalAssets', 'Total assets', 'money'],
        ['TotalLiabilities', 'Total liabilities', 'money'],
      ] },
      { title: 'Credit history', fields: [
        ['CreditScore', 'Credit score', 'num', { min: 300, max: 850, hint: '300 to 850' }],
        ['CreditCardUtilizationRate', 'Card utilisation', 'ratio', { hint: 'Share of card limit used, 0 to 1' }],
        ['DebtToIncomeRatio', 'Debt-to-income ratio', 'ratio', { hint: '0 to 1' }],
        ['PaymentHistory', 'Payment history', 'num', { hint: 'Number of on-time payments' }],
        ['LengthOfCreditHistory', 'Credit history length', 'num', { hint: 'Years' }],
        ['NumberOfOpenCreditLines', 'Open credit lines', 'num', { min: 0 }],
        ['NumberOfCreditInquiries', 'Recent credit enquiries', 'num', { min: 0 }],
        ['UtilityBillsPaymentHistory', 'Utility bills paid on time', 'ratio', { hint: 'Share, 0 to 1' }],
        ['BankruptcyHistory', 'Past bankruptcy', 'yesno'],
        ['PreviousLoanDefaults', 'Previous loan default', 'yesno'],
      ] },
      { title: 'Loan requested', fields: [
        ['LoanAmount', 'Loan amount', 'money'],
        ['LoanDuration', 'Duration', 'select', { options: [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], suffix: ' months' }],
        ['LoanPurpose', 'Purpose', 'select'],
        ['InterestRate', 'Interest rate offered', 'ratio', { optional: true, max: 1, hint: 'Annual rate, e.g. 0.12. Leave blank to estimate it from the credit score' }],
      ] },
    ],
    claim: [
      { title: 'Incident', fields: [
        ['incident_date', 'Incident date', 'date'],
        ['incident_type', 'Incident type', 'select'],
        ['incident_severity', 'Severity', 'select'],
        ['collision_type', 'Collision type', 'select', { unknown: true }],
        ['authorities_contacted', 'Authorities contacted', 'select', { unknown: true }],
        ['incident_hour_of_the_day', 'Hour of day', 'num', { min: 0, max: 23 }],
        ['number_of_vehicles_involved', 'Vehicles involved', 'num', { min: 1, max: 4 }],
        ['bodily_injuries', 'Bodily injuries', 'num', { min: 0, max: 2 }],
        ['witnesses', 'Witnesses', 'num', { min: 0, max: 3 }],
        ['property_damage', 'Property damage', 'select', { unknown: true }],
        ['police_report_available', 'Police report', 'select', { unknown: true }],
        ['incident_state', 'State', 'select'],
        ['incident_city', 'City', 'select'],
      ] },
      { title: 'Amounts claimed', fields: [
        ['injury_claim', 'Injury claim', 'money'],
        ['property_claim', 'Property claim', 'money'],
        ['vehicle_claim', 'Vehicle claim', 'money'],
        ['total_claim_amount', 'Total claim', 'money', { readonly: true, hint: 'Sum of the three claims' }],
      ] },
      { title: 'Policy', fields: [
        ['policy_bind_date', 'Policy start date', 'date'],
        ['months_as_customer', 'Months as customer', 'num', { min: 0 }],
        ['policy_annual_premium', 'Annual premium', 'money'],
        ['policy_deductable', 'Deductible', 'select', { options: [500, 1000, 2000] }],
        ['umbrella_limit', 'Umbrella limit', 'money'],
        ['policy_csl', 'Liability limit (CSL)', 'select'],
        ['policy_state', 'Policy state', 'select'],
      ] },
      { title: 'Policyholder', fields: [
        ['age', 'Age', 'num', { min: 18, max: 70 }],
        ['insured_sex', 'Sex', 'select'],
        ['insured_education_level', 'Education', 'select'],
        ['insured_occupation', 'Occupation', 'select'],
        ['insured_hobbies', 'Hobby', 'select'],
        ['insured_relationship', 'Relationship', 'select'],
        ['capital-gains', 'Capital gains', 'money'],
        ['capital-loss', 'Capital loss', 'money', { hint: 'Entered as a negative number' }],
      ] },
      { title: 'Vehicle', fields: [
        ['auto_make', 'Make', 'select'],
        ['auto_model', 'Model', 'select'],
        ['auto_year', 'Year', 'num', { min: 1990, max: 2015 }],
      ] },
    ],
    card: [
      { title: 'Balance and limit', fields: [
        ['BALANCE', 'Current balance', 'money'],
        ['CREDIT_LIMIT', 'Credit limit', 'money'],
        ['BALANCE_FREQUENCY', 'Balance update frequency', 'ratio', { hint: 'Share of months, 0 to 1' }],
        ['TENURE', 'Months as card holder', 'num', { min: 6, max: 12 }],
      ] },
      { title: 'Purchases', note: 'Six-month totals', fields: [
        ['PURCHASES', 'Total purchases', 'money'],
        ['ONEOFF_PURCHASES', 'One-off purchases', 'money'],
        ['INSTALLMENTS_PURCHASES', 'Instalment purchases', 'money'],
        ['PURCHASES_TRX', 'Purchase transactions', 'num', { min: 0 }],
        ['PURCHASES_FREQUENCY', 'Months with purchases', 'ratio', { hint: 'Share, 0 to 1' }],
        ['ONEOFF_PURCHASES_FREQUENCY', 'Months with one-off purchases', 'ratio'],
        ['PURCHASES_INSTALLMENTS_FREQUENCY', 'Months with instalments', 'ratio'],
      ] },
      { title: 'Cash advances', fields: [
        ['CASH_ADVANCE', 'Cash advance total', 'money'],
        ['CASH_ADVANCE_TRX', 'Cash-advance transactions', 'num', { min: 0 }],
        ['CASH_ADVANCE_FREQUENCY', 'Months with cash advances', 'ratio', { hint: 'Share, 0 to 1' }],
      ] },
      { title: 'Repayments', fields: [
        ['PAYMENTS', 'Total payments', 'money'],
        ['MINIMUM_PAYMENTS', 'Minimum payments made', 'money', { optional: true, hint: 'Leave blank if not recorded' }],
        ['PRC_FULL_PAYMENT', 'Months paid in full', 'ratio', { hint: 'Share, 0 to 1' }],
      ] },
    ],
  };

  const LABELS = {};
  const TYPES = {};
  Object.entries(SPECS).forEach(([t, gs]) => gs.forEach((g) => g.fields.forEach(([k, , ty]) => { TYPES[`${t}.${k}`] = ty; })));
  Object.values(SPECS).flat().forEach((g) => g.fields.forEach(([k, l]) => { LABELS[k] = l; }));
  Object.assign(LABELS, { claim_to_premium_ratio: 'Claim / premium ratio', days_bind_to_incident: 'Days from policy start to incident' });

  function optionsFor(track, key, opts) {
    if (opts.options) return opts.options;
    if (track === 'claim' && key === 'auto_model') return [];
    return (BOOT[track] && BOOT[track][key]) || [];
  }

  function fieldHTML(track, [key, label, type, opts = {}]) {
    const id = `${track}-${key}`;
    const hint = opts.hint ? `<span class="hint" id="${id}-h">${esc(opts.hint)}</span>` : '';
    const described = opts.hint ? ` aria-describedby="${id}-h"` : '';
    let control;
    if (type === 'select') {
      const list = optionsFor(track, key, opts);
      const unknown = opts.unknown ? '<option value="">Unknown</option>' : '';
      control = `<select id="${id}" name="${esc(key)}"${described}>${unknown}${list.map((o) => `<option value="${esc(o)}">${esc(o)}${esc(opts.suffix || '')}</option>`).join('')}</select>`;
    } else if (type === 'yesno') {
      return `<div class="field"><span id="${id}-l" class="lbl">${esc(label)}</span>
        <div class="seg" role="radiogroup" aria-labelledby="${id}-l">
          <label><input type="radio" name="${esc(key)}" value="0" checked><span>No</span></label>
          <label><input type="radio" name="${esc(key)}" value="1"><span>Yes</span></label>
        </div></div>`;
    } else if (type === 'date') {
      control = `<input id="${id}" name="${esc(key)}" type="date"${described}>`;
    } else {
      const step = type === 'num' ? '1' : 'any';
      const min = opts.min ?? (type === 'num' || type === 'ratio' ? 0 : '');
      const max = opts.max ?? (type === 'ratio' ? 1.5 : '');
      control = `<input id="${id}" name="${esc(key)}" type="number" inputmode="decimal" step="${step}"`
        + `${min !== '' ? ` min="${min}"` : ''}${max !== '' ? ` max="${max}"` : ''}`
        + `${opts.readonly ? ' readonly tabindex="-1"' : ''}${opts.optional ? ' placeholder="Not recorded"' : ''}${described}>`;
    }
    return `<div class="field"><label for="${id}">${esc(label)}</label>${control}${hint}</div>`;
  }

  function buildForm(track) {
    const form = document.getElementById(`form-${track}`);
    form.innerHTML = SPECS[track].map((g) => `
      <fieldset class="group">
        <legend><span>${esc(g.title)}</span>${g.note ? `<small>${esc(g.note)}</small>` : ''}</legend>
        <div class="fields">${g.fields.map((f) => fieldHTML(track, f)).join('')}</div>
      </fieldset>`).join('') + `
      <div class="form-actions">
        <button type="submit" class="btn primary">${{ loan: 'Score applicant', claim: 'Check claim', card: 'Find segment' }[track]}</button>
        <span class="note" id="note-${track}" role="status"></span>
      </div>`;
    form.addEventListener('submit', (e) => { e.preventDefault(); predict(track); });
    form.addEventListener('input', () => { $(`#note-${track}`).textContent = 'Inputs changed. Run the model again to update the result.'; state[track].actual = null; });

    if (track === 'claim') {
      const make = form.elements.auto_make;
      const fillModels = (keep) => {
        const sel = form.elements.auto_model;
        const list = BOOT.modelsByMake[make.value] || [];
        sel.innerHTML = list.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
        if (keep && list.includes(keep)) sel.value = keep;
      };
      make.addEventListener('change', () => fillModels());
      form._fillModels = fillModels;
      const sum = () => {
        const v = (k) => parseFloat(form.elements[k].value) || 0;
        form.elements.total_claim_amount.value = v('injury_claim') + v('property_claim') + v('vehicle_claim');
      };
      ['injury_claim', 'property_claim', 'vehicle_claim'].forEach((k) => form.elements[k].addEventListener('input', sum));
      form._sum = sum;
    }
  }

  function fillForm(track, inputs) {
    const form = document.getElementById(`form-${track}`);
    Object.entries(inputs).forEach(([k, v]) => {
      const el = form.elements[k];
      if (!el) return;
      if (el instanceof RadioNodeList) { el.value = String(Math.round(+v || 0)); return; }
      if (el.type === 'date' && v) { el.value = String(v).slice(0, 10); return; }
      if (el.tagName === 'SELECT') {
        if (track === 'claim' && k === 'auto_model') return;
        el.value = v === null || v === undefined ? '' : String(v);
        return;
      }
      el.value = v === null || v === undefined ? '' : (typeof v === 'number' ? +v.toFixed(TYPES[`${track}.${k}`] === 'money' ? 2 : 4) : v);
    });
    if (track === 'claim') { form._fillModels(inputs.auto_model); form._sum(); }
  }

  function readForm(track) {
    const form = document.getElementById(`form-${track}`);
    const out = {};
    new FormData(form).forEach((v, k) => { out[k] = v; });
    const algoSel = document.getElementById(`algo-${track}`);
    if (algoSel && algoSel.value) out.algorithm = algoSel.value;
    return out;
  }

  /* algorithm selector: one real saved model per option, chosen by the user, run live on submit */
  function populateAlgoSelect(track) {
    const sel = document.getElementById(`algo-${track}`);
    const info = BOOT.algorithms && BOOT.algorithms[track];
    if (!sel || !info || !info.options || !info.options.length) {
      if (sel) sel.closest('.algo-select').hidden = true;
      return;
    }
    sel.innerHTML = info.options.map((o) => `<option value="${esc(o)}"${o === info.default ? ' selected' : ''}>${esc(o)}</option>`).join('');
    sel.addEventListener('change', () => { if (loaded.has(track)) predict(track); });
  }

  /* ------------------------------------------------------------------ API */
  const state = { loan: {}, claim: {}, card: {} };

  async function api(path, body) {
    const res = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }

  async function loadSample(track, kind = 'random') {
    const btns = $$(`[data-sample="${track}"]`);
    btns.forEach((b) => { b.disabled = true; });
    try {
      const s = await api(`/api/sample/${track}?kind=${encodeURIComponent(kind)}`);
      fillForm(track, s.inputs);
      await predict(track);
      state[track].actual = s.actual ?? null;
      renderActual(track);
    } catch (err) {
      showError(track, err);
    } finally {
      btns.forEach((b) => { b.disabled = false; });
    }
  }

  async function predict(track) {
    const btn = $(`#form-${track} button[type=submit]`);
    btn.disabled = true;
    try {
      const r = await api(`/api/predict/${track}`, readForm(track));
      state[track].result = r;
      ({ loan: renderLoan, claim: renderClaim, card: renderCard })[track](r);
      $(`#note-${track}`).textContent = '';
    } catch (err) {
      showError(track, err);
    } finally {
      btn.disabled = false;
    }
  }

  function showError(track, err) {
    document.getElementById(`verdict-${track}`).innerHTML = `${bandSVG()}<div class="error-box" role="alert">${esc(err.message)}</div>`;
  }

  function renderActual(track) {
    const box = $(`#verdict-${track} .actual`);
    const a = state[track].actual;
    if (!box) return;
    if (a === null || a === undefined) { box.hidden = true; return; }
    box.hidden = false;
    box.innerHTML = track === 'loan'
      ? `This applicant's recorded score in the dataset is <strong>${fmt(a, 1)}</strong>.`
      : `This claim is recorded in the dataset as <strong>${a === 'Y' ? 'fraud' : 'genuine'}</strong>.`;
  }

  /* ------------------------------------------------------------------ verdict pieces */
  const TONE = { Low: '--low', Moderate: '--mid', Elevated: '--mid', 'Medium-High': '--mid', Medium: '--mid', High: '--high' };
  const toneOf = (band) => `var(${TONE[band] || '--mid'})`;

  /* Semicircular dial. zones: [{from, to, tone}] in 0..1; marks: [{at, label}] in 0..1. */
  function dialSVG(zones, marks) {
    const R = 118; const W = 14;
    const pt = (f, r) => { const a = Math.PI * (1 - f); return [r * Math.cos(a), -r * Math.sin(a)]; };
    const arc = (a, b, r) => { const [x1, y1] = pt(a, r); const [x2, y2] = pt(b, r); return `M${x1.toFixed(2)},${y1.toFixed(2)}A${r},${r} 0 0 1 ${x2.toFixed(2)},${y2.toFixed(2)}`; };
    const zoneP = zones.map((z) => `<path class="zone" d="${arc(z.from + 0.004, z.to - 0.004, R)}" style="stroke:var(${z.tone});opacity:.85"/>`).join('');
    let ticks = '';
    for (let i = 0; i <= 50; i += 1) {
      const f = i / 50;
      const [x1, y1] = pt(f, R - W / 2 - 4); const [x2, y2] = pt(f, R - W / 2 - (i % 5 ? 9 : 14));
      ticks += `<line class="tick" x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"/>`;
    }
    const markP = marks.map((m) => {
      const [x1, y1] = pt(m.at, R + W / 2 + 2); const [x2, y2] = pt(m.at, R - W / 2 - 16);
      const [lx, ly] = pt(m.at, R + W / 2 + 12);
      const anchor = m.at < 0.02 ? 'start' : m.at > 0.98 ? 'end' : m.at < 0.45 ? 'end' : m.at > 0.55 ? 'start' : 'middle';
      const dx = m.at < 0.02 ? -8 : m.at > 0.98 ? 8 : 0;
      return `<line class="tick major" x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"/>
        <text class="tick-label" x="${(lx + dx).toFixed(2)}" y="${(m.at < 0.02 || m.at > 0.98 ? ly + 14 : ly + 4).toFixed(2)}" text-anchor="${anchor}">${esc(m.label)}</text>`;
    }).join('');
    return `<svg class="dial" viewBox="-152 -150 304 170" aria-hidden="true">
      ${zoneP}${ticks}${markP}
      <g class="needle"><line x1="0" y1="0" x2="${-(R - 26)}" y2="0"/></g>
      <circle class="hub" r="7"/>
    </svg>`;
  }
  function setNeedle(root, track, frac) {
    const g = $('.needle', root);
    if (!g) return;
    const prev = state[track].needle ?? 0;
    const target = Math.max(0, Math.min(1, frac)) * 180;
    g.style.transition = 'none';
    g.style.transform = `rotate(${prev}deg)`;
    g.getBoundingClientRect();
    requestAnimationFrame(() => { g.style.transition = ''; g.style.transform = `rotate(${target}deg)`; });
    state[track].needle = target;
  }

  function driversHTML(drivers, unit, caption) {
    if (!drivers.length) return '';
    const max = Math.max(...drivers.map((d) => Math.abs(d.impact)), 1e-9);
    const show = (v) => (typeof v === 'number' ? fmt(v, Math.abs(v) < 2 && v % 1 ? 2 : 0) : v);
    return `<div class="v-section"><h3>What moved the result</h3><p class="cap">${esc(caption)}</p><ul class="drivers">${drivers.map((d) => {
      const w = (Math.abs(d.impact) / max) * 50;
      const up = d.impact > 0;
      const val = unit === 'pp' ? `${up ? '+' : '−'}${Math.abs(d.impact * 100).toFixed(1)}` : `${up ? '+' : '−'}${Math.abs(d.impact).toFixed(1)}`;
      return `<li><span class="d-name" title="${esc(LABELS[d.feature] || d.feature)}: ${esc(show(d.value))} (typical ${esc(show(d.typical))})">${esc(LABELS[d.feature] || d.feature)} <small>${esc(show(d.value))}</small></span>
        <span class="d-bar"><span class="d-fill" style="${up ? 'left:50%' : `left:${50 - w}%`};width:${w}%;background:var(${up ? '--high' : '--low'})"></span></span>
        <span class="d-val ${up ? 'up' : 'down'}">${val}</span></li>`;
    }).join('')}</ul></div>`;
  }

  function renderLoan(r) {
    const root = document.getElementById('verdict-loan');
    const s = r.scale;
    const at = (v) => (v - s.min) / (s.max - s.min);
    root.style.setProperty('--tone', toneOf(r.band));
    root.innerHTML = `${bandSVG()}
      <div class="v-body">
        <div class="v-head"><p class="v-label">Predicted risk score &middot; ${esc(r.algorithm)}</p><span class="v-pill">${esc(r.band)} risk</span></div>
        <div class="v-figure">
          <span class="v-number">${fmt(r.score, 1)}</span>
          <p class="v-sub">Higher than <strong>${fmt(r.percentile, 0)}%</strong> of the 20,000 applicants in the dataset.</p>
        </div>
        ${dialSVG([
          { from: 0, to: at(s.q25), tone: '--low' }, { from: at(s.q25), to: at(s.q75), tone: '--mid' }, { from: at(s.q75), to: 1, tone: '--high' }],
          [{ at: 0, label: fmt(s.min, 0) }, { at: at(s.q25), label: fmt(s.q25, 0) }, { at: at(s.q75), label: fmt(s.q75, 0) }, { at: 1, label: fmt(s.max, 0) }])}
        <p class="dial-legend">Half of all applicants score between ${fmt(s.q25, 0)} and ${fmt(s.q75, 0)}.</p>
        ${driversHTML(r.drivers, 'pts', 'Score change when this input is replaced with the typical applicant’s value. Red raises the risk, green lowers it.')}
        <div class="v-section"><h3>Calculated from your inputs</h3>
          <dl class="kv">
            <dt>Monthly income</dt><dd>${fmt(r.derived.MonthlyIncome)}</dd>
            <dt>Net worth</dt><dd>${fmt(r.derived.NetWorth)}</dd>
            <dt>Interest rate</dt><dd>${pct(r.derived.InterestRate, 2)}</dd>
            <dt>Monthly loan payment</dt><dd>${fmt(r.derived.MonthlyLoanPayment)}</dd>
            <dt>Total debt-to-income</dt><dd>${fmt(r.derived.TotalDebtToIncomeRatio, 2)}</dd>
          </dl></div>
        <p class="actual" hidden></p>
      </div>`;
    setNeedle(root, 'loan', at(r.score));
    renderActual('loan');
  }

  function renderClaim(r) {
    const root = document.getElementById('verdict-claim');
    root.style.setProperty('--tone', toneOf(r.band));
    root.innerHTML = `${bandSVG()}
      <div class="v-body">
        <div class="v-head"><p class="v-label">Chance the claim is fraudulent &middot; ${esc(r.algorithm)}</p><span class="v-pill">${esc(r.band)}</span></div>
        <div class="v-figure">
          <span class="v-number">${(r.probability * 100).toFixed(0)}<small>%</small></span>
          <p class="v-sub"><strong>${esc(r.decision)}.</strong> Claims at 50% or above are flagged. About one claim in four in the data is fraudulent.</p>
        </div>
        ${dialSVG([{ from: 0, to: 0.3, tone: '--low' }, { from: 0.3, to: 0.5, tone: '--mid' }, { from: 0.5, to: 1, tone: '--high' }],
          [{ at: 0, label: '0%' }, { at: 0.3, label: 'Review 30%' }, { at: 0.5, label: 'Flag 50%' }, { at: 1, label: '100%' }])}
        ${driversHTML(r.drivers, 'pp', 'Change in fraud probability, in percentage points, when this input is replaced with its most common value. Red raises it, green lowers it.')}
        <div class="v-section"><h3>Engineered red-flag features</h3>
          <dl class="kv">
            <dt>Claim / annual premium</dt><dd>${fmt(r.claim_to_premium_ratio, 1)}×</dd>
            <dt>Days from policy start to incident</dt><dd>${fmt(r.days_bind_to_incident)}</dd>
          </dl></div>
        <p class="actual" hidden></p>
      </div>`;
    setNeedle(root, 'claim', r.probability);
    renderActual('claim');
  }

  /* ------------------------------------------------------------------ clustering */
  const SEG_COLOR = { 'Active Transactors': '--low', 'Dormant / Low-Activity': '--dormant', 'Heavy Revolving Spenders': '--mid', 'Cash-Advance Revolvers': '--high' };
  const SEG_ORDER = ['Dormant / Low-Activity', 'Active Transactors', 'Heavy Revolving Spenders', 'Cash-Advance Revolvers'];
  let clusterMeta = null;
  let scatter = null;

  async function getClusterMeta() {
    if (!clusterMeta) clusterMeta = (await api('/api/meta')).clustering;
    return clusterMeta;
  }

  async function renderCard(r) {
    const meta = await getClusterMeta();
    const root = document.getElementById('verdict-card');
    const seg = r.segment;
    const fit = r.confidence > 0.35 ? 'Clearly inside this segment.' : r.confidence > 0.12 ? 'Inside this segment, near a neighbouring one.' : 'On the boundary between two segments.';
    const sizes = Object.fromEntries(Object.values(meta.segments).map((s) => [s.name, s.size]));
    const rows = [
      ['CREDIT_UTILIZATION', 'Credit utilisation', (v) => pct(v)],
      ['PAYMENT_TO_MIN_RATIO', 'Payments / minimum payments', (v) => `${fmt(v, 2)}×`],
      ['PURCHASES_FREQUENCY', 'Months with purchases', (v) => pct(v)],
      ['CASH_ADVANCE_FREQUENCY', 'Months with cash advances', (v) => pct(v)],
      ['PRC_FULL_PAYMENT', 'Months paid in full', (v) => pct(v)],
    ];
    root.style.setProperty('--tone', toneOf(seg.risk));
    root.innerHTML = `${bandSVG()}
      <div class="v-body">
        <div class="v-head"><p class="v-label">Segment &middot; ${esc(r.algorithm)}</p><span class="v-pill">${esc(seg.risk)} risk</span></div>
        <div class="v-figure">
          <p class="segment-name">${esc(seg.name)}</p>
          <p class="v-sub">${esc(seg.summary)} <strong>${esc(fit)}</strong></p>
          <p class="v-sub"><strong>Suggested action:</strong> ${esc(seg.action)}</p>
          ${r.algorithm !== 'K-Means' ? `<p class="v-sub cap">${esc(r.segment_source)}</p>` : ''}
        </div>
        <div>
          <ol class="ladder" aria-label="The four segments, from lower to higher risk">
            ${SEG_ORDER.map((n) => `<li class="rung${n === seg.name ? ' on' : ''}" style="--tone:var(${SEG_COLOR[n]})"${n === seg.name ? ' aria-current="true"' : ''}><i></i>${esc(n)}<span class="cap" style="margin:0">${fmt(sizes[n])} customers</span></li>`).join('')}
          </ol>
          <div class="ladder-cap"><span>Lower risk</span><span>Higher risk</span></div>
        </div>
        <div class="v-section"><h3>This customer compared with the segment</h3>
          <table class="mini"><thead><tr><th class="l">Measure</th><th>Customer</th><th>Segment median</th></tr></thead><tbody>
          ${rows.map(([k, l, f]) => `<tr><td class="l">${esc(l)}</td><td><strong>${f(r.customer[k])}</strong></td><td>${f(r.segment_median[k])}</td></tr>`).join('')}
          </tbody></table></div>
        <div class="v-section"><h3>Where the customer sits</h3>
          <p class="cap">1,500 sampled card holders on the first two principal components, coloured by segment. The ringed point is this customer.</p>
          <div class="chart-box"><canvas id="cardScatter" role="img" aria-label="Scatter plot of customers by segment with this customer highlighted"></canvas></div>
          <div class="segment-key">${SEG_ORDER.map((n) => `<span><span class="seg-swatch" style="background:var(${SEG_COLOR[n]})"></span>${esc(n)}</span>`).join('')}</div>
        </div>
      </div>`;
    drawScatter(meta, r, root);
  }

  function drawScatter(meta, r, root) {
    if (scatter) scatter.destroy();
    const byCluster = {};
    meta.points.forEach((p) => { (byCluster[p.c] ||= []).push({ x: p.x, y: p.y }); });
    const grid = css('--v-rule', root);
    const ink = css('--verdict-ink');
    const muted = css('--verdict-muted');
    const datasets = Object.entries(byCluster).map(([c, pts]) => {
      const col = css(SEG_COLOR[meta.segments[c].name] || '--dormant', root);
      return { label: meta.segments[c].name, data: pts, backgroundColor: `${col}70`, borderWidth: 0, pointRadius: 2.1, pointHoverRadius: 3 };
    });
    datasets.push({ label: 'This customer', data: [r.position], backgroundColor: css('--verdict-bg'), borderColor: ink, borderWidth: 3, pointRadius: 7, pointHoverRadius: 8, order: -1 });
    scatter = new Chart(document.getElementById('cardScatter'), {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        font: { family: 'Archivo' },
        plugins: { legend: { display: false }, tooltip: { filter: (i) => i.dataset.label === 'This customer', callbacks: { label: () => 'This customer' } } },
        scales: {
          x: { title: { display: true, text: 'First component: cash advances to the left, purchases to the right', color: muted, font: { family: 'Archivo', size: 11.5 } }, grid: { color: grid }, border: { color: grid }, ticks: { color: muted, font: { family: 'Archivo' } } },
          y: { title: { display: true, text: 'Second component', color: muted, font: { family: 'Archivo', size: 11.5 } }, grid: { color: grid }, border: { color: grid }, ticks: { color: muted, font: { family: 'Archivo' } } },
        },
      },
    });
  }

  /* ------------------------------------------------------------------ results page */
  const FIG = (file, caption) => ({ file, caption });
  /* opts: best (row index to highlight), tag (label on the best row), bar (column key drawn as an inline bar) */
  function tableHTML(cols, rows, opts = {}) {
    const best = opts.best ?? 0;
    const vals = opts.bar ? rows.map((r) => +r[opts.bar]).filter((v) => Number.isFinite(v)) : [0, 1];
    const hi = Math.max(...vals); const lo = Math.max(0, Math.min(...vals) - (hi - Math.min(...vals)) * 0.25);
    return `<div class="table-wrap"><table><thead><tr>${cols.map((c, i) => `<th class="${i === 0 || c.left ? 'l' : ''}" scope="col">${esc(c.label)}</th>`).join('')}</tr></thead><tbody>${rows.map((r, ri) => `<tr class="${ri === best ? 'best' : ''}">${cols.map((c, i) => {
      const v = r[c.key];
      const txt = esc(c.fmt ? c.fmt(v, r) : v);
      const isFirst = i === 0;
      let cell = txt;
      if (opts.bar === c.key && v !== null && v !== undefined) {
        cell = `<span class="cell-bar">${txt}<i aria-hidden="true"><b style="width:${hi > lo ? Math.max(4, ((+v - lo) / (hi - lo)) * 100) : 100}%"></b></i></span>`;
      }
      if (isFirst && ri === best && opts.tag) cell += `<span class="tag-selected">${esc(opts.tag)}</span>`;
      return `<td class="${isFirst || c.left ? 'l' : ''}">${cell}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  const f4 = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(4));
  function figsHTML(list) {
    return `<div class="figs">${list.map((f) => `<figure class="fig"><button type="button" data-fig="${esc(f.file)}" data-cap="${esc(f.caption)}" aria-label="Enlarge: ${esc(f.caption)}"><img src="/static/figures/${esc(f.file)}" alt="${esc(f.caption)}" loading="lazy"></button><figcaption>${esc(f.caption)}</figcaption></figure>`).join('')}</div>`;
  }

  function buildResults() {
    const R = BOOT.results;
    const reg = R.regression;
    $('#tab-regression').innerHTML = `
      <section class="res-block"><h2>All 10 regression algorithms</h2>
        <p>Target: the <strong>RiskScore</strong> of a loan applicant. Every model used the same 80:20 split; ranked by R² on the test set, before tuning.</p>
        ${tableHTML([{ key: 'Model', label: 'Model' }, { key: 'R2', label: 'R²', fmt: f4 }, { key: 'RMSE', label: 'RMSE', fmt: f4 }, { key: 'MAE', label: 'MAE', fmt: f4 }], reg.table, { bar: 'R2' })}</section>
      <section class="res-block"><h2>Tuning and cross-validation of the two best models</h2>
        <p>GridSearchCV with 5-fold cross-validation. The tuned Gradient Boosting Regressor is the model this app uses.</p>
        ${tableHTML([{ key: 'Model', label: 'Model' }, { key: 'Baseline R2', label: 'Baseline R²', fmt: f4 }, { key: 'Tuned R2', label: 'Tuned R²', fmt: f4 }, { key: 'cv', label: '5-fold CV R²', fmt: (v, r) => { const c = reg.cv.find((x) => x.Model === r.Model); return c ? `${c['CV R2'].toFixed(4)} ± ${c.Std.toFixed(4)}` : '—'; } }, { key: 'Best params', label: 'Best parameters', left: true, fmt: (v) => String(v).replaceAll('model__', '') }], reg.tuning, { tag: 'In this app' })}</section>
      <section class="res-block"><h2>Figures</h2>${figsHTML([
        FIG('reg_residual_pred_vs_actual.png', 'Residuals and predicted against actual, tuned Gradient Boosting'),
        FIG('reg_feature_importance_rf.png', 'Top 15 feature importances, Random Forest'),
        FIG('reg_correlation_heatmap.png', 'Correlation heatmap of the numeric features')])}</section>`;

    const clf = R.classification;
    $('#tab-classification').innerHTML = `
      <section class="res-block"><h2>Final comparison of all 10 classifiers</h2>
        <p>Target: <strong>fraud_reported</strong>. Each algorithm after GridSearchCV, with the feature representation (PCA or one-hot encoded) also chosen by cross-validation. Ranked by cross-validated F1; test-set metrics are weighted averages.</p>
        ${tableHTML([{ key: 'Model', label: 'Model' }, { key: 'Part', label: 'Part' }, { key: 'Representation', label: 'Features' }, { key: 'CV F1 (train)', label: 'CV F1', fmt: f4 }, { key: 'Accuracy', label: 'Accuracy', fmt: f4 }, { key: 'Precision', label: 'Precision', fmt: f4 }, { key: 'Recall', label: 'Recall', fmt: f4 }, { key: 'F1 (weighted)', label: 'F1', fmt: f4 }, { key: 'ROC-AUC', label: 'ROC-AUC', fmt: f4 }, { key: 'Fraud recall', label: 'Fraud recall', fmt: f4 }], clf.final, { bar: 'CV F1 (train)', tag: 'In this app' })}</section>
      <section class="res-block"><h2>Before tuning: all 10 on the shared PCA pipeline</h2>
        <p>The consolidated baseline, where every model used the 55-component PCA pipeline from Review 1. Random Forest and SVC caught no fraudulent claims at all.</p>
        ${tableHTML([{ key: 'Model', label: 'Model' }, { key: 'Part', label: 'Part' }, { key: 'Accuracy', label: 'Accuracy', fmt: f4 }, { key: 'Precision', label: 'Precision', fmt: f4 }, { key: 'Recall', label: 'Recall', fmt: f4 }, { key: 'F1 (weighted)', label: 'F1', fmt: f4 }, { key: 'ROC-AUC', label: 'ROC-AUC', fmt: f4 }, { key: 'Fraud recall', label: 'Fraud recall', fmt: f4 }], clf.baseline_pca, { bar: 'F1 (weighted)' })}</section>
      <section class="res-block"><h2>Figures</h2>${figsHTML([
        FIG('clf_final_model.png', 'Final model: confusion matrix and ROC against the Review 1 best'),
        FIG('clf_pca_vs_encoded.png', 'PCA against original encoded features for every algorithm'),
        FIG('clf_roc_curves_all10.png', 'ROC curves of all 10 algorithms, PCA baseline'),
        FIG('clf_feature_importance_rf.png', 'Top 15 feature importances, tuned Random Forest'),
        FIG('clf_confusion_matrices_part_b.png', 'Confusion matrices of the Part B algorithms')])}</section>`;

    const clu = R.clustering;
    $('#tab-clustering').innerHTML = `
      <section class="res-block"><h2>Clustering metrics</h2>
        <p>Same PCA input (8 components), k = 4. Higher Silhouette and Calinski-Harabasz are better; lower Davies-Bouldin is better. Single linkage scores well only because it puts almost every customer in one cluster.</p>
        ${tableHTML([{ key: 'Algorithm', label: 'Algorithm' }, { key: 'Silhouette', label: 'Silhouette', fmt: f4 }, { key: 'Davies-Bouldin', label: 'Davies-Bouldin', fmt: f4 }, { key: 'Calinski-Harabasz', label: 'Calinski-Harabasz', fmt: (v) => fmt(v, 1) }, { key: 'Smallest cluster', label: 'Smallest', fmt: (v) => fmt(v) }, { key: 'Largest cluster', label: 'Largest', fmt: (v) => fmt(v) }], clu.metrics, { bar: 'Calinski-Harabasz', tag: 'In this app' })}</section>
      <section class="res-block"><h2>The four segments</h2>
        <p>K-Means clusters, named from their median behaviour.</p>
        ${tableHTML([{ key: 'Segment', label: 'Segment' }, { key: 'Risk level', label: 'Risk' }, { key: 'Customers', label: 'Customers', fmt: (v) => fmt(v) }, { key: 'Share', label: 'Share', fmt: (v) => pct(v, 1) }, { key: 'Median utilisation', label: 'Median utilisation', fmt: (v) => pct(v) }, { key: 'Median payment/min ratio', label: 'Payments / minimum', fmt: (v) => `${fmt(v, 2)}×` }], clu.segments, { best: -1, bar: 'Customers' })}</section>
      <section class="res-block"><h2>Choosing k</h2>
        <p>K-Means for k = 2 to 10. The silhouette is flat from k = 3 to 8; k = 4 is where the Ward dendrogram has a natural cut.</p>
        ${tableHTML([{ key: 'k', label: 'k' }, { key: 'Inertia', label: 'Inertia', fmt: (v) => fmt(v) }, { key: 'Silhouette', label: 'Silhouette', fmt: f4 }, { key: 'Davies-Bouldin', label: 'Davies-Bouldin', fmt: f4 }, { key: 'Calinski-Harabasz', label: 'Calinski-Harabasz', fmt: (v) => fmt(v, 1) }, { key: 'Smallest cluster', label: 'Smallest cluster', fmt: (v) => fmt(v) }], clu.k_selection, { best: 2, tag: 'Chosen' })}</section>
      <section class="res-block"><h2>Figures</h2>${figsHTML([
        FIG('clu_elbow_silhouette.png', 'Elbow curve, silhouette, Davies-Bouldin and Calinski-Harabasz against k'),
        FIG('clu_dendrogram_ward.png', 'Ward dendrogram cut at 4 clusters'),
        FIG('clu_pca_2d_clusters.png', 'PCA 2-D scatter for K-Means and Agglomerative (Ward)'),
        FIG('clu_tsne_clusters.png', 't-SNE projection for both algorithms'),
        FIG('clu_linkage_comparison.png', 'Comparison of the four linkage strategies'),
        FIG('clu_segment_profile_heatmap.png', 'How each segment differs from the average customer')])}</section>`;

    $$('.tabs [role=tab]').forEach((tab, i, all) => {
      tab.addEventListener('click', () => selectTab(tab));
      tab.addEventListener('keydown', (e) => {
        const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
        if (d) { const n = all[(i + d + all.length) % all.length]; n.focus(); selectTab(n); }
      });
    });
    document.addEventListener('click', (e) => {
      const b = e.target.closest('[data-fig]');
      if (!b) return;
      const lb = $('#lightbox');
      $('img', lb).src = `/static/figures/${b.dataset.fig}`;
      $('img', lb).alt = b.dataset.cap;
      $('.lb-caption', lb).textContent = b.dataset.cap;
      lb.showModal();
    });
    $('#lightbox').addEventListener('click', (e) => { if (e.target.id === 'lightbox') e.target.close(); });
  }
  function selectTab(tab) {
    $$('.tabs [role=tab]').forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', on);
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    });
  }

  /* ------------------------------------------------------------------ routing + theme */
  const loaded = new Set();
  function route() {
    const name = (location.hash || '#overview').slice(1);
    const view = $(`[data-view="${name}"]`) ? name : 'overview';
    $$('.view').forEach((v) => { v.hidden = v.dataset.view !== view; });
    $$('[data-nav]').forEach((a) => {
      if (a.dataset.nav === view && a.closest('.mainnav')) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    if (['loan', 'claim', 'card'].includes(view) && !loaded.has(view)) {
      loaded.add(view);
      document.getElementById(`verdict-${view}`).innerHTML = `${bandSVG()}<p class="empty"><strong>Scoring a real record</strong>A record from the dataset is being loaded into the form.</p>`;
      loadSample(view, 'random');
    }
    document.title = { overview: 'Risk Desk', loan: 'Loan risk | Risk Desk', claim: 'Claim fraud | Risk Desk', card: 'Card segments | Risk Desk', results: 'Model results | Risk Desk' }[view];
    window.scrollTo({ top: 0 });
  }

  function initTheme() {
    const btn = $('#themeToggle');
    const isDark = () => document.documentElement.dataset.theme === 'dark'
      || (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    const sync = () => {
      $('span', btn).textContent = isDark() ? 'Light' : 'Dark';
      btn.setAttribute('aria-label', isDark() ? 'Switch to light theme' : 'Switch to dark theme');
    };
    btn.addEventListener('click', () => {
      const next = isDark() ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('riskdesk-theme', next); } catch (e) { /* storage unavailable */ }
      sync();
      if (state.card.result && !$('#view-card').hidden) renderCard(state.card.result);
    });
    sync();
  }

  $$('[data-rosette]').forEach((svg) => drawRosette(svg, svg.dataset.rosette));
  ['loan', 'claim', 'card'].forEach(buildForm);
  ['loan', 'claim', 'card'].forEach(populateAlgoSelect);
  $$('[data-sample]').forEach((b) => b.addEventListener('click', () => loadSample(b.dataset.sample, b.dataset.kind)));
  buildResults();
  initTheme();
  window.addEventListener('hashchange', route);
  route();
})();
