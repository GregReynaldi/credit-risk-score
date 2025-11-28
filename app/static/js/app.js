const form = document.getElementById('creditForm');
const submitBtn = document.getElementById('submitBtn');
const resetBtn = document.getElementById('resetBtn');
const probabilityValue = document.getElementById('probabilityValue');
const riskBadge = document.getElementById('riskBadge');
const heroRiskLevel = document.getElementById('heroRiskLevel');
const llmSummary = document.getElementById('llmSummary');
const latencyTag = document.getElementById('latencyTag');
const featureList = document.getElementById('featureList');

let limeChart;
let shapChart;
let globalChart;
let comparisonChart;
let riskGaugeChart;
let currentJobId = 0;
let buttonState = 'idle';

const numberFields = new Set([
  'person_age',
  'person_income',
  'person_emp_length',
  'loan_amnt',
  'loan_int_rate',
  'loan_percent_income',
  'cb_person_cred_hist_length',
]);

// Calculate loan_percent_income - backup function (inline HTML handlers are primary)
function calculateLoanPercentIncome() {
  const loanAmnt = parseFloat(document.getElementById('loan_amnt').value) || 0;
  const personIncome = parseFloat(document.getElementById('person_income').value) || 0;
  const field = document.getElementById('loan_percent_income');
  
  if (personIncome > 0 && field) {
    field.value = (loanAmnt / personIncome).toFixed(4);
  }
}

function serializeFormData(formData) {
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = numberFields.has(key) ? Number(value) : value;
  }
  return payload;
}

function setButtonState(state) {
  buttonState = state;
  if (state === 'scoring') {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Scoring…';
  } else if (state === 'narrating') {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Finishing narrative…';
  } else {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run analysis';
  }
}

function applyRiskBadge(level) {
  riskBadge.textContent = level;
  heroRiskLevel.textContent = level;
  const normalized = (level || '').toLowerCase();
  if (['low', 'moderate', 'high'].includes(normalized)) {
    riskBadge.className = 'badge ' + normalized;
  } else {
    riskBadge.className = 'badge';
  }
}

function renderChart(canvasId, features, valueKey, chartRef, title) {
  if (typeof Chart === 'undefined') {
    console.error('Chart.js is not loaded.');
    return;
  }

  const canvasEl = document.getElementById(canvasId);
  if (!canvasEl) {
    console.warn(`Canvas element ${canvasId} not found`);
    return;
  }

  const ctx = canvasEl.getContext('2d');
  if (!ctx) {
    console.warn(`Could not get 2d context for ${canvasId}`);
    return;
  }

  if (!features || features.length === 0) {
    // Clear existing chart if present
    if (chartRef && chartRef.destroy) {
      try {
        chartRef.destroy();
      } catch (e) {
        console.warn('Error destroying chart:', e);
      }
    }
    // Draw "No data" message
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.fillStyle = '#999';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No data available', canvasEl.width / 2, canvasEl.height / 2);
    return;
  }

  const labels = features.map((item) => item.feature || 'Unknown');
  const data = features.map((item) => item[valueKey] || 0);

  // Destroy existing chart if present
  if (chartRef && typeof chartRef.destroy === 'function') {
    try {
      chartRef.destroy();
    } catch (e) {
      console.warn('Error destroying existing chart:', e);
    }
  }

  // Create new chart and assign to the reference
  const newChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: title,
          data,
          backgroundColor: data.map((value) =>
            value >= 0 ? 'rgba(74, 222, 128, 0.7)' : 'rgba(248, 113, 113, 0.7)'
          ),
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: { 
          ticks: { color: '#cbd5f5' },
          grid: { color: 'rgba(255,255,255,0.08)' },
        },
        y: {
          ticks: { color: '#cbd5f5' },
        },
      },
    },
  });

  // Update the chart reference
  if (chartRef !== null && chartRef !== undefined) {
    // If chartRef is a variable reference, we need to update it via the calling function
    // For now, return the chart so the caller can assign it
    return newChart;
  }
  
  return newChart;
}

function renderLimeChart(features) {
  const chart = renderChart('limeChart', features, 'impact', limeChart, 'LIME Impact');
  if (chart) {
    limeChart = chart;
  }
}

function renderShapChart(features) {
  const chart = renderChart('shapChart', features, 'shap_value', shapChart, 'SHAP Value');
  if (chart) {
    shapChart = chart;
  }
}

function renderGlobalChart(context) {
  if (typeof Chart === 'undefined') {
    console.error('Chart.js is not loaded.');
    return;
  }
  
  if (!context || context.length === 0) {
    const canvasEl = document.getElementById('globalChart');
    if (canvasEl && globalChart) {
      try {
        globalChart.destroy();
      } catch (e) {
        console.warn('Error destroying global chart:', e);
      }
      globalChart = null;
    }
    return;
  }
  
  const canvasEl = document.getElementById('globalChart');
  if (!canvasEl) {
    console.warn('Canvas element globalChart not found');
    return;
  }
  
  const ctx = canvasEl.getContext('2d');
  if (!ctx) {
    console.warn('Could not get 2d context for globalChart');
    return;
  }
  
  if (globalChart && typeof globalChart.destroy === 'function') {
    try {
      globalChart.destroy();
    } catch (e) {
      console.warn('Error destroying existing global chart:', e);
    }
  }
  
  const features = context.map(item => item.feature);
  const shapRanks = context.map(item => item.global_shap_rank || 0);
  const limeRanks = context.map(item => item.global_lime_rank || 0);
  
  globalChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: features,
      datasets: [
        {
          label: 'SHAP Rank',
          data: shapRanks,
          backgroundColor: 'rgba(91, 124, 250, 0.7)',
        },
        {
          label: 'LIME Rank',
          data: limeRanks,
          backgroundColor: 'rgba(74, 222, 128, 0.7)',
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: true, labels: { color: '#cbd5f5' } },
      },
      scales: {
        x: { 
          ticks: { color: '#cbd5f5', reverse: true },
          grid: { color: 'rgba(255,255,255,0.08)' },
        },
        y: {
          ticks: { color: '#cbd5f5' },
        },
      },
    },
  });
}

function renderFeatureList(features, listId, valueKey = 'impact') {
  const list = document.getElementById(listId);
  if (!list) {
    console.warn(`List element ${listId} not found`);
    return;
  }
  
  list.innerHTML = '';
  
  if (!Array.isArray(features) || features.length === 0) {
    const li = document.createElement('li');
    li.innerHTML = '<span>No features available</span>';
    list.appendChild(li);
    return;
  }
  
  features.forEach((item) => {
    if (!item || typeof item !== 'object') return;
    const li = document.createElement('li');
    const featureName = item.feature || 'Unknown';
    const value = (typeof item[valueKey] === 'number') ? item[valueKey] : 0;
    li.innerHTML = `<span>${featureName}</span><strong>${value.toFixed(3)}</strong>`;
    list.appendChild(li);
  });
}

function renderRiskDrivers(increasing, decreasing) {
  const incList = document.getElementById('riskIncreasingList');
  const decList = document.getElementById('riskDecreasingList');
  
  const safeIncreasing = Array.isArray(increasing) ? increasing : [];
  const safeDecreasing = Array.isArray(decreasing) ? decreasing : [];
  
  if (incList) {
    incList.innerHTML = '';
    if (safeIncreasing.length === 0) {
      const li = document.createElement('li');
      li.innerHTML = '<span>No risk-increasing factors identified</span>';
      incList.appendChild(li);
    } else {
      safeIncreasing.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const li = document.createElement('li');
        const featureName = item.feature || 'Unknown';
        const impact = (typeof item.impact === 'number') ? item.impact : 0;
        li.innerHTML = `<span>${featureName}</span><strong>+${impact.toFixed(3)}</strong>`;
        incList.appendChild(li);
      });
    }
  }
  
  if (decList) {
    decList.innerHTML = '';
    if (safeDecreasing.length === 0) {
      const li = document.createElement('li');
      li.innerHTML = '<span>No risk-decreasing factors identified</span>';
      decList.appendChild(li);
    } else {
      safeDecreasing.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const li = document.createElement('li');
        const featureName = item.feature || 'Unknown';
        const impact = (typeof item.impact === 'number') ? item.impact : 0;
        li.innerHTML = `<span>${featureName}</span><strong>-${impact.toFixed(3)}</strong>`;
        decList.appendChild(li);
      });
    }
  }
}

function renderGlobalContext(context) {
  const table = document.getElementById('globalContextTable');
  if (!table) {
    console.warn('Global context table element not found');
    return;
  }
  
  if (!Array.isArray(context) || context.length === 0) {
    table.innerHTML = '<p>No global context data available</p>';
    return;
  }
  
  table.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Feature</th>
          <th>Global SHAP Rank</th>
          <th>Global LIME Rank</th>
          <th>Sensitivity</th>
        </tr>
      </thead>
      <tbody>
        ${context.map(item => {
          if (!item || typeof item !== 'object') return '';
          return `
          <tr>
            <td>${item.feature || 'Unknown'}</td>
            <td>${item.global_shap_rank || 'N/A'}</td>
            <td>${item.global_lime_rank || 'N/A'}</td>
            <td>${(typeof item.sensitivity === 'number') ? item.sensitivity.toFixed(4) : 'N/A'}</td>
          </tr>
        `;
        }).join('')}
      </tbody>
    </table>
  `;
}

function renderModelAgreement(agreement) {
  const table = document.getElementById('modelAgreementTable');
  if (!table) {
    console.warn('Model agreement table element not found');
    return;
  }
  
  if (!agreement || typeof agreement !== 'object') {
    table.innerHTML = '<p>No model agreement data available</p>';
    return;
  }
  
  const features = Object.keys(agreement);
  if (features.length === 0) {
    table.innerHTML = '<p>No model agreement data available</p>';
    return;
  }
  
  table.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Feature</th>
          <th>XGBoost Deep</th>
          <th>XGBoost Shallow</th>
          <th>LightGBM</th>
          <th>CatBoost</th>
          <th>Neural Network</th>
        </tr>
      </thead>
      <tbody>
        ${features.map(feat => {
          const ranks = agreement[feat];
          if (!ranks || typeof ranks !== 'object') return '';
          return `
            <tr>
              <td>${feat || 'Unknown'}</td>
              <td>${ranks.xgb_deep || 'N/A'}</td>
              <td>${ranks.xgb_shallow || 'N/A'}</td>
              <td>${ranks.lgbm || 'N/A'}</td>
              <td>${ranks.catboost || 'N/A'}</td>
              <td>${ranks.neural_network || 'N/A'}</td>
            </tr>
          `;
        }).join('')}
      </tbody>
    </table>
  `;
}

function renderComparisonChart(limeFeatures, shapFeatures, globalContext) {
  if (typeof Chart === 'undefined') {
    console.error('Chart.js is not loaded.');
    return;
  }
  
  const canvasEl = document.getElementById('comparisonChart');
  if (!canvasEl) {
    console.warn('Canvas element comparisonChart not found');
    return;
  }
  
  const ctx = canvasEl.getContext('2d');
  if (!ctx) {
    console.warn('Could not get 2d context for comparisonChart');
    return;
  }
  
  if (comparisonChart && typeof comparisonChart.destroy === 'function') {
    try {
      comparisonChart.destroy();
    } catch (e) {
      console.warn('Error destroying existing comparison chart:', e);
    }
  }
  
  const safeLime = Array.isArray(limeFeatures) ? limeFeatures : [];
  const safeShap = Array.isArray(shapFeatures) ? shapFeatures : [];
  
  const topFeatures = [...new Set([
    ...safeLime.slice(0, 8).map(f => f.feature),
    ...safeShap.slice(0, 8).map(f => f.feature)
  ])].slice(0, 8);
  
  if (topFeatures.length === 0) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.fillStyle = '#999';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No comparison data available', canvasEl.width / 2, canvasEl.height / 2);
    return;
  }
  
  const limeData = topFeatures.map(feat => {
    const item = safeLime.find(f => f.feature === feat);
    return item ? item.impact : 0;
  });
  
  const shapData = topFeatures.map(feat => {
    const item = safeShap.find(f => f.feature === feat);
    return item ? item.shap_value : 0;
  });
  
  comparisonChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: topFeatures,
      datasets: [
        {
          label: 'LIME',
          data: limeData,
          backgroundColor: 'rgba(91, 124, 250, 0.7)',
        },
        {
          label: 'SHAP',
          data: shapData,
          backgroundColor: 'rgba(74, 222, 128, 0.7)',
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: true, labels: { color: '#cbd5f5' } },
      },
      scales: {
        x: { 
          ticks: { color: '#cbd5f5' },
          grid: { color: 'rgba(255,255,255,0.08)' },
        },
        y: {
          ticks: { color: '#cbd5f5' },
        },
      },
    },
  });
}

function renderRiskGauge(probability) {
  if (typeof Chart === 'undefined') {
    console.error('Chart.js is not loaded.');
    return;
  }
  
  if (typeof probability !== 'number' || isNaN(probability)) {
    console.warn('Invalid probability value for risk gauge:', probability);
    return;
  }
  
  const canvasEl = document.getElementById('riskGaugeChart');
  if (!canvasEl) {
    console.warn('Canvas element riskGaugeChart not found');
    return;
  }
  
  const ctx = canvasEl.getContext('2d');
  if (!ctx) {
    console.warn('Could not get 2d context for riskGaugeChart');
    return;
  }
  
  if (riskGaugeChart && typeof riskGaugeChart.destroy === 'function') {
    try {
      riskGaugeChart.destroy();
    } catch (e) {
      console.warn('Error destroying existing risk gauge chart:', e);
    }
  }
  
  const riskLevel = probability >= 0.5 ? 'HIGH' : probability >= 0.3 ? 'MODERATE' : 'LOW';
  const colors = {
    LOW: 'rgba(74, 222, 128, 0.8)',
    MODERATE: 'rgba(251, 191, 36, 0.8)',
    HIGH: 'rgba(248, 113, 113, 0.8)'
  };
  
  riskGaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Low Risk (<30%)', 'Moderate Risk (30-50%)', 'High Risk (>50%)'],
      datasets: [{
        data: [
          probability < 0.3 ? probability * 100 : 0,
          probability >= 0.3 && probability < 0.5 ? probability * 100 : 0,
          probability >= 0.5 ? probability * 100 : 0
        ],
        backgroundColor: [
          'rgba(74, 222, 128, 0.3)',
          'rgba(251, 191, 36, 0.3)',
          'rgba(248, 113, 113, 0.3)'
        ],
        borderColor: colors[riskLevel],
        borderWidth: 3
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true, labels: { color: '#cbd5f5' } },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.label}: ${probability.toFixed(2)}%`;
            }
          }
        }
      }
    }
  });
}

async function requestPrediction(payload, includeLLM) {
  const response = await fetch(`/api/predict?include_llm=${includeLLM}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail?.detail || 'Prediction failed');
  }

  return response.json();
}

async function triggerLLM(payload, jobId) {
  try {
    const llmResult = await requestPrediction(payload, true);
    if (jobId !== currentJobId) return;
    llmSummary.textContent =
      llmResult.llm_summary || 'LLM explanation unavailable for this deployment.';
    if (jobId === currentJobId) {
      setButtonState('idle');
    }
  } catch (error) {
    if (jobId === currentJobId) {
      llmSummary.textContent = error.message;
      setButtonState('idle');
    }
  }
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tabName = btn.dataset.tab;
    
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    btn.classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (buttonState !== 'idle') return;
  
  calculateLoanPercentIncome();
  
  setButtonState('scoring');
  const payload = serializeFormData(new FormData(form));
  const jobId = ++currentJobId;
  const start = performance.now();

  try {
    const result = await requestPrediction(payload, false);
    if (jobId !== currentJobId) return;

    probabilityValue.textContent = (result.probability * 100).toFixed(2) + '%';
    applyRiskBadge(result.risk_level);
    latencyTag.textContent = `${Math.round(performance.now() - start)} ms`;

    // Render LIME features (always render, even if empty)
    const limeFeatures = Array.isArray(result.lime_features) ? result.lime_features : [];
    renderLimeChart(limeFeatures);
    if (limeFeatures.length > 0) {
      renderFeatureList(limeFeatures, 'featureList', 'impact');
    } else {
      const list = document.getElementById('featureList');
      if (list) list.innerHTML = '<li>No LIME features available</li>';
    }

    // Render SHAP features (always render, even if empty)
    const shapFeatures = Array.isArray(result.shap_features) ? result.shap_features : [];
    renderShapChart(shapFeatures);
    if (shapFeatures.length > 0) {
      renderFeatureList(shapFeatures, 'shapFeatureList', 'shap_value');
    } else {
      const list = document.getElementById('shapFeatureList');
      if (list) list.innerHTML = '<li>No SHAP features available</li>';
    }

    // Render risk drivers
    const riskIncreasing = Array.isArray(result.risk_drivers_increasing) ? result.risk_drivers_increasing : [];
    const riskDecreasing = Array.isArray(result.risk_drivers_decreasing) ? result.risk_drivers_decreasing : [];
    renderRiskDrivers(riskIncreasing, riskDecreasing);

    // Render global context
    const globalContext = Array.isArray(result.global_context) ? result.global_context : [];
    if (globalContext.length > 0) {
      renderGlobalContext(globalContext);
      renderGlobalChart(globalContext);
    } else {
      const table = document.getElementById('globalContextTable');
      if (table) table.innerHTML = '<p>No global context available</p>';
    }

    // Render model agreement
    if (result.model_agreement && typeof result.model_agreement === 'object') {
      renderModelAgreement(result.model_agreement);
    } else {
      const table = document.getElementById('modelAgreementTable');
      if (table) table.innerHTML = '<p>No model agreement data available</p>';
    }

    // Render comparison chart
    if (limeFeatures.length > 0 || shapFeatures.length > 0) {
      renderComparisonChart(limeFeatures, shapFeatures, globalContext);
    }

    // Render risk gauge
    if (typeof result.probability === 'number' && !isNaN(result.probability)) {
      renderRiskGauge(result.probability);
    }

    // Handle LLM summary
    if (result.llm_summary && result.llm_summary.trim().length > 0) {
      llmSummary.textContent = result.llm_summary;
      setButtonState('idle');
    } else {
      llmSummary.textContent = 'Generating narrative...';
      setButtonState('narrating');
      triggerLLM(payload, jobId);
    }
  } catch (error) {
    if (jobId === currentJobId) {
      llmSummary.textContent = error.message;
      setButtonState('idle');
    }
  } 
});

resetBtn.addEventListener('click', () => {
  form.reset();
  setTimeout(() => {
    calculateLoanPercentIncome();
  }, 10);
  probabilityValue.textContent = '--%';
  applyRiskBadge('--');
  llmSummary.textContent = 'Submit the form to generate an analyst summary.';
  featureList.innerHTML = '';
  
  const shapList = document.getElementById('shapFeatureList');
  if (shapList) shapList.innerHTML = '';
  const incList = document.getElementById('riskIncreasingList');
  if (incList) incList.innerHTML = '';
  const decList = document.getElementById('riskDecreasingList');
  if (decList) decList.innerHTML = '';
  const contextTable = document.getElementById('globalContextTable');
  if (contextTable) contextTable.innerHTML = '';
  const agreementTable = document.getElementById('modelAgreementTable');
  if (agreementTable) agreementTable.innerHTML = '';
  
  latencyTag.textContent = '-- ms';
  currentJobId++;
  setButtonState('idle');
  
  if (limeChart) { limeChart.destroy(); limeChart = null; }
  if (shapChart) { shapChart.destroy(); shapChart = null; }
  if (globalChart) { globalChart.destroy(); globalChart = null; }
  if (comparisonChart) { comparisonChart.destroy(); comparisonChart = null; }
  if (riskGaugeChart) { riskGaugeChart.destroy(); riskGaugeChart = null; }
  
  const localTab = document.querySelector('.tab-btn[data-tab="local"]');
  if (localTab) localTab.click();
});

