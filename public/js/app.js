// ==========================================================================
// AirQuality Research Platform - Core Client Application Controller
// Orchestrates API telemetry, Chart.js multi-horizon rendering, data pipeline,
// Ollama AI assistant dialogue, and automated academic PDF dossier compilation.
// ==========================================================================

let currentTelemetry = null;
let currentLocation = { city: 'Delhi', name: 'Delhi', lat: 28.6139, lon: 77.2090 };
let chartInstances = {};
let conversationHistory = [];
let activeTrajectoryMode = 'aqi'; // 'aqi' or 'pm25'
let showConfidenceBand = true;

document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  checkOllamaStatus();
  updateChatSuggestions(currentLocation, null);
  // Load the default city.
  loadAnalysisData({ city: 'Delhi' });
});

// ==========================================================================
// Event Listeners & UI Binding
// ==========================================================================
function initEventListeners() {
  // Theme toggle
  document.getElementById('btnThemeToggle').addEventListener('click', toggleTheme);

  // Search input & button
  const searchInput = document.getElementById('citySearchInput');
  const btnSearch = document.getElementById('btnSearch');
  const btnClear = document.getElementById('btnClearSearch');

  btnSearch.addEventListener('click', () => {
    const q = searchInput.value.trim();
    if (q) performSearch(q);
  });

  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const q = searchInput.value.trim();
      if (q) performSearch(q);
    }
  });

  searchInput.addEventListener('input', () => {
    btnClear.style.display = searchInput.value ? 'block' : 'none';
  });

  btnClear.addEventListener('click', () => {
    searchInput.value = '';
    btnClear.style.display = 'none';
    document.getElementById('searchSuggestions').style.display = 'none';
  });

  // Quick Preset City Buttons
  document.querySelectorAll('.hub-pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
      document.querySelectorAll('.hub-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const city = pill.getAttribute('data-city');
      loadAnalysisData({ city });
    });
  });

  // Geolocation button
  document.getElementById('btnGeoLocation').addEventListener('click', detectUserLocation);

  // Refresh button
  document.getElementById('btnRefresh').addEventListener('click', () => {
    if (currentLocation) {
      loadAnalysisData({ lat: currentLocation.lat, lon: currentLocation.lon }, true);
    }
  });

  // Research Tabs Switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(btn.getAttribute('data-tab'));
      if (targetPane) targetPane.classList.add('active');

      // Trigger chart resize on tab switch
      setTimeout(() => {
        Object.values(chartInstances).forEach(chart => {
          if (chart && typeof chart.resize === 'function') chart.resize();
        });
      }, 50);
    });
  });

  // 72h Forecast Chart Toggles
  document.getElementById('toggleAqiTrajectory').addEventListener('click', () => {
    activeTrajectoryMode = 'aqi';
    document.getElementById('toggleAqiTrajectory').classList.add('active');
    document.getElementById('togglePm25Trajectory').classList.remove('active');
    renderPredictionChart();
  });

  document.getElementById('togglePm25Trajectory').addEventListener('click', () => {
    activeTrajectoryMode = 'pm25';
    document.getElementById('togglePm25Trajectory').classList.add('active');
    document.getElementById('toggleAqiTrajectory').classList.remove('active');
    renderPredictionChart();
  });

  document.getElementById('toggleConfidenceBand').addEventListener('click', () => {
    showConfidenceBand = !showConfidenceBand;
    document.getElementById('toggleConfidenceBand').classList.toggle('active', showConfidenceBand);
    renderPredictionChart();
  });

  // Chatbot Dock Controls
  const chatDock = document.getElementById('chatDock');
  const chatFloatingBtn = document.getElementById('chatFloatingBtn');
  const btnChatMinimize = document.getElementById('btnChatMinimize');

  btnChatMinimize.addEventListener('click', () => {
    chatDock.classList.add('minimized');
    chatFloatingBtn.style.display = 'flex';
  });

  chatFloatingBtn.addEventListener('click', () => {
    chatDock.classList.remove('minimized');
    chatFloatingBtn.style.display = 'none';
    document.getElementById('chatInput').focus();
  });

  // Chat Form Submit
  document.getElementById('chatInputForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (msg) {
      sendChatMessage(msg);
      input.value = '';
    }
  });

  // City-specific assistant suggestions (delegated so they can be refreshed).
  document.getElementById('chatSuggestions').addEventListener('click', event => {
    const chip = event.target.closest('.chip-btn');
    if (chip?.dataset.query) sendChatMessage(chip.dataset.query);
  });

  // Copy Abstract Button
  document.getElementById('btnCopyAbstract').addEventListener('click', () => {
    if (currentTelemetry && currentTelemetry.researchAbstract) {
      const abs = currentTelemetry.researchAbstract;
      const text = `${abs.title}\n\nAuthors: ${abs.authors.join(', ')}\n\nAbstract:\n${abs.abstractText}\n\nKeywords: ${abs.keywords.join(', ')}`;
      navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('btnCopyAbstract');
        btn.textContent = '✅ Copied to Clipboard';
        setTimeout(() => { btn.textContent = '📋 Copy Abstract'; }, 2000);
      });
    }
  });

  // PDF Dossier Export
  document.getElementById('btnExportPdf').addEventListener('click', exportResearchDossierPdf);

  // Train ML Model Button with Live Progress Polling
  const btnTrain = document.getElementById('btnTrainModel');
  if (btnTrain) {
    let pollingInterval = null;
    let trainingRequested = false;

    const pollTrainingProgress = (city) => {
      if (pollingInterval) clearInterval(pollingInterval);
      pollingInterval = setInterval(async () => {
        try {
          const res = await fetch(`/api/ml/status?city=${encodeURIComponent(city)}`);
          const data = await res.json();
          if (data.trainingFailed) {
            btnTrain.innerHTML = data.isTrained ? `<span>⚠</span> ${city} retrain failed` : '<span>⚠</span> Training failed';
            btnTrain.title = data.progress?.detail || 'Training failed';
            btnTrain.disabled = false;
            clearInterval(pollingInterval);
            pollingInterval = null;
            trainingRequested = false;
            return;
          }
          if (data.isTrained) {
            btnTrain.innerHTML = `<span>✅</span> ${city} ML Ready`;
            btnTrain.disabled = false;
            clearInterval(pollingInterval);
            pollingInterval = null;
            if (trainingRequested) {
              trainingRequested = false;
              loadAnalysisData({ lat: currentTelemetry?.meta?.coordinates?.lat, lon: currentTelemetry?.meta?.coordinates?.lon }, true);
            }
            setTimeout(() => {
              btnTrain.innerHTML = '<span>🧠</span> Train ML Model';
            }, 6000);
            return;
          }
          if (data.progress && data.progress.percent > 0) {
            btnTrain.innerHTML = `<span>⏳</span> ${data.progress.percent}% (${data.progress.phase})`;
            btnTrain.title = data.progress.detail || '';
          }
        } catch (_) {}
      }, 3000);
    };

    btnTrain.addEventListener('click', async () => {
      const city = currentLocation ? currentLocation.city : 'Delhi';
      if (!confirm(`Refresh Open-Meteo historical data and train AQI and PM2.5 XGBoost models for ${city}?\n\nThis runs in the background on the server.`)) return;
      btnTrain.disabled = true;
      trainingRequested = true;
      btnTrain.innerHTML = '<span>⏳</span> Starting...';
      try {
        const res = await fetch('/api/ml/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ city, lat: currentTelemetry?.meta?.coordinates?.lat, lon: currentTelemetry?.meta?.coordinates?.lon })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Training request was rejected.');
        pollTrainingProgress(city);
      } catch (err) {
        alert('Could not start training: ' + err.message);
        btnTrain.disabled = false;
        trainingRequested = false;
        btnTrain.innerHTML = '<span>🧠</span> Train ML Model';
      }
    });

    // Initial check on load and city change
    window.checkCurrentCityML = () => {
      const city = currentLocation ? currentLocation.city : 'Delhi';
      pollTrainingProgress(city);
    };
    setTimeout(window.checkCurrentCityML, 1500);
  }
}



// ==========================================================================
// Theme Management
// ==========================================================================
function toggleTheme() {
  const isLight = document.body.classList.toggle('theme-light');
  document.body.classList.toggle('theme-dark', !isLight);
  document.getElementById('themeIcon').textContent = isLight ? '🌙' : '☀️';
  // Redraw charts with adapted themes
  renderAllCharts();
}

// ==========================================================================
// Data Fetching & Pipeline Execution
// ==========================================================================
async function loadAnalysisData(params = {}, force = false) {
  showLoading(true, 'Loading AQI and weather...', 'Fetching city-level data from Open-Meteo...');
  hideError();

  try {
    let url = '/api/analysis?';
    if (params.city) url += `city=${encodeURIComponent(params.city)}&`;
    if (params.lat != null && params.lon != null) url += `lat=${params.lat}&lon=${params.lon}&`;
    if (force) url += 'force=true';

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Analysis Pipeline HTTP ${res.status}`);
    const data = await res.json();

    currentTelemetry = data;
    const previousCity = currentLocation?.city;
    if (previousCity && previousCity !== data.meta.location.city) {
      conversationHistory = [];
      document.querySelectorAll('#chatMessagesContainer .chat-msg:not(.system)').forEach(message => message.remove());
    }
    currentLocation = data.meta.location;
    updateChatSuggestions(currentLocation, data.predictionsData);

    updateHeroRibbon(data);
    updateChemicalViews(data);
    updateModelAndValidation(data);
    updateHealthAndOperations(data);
    renderAllCharts();
    if (window.checkCurrentCityML) window.checkCurrentCityML();

  } catch (err) {
    console.error('Data loading error:', err);
    showError(`Failed to load AQI and weather data: ${err.message}`);
  } finally {
    showLoading(false);
  }
}

function performSearch(query) {
  const q = query.trim();
  loadAnalysisData({ city: q });
}

function detectUserLocation() {
  if (!navigator.geolocation) {
    showError('Geolocation is not supported by your browser.');
    return;
  }

  showLoading(true, 'Getting Your Location...', 'Loading AQI and weather for your coordinates...');
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      loadAnalysisData({ lat, lon });
    },
    (err) => {
      showLoading(false);
      showError(`Unable to retrieve GPS coordinates: ${err.message}`);
    }
  );
}

// Check Ollama Service
async function checkOllamaStatus() {
  try {
    const res = await fetch('/api/ollama/status');
    const data = await res.json();
    const badge = document.getElementById('chatEngineBadge');
    if (data.available && data.models.length > 0) {
      badge.textContent = `Ollama (${data.activeModel})`;
      badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      badge.style.color = '#10b981';
    } else {
      badge.textContent = 'Scientific Engine (Active)';
      badge.style.borderColor = 'rgba(6, 182, 212, 0.4)';
      badge.style.color = '#06b6d4';
    }
  } catch (_) {}
}

// ==========================================================================
// UI View Updaters
// ==========================================================================
function updateHeroRibbon(data) {
  const location = data.meta.location;
  const current = data.chemistryData.current;
  const weather = current.meteorology;
  const safeSet = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };
  const safeClass = (id, className) => { const el = document.getElementById(id); if (el) el.className = className; };

  safeSet('activeCityName', location.city);
  safeSet('activeRegionName', [location.state, location.country].filter(Boolean).join(', ') || 'Selected city');
  safeSet('badgeWeatherCondition', weather.weatherCondition || 'Weather');
  safeSet('valTemp', weather.surfaceTemperatureC ?? '?');
  safeSet('valWindMs', weather.windSpeedMs ?? '?');
  safeSet('valHumidity', weather.relativeHumidityPercent ?? '?');
  safeSet('valWeatherCondition', weather.weatherCondition || 'Unavailable');
  safeSet('valPressure', weather.surfacePressureHpa ?? '?');
  safeSet('valCloudCover', weather.cloudCoverPercent ?? '?');
  safeSet('valWeatherSourceStatus', 'Open-Meteo current weather');

  safeSet('valUsAqi', current.usAqi ?? '?');
  safeSet('badgeAqiCategory', current.category.label);
  safeClass('badgeAqiCategory', `badge-aqi aqi-${current.category.code}`);
  safeSet('valIndiaAqi', current.indiaAqi ?? '?');
  safeSet('valPrimaryPollutant', current.primaryPollutant || 'Unavailable');

  const pm25 = current.pollutants.pm25;
  safeSet('valPm25Conc', pm25.value ?? '?');
  safeSet('badgeWhoExceed', pm25.ratio == null ? '?' : `${pm25.ratio}x WHO guideline`);
  safeSet('valPm10Conc', `${current.pollutants.pm10.value ?? '?'} ug/m3`);
  safeSet('valNo2Conc', `${current.pollutants.no2.value ?? '?'} ug/m3`);
  safeSet('valAod', current.pollutants.aod.value ?? 'Unavailable');
}

function updateChemicalViews(data) {
  const chem = data.chemistryData.current;
  const tbody = document.getElementById('chemicalTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const specs = [
    { name: 'Fine Particulate Matter', formula: 'PM2.5', key: 'pm25', toxicity: 'Penetrates deep alveoli and enters capillary bloodstream' },
    { name: 'Coarse Particulate Matter', formula: 'PM10', key: 'pm10', toxicity: 'Upper bronchial deposition, tracheobronchial irritation' },
    { name: 'Nitrogen Dioxide', formula: 'NO₂', key: 'no2', toxicity: 'Airway hyperresponsiveness, secondary nitrate aerosol precursor' },
    { name: 'Ground-Level Ozone', formula: 'O₃', key: 'o3', toxicity: 'Potent oxidant, induces chest tightness and reduces FEV1' },
    { name: 'Sulphur Dioxide', formula: 'SO₂', key: 'so2', toxicity: 'Bronchoconstriction in asthmatics, acid rain precursor' },
    { name: 'Carbon Monoxide', formula: 'CO', key: 'co', toxicity: 'Carboxyhemoglobin formation, reduces tissue oxygen delivery' }
  ];

  specs.forEach(s => {
    const pol = chem.pollutants[s.key];
    if (!pol) return;
    const row = document.createElement('tr');
    const isPrimary = chem.primaryPollutant.toUpperCase().includes(s.formula.toUpperCase());
    row.innerHTML = `
      <td><strong>${s.name}</strong> ${isPrimary ? '<span class="badge-research">PRIMARY</span>' : ''}</td>
      <td><code>${s.formula}</code></td>
      <td><strong>${pol.value}</strong> ${pol.unit}</td>
      <td>${pol.whoLimit} ${pol.unit}</td>
      <td><span class="${pol.ratio > 1 ? 'text-danger' : 'text-success'}"><strong>${pol.ratio}x</strong></span></td>
      <td>${pol.subIndex || '—'}</td>
      <td style="font-size: 0.8rem; color: var(--text-muted);">${s.toxicity}</td>
    `;
    tbody.appendChild(row);
  });
}

function updateModelAndValidation(data) {
  const pred = data.predictionsData;
  const metrics = pred.modelValidationMetrics || {};
  const abs = data.researchAbstract;
  const safeSet = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };

  const trained = pred.modelStatus === 'ready';
  safeSet('valModelR2', trained ? metrics.rSquaredScore : '—');
  safeSet('valModelRmse', trained ? metrics.rmse + ' AQI' : '—');
  safeSet('valModelMae', trained ? metrics.mae + ' AQI' : '?');
  safeSet('valModelDirAcc', trained ? metrics.directionAccuracyPercent + '%' : '—');
  safeSet('valModelArch', trained ? metrics.architecture : 'Model not trained');
  safeSet('valModelMethod', trained ? metrics.validationNote : 'No forecast until trained');
  safeSet('valModelStatus', trained ? 'Ready' : 'Unavailable');
  safeSet('badgeModelStatus', trained ? 'Ready' : 'No model');
  safeSet('valModelStatusDetail', trained ? 'Validated model for ' + pred.city : (pred.error || 'Train a model for this city'));
  safeSet('modelCityVal', trained ? pred.city : '—');
  safeSet('modelTrainRowsVal', trained ? Number(metrics.trainPairCount).toLocaleString() : '—');
  safeSet('modelTestRowsVal', trained ? Number(metrics.testPairCount).toLocaleString() : '—');
  safeSet('modelPersistenceMaeVal', trained ? metrics.persistenceMae + ' AQI' : '—');
  safeSet('modelTrainedAtVal', trained && metrics.trainedAt ? new Date(metrics.trainedAt).toLocaleString() : '—');
  const horizonContainer = document.getElementById('modelHorizonMetrics');
  if (horizonContainer) {
    horizonContainer.innerHTML = '';
    if (trained && metrics.horizonMetrics) {
      const table = document.createElement('table');
      table.className = 'model-horizon-table';
      table.innerHTML = '<thead><tr><th>Hours</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>';
      const body = document.createElement('tbody');
      Object.entries(metrics.horizonMetrics).forEach(([horizon, value]) => {
        const row = document.createElement('tr');
        [horizon, value.mae, value.rmse, value.r2].forEach(cell => {
          const td = document.createElement('td');
          td.textContent = String(cell ?? '—');
          row.appendChild(td);
        });
        body.appendChild(row);
      });
      table.appendChild(body);
      horizonContainer.appendChild(table);
    } else {
      horizonContainer.textContent = 'No trained model metrics available.';
    }
  }

  if (abs) {
    safeSet('abstractPaperTitle', abs.title);
    safeSet('abstractPaperAuthors', 'Authors: ' + abs.authors.join(', '));
    safeSet('abstractPaperBody', abs.abstractText);
    const kwContainer = document.getElementById('abstractPaperKeywords');
    if (kwContainer) {
      kwContainer.innerHTML = '';
      abs.keywords.forEach(k => {
        const span = document.createElement('span');
        span.className = 'keyword-tag';
        span.textContent = k;
        kwContainer.appendChild(span);
      });
    }
  }

  const dailyContainer = document.getElementById('dailyForecastCards');
  if (dailyContainer && pred.dailySummary) {
    dailyContainer.innerHTML = '';
    pred.dailySummary.forEach(d => {
      const card = document.createElement('div');
      card.className = 'daily-card';
      const aqiValue = d.avgAqi == null ? 'ML unavailable' : d.avgAqi;
      const aqiColor = d.avgAqi == null ? '#94a3b8' : getAqiHex(d.avgAqi);
      card.innerHTML = `
        <div class="daily-day-name">${d.dayName} <span class="daily-full-name">(${d.dateStr})</span></div>
        <div class="daily-wx-pill"><span class="daily-wx-icon">${d.weatherIcon || '⛅'}</span><span class="daily-wx-label">${d.weatherCondition || 'Weather unavailable'}</span></div>
        <div class="daily-temp-row"><span class="daily-temp-high">${d.tempMax ?? '—'}°C</span><span class="daily-temp-divider">/</span><span class="daily-temp-low">${d.tempMin ?? '—'}°C</span></div>
        <div class="daily-rain-pill ${d.totalPrecipitationMm > 5 ? 'rain-high' : ''}">💧 ${d.totalPrecipitationMm ?? '—'} mm rain</div>
        <div style="font-size:0.72rem;color:var(--text-secondary);margin-top:4px">Humidity ${d.avgHumidityPercent ?? '—'}% · Wind max ${d.maxWindSpeedMs ?? '—'} m/s</div>
        <div class="daily-aqi-row"><div class="daily-aqi-num" style="color: ${aqiColor}">${aqiValue}</div><span class="badge-aqi aqi-${d.categoryClass || 'moderate'}" style="font-size: 0.68rem; padding: 2px 6px;">${d.aqiCategory || 'No model'}</span></div>
        <div class="daily-tip-text">${d.healthTip || (trained ? 'Forecast from the trained Open-Meteo AQI model.' : (pred.error || 'Train a city model to generate AQI predictions.'))}</div>
      `;
      dailyContainer.appendChild(card);
    });
  }
}
function updateHealthAndOperations(data) {
  const health = data.healthAndOpsData;
  if (!health) return;
  const setText = (id, value) => { const element = document.getElementById(id); if (element) element.textContent = value; };
  setText('healthSeverityBadge', `AQI ${health.currentAqi}: ${health.advisoryTone}`);
  setText('healthPathologyText', health.guidanceNote);
  const container = document.getElementById('cohortsGrid');
  if (!container) return;
  container.innerHTML = '';
  (health.vulnerableCohorts || []).forEach(group => {
    const card = document.createElement('div');
    card.className = 'cohort-card';
    const heading = document.createElement('strong');
    heading.textContent = `${group.group} ? ${group.status}`;
    const advice = document.createElement('p');
    advice.textContent = group.directive;
    card.append(heading, advice);
    container.appendChild(card);
  });
}

function getAqiHex(aqi) {
  if (aqi <= 50) return '#10b981';
  if (aqi <= 100) return '#fbbf24';
  if (aqi <= 150) return '#fb923c';
  if (aqi <= 200) return '#f87171';
  if (aqi <= 300) return '#c084fc';
  return '#fda4af';
}

// ==========================================================================
// Chart.js Rendering Functions
// ==========================================================================
function renderAllCharts() {
  if (!currentTelemetry) return;
  renderPredictionChart();
  renderSpeciationRadarChart();
  renderWhoComparisonBarChart();
  renderFeatureImportanceChart();
}

// 1. 72-Hour Predictive Chart with Confidence Bounds
function renderPredictionChart() {
  const ctx = document.getElementById('predictionChart');
  if (!ctx) return;

  if (chartInstances.prediction) chartInstances.prediction.destroy();

  const series = currentTelemetry.predictionsData.hourlyTrajectory || [];
  const labels = series.map((s, i) => i % 12 === 0 ? s.displayTime : '');
  const isAqi = activeTrajectoryMode === 'aqi';

  const mainData = isAqi ? series.map(s => s.predictedAqi) : series.map(s => s.predictedPm25);
  const upperData = isAqi ? series.map(s => s.upperBound95) : series.map(s => s.pm25UpperBound95);
  const lowerData = isAqi ? series.map(s => s.lowerBound95) : series.map(s => s.pm25LowerBound95);

  const datasets = [
    {
      label: isAqi ? 'Predicted AQI (XGBoost)' : 'Predicted PM2.5 (XGBoost)',
      data: mainData,
      borderColor: isAqi ? '#06b6d4' : '#f97316',
      backgroundColor: isAqi ? 'rgba(6, 182, 212, 0.15)' : 'rgba(249, 115, 22, 0.15)',
      fill: false,
      tension: 0.35,
      borderWidth: 2.5,
      pointRadius: 2,
      pointHoverRadius: 6
    }
  ];

  if (showConfidenceBand) {
    datasets.unshift({
      label: '95% Confidence Upper Bound',
      data: upperData,
      borderColor: 'transparent',
      backgroundColor: isAqi ? 'rgba(6, 182, 212, 0.08)' : 'rgba(249, 115, 22, 0.08)',
      fill: '+1',
      pointRadius: 0,
      tension: 0.35
    });
    datasets.push({
      label: '95% Confidence Lower Bound',
      data: lowerData,
      borderColor: 'transparent',
      backgroundColor: 'transparent',
      fill: false,
      pointRadius: 0,
      tension: 0.35
    });
  }

  chartInstances.prediction = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const idx = items[0].dataIndex;
              return `${series[idx].displayTime} (+${series[idx].horizonHours}h)`;
            },
            afterBody: (items) => {
              const idx = items[0].dataIndex;
              return `Source: trained XGBoost model | Horizon: ${series[idx].horizonHours}h`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: false,
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8' }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', maxRotation: 0 }
        }
      }
    }
  });
}

// 2. Chemical Speciation Radar Chart
function renderSpeciationRadarChart() {
  const ctx = document.getElementById('speciationRadarChart');
  if (!ctx) return;
  if (chartInstances.radar) chartInstances.radar.destroy();

  const pol = currentTelemetry.chemistryData.current.pollutants;
  const labels = ['PM2.5', 'PM10', 'NO₂', 'SO₂', 'CO (×0.01)', 'O₃'];
  const values = [
    pol.pm25.value,
    pol.pm10.value,
    pol.no2.value,
    pol.so2.value,
    Math.round(pol.co.value * 0.01),
    pol.o3.value
  ];

  chartInstances.radar = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        label: 'Aerosol & Criteria Gases (µg/m³)',
        data: values,
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.25)',
        borderWidth: 2,
        pointBackgroundColor: '#06b6d4',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
          pointLabels: { color: '#94a3b8', font: { size: 11, family: 'Inter' } },
          ticks: { backdropColor: 'transparent', color: '#64748b' }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// 3. WHO Guidelines Comparison Bar Chart
function renderWhoComparisonBarChart() {
  const ctx = document.getElementById('whoComparisonBarChart');
  if (!ctx) return;
  if (chartInstances.who) chartInstances.who.destroy();

  const pol = currentTelemetry.chemistryData.current.pollutants;
  const labels = ['PM2.5', 'PM10', 'NO₂', 'SO₂', 'O₃'];
  const observed = [pol.pm25.value, pol.pm10.value, pol.no2.value, pol.so2.value, pol.o3.value];
  const whoLimits = [pol.pm25.whoLimit, pol.pm10.whoLimit, pol.no2.whoLimit, pol.so2.whoLimit, pol.o3.whoLimit];

  chartInstances.who = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Observed Concentration',
          data: observed,
          backgroundColor: '#06b6d4',
          borderRadius: 4
        },
        {
          label: 'WHO 2021 Guideline',
          data: whoLimits,
          backgroundColor: '#ef4444',
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8' }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#94a3b8', font: { size: 11 } }
        }
      }
    }
  });
}

// 4. Feature Importance Horizontal Bar Chart
function renderFeatureImportanceChart() {
  const ctx = document.getElementById('featureImportanceChart');
  if (!ctx) return;
  if (chartInstances.features) chartInstances.features.destroy();

  const feat = currentTelemetry.predictionsData.modelValidationMetrics.featureImportance || [];
  const labels = feat.map(f => f.feature);
  const data = feat.map(f => Math.round(f.importance * 100));

  chartInstances.features = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Relative Predictive Weight (%)',
        data,
        backgroundColor: [
          '#06b6d4',
          '#3b82f6',
          '#8b5cf6',
          '#f59e0b',
          '#10b981',
          '#64748b'
        ],
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8' },
          max: 40
        },
        y: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ==========================================================================
// AI Scientific Assistant Dialogue
// ==========================================================================
function updateChatSuggestions(location, predictions) {
  const container = document.getElementById('chatSuggestions');
  if (!container) return;
  const city = location?.city || 'this city';
  const forecastLabel = predictions?.modelStatus === 'ready' ? `7-day AQI & weather in ${city}` : `7-day weather in ${city}`;
  const forecastQuestion = predictions?.modelStatus === 'ready'
    ? `Show me the AQI and weather outlook for ${city} for the next 7 days.`
    : `Show me the 7-day weather outlook for ${city}, and tell me whether an AQI model is available.`;
  const suggestions = [
    { label: `Current AQI in ${city}`, query: `What is the current AQI in ${city}, and what does its category mean?` },
    { label: `Main pollutant in ${city}`, query: `Which pollutant is contributing most to the AQI in ${city}?` },
    { label: forecastLabel, query: forecastQuestion },
    { label: `Health guidance for ${city}`, query: `What general precautions should people in ${city} consider based on the current AQI?` },
  ];
  container.replaceChildren(...suggestions.map(item => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip-btn';
    button.dataset.query = item.query;
    button.textContent = item.label;
    return button;
  }));
}

async function sendChatMessage(promptText) {
  const container = document.getElementById('chatMessagesContainer');

  // Append user message
  const userMsg = document.createElement('div');
  userMsg.className = 'chat-msg user';
  userMsg.innerHTML = `
    <div class="msg-avatar">👤</div>
    <div class="msg-bubble">${escapeHtml(promptText)}</div>
  `;
  container.appendChild(userMsg);
  container.scrollTop = container.scrollHeight;

  // Append thinking placeholder
  const botMsg = document.createElement('div');
  botMsg.className = 'chat-msg bot';
  botMsg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble">
      <span class="status-dot"></span> <em>Preparing an answer from current AQI and weather data...</em>
    </div>
  `;
  container.appendChild(botMsg);
  container.scrollTop = container.scrollHeight;

  const contextSummary = currentTelemetry ? {
    location: currentTelemetry.meta.location,
    chemistryData: { current: currentTelemetry.chemistryData.current, dailyWeather: currentTelemetry.chemistryData.dailyWeather },
    predictionsData: currentTelemetry.predictionsData,
    healthAndOpsData: currentTelemetry.healthAndOpsData,
    researchAbstract: currentTelemetry.researchAbstract
  } : null;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: promptText,
        history: conversationHistory,
        context: contextSummary
      })
    });

    if (!res.ok) throw new Error(`Assistant HTTP ${res.status}`);
    const data = await res.json();

    const formattedReply = renderMarkdown(data.reply);
    botMsg.querySelector('.msg-bubble').innerHTML = formattedReply;
    conversationHistory.push({ role: 'user', content: promptText });
    conversationHistory.push({ role: 'assistant', content: data.reply });
  } catch (err) {
    botMsg.querySelector('.msg-bubble').innerHTML = `
      <span class="text-danger">⚠️ Error communicating with assistant: ${err.message}</span>
    `;
  }

  container.scrollTop = container.scrollHeight;
}

// Simple Markdown Renderer
function renderMarkdown(text) {
  if (!text) return '';
  const lines = escapeHtml(String(text)).replace(/\r/g, '').split('\n');
  const output = [];
  let paragraph = [];
  let listOpen = false;
  const inline = value => value
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
  const flushParagraph = () => {
    if (paragraph.length) output.push(`<p>${inline(paragraph.join(' '))}</p>`);
    paragraph = [];
  };
  const closeList = () => { if (listOpen) output.push('</ul>'); listOpen = false; };
  const cells = line => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) { flushParagraph(); closeList(); continue; }
    if (line.startsWith('|') && i + 1 < lines.length && /^\|?\s*:?-{3,}/.test(lines[i + 1].trim())) {
      flushParagraph(); closeList();
      const headers = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(cells(lines[i++]));
      i--;
      output.push(`<div class="chat-table-wrap"><table class="chat-response-table"><thead><tr>${headers.map(cell => `<th>${inline(cell)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${headers.map((_, index) => `<td>${inline(row[index] || 'Unavailable')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph(); closeList();
      const level = Math.min(heading[1].length + 2, 5);
      output.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    const item = line.match(/^[-*]\s+(.+)$/);
    if (item) {
      flushParagraph();
      if (!listOpen) { output.push('<ul>'); listOpen = true; }
      output.push(`<li>${inline(item[1])}</li>`);
      continue;
    }
    paragraph.push(line);
  }
  flushParagraph();
  closeList();
  return output.join('');
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ==========================================================================
// Academic Research Dossier PDF Generator
// ==========================================================================
async function exportResearchDossierPdf() {
  if (!currentTelemetry) {
    alert('Please wait for city data to finish loading.');
    return;
  }
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' });
  const location = currentTelemetry.meta.location;
  const current = currentTelemetry.chemistryData.current;
  const predictions = currentTelemetry.predictionsData;
  const abstract = currentTelemetry.researchAbstract;
  const weather = current.meteorology || {};
  const metrics = predictions.modelValidationMetrics || {};
  const daily = predictions.dailySummary || [];
  const weatherMetrics = metrics.weatherValidationMetrics || {};
  const colors = {
    navy: [11, 24, 43], ink: [24, 39, 57], muted: [92, 108, 124],
    cyan: [6, 182, 212], pale: [239, 247, 250], line: [220, 229, 235], white: [255, 255, 255],
  };
  const value = (input, digits = 1) => {
    if (input == null || input === '' || !Number.isFinite(Number(input))) return 'N/A';
    const n = Number(input);
    return Number.isInteger(n) ? String(n) : n.toFixed(digits);
  };
  const aqiColor = aqi => {
    if (aqi <= 50) return [16, 150, 105];
    if (aqi <= 100) return [190, 139, 0];
    if (aqi <= 150) return [220, 105, 35];
    if (aqi <= 200) return [200, 55, 55];
    if (aqi <= 300) return [130, 70, 150];
    return [105, 45, 65];
  };
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const generatedAt = new Date();
  const generatedLabel = generatedAt.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  doc.setProperties({
    title: `Air Quality and Weather Report - ${location.city || 'Selected City'}`,
    subject: 'Current conditions and machine-learning forecast',
    author: 'Air Quality and Weather Research Project',
  });

  const drawHeader = (eyebrow, title, subtitle) => {
    doc.setFillColor(...colors.navy);
    doc.rect(0, 0, pageWidth, 29, 'F');
    doc.setFillColor(...colors.cyan);
    doc.rect(0, 28, pageWidth, 1.5, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(...colors.cyan);
    doc.text(eyebrow.toUpperCase(), 15, 9);
    doc.setFontSize(17);
    doc.setTextColor(...colors.white);
    doc.text(title, 15, 18);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(205, 219, 232);
    doc.text(subtitle, 15, 25);
    doc.setTextColor(...colors.white);
    doc.setFontSize(8);
    doc.text(generatedLabel, pageWidth - 15, 18, { align: 'right' });
  };
  const drawFooter = pageNumber => {
    doc.setDrawColor(...colors.line);
    doc.line(15, pageHeight - 10, pageWidth - 15, pageHeight - 10);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7);
    doc.setTextColor(...colors.muted);
    const footerNote = predictions.modelStatus === 'ready'
      ? 'Current observations and historical targets: Open-Meteo; forecasts are city-specific ML estimates, not official alerts.'
      : 'Current observations and weather outlook: Open-Meteo; AQI ML forecast is unavailable for this city.';
    doc.text(footerNote, 15, pageHeight - 5);
    doc.text(`${pageNumber} / 2`, pageWidth - 15, pageHeight - 5, { align: 'right' });
  };
  const sectionTitle = (title, x, y, note = '') => {
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(...colors.ink);
    doc.text(title, x, y);
    if (note) {
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7.5);
      doc.setTextColor(...colors.muted);
      doc.text(note, pageWidth - 15, y, { align: 'right' });
    }
  };
  const drawCard = (x, y, w, h, label, main, sub, accent = colors.cyan) => {
    doc.setFillColor(249, 252, 253);
    doc.setDrawColor(...colors.line);
    doc.roundedRect(x, y, w, h, 2, 2, 'FD');
    doc.setFillColor(...accent);
    doc.roundedRect(x, y, 2.2, h, 1, 1, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(7.5);
    doc.setTextColor(...colors.muted);
    doc.text(label.toUpperCase(), x + 7, y + 7);
    doc.setFontSize(16);
    doc.setTextColor(...colors.ink);
    doc.text(String(main), x + 7, y + 16);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(...colors.muted);
    doc.text(String(sub || ''), x + 7, y + h - 4);
  };

  // Page 1: a concise city snapshot and the complete forecast table.
  drawHeader('City report', 'Air Quality & Weather', `${location.city || 'Selected location'}${location.country ? `, ${location.country}` : ''}  |  Current conditions and 7-day outlook`);
  const cardY = 36;
  const cardGap = 4;
  const cardWidth = (pageWidth - 30 - cardGap * 3) / 4;
  const aqiValueNow = current.usAqi;
  const cardData = [
    ['US AQI', value(aqiValueNow, 0), current.category?.label || 'Category unavailable', aqiColor(Number(aqiValueNow) || 0)],
    ['PM2.5', `${value(current.pollutants?.pm25?.value)} ug/m3`, `PM10 ${value(current.pollutants?.pm10?.value)} ug/m3`, [6, 182, 212]],
    ['Temperature', `${value(weather.surfaceTemperatureC)} deg C`, weather.weatherCondition || 'Current weather', [235, 145, 55]],
    ['Humidity & wind', `${value(weather.relativeHumidityPercent, 0)}%`, `Wind ${value(weather.windSpeedMs)} m/s`, [79, 124, 172]],
  ];
  cardData.forEach((card, i) => drawCard(15 + i * (cardWidth + cardGap), cardY, cardWidth, 27, ...card));

  sectionTitle('City summary', 15, 73, `Primary pollutant: ${current.primaryPollutant || 'N/A'}  |  India NAQI: ${value(current.indiaAqi, 0)}`);
  const summaryText = abstract?.abstractText || 'Current AQI and weather readings are shown above. Daily values below are generated from the selected city forecast.';
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(...colors.ink);
  const allSummaryLines = doc.splitTextToSize(summaryText, pageWidth - 30);
  const summaryLines = allSummaryLines.length > 3 ? [...allSummaryLines.slice(0, 2), '...'] : allSummaryLines;
  doc.text(summaryLines, 15, 80, { lineHeightFactor: 1.3 });

  sectionTitle('7-day forecast', 15, 94, predictions.modelStatus === 'ready' ? 'City-specific ML predictions' : 'Weather outlook available; AQI ML model unavailable');
  const columns = [
    { title: 'DAY', width: 24 }, { title: 'DATE', width: 25 }, { title: 'AVG AQI', width: 27 },
    { title: 'PM2.5', width: 27 }, { title: 'CONDITIONS', width: 38 }, { title: 'HIGH / LOW', width: 31 },
    { title: predictions.modelStatus === 'ready' ? 'RAIN' : 'RAIN CHANCE', width: 27 }, { title: 'HUMIDITY', width: 28 }, { title: 'MAX WIND', width: 25 },
  ];
  let tableX = 15;
  let tableY = 99;
  const tableWidth = columns.reduce((sum, column) => sum + column.width, 0);
  doc.setFillColor(...colors.navy);
  doc.roundedRect(tableX, tableY, tableWidth, 8, 1.5, 1.5, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7);
  doc.setTextColor(...colors.white);
  let cursorX = tableX;
  columns.forEach(column => {
    doc.text(column.title, cursorX + 2, tableY + 5.2);
    cursorX += column.width;
  });
  tableY += 8;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7.5);
  (daily.slice(0, 7)).forEach((day, index) => {
    if (index % 2 === 0) {
      doc.setFillColor(...colors.pale);
      doc.rect(tableX, tableY, tableWidth, 8.5, 'F');
    }
    const values = [
      day.dayName || 'N/A', day.dateStr || 'N/A', value(day.avgAqi, 0), value(day.avgPm25),
      day.weatherCondition || 'N/A', `${value(day.tempMax, 0)} / ${value(day.tempMin, 0)} deg C`,
      predictions.modelStatus === 'ready' ? `${value(day.totalPrecipitationMm)} mm` : `${value(day.rainProb, 0)}%`,
      `${value(day.avgHumidityPercent, 0)}%`, `${value(day.maxWindSpeedMs)} m/s`,
    ];
    cursorX = tableX;
    values.forEach((cell, cellIndex) => {
      if (cellIndex === 2 && day.avgAqi != null) doc.setTextColor(...aqiColor(Number(day.avgAqi)));
      else doc.setTextColor(...colors.ink);
      const maxWidth = columns[cellIndex].width - 4;
      doc.text(doc.splitTextToSize(String(cell), maxWidth).slice(0, 1), cursorX + 2, tableY + 5.5);
      cursorX += columns[cellIndex].width;
    });
    tableY += 8.5;
  });
  if (!daily.length) {
    doc.setTextColor(...colors.muted);
    doc.text('No forecast rows are available for this city.', tableX + 2, tableY + 6);
  }
  drawFooter(1);

  // Page 2: validation scores, methodology, and data interpretation notes.
  doc.addPage('a4', 'landscape');
  drawHeader('Model notes', 'Validation & data', `${location.city || 'Selected location'}  |  Model version ${metrics.architecture ? '5' : 'unavailable'}`);
  sectionTitle('AQI model holdout performance', 15, 39, 'Regression errors are measured in AQI points');
  const scoreY = 45;
  const scoreGap = 4;
  const scoreWidth = (pageWidth - 30 - scoreGap * 3) / 4;
  const scoreCards = [
    ['AQI MAE', `${value(metrics.mae)} points`, 'Mean absolute error', [6, 182, 212]],
    ['AQI RMSE', `${value(metrics.rmse)} points`, 'Penalizes larger misses', [79, 124, 172]],
    ['R-squared', value(metrics.rSquaredScore, 3), 'Holdout fit score', [119, 106, 180]],
    ['Direction match', `${value(metrics.directionAccuracyPercent, 1)}%`, 'Correct rise / fall direction', [235, 145, 55]],
  ];
  scoreCards.forEach((card, i) => drawCard(15 + i * (scoreWidth + scoreGap), scoreY, scoreWidth, 27, ...card));

  sectionTitle('Weather model holdout errors', 15, 82, 'MAE and persistence comparison use each variable native unit');
  const weatherLabels = {
    temperature_2m: ['Temperature', 'deg C'], relative_humidity_2m: ['Humidity', 'percentage points'],
    wind_speed_10m: ['Wind speed', 'm/s'], precipitation: ['Precipitation', 'mm/hour'],
    cloud_cover: ['Cloud cover', 'percentage points'], surface_pressure: ['Surface pressure', 'hPa'],
  };
  const wxCols = [
    { title: 'VARIABLE', width: 75 }, { title: 'MAE', width: 52 },
    { title: 'RMSE', width: 52 }, { title: 'R-SQUARED', width: 42 }, { title: 'PERSISTENCE MAE', width: 46 },
  ];
  let wxX = 15;
  let wxY = 87;
  const wxWidth = wxCols.reduce((sum, column) => sum + column.width, 0);
  doc.setFillColor(...colors.navy);
  doc.roundedRect(wxX, wxY, wxWidth, 8, 1.5, 1.5, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7);
  doc.setTextColor(...colors.white);
  let wxCursor = wxX;
  wxCols.forEach(column => { doc.text(column.title, wxCursor + 2, wxY + 5.2); wxCursor += column.width; });
  wxY += 8;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  for (const [target, [label, unit]] of Object.entries(weatherLabels)) {
    const score = weatherMetrics[target];
    if (!score) continue;
    if (Object.keys(weatherMetrics).indexOf(target) % 2 === 0) {
      doc.setFillColor(...colors.pale);
      doc.rect(wxX, wxY, wxWidth, 8.5, 'F');
    }
    const row = [label, `${value(score.mae)} ${unit}`, `${value(score.rmse)} ${unit}`, value(score.r2, 3), `${value(score.persistence_mae)} ${unit}`];
    wxCursor = wxX;
    row.forEach((cell, i) => { doc.setTextColor(...colors.ink); doc.text(doc.splitTextToSize(cell, wxCols[i].width - 4).slice(0, 1), wxCursor + 2, wxY + 5.5); wxCursor += wxCols[i].width; });
    wxY += 8.5;
  }

  const notesY = Math.max(wxY + 5, 157);
  sectionTitle('Method and interpretation', 15, notesY);
  const methodLines = [
    `Model: ${metrics.architecture || 'No compatible trained model is available for this city.'}`,
    `Evaluation: ${metrics.validationNote || 'Model holdout statistics are unavailable.'} AQI persistence MAE: ${value(metrics.persistenceMae)} points.`,
    `Training data: ${metrics.targetSource || 'Open-Meteo Air Quality API'} and Open-Meteo Historical Weather API. Trained: ${metrics.trainedAt ? new Date(metrics.trainedAt).toLocaleString() : 'N/A'}. AQI test samples: ${value(metrics.testPairCount, 0)}.`,
    'These results measure agreement with the historical data source. They do not guarantee future performance or replace official measurements and health guidance.',
  ];
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7.5);
  doc.setTextColor(...colors.ink);
  const methodText = doc.splitTextToSize(methodLines.join('\n'), pageWidth - 30);
  doc.text(methodText, 15, notesY + 6, { lineHeightFactor: 1.25 });
  drawFooter(2);

  const safeCity = String(location.city || 'City').replace(/[^a-z0-9_-]+/gi, '_');
  doc.save(`Air_Quality_Weather_Report_${safeCity}_${generatedAt.toISOString().slice(0, 10)}.pdf`);
}

// Helpers
function showLoading(show, title = '', sub = '') {
  const overlay = document.getElementById('loadingOverlay');
  if (show) {
    if (title) document.querySelector('.loader-title').textContent = title;
    if (sub) document.getElementById('loaderSubtext').textContent = sub;
    overlay.style.display = 'flex';
  } else {
    overlay.style.display = 'none';
  }
}

function showError(msg) {
  const banner = document.getElementById('errorBanner');
  document.getElementById('errorMessage').textContent = msg;
  banner.style.display = 'flex';
}

function hideError() {
  document.getElementById('errorBanner').style.display = 'none';
}
