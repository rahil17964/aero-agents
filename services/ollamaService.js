const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://127.0.0.1:11434';

export async function getOllamaStatus() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const res = await fetch(`${OLLAMA_HOST}/api/tags`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
    const data = await res.json();
    const models = (data.models || []).map(model => model.name);
    return { available: true, host: OLLAMA_HOST, models, activeModel: models.find(name => name.includes('qwen2.5:3b')) || models[0] || 'qwen2.5:3b' };
  } catch (err) {
    return { available: false, host: OLLAMA_HOST, models: [], error: err.message, activeModel: 'Built-in AQI and Weather Assistant' };
  }
}

export async function generateAssistantResponse({ prompt, conversationHistory = [], currentLocationContext = null, requestedModel = null }) {
  const status = await getOllamaStatus();
  if (status.available && requestedModel !== 'scientific-engine') {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 45000);
      const res = await fetch(`${OLLAMA_HOST}/api/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({
          model: requestedModel || status.activeModel,
          messages: [{ role: 'system', content: constructSystemPrompt(currentLocationContext) }, ...conversationHistory.slice(-6), { role: 'user', content: prompt }],
          stream: false, options: { temperature: 0.4, top_p: 0.9, num_predict: 600 },
        }),
      });
      clearTimeout(timeout);
      if (res.ok) {
        const data = await res.json();
        const reply = data.message?.content || data.response || '';
        if (reply.trim()) return { reply, engine: `Local Ollama (${requestedModel || status.activeModel})`, confidence: 0.95, assistantSource: 'AQI and Weather Assistant' };
      }
    } catch (err) {
      console.warn('Ollama query error, using built-in assistant:', err.message);
    }
  }
  return { reply: generateBuiltInReply(prompt, currentLocationContext), engine: 'Built-in AQI and Weather Assistant', confidence: 0.98, assistantSource: 'AQI and Weather Assistant' };
}

function constructSystemPrompt(ctx) {
  if (!ctx) return 'You are an assistant for an air-quality and weather research project. Give clear, concise answers with headings and bullet points when useful. Show AQI as a number plus category, pollutant concentrations in micrograms per cubic meter, temperature in degrees Celsius, wind in meters per second, and humidity as a percentage. Refer to the selected city by name. Use only supplied data for current values; do not invent measurements or claim correlation proves a cause.';
  const location = ctx.location || {};
  const current = ctx.chemistryData?.current || {};
  const weather = current.meteorology || {};
  const pollutants = current.pollutants || {};
  const predictions = ctx.predictionsData || {};
  const forecast = predictions.modelStatus === 'ready'
    ? (predictions.dailySummary || []).map(day => `${day.dateStr}: ML weather ${day.weatherCondition}, temperature ${formatValue(day.tempMax)} / ${formatValue(day.tempMin)} degrees C, total rain ${formatValue(day.totalPrecipitationMm)} mm, humidity ${formatValue(day.avgHumidityPercent, 0)} percent, max wind ${formatValue(day.maxWindSpeedMs)} m/s, predicted AQI ${formatValue(day.avgAqi, 0)} (${day.aqiCategory || 'unavailable'}), PM2.5 ${formatValue(day.avgPm25)} micrograms per cubic meter`).join('\n')
    : (ctx.chemistryData?.dailyWeather || []).map(day => `${day.date}: weather ${day.weatherCondition}, temperature ${formatValue(day.tempMax)} / ${formatValue(day.tempMin)} degrees C, rain chance ${formatValue(day.rainProb, 0)} percent`).join('\n');
  const metrics = predictions.modelValidationMetrics || {};
  return `You are an assistant for an air-quality and weather research project. Answer clearly and concisely. Use Markdown headings, short bullet lists, or a small table when useful. Format measurements with labels and units: pollutant concentrations in micrograms per cubic meter, temperature in degrees Celsius, wind in meters per second, humidity in percent, and AQI with its category. Always use the selected city name. For forecast questions, use only the forecast rows below. If no AQI model is trained, state that AQI predictions are unavailable and report only the weather outlook. Do not invent data or claim current correlations prove causes.\n\nCURRENT DATA\nCity: ${location.city || 'Unknown'}\nUS AQI: ${current.usAqi ?? 'unavailable'} (${current.category?.label || 'unavailable'})\nIndia NAQI: ${current.indiaAqi ?? 'unavailable'} (${current.indiaCategory?.label || 'unavailable'})\nMain pollutant: ${current.primaryPollutant || 'unavailable'}\nPM2.5: ${formatValue(pollutants.pm25?.value)} ${concentrationUnit}\nPM10: ${formatValue(pollutants.pm10?.value)} ${concentrationUnit}\nNO2: ${formatValue(pollutants.no2?.value)} micrograms per cubic meter\nWeather: ${weather.weatherCondition || 'unavailable'}\nTemperature: ${formatValue(weather.surfaceTemperatureC)} degrees C\nWind: ${formatValue(weather.windSpeedMs)} meters per second\nHumidity: ${formatValue(weather.relativeHumidityPercent)} percent\nPressure: ${formatValue(weather.surfacePressureHpa)} hPa\nCloud cover: ${formatValue(weather.cloudCoverPercent)} percent\nAQI model status: ${predictions.modelStatus || 'unavailable'}\nAQI holdout MAE: ${formatValue(metrics.mae)} AQI points\nAQI holdout R-squared: ${formatValue(metrics.rSquaredScore, 2)}\n\nFORECAST ROWS\n${forecast || 'No forecast rows available.'}`;
}

function formatValue(value, digits = 1) {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return 'unavailable';
  const number = Number(value);
  return Number.isInteger(number) ? String(number) : number.toFixed(digits);
}

function generateBuiltInReply(prompt, ctx) {
  if (!ctx) return 'Select a city to load its current AQI and weather data.';
  const question = String(prompt || '').toLowerCase();
  const location = ctx.location || {};
  const city = location.city || 'selected city';
  const current = ctx.chemistryData?.current || {};
  const pollutants = current.pollutants || {};
  const weather = current.meteorology || {};
  const predictions = ctx.predictionsData || {};
  const health = ctx.healthAndOpsData || {};
  const celsius = String.fromCharCode(176) + 'C';
  const concentrationUnit = String.fromCharCode(181) + 'g/m' + String.fromCharCode(179);
  const weatherLine = `- **Conditions:** ${weather.weatherCondition || 'unavailable'}\n- **Temperature:** ${formatValue(weather.surfaceTemperatureC)} ${celsius}\n- **Wind:** ${formatValue(weather.windSpeedMs)} m/s\n- **Humidity:** ${formatValue(weather.relativeHumidityPercent)}%`;

  if (question.includes('forecast') || question.includes('predict') || question.includes('tomorrow') || question.includes('next week') || question.includes('7 day')) {
    if (predictions.modelStatus !== 'ready') {
      const days = (ctx.chemistryData?.dailyWeather || []).map(day => `- **${day.date}:** ${day.weatherCondition}, ${formatValue(day.tempMax)} / ${formatValue(day.tempMin)} ${celsius}, rain chance ${formatValue(day.rainProb, 0)}% (Open-Meteo fallback; this city has no trained ML model)`).join('\n');
      return `### 7-day outlook for ${city}\n\n**Weather forecast**\n${days || 'Weather forecast is unavailable.'}\n\nAQI forecast is unavailable because a trained model has not been loaded for this city.`;
    }
    const rows = (predictions.dailySummary || []).map(day => `| ${day.dayName} (${day.dateStr}) | ${day.weatherCondition || 'unavailable'} | ${formatValue(day.tempMax)} / ${formatValue(day.tempMin)} ${celsius} | ${formatValue(day.totalPrecipitationMm)} mm | ${formatValue(day.avgHumidityPercent, 0)}% | ${formatValue(day.maxWindSpeedMs)} m/s | ${formatValue(day.avgAqi, 0)} | ${day.aqiCategory || 'unavailable'} |`).join('\n');
    return `### 7-day AQI and weather outlook for ${city}\n\n| Day | Weather | High / Low | Rain | Humidity | Max wind | Predicted AQI | Category |\n|---|---|---:|---:|---:|---:|---:|---|\n${rows || '| Unavailable | unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | unavailable |'}\n\nAQI and weather values are ML predictions; performance varies by forecast horizon.`;
  }

  if (question.includes('health') || question.includes('precaution') || question.includes('mask') || question.includes('asthma') || question.includes('children') || question.includes('safe')) {
    return `### AQI guidance for ${city}\n\n- **Current AQI:** ${formatValue(current.usAqi, 0)} (${current.category?.label || 'unavailable'})\n${(health.vulnerableCohorts || []).map(group => `- **${group.group}:** ${group.directive}`).join('\n')}\n\n${health.guidanceNote || 'Follow local public-health guidance.'}`;
  }

  if (question.includes('why') || question.includes('cause') || question.includes('spik')) {
    return `### Current air quality in ${city}\n\n- **US AQI:** ${formatValue(current.usAqi, 0)} (${current.category?.label || 'unavailable'})\n- **Main pollutant:** ${current.primaryPollutant || 'unavailable'}\n- **PM2.5:** ${formatValue(pollutants.pm25?.value)} ${concentrationUnit}\n\nThese values describe current conditions but do not identify the cause of pollution. Studying causes requires comparing time-aligned pollution, weather, and emissions data.`;
  }

  return `### Current conditions in ${city}\n\n**Air quality**\n- **US AQI:** ${formatValue(current.usAqi, 0)} (${current.category?.label || 'category unavailable'})\n- **India NAQI:** ${formatValue(current.indiaAqi, 0)} (${current.indiaCategory?.label || 'category unavailable'})\n- **Main pollutant:** ${current.primaryPollutant || 'unavailable'}\n- **PM2.5:** ${formatValue(pollutants.pm25?.value)} ${concentrationUnit}\n- **PM10:** ${formatValue(pollutants.pm10?.value)} ${concentrationUnit}\n\n**Weather**\n${weatherLine}`;
}
