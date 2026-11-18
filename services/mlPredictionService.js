/** Runs the trained Python AQI/weather models and manages background model training. */
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ML_DIR = path.join(__dirname, '..', 'ml');
const PYTHON = process.env.PYTHON_EXECUTABLE || (process.platform === 'win32' ? 'py' : 'python3');
const PYTHON_ARGS = process.platform === 'win32' && !process.env.PYTHON_EXECUTABLE ? ['-3.14', '-X', 'utf8'] : [];

function pythonScript(script, args = [], { timeoutMs = 120000, input = null, onOutput = null } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON, [...PYTHON_ARGS, script, ...args], {
      cwd: ML_DIR,
      env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
    });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      proc.kill();
      reject(new Error(`Python process timed out after ${timeoutMs / 1000}s`));
    }, timeoutMs);
    proc.stdout.on('data', chunk => {
      const value = chunk.toString();
      stdout += value;
      onOutput?.('stdout', value);
    });
    proc.stderr.on('data', chunk => {
      const value = chunk.toString();
      stderr += value;
      onOutput?.('stderr', value);
    });
    proc.on('error', error => { clearTimeout(timer); reject(error); });
    proc.on('close', code => {
      clearTimeout(timer);
      if (code === 0) resolve(stdout);
      else reject(new Error(stderr.trim() || `Python exited with status ${code}`));
    });
    if (input != null) proc.stdin.end(JSON.stringify(input));
    else proc.stdin.end();
  });
}

export async function runMLPrediction(aqi, city = 'Delhi') {
  const series = aqi.hourly || {};
  const asUtcIso = value => {
    const text = String(value);
    return /(?:Z|[+-]\d{2}:?\d{2})$/i.test(text) ? text : `${text}Z`;
  };
  const currentTime = asUtcIso(aqi.current.time || new Date().toISOString());
  const cutoff = new Date(currentTime).getTime();
  const history = (series.time || []).map((time, i) => ({
    time: asUtcIso(time),
    usAqi: series.usAqi?.[i],
    pm25: series.pm25?.[i],
  })).filter(row => new Date(row.time).getTime() <= cutoff).slice(-169);
  const weatherHistory = (series.time || []).map((time, i) => ({
    time: asUtcIso(time),
    temperature_2m: series.temperature?.[i],
    relative_humidity_2m: series.relativeHumidity?.[i],
    wind_speed_10m: series.windSpeed?.[i],
    precipitation: series.precipitation?.[i],
    cloud_cover: series.cloudCover?.[i],
    surface_pressure: series.surfacePressure?.[i],
  })).filter(row => new Date(row.time).getTime() <= cutoff).slice(-169);

  const m = aqi.current.meteorology;
  const payload = {
    city,
    currentTime,
    history,
    weatherHistory,
    weatherCurrent: {
      temperature_2m: m.surfaceTemperatureC,
      relative_humidity_2m: m.relativeHumidityPercent,
      wind_speed_10m: m.windSpeedMs,
      wind_direction_10m: m.windDirectionDeg,
      surface_pressure: m.surfacePressureHpa,
      precipitation: m.precipitationMm,
      cloud_cover: m.cloudCoverPercent,
      boundary_layer_height: m.boundaryLayerHeightMeters,
      shortwave_radiation: m.shortwaveRadiation,
    },
    dailyWeather: aqi.dailyWeather || [],
  };

  try {
    const output = await pythonScript(path.join(ML_DIR, 'predict.py'), [], { input: payload, timeoutMs: 30000 });
    const result = JSON.parse(output.trim());
    if (result.error) {
      console.warn(`[ML] ${result.error}`);
      return { modelStatus: 'unavailable', error: result.error };
    }
    return result;
  } catch (error) {
    console.warn('[ML] Prediction unavailable:', error.message);
    return { modelStatus: 'unavailable', error: `ML prediction failed: ${error.message}` };
  }
}

export function triggerTraining(city = 'Delhi', coordinates = null) {
  const progressPath = path.join(ML_DIR, 'data', `${city.toLowerCase().replace(/\s+/g, '_')}_progress.json`);
  fs.mkdirSync(path.dirname(progressPath), { recursive: true });
  fs.writeFileSync(progressPath, JSON.stringify({
    phase: 'starting', percent: 1, detail: 'Starting historical data refresh and model training',
    updated_at: new Date().toISOString(),
  }));
  console.log(`[ML] Starting data refresh and model training for ${city}...`);
  const output = (stream, value) => process[stream === 'stderr' ? 'stderr' : 'stdout'].write(`[ML ${stream}] ${value}`);
  const fetchArgs = [city];
  if (Number.isFinite(coordinates?.lat) && Number.isFinite(coordinates?.lon)) {
    fetchArgs.push('--lat', String(coordinates.lat), '--lon', String(coordinates.lon));
  }
  pythonScript(path.join(ML_DIR, 'fetch_data.py'), fetchArgs, { timeoutMs: 45 * 60 * 1000, onOutput: output })
    .then(() => {
      console.log(`[ML] Data refresh complete; training AQI, PM2.5, and weather models for ${city}.`);
      return pythonScript(path.join(ML_DIR, 'train_model.py'), [city], { timeoutMs: 45 * 60 * 1000, onOutput: output });
    })
    .then(() => console.log(`[ML] Training complete for ${city}.`))
    .catch(error => {
      fs.writeFileSync(progressPath, JSON.stringify({
        phase: 'training_failed', percent: 0, detail: error.message,
        updated_at: new Date().toISOString(),
      }));
      console.error(`[ML] Training failed for ${city}: ${error.message}`);
    });
}
