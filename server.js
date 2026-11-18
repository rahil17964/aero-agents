// Express API Server for Air Quality and Weather Platform
import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

import { geocodeLocation, listPopularCities } from './services/cityLocator.js';
import { runAnalysisPipeline } from './services/analysisService.js';
import { getOllamaStatus, generateAssistantResponse } from './services/ollamaService.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Cache for recent pipeline runs to keep latency low
const analysisCache = new Map();
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

// 1. Search cities
app.get('/api/locations/search', async (req, res) => {
  try {
    const q = req.query.q || '';
    if (!q.trim()) return res.json({ results: listPopularCities() });
    res.json({ results: [await geocodeLocation(q)] });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 2. Current location by coordinates
app.get('/api/locations/current', (req, res) => {
  const lat = parseFloat(req.query.lat);
  const lon = parseFloat(req.query.lon);

  if (isNaN(lat) || isNaN(lon)) {
    return res.status(400).json({ error: 'Valid latitude and longitude required' });
  }

  res.json({
    location: { city: 'My Location', name: 'My Location', lat, lon },
    coordinates: { lat, lon }
  });
});

// 3. AQI and weather analysis
app.get('/api/analysis', async (req, res) => {
  try {
    let { city, lat, lon } = req.query;
    let targetLocation = null;
    let targetCoords = null;

    if (city) {
      targetLocation = await geocodeLocation(city);
      targetCoords = { lat: targetLocation.lat, lon: targetLocation.lon };
    }

    if (!targetLocation && lat && lon) {
      targetCoords = { lat: parseFloat(lat), lon: parseFloat(lon) };
      if (!Number.isFinite(targetCoords.lat) || !Number.isFinite(targetCoords.lon)) {
        return res.status(400).json({ error: 'Valid latitude and longitude required.' });
      }
      targetLocation = { city: 'My Location', name: 'My Location', ...targetCoords };
    }

    if (!targetLocation) {
      targetLocation = await geocodeLocation('Delhi');
      targetCoords = { lat: targetLocation.lat, lon: targetLocation.lon };
    }

    if (!targetCoords) {
      targetCoords = { lat: targetLocation.lat, lon: targetLocation.lon };
    }

    // Check cache
    const cacheKey = `${targetLocation.city}_${Math.round(targetCoords.lat * 100)}_${Math.round(targetCoords.lon * 100)}`;
    const cached = analysisCache.get(cacheKey);
    const now = Date.now();

    if (cached && (now - cached.timestamp < CACHE_TTL_MS) && !req.query.force) {
      return res.json({ ...cached.data, fromCache: true });
    }

    // Run the AQI and weather analysis
    const result = await runAnalysisPipeline(targetLocation, targetCoords);

    // Save to cache
    analysisCache.set(cacheKey, { timestamp: now, data: result });

    res.json(result);
  } catch (err) {
    console.error('Analysis pipeline error:', err);
    res.status(500).json({ error: err.message });
  }
});

// 4. AQI and Weather Assistant Endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { prompt, history, context, requestedModel } = req.body;
    if (!prompt || !prompt.trim()) {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    const response = await generateAssistantResponse({
      prompt,
      conversationHistory: history || [],
      currentLocationContext: context || null,
      requestedModel
    });

    res.json(response);
  } catch (err) {
    console.error('Chat assistant error:', err);
    res.status(500).json({ error: err.message });
  }
});

// 5. Ollama Status & Available Models
app.get('/api/ollama/status', async (req, res) => {
  try {
    const status = await getOllamaStatus();
    res.json(status);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 6. Health Check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ONLINE',
    version: '1.0.0',
    service: 'Air Quality and Weather Platform',
    timestamp: new Date().toISOString()
  });
});

// 7. ML Model Training Status & Trigger
const ML_DIR = path.join(__dirname, 'ml');
const MODEL_DIR = path.join(ML_DIR, 'models');
import fs from 'fs';
import { triggerTraining } from './services/mlPredictionService.js';

app.get('/api/ml/status', (req, res) => {
  try {
    const city = (req.query.city || 'Delhi').trim().toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_-]/g, '');
    const aqiModelPath = path.join(MODEL_DIR, `${city}_us_aqi_xgb.joblib`);
    const pm25ModelPath = path.join(MODEL_DIR, `${city}_pm2_5_xgb.joblib`);
    const weatherTargets = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation', 'cloud_cover', 'surface_pressure'];
    const weatherArtifactsReady = weatherTargets.every(target =>
      fs.existsSync(path.join(MODEL_DIR, `${city}_${target}_xgb.joblib`)) &&
      fs.existsSync(path.join(MODEL_DIR, `${city}_${target}_metadata.json`))
    );
    const metaPath = path.join(MODEL_DIR, `${city}_metadata.json`);
    const progressPath = path.join(ML_DIR, 'data', `${city}_progress.json`);
    const artifactsReady = fs.existsSync(aqiModelPath) && fs.existsSync(pm25ModelPath) && weatherArtifactsReady;
    
    let meta = null;
    if (artifactsReady && fs.existsSync(metaPath)) {
      meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    }
    const modelReady = artifactsReady && meta?.model_version === 5;

    let progress = null;
    if (fs.existsSync(progressPath)) {
      try {
        progress = JSON.parse(fs.readFileSync(progressPath, 'utf8'));
      } catch (_) {}
    }

    const trainingActive = progress && progress.phase !== 'complete' && !progress.phase?.includes?.('failed') && progress.phase !== 'idle';
    if (progress?.phase === 'complete' && !modelReady) progress = null;
    res.json({
      city,
      isTrained: modelReady && !trainingActive,
      isTraining: Boolean(trainingActive),
      trainingFailed: Boolean(progress?.phase?.includes?.('failed')),
      progress: progress || (modelReady ? { percent: 100, phase: 'complete', detail: 'Model ready' } : { percent: 0, phase: 'idle', detail: 'Not trained' }),
      metadata: meta
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


app.post('/api/ml/train', (req, res) => {
  try {
    const city = (req.body.city || 'Delhi').trim();
    if (!/^[\p{L} .'-]+$/u.test(city)) return res.status(400).json({ error: 'Invalid city name.' });
    const lat = Number(req.body.lat);
    const lon = Number(req.body.lon);
    const coordinates = Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : null;
    triggerTraining(city, coordinates);
    res.json({ success: true, message: `Training triggered in background for ${city}` });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Fallback to index.html for SPA
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`ðŸš€ Air Quality and Weather Platform Server Running on:`);
  console.log(`   http://localhost:${PORT}`);
  console.log(`   AQI and weather data: Open-Meteo`);
  console.log(`====================================================`);
});
