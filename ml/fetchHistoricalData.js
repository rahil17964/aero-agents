/**
 * fetchHistoricalData.js
 * Fetches up to 10 years of hourly AQI + weather data from Open-Meteo archives.
 * Weather (ERA5) goes back to 1940; AQI satellite data available from ~2015.
 * Output: JSON file with merged hourly records ready for ML training.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Configuration ───────────────────────────────────────────────────────────
const CITY_CONFIGS = {
  Mumbai:    { lat: 19.0760, lon: 72.8777 },
  Delhi:     { lat: 28.6139, lon: 77.2090 },
  Bangalore: { lat: 12.9716, lon: 77.5946 },
  Kolkata:   { lat: 22.5726, lon: 88.3639 },
  Chennai:   { lat: 13.0827, lon: 80.2707 },
};

const AQI_START_YEAR  = 2015; // Earliest reliable CAMS satellite AQI data
const WEATHER_EXTRA_YEARS = 10; // Extra years of weather-only data for patterns
const WEATHER_START_YEAR  = AQI_START_YEAR - WEATHER_EXTRA_YEARS; // 2005
const END_DATE = new Date();

const WEATHER_VARS = [
  'temperature_2m',
  'relative_humidity_2m',
  'wind_speed_10m',
  'wind_direction_10m',
  'surface_pressure',
  'precipitation',
  'cloud_cover',
  'apparent_temperature',
  'dew_point_2m',
  'vapour_pressure_deficit',
  'boundary_layer_height',
].join(',');

const AQI_VARS = [
  'pm2_5',
  'pm10',
  'nitrogen_dioxide',
  'ozone',
  'carbon_monoxide',
  'sulphur_dioxide',
  'us_aqi',
].join(',');

// ─── Helpers ─────────────────────────────────────────────────────────────────
function dateStr(d) {
  return d.toISOString().split('T')[0];
}

async function fetchWithRetry(url, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      if (i === retries - 1) throw err;
      await new Promise(r => setTimeout(r, 2000 * (i + 1)));
    }
  }
}

/** Fetch one year of hourly weather data (ERA5 reanalysis) */
async function fetchWeatherYear(lat, lon, year) {
  const start = `${year}-01-01`;
  const end   = year === END_DATE.getFullYear()
    ? dateStr(new Date(Date.now() - 86400000)) // yesterday
    : `${year}-12-31`;

  const url = `https://archive-api.open-meteo.com/v1/archive?`
    + `latitude=${lat}&longitude=${lon}`
    + `&start_date=${start}&end_date=${end}`
    + `&hourly=${WEATHER_VARS}`
    + `&timezone=auto`;

  console.log(`  [Weather] Fetching ${year}...`);
  const data = await fetchWithRetry(url);
  return data.hourly;
}

/** Fetch one year of hourly AQI data (CAMS) */
async function fetchAqiYear(lat, lon, year) {
  const start = `${year}-01-01`;
  const end   = year === END_DATE.getFullYear()
    ? dateStr(new Date(Date.now() - 86400000))
    : `${year}-12-31`;

  const url = `https://air-quality-api.open-meteo.com/v1/air-quality?`
    + `latitude=${lat}&longitude=${lon}`
    + `&start_date=${start}&end_date=${end}`
    + `&hourly=${AQI_VARS}`
    + `&timezone=auto`;

  console.log(`  [AQI]     Fetching ${year}...`);
  const data = await fetchWithRetry(url);
  return data.hourly;
}

/** Merge weather + AQI hourly records by timestamp */
function mergeHourly(weather, aqi) {
  const aqiMap = new Map();
  if (aqi && aqi.time) {
    aqi.time.forEach((t, i) => {
      aqiMap.set(t, {
        pm25:  aqi.pm2_5?.[i]       ?? null,
        pm10:  aqi.pm10?.[i]        ?? null,
        no2:   aqi.nitrogen_dioxide?.[i] ?? null,
        o3:    aqi.ozone?.[i]       ?? null,
        co:    aqi.carbon_monoxide?.[i]  ?? null,
        so2:   aqi.sulphur_dioxide?.[i]  ?? null,
        usAqi: aqi.us_aqi?.[i]      ?? null,
      });
    });
  }

  const rows = [];
  weather.time.forEach((t, i) => {
    const aqiRow = aqiMap.get(t) || {};
    const dt = new Date(t);
    rows.push({
      time:       t,
      hour:       dt.getHours(),
      dayOfWeek:  dt.getDay(),
      month:      dt.getMonth() + 1,
      dayOfYear:  Math.floor((dt - new Date(dt.getFullYear(), 0, 0)) / 86400000),
      // Weather features
      temp:       weather.temperature_2m?.[i]         ?? null,
      humidity:   weather.relative_humidity_2m?.[i]   ?? null,
      windSpeed:  weather.wind_speed_10m?.[i]         ?? null,
      windDir:    weather.wind_direction_10m?.[i]     ?? null,
      pressure:   weather.surface_pressure?.[i]       ?? null,
      precip:     weather.precipitation?.[i]          ?? null,
      cloudCover: weather.cloud_cover?.[i]            ?? null,
      apparentTemp: weather.apparent_temperature?.[i] ?? null,
      dewPoint:   weather.dew_point_2m?.[i]           ?? null,
      vpd:        weather.vapour_pressure_deficit?.[i]?? null,
      pbl:        weather.boundary_layer_height?.[i]  ?? null,
      // AQI features (may be null for pre-2015 data)
      ...aqiRow
    });
  });
  return rows;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export async function fetchAllHistoricalData(cityName = 'Delhi', outputDir = path.join(__dirname, 'data')) {
  const cfg = CITY_CONFIGS[cityName];
  if (!cfg) throw new Error(`Unknown city: ${cityName}. Options: ${Object.keys(CITY_CONFIGS).join(', ')}`);

  fs.mkdirSync(outputDir, { recursive: true });
  const outFile = path.join(outputDir, `${cityName.toLowerCase()}_historical.json`);

  console.log(`\n📡 Fetching historical data for ${cityName} (${cfg.lat}, ${cfg.lon})`);
  console.log(`   Weather: ${WEATHER_START_YEAR}–present | AQI: ${AQI_START_YEAR}–present\n`);

  const allRows = [];
  const currentYear = END_DATE.getFullYear();

  for (let year = WEATHER_START_YEAR; year <= currentYear; year++) {
    const weather = await fetchWeatherYear(cfg.lat, cfg.lon, year);

    let aqi = null;
    if (year >= AQI_START_YEAR) {
      try {
        aqi = await fetchAqiYear(cfg.lat, cfg.lon, year);
      } catch (e) {
        console.warn(`  [AQI] ${year} failed: ${e.message}`);
      }
    }

    const rows = mergeHourly(weather, aqi);
    allRows.push(...rows);
    console.log(`  ✓ ${year}: ${rows.length} hourly records (total: ${allRows.length})`);

    // Small pause to respect rate limits
    await new Promise(r => setTimeout(r, 500));
  }

  console.log(`\n✅ Total records: ${allRows.length}`);
  fs.writeFileSync(outFile, JSON.stringify({ city: cityName, coords: cfg, records: allRows }, null, 0));
  console.log(`💾 Saved to: ${outFile}`);
  return outFile;
}

// Run directly: node ml/fetchHistoricalData.js Delhi
if (process.argv[2]) {
  fetchAllHistoricalData(process.argv[2]).catch(console.error);
}
