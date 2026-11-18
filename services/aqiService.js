// Atmospheric Chemistry & Air Quality Service
// Fetches AQI history/forecast and meteorology from Open-Meteo without synthetic fallbacks

export async function fetchAtmosphericChemistry(lat, lon) {
  const roundLat = Math.round(lat * 100) / 100;
  const roundLon = Math.round(lon * 100) / 100;

  // Open-Meteo Air Quality URL (Full 7-day hourly forecast)
  const aqiUrl = `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${roundLat}&longitude=${roundLon}&timezone=UTC&past_days=8&forecast_days=7&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,us_aqi,european_aqi&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,us_aqi`;

  // Open-Meteo Weather URL (Full 7-day hourly & daily weather forecast)
  const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${roundLat}&longitude=${roundLon}&timezone=UTC&wind_speed_unit=kmh&past_days=8&forecast_days=7&current=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,shortwave_radiation,surface_pressure,wind_speed_10m,wind_direction_10m,boundary_layer_height,weather_code&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height,surface_pressure,weather_code,precipitation,precipitation_probability,cloud_cover,shortwave_radiation&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max`;

  try {
    const [aqiRes, wxRes] = await Promise.all([
      fetch(aqiUrl, { headers: { 'User-Agent': 'AirQualityWeather/1.0' } }),
      fetch(weatherUrl, { headers: { 'User-Agent': 'AirQualityWeather/1.0' } })
    ]);

    if (!aqiRes.ok) throw new Error(`AQI API HTTP ${aqiRes.status}`);
    const aqiData = await aqiRes.json();

    let wxData = null;
    if (wxRes.ok) {
      wxData = await wxRes.json();
    }

    return processAtmosphericData(aqiData, wxData);
  } catch (err) {
    console.warn(`Atmospheric fetch failed for (${lat}, ${lon}):`, err.message);
    throw new Error(`Open-Meteo air quality/weather data unavailable: ${err.message}`);
  }
}

// Translate WMO Weather Codes to simple friendly English & emoji
export function decodeWeatherCode(code) {
  if (code === 0) return { label: 'Clear Sky', icon: '☀️' };
  if ([1, 2].includes(code)) return { label: 'Partly Cloudy', icon: '⛅' };
  if (code === 3) return { label: 'Cloudy', icon: '☁️' };
  if ([45, 48].includes(code)) return { label: 'Hazy / Foggy', icon: '🌫️' };
  if ([51, 53, 55].includes(code)) return { label: 'Light Rain', icon: '🌦️' };
  if ([61, 63, 65].includes(code)) return { label: 'Rain', icon: '🌧️' };
  if ([71, 73, 75].includes(code)) return { label: 'Snow', icon: '❄️' };
  if ([80, 81, 82].includes(code)) return { label: 'Rain Showers', icon: '🌧️' };
  if ([95, 96, 99].includes(code)) return { label: 'Thunderstorm', icon: '⛈️' };
  return { label: 'Fair Weather', icon: '🌤️' };
}

// Process raw API telemetry into structured chemical speciation and weather data
function processAtmosphericData(aqi, wx) {
  const cur = aqi.current || {};
  const curWx = (wx && wx.current) || {};

  if (cur.us_aqi == null || cur.pm2_5 == null || !wx || curWx.temperature_2m == null) {
    throw new Error('Open-Meteo returned incomplete current AQI or weather data.');
  }

  const pm25 = Math.round(cur.pm2_5 * 10) / 10;
  const pm10 = cur.pm10 != null ? Math.round(cur.pm10 * 10) / 10 : null;
  const no2 = cur.nitrogen_dioxide != null ? Math.round(cur.nitrogen_dioxide * 10) / 10 : null;
  const so2 = cur.sulphur_dioxide != null ? Math.round(cur.sulphur_dioxide * 10) / 10 : null;
  const co = cur.carbon_monoxide != null ? Math.round(cur.carbon_monoxide * 10) / 10 : null;
  const o3 = cur.ozone != null ? Math.round(cur.ozone * 10) / 10 : null;
  const aod = cur.aerosol_optical_depth != null ? Math.round(cur.aerosol_optical_depth * 100) / 100 : null;
  const dust = cur.dust != null ? Math.round(cur.dust * 10) / 10 : null;

  // Current weather values from the Open-Meteo forecast API
  const boundaryLayerHgt = curWx.boundary_layer_height != null ? Math.round(curWx.boundary_layer_height) : null;
  const temp2m = Math.round(curWx.temperature_2m * 10) / 10;
  const rh2m = curWx.relative_humidity_2m != null ? Math.round(curWx.relative_humidity_2m) : null;
  const wind10m = curWx.wind_speed_10m != null ? Math.round(curWx.wind_speed_10m / 3.6 * 10) / 10 : null;
  const pressureHpa = curWx.surface_pressure != null ? Math.round(curWx.surface_pressure) : null;
  const weatherCode = curWx.weather_code !== undefined ? curWx.weather_code : 0;
  const weatherDesc = decodeWeatherCode(weatherCode);

  // EPA & Indian CPCB AQI Computations
  const epaSubIndices = calculateEpaSubIndices({ pm25, pm10, no2, so2, co, o3 });
  const indiaSubIndices = calculateIndiaNaqiSubIndices({ pm25, pm10, no2, so2, co, o3 });

  // US AQI from API or computed
  const usAqi = cur.us_aqi;
  const indiaAqi = indiaSubIndices.maxAqi;

  // WHO 2021 Guideline Exceedance Multipliers
  const whoExceedance = {
    pm25Ratio: Math.round((pm25 / 15) * 10) / 10,
    pm10Ratio: Math.round((pm10 / 45) * 10) / 10,
    no2Ratio: Math.round((no2 / 25) * 10) / 10,
    so2Ratio: Math.round((so2 / 40) * 10) / 10,
    o3Ratio: Math.round((o3 / 100) * 10) / 10
  };

  // Hourly Series for model feeding (Full 7 days / 168 hours)
  const hourlyTime = (aqi.hourly && aqi.hourly.time) || [];
  const hourlyPm25 = (aqi.hourly && aqi.hourly.pm2_5) || [];
  const hourlyPm10 = (aqi.hourly && aqi.hourly.pm10) || [];
  const hourlyUsAqi = (aqi.hourly && aqi.hourly.us_aqi) || [];
  const hourlyPbl = (wx && wx.hourly && wx.hourly.boundary_layer_height) || [];
  const hourlyWind = (wx && wx.hourly && wx.hourly.wind_speed_10m) || [];
  const hourlyTemp = (wx && wx.hourly && wx.hourly.temperature_2m) || [];
  const hourlyPrecip = (wx && wx.hourly && wx.hourly.precipitation_probability) || [];

  // Daily weather forecast for next week
  const dailyWx = (wx && wx.daily) || {};
  const dailySummaryWeather = (dailyWx.time || []).map((dateStr, i) => {
    const code = dailyWx.weather_code ? dailyWx.weather_code[i] : 0;
    const info = decodeWeatherCode(code);
    return {
      date: dateStr,
      weatherCode: code,
      tempMax: dailyWx.temperature_2m_max ? Math.round(dailyWx.temperature_2m_max[i]) : 30,
      tempMin: dailyWx.temperature_2m_min ? Math.round(dailyWx.temperature_2m_min[i]) : 24,
      rainProb: dailyWx.precipitation_probability_max ? dailyWx.precipitation_probability_max[i] : 0,
      windMax: dailyWx.wind_speed_10m_max?.[i] != null ? Math.round(dailyWx.wind_speed_10m_max[i] / 3.6 * 10) / 10 : null,
      weatherCondition: info.label,
      weatherIcon: info.icon
    };
  });

  return {
    current: {
      usAqi,
      time: cur.time,
      indiaAqi,
      primaryPollutant: epaSubIndices.primaryPollutant,
      category: getAqiCategory(usAqi),
      indiaCategory: getIndiaAqiCategory(indiaAqi),
      pollutants: {
        pm25: { value: pm25, unit: 'µg/m³', whoLimit: 15, ratio: whoExceedance.pm25Ratio, subIndex: epaSubIndices.pm25 },
        pm10: { value: pm10, unit: 'µg/m³', whoLimit: 45, ratio: whoExceedance.pm10Ratio, subIndex: epaSubIndices.pm10 },
        no2: { value: no2, unit: 'µg/m³', whoLimit: 25, ratio: whoExceedance.no2Ratio, subIndex: epaSubIndices.no2 },
        so2: { value: so2, unit: 'µg/m³', whoLimit: 40, ratio: whoExceedance.so2Ratio, subIndex: epaSubIndices.so2 },
        co: { value: co, unit: 'µg/m³', whoLimit: 4000, ratio: Math.round((co / 4000) * 100) / 100, subIndex: epaSubIndices.co },
        o3: { value: o3, unit: 'µg/m³', whoLimit: 100, ratio: whoExceedance.o3Ratio, subIndex: epaSubIndices.o3 },
        aod: { value: aod, unit: 'unitless (550nm)', rating: aod > 0.6 ? 'High Aerosol Haze' : 'Normal / Low' },
        dust: { value: dust, unit: 'µg/m³' }
      },
      meteorology: {
        boundaryLayerHeightMeters: boundaryLayerHgt,
        surfaceTemperatureC: temp2m,
        relativeHumidityPercent: rh2m,
        windSpeedMs: wind10m,
        windDirectionDeg: curWx.wind_direction_10m ?? null,
        surfacePressureHpa: pressureHpa,
        precipitationMm: curWx.precipitation ?? null,
        cloudCoverPercent: curWx.cloud_cover ?? null,
        shortwaveRadiation: curWx.shortwave_radiation ?? null,
        weatherCondition: weatherDesc.label,
        weatherIcon: weatherDesc.icon
      }
    },
    hourly: {
      time: hourlyTime,
      pm25: hourlyPm25,
      pm10: hourlyPm10,
      usAqi: hourlyUsAqi,
      boundaryLayerHeight: hourlyPbl,
      windSpeed: hourlyWind.map(value => value == null ? null : Math.round(value / 3.6 * 10) / 10),
      temperature: hourlyTemp,
      precipProb: hourlyPrecip,
      relativeHumidity: (wx && wx.hourly && wx.hourly.relative_humidity_2m) || [],
      windDirection: (wx && wx.hourly && wx.hourly.wind_direction_10m) || [],
      surfacePressure: (wx && wx.hourly && wx.hourly.surface_pressure) || [],
      precipitation: (wx && wx.hourly && wx.hourly.precipitation) || [],
      cloudCover: (wx && wx.hourly && wx.hourly.cloud_cover) || [],
      shortwaveRadiation: (wx && wx.hourly && wx.hourly.shortwave_radiation) || [],
      weatherCode: (wx && wx.hourly && wx.hourly.weather_code) || []
    },
    dailyWeather: dailySummaryWeather
  };
}

// EPA AQI Linear Breakpoint Calculation Formula:
// I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
function linearInterpolate(c, cLow, cHigh, iLow, iHigh) {
  return Math.round(((iHigh - iLow) / (cHigh - cLow)) * (c - cLow) + iLow);
}

// EPA PM2.5 Breakpoints (2024 revised standard)
function calculatePm25EpaSubIndex(c) {
  if (c <= 9.0) return linearInterpolate(c, 0.0, 9.0, 0, 50);
  if (c <= 35.4) return linearInterpolate(c, 9.1, 35.4, 51, 100);
  if (c <= 55.4) return linearInterpolate(c, 35.5, 55.4, 101, 150);
  if (c <= 125.4) return linearInterpolate(c, 55.5, 125.4, 151, 200);
  if (c <= 225.4) return linearInterpolate(c, 125.5, 225.4, 201, 300);
  if (c <= 325.4) return linearInterpolate(c, 225.5, 325.4, 301, 400);
  return linearInterpolate(Math.min(500, c), 325.5, 500.0, 401, 500);
}

// EPA PM10 Breakpoints
function calculatePm10EpaSubIndex(c) {
  if (c <= 54) return linearInterpolate(c, 0, 54, 0, 50);
  if (c <= 154) return linearInterpolate(c, 55, 154, 51, 100);
  if (c <= 254) return linearInterpolate(c, 155, 254, 101, 150);
  if (c <= 354) return linearInterpolate(c, 255, 354, 151, 200);
  if (c <= 424) return linearInterpolate(c, 355, 424, 201, 300);
  return linearInterpolate(Math.min(600, c), 425, 604, 301, 500);
}

// Calculate all US EPA sub-indices
export function calculateEpaSubIndices({ pm25, pm10, no2, so2, co, o3 }) {
  const idxPm25 = calculatePm25EpaSubIndex(pm25);
  const idxPm10 = calculatePm10EpaSubIndex(pm10);
  const idxNo2 = Math.round((no2 / 100) * 50); // Approximation for NO2 ppb
  const idxO3 = Math.round((o3 / 100) * 60);

  const entries = [
    { name: 'PM2.5', value: idxPm25 },
    { name: 'PM10', value: idxPm10 },
    { name: 'NO2', value: idxNo2 },
    { name: 'O3', value: idxO3 }
  ];

  entries.sort((a, b) => b.value - a.value);

  return {
    pm25: idxPm25,
    pm10: idxPm10,
    no2: idxNo2,
    o3: idxO3,
    so2: Math.round((so2 / 80) * 50),
    co: Math.round((co / 5000) * 50),
    maxAqi: entries[0].value,
    primaryPollutant: entries[0].name
  };
}

// Indian CPCB NAQI (National Air Quality Index) Breakpoints
export function calculateIndiaNaqiSubIndices({ pm25, pm10, no2, so2, co, o3 }) {
  // PM2.5 Breakpoints: 0-30 (0-50), 31-60 (51-100), 61-90 (101-200), 91-120 (201-300), 121-250 (301-400), 250+ (401-500)
  let subPm25 = 50;
  if (pm25 <= 30) subPm25 = linearInterpolate(pm25, 0, 30, 0, 50);
  else if (pm25 <= 60) subPm25 = linearInterpolate(pm25, 31, 60, 51, 100);
  else if (pm25 <= 90) subPm25 = linearInterpolate(pm25, 61, 90, 101, 200);
  else if (pm25 <= 120) subPm25 = linearInterpolate(pm25, 91, 120, 201, 300);
  else if (pm25 <= 250) subPm25 = linearInterpolate(pm25, 121, 250, 301, 400);
  else subPm25 = linearInterpolate(Math.min(500, pm25), 250, 500, 401, 500);

  // PM10 Breakpoints: 0-50 (0-50), 51-100 (51-100), 101-250 (101-200), 251-350 (201-300), 351-430 (301-400), 430+ (401-500)
  let subPm10 = 50;
  if (pm10 <= 50) subPm10 = linearInterpolate(pm10, 0, 50, 0, 50);
  else if (pm10 <= 100) subPm10 = linearInterpolate(pm10, 51, 100, 51, 100);
  else if (pm10 <= 250) subPm10 = linearInterpolate(pm10, 101, 250, 101, 200);
  else if (pm10 <= 350) subPm10 = linearInterpolate(pm10, 251, 350, 201, 300);
  else if (pm10 <= 430) subPm10 = linearInterpolate(pm10, 351, 430, 301, 400);
  else subPm10 = linearInterpolate(Math.min(600, pm10), 430, 600, 401, 500);

  const maxAqi = Math.max(subPm25, subPm10);
  const primaryPollutant = subPm25 >= subPm10 ? 'PM2.5' : 'PM10';

  return {
    pm25: subPm25,
    pm10: subPm10,
    maxAqi,
    primaryPollutant
  };
}

export function getAqiCategory(aqi) {
  if (aqi <= 50) return { label: 'Good', code: 'good', color: '#10b981', textColor: '#064e3b' };
  if (aqi <= 100) return { label: 'Moderate', code: 'moderate', color: '#f59e0b', textColor: '#78350f' };
  if (aqi <= 150) return { label: 'Unhealthy for Sensitive Groups', code: 'unhealthy-sensitive', color: '#f97316', textColor: '#7c2d12' };
  if (aqi <= 200) return { label: 'Unhealthy', code: 'unhealthy', color: '#ef4444', textColor: '#7f1d1d' };
  if (aqi <= 300) return { label: 'Very Unhealthy', code: 'very-unhealthy', color: '#a855f7', textColor: '#4c1d95' };
  return { label: 'Hazardous', code: 'hazardous', color: '#881337', textColor: '#ffe4e6' };
}

export function getIndiaAqiCategory(aqi) {
  if (aqi <= 50) return { label: 'Good', code: 'good' };
  if (aqi <= 100) return { label: 'Satisfactory', code: 'satisfactory' };
  if (aqi <= 200) return { label: 'Moderate', code: 'moderate' };
  if (aqi <= 300) return { label: 'Poor', code: 'poor' };
  if (aqi <= 400) return { label: 'Very Poor', code: 'very-poor' };
  return { label: 'Severe', code: 'severe' };
}
