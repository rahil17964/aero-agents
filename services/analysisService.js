import { fetchAtmosphericChemistry } from './aqiService.js';
import { evaluateHealthAndAirQuality } from './healthEpidemiology.js';
import { runMLPrediction } from './mlPredictionService.js';

export async function runAnalysisPipeline(location, coordinates) {
  const aqi = await fetchAtmosphericChemistry(coordinates.lat, coordinates.lon);
  const current = aqi.current;

  let predictions = await runMLPrediction(aqi, location.city);
  if (!predictions || !predictions.hourlyTrajectory?.length) {
    const modelError = predictions?.error || `No trained AQI model is available for ${location.city}. Train this city model to enable AQI predictions.`;
    predictions = {
      modelType: 'Unavailable', modelStatus: 'unavailable', error: modelError,
      hourlyTrajectory: [], dailySummary: (aqi.dailyWeather || []).map((weather, index) => ({
        dayName: index === 0 ? 'Today' : (index === 1 ? 'Tomorrow' : weather.date),
        dateStr: weather.date, avgAqi: null, minAqi: null, maxAqi: null, avgPm25: null,
        aqiCategory: 'Model not trained', categoryClass: 'moderate', tempMax: weather.tempMax,
        tempMin: weather.tempMin, rainProb: weather.rainProb,
        weatherCondition: weather.weatherCondition, weatherIcon: weather.weatherIcon,
      })),
      modelValidationMetrics: {
        architecture: 'Untrained', targetSource: 'Open-Meteo Air Quality API',
        rSquaredScore: null, rmse: null, mae: null, directionAccuracyPercent: null,
        featureImportance: [],
      },
    };
  }
  const health = evaluateHealthAndAirQuality(aqi);

  const researchAbstract = makeResearchAbstract({ location, aqi, predictions });

  return {
    meta: { generatedAt: new Date().toISOString(), location, coordinates },
    chemistryData: aqi,
    predictionsData: predictions,
    healthAndOpsData: health,
    researchAbstract,
  };
}

function makeResearchAbstract({ location, aqi, predictions }) {
  const current = aqi.current;
  const pm25 = current.pollutants.pm25.value;
  const validation = predictions.modelStatus === 'ready'
    ? `The AQI model uses a chronological 90-day holdout with a 168-hour purge; holdout MAE is ${predictions.modelValidationMetrics.mae} AQI points and RMSE is ${predictions.modelValidationMetrics.rmse}.`
    : 'No trained model is available, so no AQI forecast or performance claim is reported.';
  return {
    title: `AQI and Weather Analysis: ${location.city}`,
    authors: ['Air Quality and Weather Research Platform'],
    abstractText: `Open-Meteo reports a current US AQI of ${current.usAqi} (${current.category.label}) for ${location.city}; PM2.5 is ${pm25} µg/m³. Current weather is ${current.meteorology.weatherCondition} at ${current.meteorology.surfaceTemperatureC}°C, with wind speed ${current.meteorology.windSpeedMs} m/s. ${validation} When trained, the horizon-calibrated ML system predicts AQI and weather from historical Open-Meteo data.`,
    keywords: ['AQI', 'Air Quality', 'Weather', 'Open-Meteo', 'XGBoost', 'Chronological Validation', location.city],
  };
}
