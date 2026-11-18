# Air Quality and Weather Dashboard

A city-level air-quality and weather dashboard using Open-Meteo data. Current conditions include AQI, pollutant concentrations, and weather. AQI and PM2.5 forecasts are produced by city-specific horizon-conditioned XGBoost models trained on Open-Meteo historical data; weather forecasts remain sourced from Open-Meteo.

## Run locally

1. Install Node.js dependencies with `npm install`.
2. Install Python dependencies from `ml/requirements.txt` using Python 3.14, or set `PYTHON_EXECUTABLE` to the Python executable that has those packages installed.
3. Run `npm start` and open `http://localhost:3000`.

## AQI model

Use **Train ML Model** to refresh historical Open-Meteo data and train AQI and PM2.5 models for the selected city. A model forecast is only displayed after a compatible model has been trained. If data or a trained model is unavailable, the dashboard reports that state rather than substituting a synthetic forecast.

The current target is Open-Meteo's modeled US AQI product, not direct regulatory-monitor observations. Validation measures performance against that modeled product. Metrics are separated by forecast horizon and use a chronological holdout; long-range results may be weak.

## Data and model notes

- City search uses Open-Meteo geocoding; current AQI and weather use Open-Meteo APIs.
- The forecast model is a horizon-conditioned XGBoost regressor with persistence baselines and holdout-derived intervals.

## Optional local assistant

If Ollama is installed and running, the assistant can use a local model. Without it, a built-in response helper uses the dashboard's current AQI, weather, forecast, and health-guidance data.
