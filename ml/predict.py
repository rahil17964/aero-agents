"""Generate a 168-hour AQI and PM2.5 forecast from trained XGBoost models."""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

warnings.filterwarnings("ignore")
ML_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ML_DIR, "models")
LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING = [6, 24, 72]
WEATHER_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation",
    "cloud_cover", "boundary_layer_height", "shortwave_radiation",
]
WEATHER_TARGETS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "precipitation", "cloud_cover", "surface_pressure",
]
WEATHER_LAGS = [1, 3, 6, 12, 24, 48, 72, 168]
WEATHER_ROLLING = [6, 24, 72]


def model_paths(city):
    key = city.lower().replace(" ", "_")
    return key, os.path.join(MODEL_DIR, f"{key}_{{target}}_xgb.joblib"), os.path.join(MODEL_DIR, f"{key}_{{target}}_metadata.json")


def cyclical_features(dt):
    return {
        "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "weekday_sin": np.sin(2 * np.pi * dt.weekday() / 7),
        "weekday_cos": np.cos(2 * np.pi * dt.weekday() / 7),
        "year_day_sin": np.sin(2 * np.pi * dt.timetuple().tm_yday / 365.25),
        "year_day_cos": np.cos(2 * np.pi * dt.timetuple().tm_yday / 365.25),
    }


def build_row(origin, horizon, weather, history, weather_history):
    target_time = origin + timedelta(hours=horizon)
    row = cyclical_features(target_time)
    for col in WEATHER_COLUMNS:
        if col == "wind_direction_10m":
            continue
        val = weather.get(col)
        row[col] = float(val) if val is not None else np.nan
    wind_dir = weather.get("wind_direction_10m")
    if wind_dir is None:
        row["wind_dir_sin"] = np.nan
        row["wind_dir_cos"] = np.nan
    else:
        direction_rad = np.deg2rad(float(wind_dir))
        row["wind_dir_sin"] = np.sin(direction_rad)
        row["wind_dir_cos"] = np.cos(direction_rad)

    ordered = sorted(history, key=lambda item: item["time"])
    for name, key in [("us_aqi", "usAqi"), ("pm2_5", "pm25")]:
        values = [float(item[key]) for item in ordered if item.get(key) is not None and np.isfinite(float(item[key]))]
        if not values:
            raise ValueError(f"No recent {name} history is available.")
        values = values[-169:]
        values.reverse()  # value[0] is the latest observed value at forecast origin
        for lag in LAGS:
            if lag >= len(values):
                raise ValueError(f"At least {lag + 1} hourly {name} observations are needed; got {len(values)}.")
            row[f"{name}_lag{lag}"] = values[lag]
        for window in ROLLING:
            window_values = values[:window]
            if len(window_values) < window:
                raise ValueError(f"At least {window} hourly {name} observations are needed.")
            row[f"{name}_roll{window}_mean"] = float(np.mean(window_values))
            row[f"{name}_roll{window}_std"] = float(np.std(window_values, ddof=1))
    ordered_weather = sorted(weather_history, key=lambda item: item["time"])
    for name in WEATHER_TARGETS:
        values = [float(item[name]) for item in ordered_weather if item.get(name) is not None and np.isfinite(float(item[name]))]
        if not values:
            raise ValueError(f"No recent {name} weather history is available.")
        values = values[-169:]
        values.reverse()
        for lag in WEATHER_LAGS:
            if lag >= len(values):
                raise ValueError(f"At least {lag + 1} hourly {name} weather observations are needed; got {len(values)}.")
            row[f"{name}_lag{lag}"] = values[lag]
        for window in WEATHER_ROLLING:
            window_values = values[:window]
            if len(window_values) < window:
                raise ValueError(f"At least {window} hourly {name} weather observations are needed.")
            row[f"{name}_roll{window}_mean"] = float(np.mean(window_values))
            row[f"{name}_roll{window}_std"] = float(np.std(window_values, ddof=1))
    row["forecast_horizon_hours"] = float(horizon)
    return row


def interval_for(metadata, horizon):
    values = metadata.get("residual_quantiles_by_horizon", {})
    keys = sorted((int(key) for key in values), key=int)
    if not keys:
        return -25.0, 25.0
    left = max((h for h in keys if h <= horizon), default=keys[0])
    right = min((h for h in keys if h >= horizon), default=keys[-1])
    lo1, hi1 = values[str(left)]["lower"], values[str(left)]["upper"]
    lo2, hi2 = values[str(right)]["lower"], values[str(right)]["upper"]
    if left == right:
        return lo1, hi1
    weight = (horizon - left) / (right - left)
    return lo1 + (lo2 - lo1) * weight, hi1 + (hi2 - hi1) * weight


def correction_scale_for(metadata, horizon):
    values = metadata.get("correction_scales_by_horizon", {})
    keys = sorted(int(key) for key in values)
    if not keys:
        return 1.0
    left = max((key for key in keys if key <= horizon), default=keys[0])
    right = min((key for key in keys if key >= horizon), default=keys[-1])
    if left == right:
        return float(values[str(left)])
    weight = (horizon - left) / (right - left)
    return float(values[str(left)]) + (float(values[str(right)]) - float(values[str(left)])) * weight


def forecast_bounds(target):
    return {
        "us_aqi": (0, 500), "pm2_5": (0, 500),
        "temperature_2m": (-60, 60), "relative_humidity_2m": (0, 100),
        "wind_speed_10m": (0, 100), "precipitation": (0, 100),
        "cloud_cover": (0, 100), "surface_pressure": (800, 1100),
    }[target]


def condition_from_weather(precipitation, cloud_cover):
    if precipitation >= 0.1:
        return "Rain", "🌧️"
    if cloud_cover >= 80:
        return "Cloudy", "☁️"
    if cloud_cover >= 35:
        return "Partly Cloudy", "⛅"
    return "Clear Sky", "☀️"


def aqi_category(value):
    if value <= 50: return "Good", "good"
    if value <= 100: return "Moderate", "moderate"
    if value <= 150: return "Unhealthy for Sensitive Groups", "unhealthy-sensitive"
    if value <= 200: return "Unhealthy", "unhealthy"
    if value <= 300: return "Very Unhealthy", "very-unhealthy"
    return "Hazardous", "hazardous"


def weather_code(code):
    if code is None: return "Fair Weather", "☀️"
    code = int(code)
    if code == 0: return "Clear Sky", "☀️"
    if code in (1, 2): return "Partly Cloudy", "⛅"
    if code == 3: return "Cloudy", "☁️"
    if code in (45, 48): return "Fog", "🌫️"
    if code in (51, 53, 55, 61, 63, 65, 80, 81, 82): return "Rain", "🌧️"
    if code in (95, 96, 99): return "Thunderstorm", "⛈️"
    return "Fair Weather", "🌤️"


def predict(payload):
    city = payload["city"]
    key, model_template, metadata_template = model_paths(city)
    targets = {}
    for target in ("us_aqi", "pm2_5", *WEATHER_TARGETS):
        model_path = model_template.format(target=target)
        meta_path = metadata_template.format(target=target)
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            return {"error": f"No trained {target} model for {city}. Train it from the dashboard or run scripts/trainCity.js {city}."}
        with open(meta_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("model_version") != 5:
            return {"error": f"The {city} {target} model uses an obsolete feature schema. Retrain it from the dashboard."}
        targets[target] = (joblib.load(model_path), metadata)

    origin = datetime.fromisoformat(payload["currentTime"].replace("Z", "+00:00"))
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=timezone.utc)
    weather = payload["weatherCurrent"]
    weather_history = payload.get("weatherHistory", [])
    history = payload.get("history", [])
    if len(history) < 169:
        return {"error": f"Need at least 169 recent hourly AQI records; received {len(history)}."}
    history = sorted(history, key=lambda item: item["time"])[-169:]
    if len(weather_history) < 169:
        return {"error": f"Need at least 169 recent hourly weather records; received {len(weather_history)}."}
    weather_history = sorted(weather_history, key=lambda item: item["time"])[-169:]
    for index, item in enumerate(history):
        if item.get("usAqi") is None or item.get("pm25") is None:
            return {"error": "Recent hourly AQI history contains missing AQI or PM2.5 values."}
        if index:
            previous = datetime.fromisoformat(history[index - 1]["time"].replace("Z", "+00:00"))
            current = datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
            if current - previous != timedelta(hours=1):
                return {"error": "Recent AQI history has a time gap; refusing to treat irregular rows as hourly lags."}

    hourly = []
    latest_aqi = float(history[-1]["usAqi"])
    latest_pm25 = float(history[-1]["pm25"])
    rows = [build_row(origin, horizon, weather, history, weather_history) for horizon in range(1, 169)]
    feature_names = targets["us_aqi"][1]["feature_names"]
    vectors = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
    model_deltas = {target: model.predict(vectors).astype(float) for target, (model, _) in targets.items()}
    for index, horizon in enumerate(range(1, 169)):
        aqi_scale = correction_scale_for(targets["us_aqi"][1], horizon)
        pm25_scale = correction_scale_for(targets["pm2_5"][1], horizon)
        aqi = float(np.clip(latest_aqi + model_deltas["us_aqi"][index] * aqi_scale, 0, 500))
        pm25 = float(np.clip(latest_pm25 + model_deltas["pm2_5"][index] * pm25_scale, 0, 500))
        weather_values = {}
        for target in WEATHER_TARGETS:
            model, metadata = targets[target]
            latest = float(weather[target])
            scale = correction_scale_for(metadata, horizon)
            low, high = forecast_bounds(target)
            weather_values[target] = float(np.clip(latest + model_deltas[target][index] * scale, low, high))
        aqi_lo, aqi_hi = interval_for(targets["us_aqi"][1], horizon)
        pm_lo, pm_hi = interval_for(targets["pm2_5"][1], horizon)
        dt = origin + timedelta(hours=horizon)
        label, code = aqi_category(aqi)
        condition, weather_icon = condition_from_weather(weather_values["precipitation"], weather_values["cloud_cover"])
        hourly.append({
            "horizonHours": horizon,
            "timestamp": dt.isoformat(),
            "displayTime": dt.strftime("%a %d %b %H:00 UTC"),
            "date": dt.strftime("%b %d"),
            "predictedAqi": int(round(aqi)),
            "predictedPm25": round(pm25, 1),
            "predictedTemperatureC": round(weather_values["temperature_2m"], 1),
            "predictedHumidityPercent": int(round(weather_values["relative_humidity_2m"])),
            "predictedWindSpeedMs": round(weather_values["wind_speed_10m"], 1),
            "predictedPrecipitationMm": round(weather_values["precipitation"], 2),
            "predictedCloudCoverPercent": int(round(weather_values["cloud_cover"])),
            "predictedSurfacePressureHpa": round(weather_values["surface_pressure"], 1),
            "weatherCondition": condition,
            "weatherIcon": weather_icon,
            "lowerBound95": int(max(0, round(aqi + aqi_lo))),
            "upperBound95": int(min(500, round(aqi + aqi_hi))),
            "pm25LowerBound95": round(max(0, pm25 + pm_lo), 1),
            "pm25UpperBound95": round(min(500, pm25 + pm_hi), 1),
            "aqiCategory": label,
            "categoryClass": code,
            "primaryDriver": "Model output",
        })

    daily = []
    for day in range(7):
        start, end = day * 24, (day + 1) * 24
        rows = hourly[start:end]
        temperatures = [r["predictedTemperatureC"] for r in rows]
        rain_hours = sum(r["predictedPrecipitationMm"] >= 0.1 for r in rows)
        conditions = [r["weatherCondition"] for r in rows]
        condition = max(set(conditions), key=conditions.count)
        condition_icon = next(r["weatherIcon"] for r in rows if r["weatherCondition"] == condition)
        avg = int(round(np.mean([r["predictedAqi"] for r in rows])))
        previous_avg = daily[-1]["avgAqi"] if daily else None
        label, code = aqi_category(avg)
        daily.append({
            "dayIndex": day + 1,
            "dayName": "Today" if day == 0 else ("Tomorrow" if day == 1 else (origin + timedelta(days=day)).strftime("%a")),
            "dateStr": (origin + timedelta(days=day)).strftime("%b %d"),
            "avgAqi": avg,
            "minAqi": min(r["predictedAqi"] for r in rows),
            "maxAqi": max(r["predictedAqi"] for r in rows),
            "lowerBound95": int(round(np.mean([r["lowerBound95"] for r in rows]))),
            "upperBound95": int(round(np.mean([r["upperBound95"] for r in rows]))),
            "avgPm25": round(float(np.mean([r["predictedPm25"] for r in rows])), 1),
            "aqiCategory": label,
            "categoryClass": code,
            "tempMax": round(max(temperatures)),
            "tempMin": round(min(temperatures)),
            "rainProb": int(round(rain_hours / len(rows) * 100)),
            "weatherCondition": condition,
            "weatherIcon": condition_icon,
            "avgHumidityPercent": int(round(np.mean([r["predictedHumidityPercent"] for r in rows]))),
            "maxWindSpeedMs": round(max(r["predictedWindSpeedMs"] for r in rows), 1),
            "totalPrecipitationMm": round(sum(r["predictedPrecipitationMm"] for r in rows), 1),
            "healthTip": "Forecast is model output trained against Open-Meteo AQI history.",
            "trend": "stable" if previous_avg is None or avg == previous_avg else ("worsening" if avg > previous_avg else "improving"),
        })

    aqi_meta = targets["us_aqi"][1]
    pm_meta = targets["pm2_5"][1]
    val = aqi_meta["aggregate_validation"]
    metrics = {
        "architecture": aqi_meta["model_type"],
        "targetSource": aqi_meta["target_source"],
        "rSquaredScore": val.get("r2"),
        "rmse": val["rmse"], "mae": val["mae"],
        "directionAccuracyPercent": val["direction_accuracy_percent"],
        "persistenceMae": val["persistence_mae"],
        "pm25Rmse": pm_meta["aggregate_validation"]["rmse"],
        "pm25Mae": pm_meta["aggregate_validation"]["mae"],
        "validationNote": aqi_meta["validation_method"],
        "trainedAt": aqi_meta["trained_at"],
        "trainPairCount": aqi_meta["train_pair_count"],
        "testPairCount": aqi_meta["test_pair_count"],
        "horizonMetrics": aqi_meta["horizon_metrics"],
        "featureImportance": aqi_meta["feature_importance"],
        "weatherValidationMetrics": {target: metadata["aggregate_validation"] for target, (_, metadata) in targets.items() if target in WEATHER_TARGETS},
    }
    return {
        "modelType": metrics["architecture"], "modelStatus": "ready", "city": city,
        "hourlyTrajectory": hourly, "dailySummary": daily,
        "modelValidationMetrics": metrics,
    }


if __name__ == "__main__":
    try:
        payload = json.load(sys.stdin)
        print(json.dumps(predict(payload), allow_nan=False))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
