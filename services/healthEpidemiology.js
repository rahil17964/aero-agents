export function evaluateHealthAndAirQuality(aqiData) {
  const aqi = Number(aqiData.current.usAqi);
  const pm25 = Number(aqiData.current.pollutants.pm25.value);
  const advisoryTone = aqi > 200 ? 'danger' : aqi > 150 ? 'warning' : aqi > 100 ? 'caution' : aqi > 50 ? 'moderate' : 'good';

  const vulnerableCohorts = [
    {
      group: 'Children and older adults',
      status: aqi > 100 ? 'Take extra care' : 'Usual precautions',
      directive: aqi > 150 ? 'Reduce prolonged or strenuous outdoor activity.' : aqi > 100 ? 'Consider limiting prolonged outdoor activity.' : 'Follow local air-quality guidance.',
    },
    {
      group: 'People with asthma or heart and lung conditions',
      status: aqi > 100 ? 'Take extra care' : 'Usual precautions',
      directive: aqi > 150 ? 'Follow your care plan and reduce strenuous outdoor activity.' : 'Follow your care plan and monitor symptoms.',
    },
    {
      group: 'Outdoor workers and active people',
      status: aqi > 150 ? 'Reduce strenuous exposure' : 'Check current conditions',
      directive: aqi > 150 ? 'Take breaks indoors or reduce strenuous activity outdoors.' : 'Check the AQI before extended outdoor activity.',
    },
  ];

  return {
    advisoryTone,
    currentAqi: aqi,
    pm25MicrogramsPerCubicMeter: pm25,
    vulnerableCohorts,
    guidanceNote: 'General guidance only. Follow local public-health advice and seek medical care for concerning symptoms.',
    engineeringMitigation: {
      recommendedCadrCfm: aqi > 100 ? Math.round((56 * (aqi > 200 ? 5 : 4) * 0.5886) * 1.2) : null,
      optimalVentilationTiming: 'Check current and forecast AQI before opening windows; choose a lower-AQI period.',
    },
  };
}
