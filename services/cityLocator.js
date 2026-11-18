const POPULAR_CITIES = [
  { city: 'Delhi', lat: 28.6139, lon: 77.2090, state: 'Delhi', country: 'India' },
  { city: 'Mumbai', lat: 19.0760, lon: 72.8777, state: 'Maharashtra', country: 'India' },
  { city: 'Bengaluru', lat: 12.9716, lon: 77.5946, state: 'Karnataka', country: 'India' },
  { city: 'Pune', lat: 18.5204, lon: 73.8567, state: 'Maharashtra', country: 'India' },
  { city: 'Kolkata', lat: 22.5726, lon: 88.3639, state: 'West Bengal', country: 'India' },
  { city: 'Chennai', lat: 13.0827, lon: 80.2707, state: 'Tamil Nadu', country: 'India' },
  { city: 'Hyderabad', lat: 17.3850, lon: 78.4867, state: 'Telangana', country: 'India' },
  { city: 'Ahmedabad', lat: 23.0225, lon: 72.5714, state: 'Gujarat', country: 'India' },
  { city: 'Jaipur', lat: 26.9124, lon: 75.7873, state: 'Rajasthan', country: 'India' },
  { city: 'Lucknow', lat: 26.8467, lon: 80.9462, state: 'Uttar Pradesh', country: 'India' },
  { city: 'London', lat: 51.5072, lon: -0.1276, state: 'England', country: 'United Kingdom' },
  { city: 'New York', lat: 40.7128, lon: -74.0060, state: 'New York', country: 'United States' },
];

export function listPopularCities() {
  return POPULAR_CITIES.map(city => ({ ...city, name: city.city }));
}

export async function geocodeLocation(query) {
  const q = String(query || '').trim();
  if (!q) throw new Error('Enter a city name.');

  const local = POPULAR_CITIES.find(item => item.city.toLowerCase() === q.toLowerCase());
  if (local) return { ...local, name: local.city, fullName: `${local.city}, ${local.state}, ${local.country}` };

  const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q)}&count=1&language=en&format=json`;
  const res = await fetch(url, { headers: { 'User-Agent': 'AirQualityWeatherResearch/1.0' } });
  if (!res.ok) throw new Error(`Location search failed (HTTP ${res.status}).`);
  const data = await res.json();
  const result = data.results?.[0];
  if (!result) throw new Error(`City not found: "${q}"`);

  return {
    city: result.name,
    name: result.name,
    fullName: [result.name, result.admin1, result.country].filter(Boolean).join(', '),
    lat: result.latitude,
    lon: result.longitude,
    state: result.admin1 || '',
    country: result.country || '',
    elevation: result.elevation ?? null,
  };
}
