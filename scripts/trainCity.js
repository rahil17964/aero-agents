/**
 * trainCity.js
 * Convenience script to fetch historical data + train ML model for a city.
 * Usage: node scripts/trainCity.js Delhi
 *        node scripts/trainCity.js Mumbai
 */

import { execFileSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ML_DIR = path.join(__dirname, '..', 'ml');
const PYTHON = process.env.PYTHON_EXECUTABLE || (process.platform === 'win32' ? 'py' : 'python3');
const PYTHON_ARGS = process.platform === 'win32' && !process.env.PYTHON_EXECUTABLE ? ['-3.14', '-X', 'utf8'] : [];

const city = process.argv[2] || 'Delhi';

console.log(`\n═══════════════════════════════════════════`);
console.log(` Air Quality & Weather ML Training Pipeline`);
console.log(` City: ${city}`);
console.log(`═══════════════════════════════════════════\n`);

// Step 1: Fetch historical data
console.log('Step 1/2 — Fetching historical data (this may take 5-10 minutes)...\n');
try {
  execFileSync(PYTHON, [...PYTHON_ARGS, path.join(ML_DIR, 'fetch_data.py'), city], {
    stdio: 'inherit',
    cwd: ML_DIR,
    env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' }
  });
} catch (e) {
  console.error('Data fetch failed:', e.message);
  process.exit(1);
}

// Step 2: Train ML model
console.log('\nStep 2/2 — Training XGBoost model (this may take 5-15 minutes)...\n');
try {
  execFileSync(PYTHON, [...PYTHON_ARGS, path.join(ML_DIR, 'train_model.py'), city], {
    stdio: 'inherit',
    cwd: ML_DIR,
    env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' }
  });
} catch (e) {
  console.error('Training failed:', e.message);
  process.exit(1);
}

// Check output
const cityKey = city.toLowerCase().replace(/\s+/g, '_');
const aqiModelPath = path.join(ML_DIR, 'models', `${cityKey}_us_aqi_xgb.joblib`);
const pm25ModelPath = path.join(ML_DIR, 'models', `${cityKey}_pm2_5_xgb.joblib`);
if (fs.existsSync(aqiModelPath) && fs.existsSync(pm25ModelPath)) {
  console.log(`\n✅ Model ready! The dashboard will now use real ML predictions for ${city}.`);
  console.log(`   Restart the server: node server.js\n`);
} else {
  console.error('\n❌ Model file not found. Check errors above.\n');
}
