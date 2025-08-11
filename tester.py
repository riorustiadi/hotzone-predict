import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc
import logging
import datetime as dt
from datetime import datetime
import re

gc.collect()

VAULT_DIR = "vault"
os.makedirs(VAULT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(VAULT_DIR, 'tester.log')), 
        logging.StreamHandler()])

logger = logging.getLogger(__name__)

def load_hourly_data(folder):
    dfs = []
    for hour in range(24):
        file_path = f'data/{folder}/hour_{hour:02d}.parquet'
        if os.path.exists(file_path):
            dfs.append(pd.read_parquet(file_path))
        else:
            logger.warning(f"File not found: {file_path}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

logger.info("Loading test data...")
df = load_hourly_data('test')

if df.empty:
    logger.error("Test data is empty! Pipeline stopped.")
    raise ValueError("Test data is empty.")

logger.info("Performing feature engineering")

zone_encoder_path = os.path.join(VAULT_DIR, 'complete_zone_encoder.joblib')
if os.path.exists(zone_encoder_path):
    zone_encoder = joblib.load(zone_encoder_path)
    logger.info(f"Loaded zone encoder from {zone_encoder_path}")
else:
    zone_encoder = LabelEncoder()
    zone_encoder.fit(df['zone'])
    joblib.dump(zone_encoder, zone_encoder_path)
    logger.info(f"Created and saved new zone encoder to {zone_encoder_path}")

try:
    df['zone_encoded'] = zone_encoder.transform(df['zone'])
except Exception as e:
    logger.error(f"Zone encoding failed: {e}")
    logger.error("Zone in encoded data is different from training data !")
    raise

logger.info("Encoding datetime with cyclical encoders...")

df['is_weekend'] = (df['pickup_weekday'] >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * (df['pickup_day'] - 1) / 31)
df['day_cos'] = np.cos(2 * np.pi * (df['pickup_day'] - 1) / 31)
df['month_sin'] = np.sin(2 * np.pi * (df['pickup_month'] - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df['pickup_month'] - 1) / 12)
df['frequency_log'] = np.log1p(df['frequency'])
df.drop('frequency', axis=1, inplace=True)

logger.info("Feature engineering completed.")
logger.info("Converting data types for memory efficiency...")

df['pickup_hour'] = df['pickup_hour'].astype('int8')
df['pickup_weekday'] = df['pickup_weekday'].astype('int8')
df['pickup_day'] = df['pickup_day'].astype('int8')
df['pickup_month'] = df['pickup_month'].astype('int8')
df['is_holiday'] = df['is_holiday'].astype('int8')
df['frequency_log'] = df['frequency_log'].astype('float32')
df['is_weekend'] = df['is_weekend'].astype('int8')
df['zone_encoded'] = df['zone_encoded'].astype('category')
df['hour_sin'] = df['hour_sin'].astype('float32')
df['hour_cos'] = df['hour_cos'].astype('float32')
df['day_sin'] = df['day_sin'].astype('float32')
df['day_cos'] = df['day_cos'].astype('float32')
df['month_sin'] = df['month_sin'].astype('float32')
df['month_cos'] = df['month_cos'].astype('float32')

logger.info("Splitting data into features and target...")

feature_cols = ['is_holiday', 'pickup_hour', 'pickup_weekday', 'pickup_day', 'pickup_month',
                'zone_encoded', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

X_test, y_test = df[feature_cols], df['frequency_log']

del df
gc.collect()

logger.info("Data preparation completed. Starting model testing...")

logger.info("Loading LightGBM model for testing...")
model_files = [f for f in os.listdir(VAULT_DIR) if f.startswith("hotzone_model_") and f.endswith(".joblib")]
if not model_files:
    raise FileNotFoundError("No model files found in vault.")

logger.info(f"Finding latest model file...")
def extract_datetime(filename):
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return datetime.strptime(match.group(1), "%d%m%Y_%H%M%S")
    return datetime.min

model_files.sort(key=extract_datetime)
latest_model_file = model_files[-1]
logger.info(f"Latest model file found: {latest_model_file}")

model_path = os.path.join(VAULT_DIR, latest_model_file)
model_lgb = joblib.load(model_path)
logger.info("Latest model loaded successfully.")

logger.info("Making predictions on test data...")

y_pred = model_lgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mask = y_test != 0
if mask.sum() > 0:
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    logger.info(f"LightGBM Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
else:
    logger.info(f"LightGBM Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: N/A (zero values in target)")

logger.info("Saving predictions to CSV...")

datetime_str = dt.datetime.now().strftime("%d%m%Y_%H%M%S")
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
    'absolute_error': np.abs(y_test - y_pred)
})
predictions_path = os.path.join(VAULT_DIR, f"test_predictions_{datetime_str}.csv")
predictions_df.to_csv(predictions_path, index=False)
logger.info(f"Predictions saved to {predictions_path}")

logger.info("Testing completed successfully.")