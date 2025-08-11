import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import lightgbm as lgb
import optuna
import gc
import logging
import datetime as dt

gc.collect()

VAULT_DIR = "vault"
os.makedirs(VAULT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(VAULT_DIR, 'trainer.log')), 
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

logger.info("Loading training data...")
train_df = load_hourly_data('train')

logger.info("Loading validation data...")
val_df = load_hourly_data('val')

if train_df.empty or val_df.empty:
    logger.error("Train or validation data is empty! Pipeline stopped.")
    raise ValueError("Train or validation data is empty.")

logger.info("Performing feature engineering")

zone_encoder = LabelEncoder()
zone_encoder.fit(train_df['zone'])
train_df['zone_encoded'] = zone_encoder.transform(train_df['zone'])
val_df['zone_encoded'] = zone_encoder.transform(val_df['zone'])
joblib.dump(zone_encoder, os.path.join(VAULT_DIR, 'zone_encoder.joblib'))

logger.info("Encoding datetime with cyclical encoders...")
for df in [train_df, val_df]:
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

for df in [train_df, val_df]:
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

X_train, y_train = train_df[feature_cols], train_df['frequency_log']
X_val, y_val = val_df[feature_cols], val_df['frequency_log']

del train_df, val_df
gc.collect()

logger.info("Data preparation completed. Starting model training...")

datetime_str = dt.datetime.now().strftime("%d%m%Y_%H%M%S")
model_name = f"hotzone_model_{datetime_str}"

study = optuna.load_study(
    study_name="hotzone_tuning_10082025_203405",
    storage="sqlite:///vault/hotzone_tune.db"
)
best_params_lgb = study.best_params

params = best_params_lgb.copy()
params.update({
    'objective': 'regression', 
    'metric': 'rmse', 
    'boosting_type': 'gbdt', 
    'verbose': -1,
    'random_state': 42,
    'force_row_wise': True
})

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model_lgb = lgb.train(
    params, 
    train_data, 
    valid_sets=[val_data], 
    num_boost_round=3000,
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)])

model_path = os.path.join(VAULT_DIR, f"{model_name}.joblib")
joblib.dump(model_lgb, model_path)
logger.info(f"Model saved as {model_path}")

json_model_path = os.path.join(VAULT_DIR, f"{model_name}.json")
model_lgb.save_model(json_model_path, format='json')
logger.info(f"Model also saved as JSON: {json_model_path}")

logger.info(f"Best iteration: {model_lgb.best_iteration}")
logger.info(f"Best score: {model_lgb.best_score['valid_0']['rmse']}")
logger.info("Training completed successfully.")