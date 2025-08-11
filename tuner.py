import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
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
        logging.FileHandler(os.path.join(VAULT_DIR, 'tuner.log')), 
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

zone_encoder_path = os.path.join(VAULT_DIR, 'complete_zone_encoder.joblib')
if os.path.exists(zone_encoder_path):
    zone_encoder = joblib.load(zone_encoder_path)
    logger.info(f"Loaded zone encoder from {zone_encoder_path}")
else:
    zone_encoder = LabelEncoder()
    zone_encoder.fit(train_df['zone'])
    joblib.dump(zone_encoder, zone_encoder_path)
    logger.info(f"Created and saved new zone encoder to {zone_encoder_path}")

logger.info("Try encoding zones in training data...")
try:
    train_df['zone_encoded'] = zone_encoder.transform(train_df['zone'])
except Exception as e:
    logger.error(f"Zone encoding failed: {e}")
    logger.error("Zone encoded data is different from training data !")
    raise
logger.info("Zone encoding completed successfully for training data.")

logger.info("Try encoding zones in validation data...")
try:
    val_df['zone_encoded'] = zone_encoder.transform(val_df['zone'])
except Exception as e:
    logger.error(f"Zone encoding failed: {e}")
    logger.error("Zone encoded data is different from validation data !")
    raise
logger.info("Zone encoding completed successfully for validation data.")

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

logger.info("Data preparation completed. Starting hyperparameter tuning...")

storage = f"sqlite:///{os.path.join(VAULT_DIR, 'hotzone_tune.db')}"
datetime_str = dt.datetime.now().strftime("%d%m%Y_%H%M%S")
study_name = f"hotzone_tuning_{datetime_str}"

def objective_lightgbm(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'num_leaves': trial.suggest_int('num_leaves', 20, 35),
        'learning_rate': trial.suggest_float('learning_rate', 0.06, 0.12, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.95, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.75),
        'bagging_freq': trial.suggest_int('bagging_freq', 5, 7),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 45, 70),
        'lambda_l1': trial.suggest_float('lambda_l1', 6.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.5, 2.0),
        'max_depth': trial.suggest_int('max_depth', 10, 12),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 8.0, 13.0),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000,
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0), pruning_callback])
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

study = optuna.create_study(
    direction='minimize',
    study_name=study_name,
    storage=storage,
    load_if_exists=True)

study.optimize(objective_lightgbm, n_trials=20)
best_params_lgb = study.best_params

logger.info(f"Best parameters found: {best_params_lgb}")
logger.info(f"Best trial value: {study.best_value}")
logger.info("Hyperparameter tuning completed.")