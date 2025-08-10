# HotZone Tuner 🔥

> A machine learning pipeline for hotzone prediction - currently in hyperparameter tuning phase

## What's This About?

This is a step-by-step ML pipeline that I'm building to predict taxi pickup hotspots. Right now we're at the **hyperparameter tuning** stage using Optuna, but the plan is to expand it into a full training and testing pipeline.

## Current Features

- **Data Loading**: Processes hourly parquet files for train/val datasets
- **Feature Engineering**: Cyclical encoding for time features, zone encoding, log transforms
- **Hyperparameter Tuning**: Uses Optuna with LightGBM for regression
- **Smart Storage**: All results saved to `vault/` folder (logs, models, study database)
- **Docker Ready**: Containerized for easy deployment

## Pipeline Roadmap

- [x] **Phase 1**: Hyperparameter Tuning (current)
- [ ] **Phase 2**: Full Model Training
- [ ] **Phase 3**: Model Testing & Evaluation
- [ ] **Phase 4**: Prediction API

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the tuner
python tuner.py
```

### Docker (Recommended)
```bash
# Build and run
docker-compose up --build

# Results will be saved to your mounted vault directory
```

## What Gets Generated

After running, you'll find these in your `vault` folder:
- `tuner.log` - Detailed pipeline logs
- `zone_encoder.joblib` - Fitted label encoder for zones
- `hotzone_tune.db` - Optuna study database with all trials

## Data Structure Expected

```
data/
├── train/
│   ├── hour_00.parquet
│   ├── hour_01.parquet
│   └── ... (up to hour_23.parquet)
└── val/
    ├── hour_00.parquet
    ├── hour_01.parquet
    └── ... (up to hour_23.parquet)
```

## Tech Stack

- **ML Framework**: LightGBM
- **Hyperparameter Optimization**: Optuna
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Storage**: SQLite (via Optuna RDBStorage)
- **Containerization**: Docker

## Features in the Pipeline

- Cyclical encoding for temporal features (hour, day, month)
- Zone-based categorical encoding
- Log transformation for frequency data
- Memory-efficient data type conversion
- Optuna pruning for faster hyperparameter search
- Comprehensive logging throughout the process

---

*This is an evolving project - more phases coming soon! 🚀*
