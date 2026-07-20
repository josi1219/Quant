# Antigravity Quantitative Trading Execution Engine (EURUSD)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Advanced-orange)
![MetaTrader 5](https://img.shields.io/badge/MetaTrader5-IPC_Bridge-green)
![Quant](https://img.shields.io/badge/Status-Production_Ready-success)

A production-grade, asynchronous quantitative trading execution engine for EURUSD. This project implements advanced machine learning frameworks primarily popularized by Marcos Lopez de Prado in *Advances in Financial Machine Learning*, bridging the gap between rigorous academic research and live algorithmic execution.

## 🧠 Core Quantitative Methodologies

This architecture deliberately abandons traditional retail trading concepts (like simple time bars and moving average crossovers) in favor of institutional-grade data engineering:

### 1. The Volume Clock (Market Microstructure)
Traditional time bars (e.g., 5-minute candles) oversample slow, illiquid markets and undersample highly active sessions, causing poor statistical properties. This pipeline constructs **Volume Bars** (sampling the market only after a fixed volume threshold is reached). This allows the model to naturally "speed up" during high-activity news events and "slow down" during the Asian session, stabilizing the statistical properties of the data.

### 2. Fractional Differencing
Achieving stationarity in financial time series typically requires integer differencing (e.g., $P_t - P_{t-1}$), which completely destroys the memory (long-term trends) of the series. We implemented **Fractional Differencing** ($d \approx 0.150$) to make the price series stationary enough for machine learning algorithms while retaining maximal memory of the original price action.

### 3. The Triple-Barrier Method & Meta-Labeling
Instead of classifying simple "Up" or "Down" moves, this model is trained on a dynamic **Triple-Barrier Framework**:
- **Upper Barrier (Take Profit):** Dynamically scaled by rolling daily volatility.
- **Lower Barrier (Stop Loss):** Dynamically scaled by rolling daily volatility.
- **Vertical Barrier (Max Holding Time):** Enforced at 120 minutes to prevent the model from holding stale, zero-edge predictions.

A **Meta-Model** structure is then applied: a primary model searches for anomalies (trade setups), and a secondary model sizes the position (or vetoes the trade entirely) based on statistical confidence.

## 🏗️ Machine Learning Pipeline

### Multi-Timeframe Alignment
High-timeframe (H1) macroeconomic and structural context is shifted and merged onto the execution timeframe (M5) using strict `merge_asof` joins. This guarantees **zero look-ahead bias**—the model is completely blind to future data during training.

### Purged Walk-Forward Cross-Validation (CV)
Standard k-fold CV fails in finance due to extreme serial correlation. This pipeline utilizes a **Purged Walk-Forward CV**. Whenever the data is split into train and test sets, any training samples whose evaluation windows overlap with the test set are actively "embargoed" and purged. This ensures a leak-proof evaluation environment.

### Cost-Aware Hyperparameter Optimization (Optuna)
Optuna is utilized to run joint optimization sweeps across both the LightGBM hyperparameters and the Triple-Barrier thresholds (Take-Profit distance, Stop-Loss distance). 
Crucially, the objective function maximizes **Net Cost-Adjusted Expected Return**. The model is penalized for high-frequency trading behaviors that look profitable on paper but lose money in production due to broker spreads and slippage.

## 🚀 Live Execution Engine (`LiveExecutor`)

The `LiveExecutor` is a robust, asynchronous Python bridge that communicates directly with the MetaTrader 5 terminal via IPC. 

It is designed with strict risk management safety nets:
- **Trade Manager (Vertical Barrier Enforcer):** Actively monitors open trades every 5 minutes. If a trade exceeds the maximum holding period, the executor automatically transmits an opposing market order to flatten the position.
- **Dynamic Risk Scaling:** Position sizing (Lot Size) is dynamically assigned based on the meta-model's prediction probability. 
- **Spread Safety:** Vetoes trades immediately if the live broker spread exceeds acceptable tolerances.
- **Magic Number Isolation:** The bot tracks only its own proprietary executions, allowing simultaneous manual trading or external EAs to run on the same account without interference.

## 💻 Installation & Quick Start

```bash
# Clone the repository
git clone https://github.com/josi1219/Quant.git
cd Quant

# Install the Python package and dependencies
pip install -e ".[dev,export,tune]"

# Run the live execution engine
python run_live.py
```

## 📁 Repository Structure

```text
src/              ← Core Python Package
  config.py       ← Central hyperparameters, constants, and risk management limits
  data/           ← Data loading, cleaning, and Volume Bar construction
  features/       ← Feature engineering (Fractional Diff, Microstructure, Regimes)
  labels/         ← Triple-barrier labeling logic
  training/       ← Purged Walk-Forward CV, Optuna tuner, and Trainer
  evaluation/     ← Cost-aware performance metrics
  live/           ← MT5 IPC Client, LiveExecutor, and Trade Manager

run_live.py       ← Entry point for live algorithmic trading
tests/            ← Unit tests ensuring pipeline integrity
models/           ← Serialized LightGBM booster artifacts (.lgbm)
```
