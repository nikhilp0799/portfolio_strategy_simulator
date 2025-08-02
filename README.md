# AI Portfolio Strategy Simulator

🎯 A Multi-Agent System for Market Forecasting, Trading Signal Generation, and Risk-Aware Portfolio Management

---

## 📘 Overview

**AI Portfolio Strategy Simulator** is a modular, production-grade system that replicates the decision-making and control flow of a real-world trading and investment platform. Built around a multi-agent architecture and a professional frontend-backend stack, it combines financial forecasting, signal generation, risk analytics, and explainability into an integrated portfolio strategy simulation framework.

Designed to mirror the internal tooling used by quant funds, hedge funds, and trading firms, this system:

- Predicts asset prices using multiple models (RF, XGBoost, GRU)
- Generates real-time trade signals with logic-based confidence scoring
- Runs backtests with key performance metrics (Sharpe, Drawdown, CAGR)
- Detects macro regimes and adjusts strategy behavior accordingly
- Explains model decisions using SHAP values
- Calculates Value at Risk (VaR) and monitors portfolio exposure
- Maintains a detailed trade and asset logbook for traceability
- Provides a React-based orchestration dashboard for strategy control

---

## 🧠 Core Agents

| Agent | Description |
|-------|-------------|
| 🧠 **Forecasting Agent** | Predicts future asset prices using ML/DL models |
| 📈 **Signal Agent** | Converts predictions into Buy/Sell/Hold signals |
| 🔁 **Backtesting Agent** | Evaluates strategy performance over historical data |
| 📉 **Risk Agent** | Computes Value at Risk, drawdown, exposure levels |
| 🧠 **Explainability Agent** | Generates SHAP plots to interpret model drivers |
| 🧮 **Portfolio Optimization Agent** *(coming soon)* | Allocates capital across assets using optimization logic |
| 📓 **Logbook Agent** | Tracks trades, asset positions, execution rationale |
| 🧭 **Orchestrator Agent** | Central command unit coordinating all other agents and serving data to the React frontend |

---

## 📊 Features

- 🔮 **Forecasting**: Price prediction using Random Forest, XGBoost, GRU
- 📈 **Signal Generation**: Converts predictions into actionable decisions
- 🔁 **Backtesting**: Calculates Sharpe, CAGR, Drawdown, and trade stats
- 📉 **Risk Management**: Computes VaR, monitors stop-loss and exposure levels
- 🧠 **Explainability (SHAP)**: Feature attribution and model transparency
- 📓 **Audit-Grade Trade Log**: Tracks trade history, strategy, and agent trigger
- 🧮 **Portfolio Management** *(upcoming)*: Intelligent capital allocation
- 📉 **Macro Regime Modeling** *(upcoming)*: Bull/Bear/Neutral awareness

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| 💻 Frontend | **React.js** (Next.js optional) |
| 🔌 Backend API | **FastAPI** |
| 🧠 Core Agents | Python (modular) |
| 📦 Models | scikit-learn, XGBoost, TensorFlow |
| 📊 Visualization | Plotly, SHAP, Dash (internal use only) |
| 📁 Data Ingestion | `yfinance`, `fredapi`, `newsapi-python`, `tweepy` |
| 🔍 Explainability | SHAP |
| ⚖️ Risk/Backtest | Backtrader, NumPy, Pandas |

---

## 📁 Directory Structure (Planned)

