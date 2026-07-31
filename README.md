# VPIN & Order Flow Toxicity Modeler (HFT Analysis)

This repository contains a Python-based implementation of the **Volume-Synchronized Probability of Informed Trading (VPIN)** model, derived from the seminal research paper *"Flow Toxicity and Liquidity in a High Frequency World"* by Easley, Lopez de Prado, and O'Hara (2012).

## Project Overview

In High-Frequency Trading (HFT) and market microstructure environments, standard clock-time metrics often fail to capture the true nature of information arrival. This project implements a **volume-time** approach to estimate order flow toxicity, providing a critical risk-management tool for market makers anticipating adverse selection and toxicity-induced volatility (e.g., Flash Crashes).

This pipeline is designed to be conceptually compatible with high-frequency data architectures (like ClickHouse) used for processing exchange ITCH data. For the purposes of this demonstration, it utilizes nanosecond-level crypto tick data (e.g., Binance BTC/USDT) as a proxy for commercial HFT data (such as Borsa Istanbul tick data) to showcase the model's resilience against real-world, noisy microstructure environments.

## Core Features

1.  **Volume-Time Bucketing:** Mitigates volatility clustering by sampling data based on trade intensity rather than chronological time.
2.  **Bulk Volume Classification:** Abandons outdated tick-by-tick classification (e.g., Lee-Ready) in favor of a probabilistic volume classification based on standard deviations of price changes within time bars, effectively filtering HFT noise and order splitting.
3.  **Real-Time Toxicity Metric (VPIN):** Calculates the continuous VPIN metric to detect informed trading saturation points prior to major market movements.

## Methodology Highlights

*   **No Parameter Estimation:** Unlike traditional PIN models requiring computationally heavy Maximum Likelihood Estimation for Poisson distributions, this VPIN implementation relies on direct analytic estimation over equal-volume buckets, allowing for ultra-low latency execution.
*   **Adverse Selection Indicator:** High VPIN values do not predict directional price moves, but rather signal an impending withdrawal of liquidity providers due to toxic order flow, invariably leading to extreme volatility spikes.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/harunalkan/VPIN-Order-Flow-Toxicity-Model.git](https://github.com/harunalkan/VPIN-Order-Flow-Toxicity-Model.git)
    cd VPIN-Order-Flow-Toxicity-Model
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib scipy
    ```
3.  **Data Requirements:**
    Download high-frequency tick data (e.g., Binance BTCUSDT trades from Kaggle) and place the `.csv` file in the root directory. Ensure the CSV contains `time` (or timestamp), `price`, and `quantity` (or volume) columns.
4.  **Run the Modeler:**
    Modify the `veri_dosyasi` variable in the `__main__` block to match your dataset's filename (e.g., `BTCUSDT.csv`), then execute:
    ```bash
    python vpin_estimator.py
    ```

## Author
**Harun Alkan** | Mathematical Engineering Student at ITU | Quantitative Finance Enthusiast
