# Systematic Stock Selection Engine: High-Efficiency Filtering with XGBoost

## Problem Statement
In the modern financial landscape, market participants are overwhelmed by a "sea of noise." Manually screening hundreds of daily stock tickers to identify high-potential assets 
is not only time-consuming but also prone to emotional bias. There is a critical need for a systematic, data-driven approach that can efficiently filter and rank assets based on 
their predicted performance, allowing for faster and more objective decision-making.

## Feature Engineering Approach
The core of this system lies in its ability to transform raw market data into high-signal inputs. The process begins by ingesting raw daily price and volume data, which is then passed through a feature engineering pipeline. During this stage, a series of new technical indicators and specialized features are generated to capture market momentum, volatility, and historical patterns. This transformation process is essential for highlighting the underlying signals within the noise, providing the model with a refined dataset optimized for daily asset ranking.

## Model Architecture
The core engine is built using XGBoost Regression, a state-of-the-art Gradient Boosting algorithm known for its speed and performance with tabular data. The model has undergone 
extensive hyperparameter tuning to optimize its ability to predict a rank percentage, transforming a standard regression problem into a high-efficiency asset 
ranking system.

## Model Evaluation
<div style="display: flex; justify-content: center;">
  <img src="/Research Results/Growth Simulation Result.png" style="width:100%; height:auto;">
</div>
This chart demonstrates the cumulative PnL (%) of the XGBoost-driven strategy against major benchmarks (IHSG, Gold, and LQ45). Throughout 2025, the model significantly outperformed 
the market, proving its ability to generate consistent Alpha even during periods where traditional indices remained stagnant or declined.
<br><br>

<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="/Research Results/Scatter Plot Result.png" style="width:42%; height:auto;">
  <img src="/Research Results/Residual Plot Result.png" style="width:54%; height:auto;">
</div>
To ensure the model’s robustness, we analyzed the correlation between predicted rankings and actual market outcomes alongside the error distribution. The Actual vs. Predicted 
scatter plot shows a clear positive trend, indicating the model’s effectiveness in distinguishing high-performers from laggards. Complementing this, the Residuals plot confirms 
a balanced error distribution across the prediction range, suggesting that the model maintains consistent accuracy without significant systematic bias or volatility-induced distortion.

## Insights
- Efficiency at Scale: The system successfully narrows down a vast universe of stocks into a high-probability "watchlist" in seconds, drastically reducing manual research time.
- Benchmark Superiority: The model's strategy yielded a significantly higher total return compared to passive indexing (LQ45/IHSG) and safe-haven assets (Gold).
- Systematic Discipline: By relying on XGBoost's ranking logic, the system eliminates human error and provides a repeatable, objective framework for daily market participation.
