# Intelligent Housing Purchase Advisory System

Welcome to the **Intelligent Housing Purchase Advisory System** repository! This project uses machine learning models to predict house prices and analyze market trends, helping users make informed decisions.

## Project Overview
This project combines regression techniques and clustering models to provide accurate house price predictions and segment the housing market. It is based on data from the [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview).

## Features
1. **Data Preparation**:
   - Feature selection: Variables include `LotArea`, `OverallQual`, `GarageArea`, etc.
   - Data cleaning: Handles missing values, outliers, and scaling.

2. **Regression Models**:
   - Models used: Linear Regression, Lasso, Elastic Net, Random Forest, and SVR.
   - Evaluates performance using standard metrics.

3. **Clustering Techniques**:
   - Algorithms: KMeans, Birch, and MeanShift.
   - Helps in market segmentation and trend analysis.

4. **Hybrid Approach**:
   - Combines clustering outputs with regression models to improve predictions.

5. **Visualization**:
   - Plots feature importance, data distributions, and model results.

## Technologies Used
- **Language**: Python
- **Libraries**:
  - Data Handling: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
