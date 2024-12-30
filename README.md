# Car-Dheko---Used-Car-Price-Prediction
CarDekho Model Training Notebook

This README provides a detailed explanation of the notebook that builds a car price prediction model using a Random Forest algorithm.

Table of Contents

Introduction

Dataset Overview

Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Model Training

Model Evaluation

Conclusion

1. Introduction

This notebook is designed to predict car prices based on various features such as mileage, years of usage, ownership details, and other specifications. It uses Random Forest as the machine learning model to deliver predictions.

2. Dataset Overview

The dataset consists of columns such as:

Fuel Type (ft): Type of fuel used by the car.

City: Location where the car is listed for sale.

Kilometers Driven (km): Total kilometers driven.

No of Years: The car's age calculated as Current Year - Model Year.

Owner Number (ownerNo): Ownership details.

Top Mileage: Car mileage.

Transmission: Whether the car is manual or automatic.

Insurance Validity: Whether the car's insurance is valid.

Target variable: Price (dependent variable).

3. Preprocessing

Tasks:

Handling Missing Values:

Missing values are identified and appropriately filled or removed.

Some categorical columns are label-encoded for model compatibility.

Feature Scaling:

Numerical features are scaled using MinMaxScaler to ensure consistent ranges for the machine learning model.

Note: Exploratory data analysis is conducted on the original dataset (before scaling).

4. Exploratory Data Analysis (EDA)

Key insights include:

Distribution plots for numerical features like mileage, kilometers driven, and years of usage.

Relationship analysis between features and the target variable (Price).

Categorical feature distributions (e.g., fuel type, transmission).

Plots are generated on the original, unscaled data for better interpretability.

5. Feature Engineering

New features are created to improve model performance:

No of Years replaces the original Model Year feature.

Additional transformations are applied to ensure categorical features like Fuel Type and City are label-encoded.

6. Model Training

Algorithm: Random Forest

Libraries: sklearn.ensemble.RandomForestRegressor

Parameters:

Default parameters are used initially.

Grid Search or manual tuning can be applied for optimization.

Inputs:

Features: ft, city, km, No of Years, ownerNo, top_Mileage, Transmission, Insurance Validity.

Target: Price

Outputs:

Trained Random Forest model saved for later use.

7. Model Evaluation

Metrics:

Mean Absolute Error (MAE)

Root Mean Square Error (RMSE)

R² Score

Results are displayed to assess model performance.

8. Conclusion

The notebook successfully predicts car prices using a Random Forest model.
