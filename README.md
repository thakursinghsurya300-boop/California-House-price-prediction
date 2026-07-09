# California-House-price-prediction
End-to-end machine learning project to predict California house prices using Random Forest Regression.
# 🏠 California House Price Prediction using Random Forest Regression

## 📌 Project Overview

This project predicts California house prices using the California Housing dataset provided by Scikit-learn. It demonstrates the complete machine learning workflow, including exploratory data analysis (EDA), feature selection, model training, evaluation, and prediction.

---

## 🎯 Objective

The objective of this project is to build a machine learning model capable of predicting the median house value of a California district based on demographic and housing-related features.

---

## 📊 Dataset

**Dataset:** California Housing Dataset

**Source:** Scikit-learn (`fetch_california_housing()`)

**Target Variable:**
- **MedHouseVal** – Median House Value

### Features

| Feature | Description |
|----------|-------------|
| MedInc | Median income of households |
| HouseAge | Median age of houses |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Population of the block |
| AveOccup | Average occupancy per household |
| Latitude | Latitude of the district |
| Longitude | Longitude of the district |

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

## 📈 Exploratory Data Analysis

The following visualizations were created to understand the dataset:

- House Prices Across California
- Median Income vs House Value
- Average Rooms vs House Value
- Population vs House Value

These visualizations helped identify relationships between housing characteristics and house prices before training the model.

---

## 🤖 Machine Learning Workflow

1. Load Dataset
2. Data Exploration
3. Feature Selection
4. Train-Test Split
5. Train Random Forest Regressor
6. Predict House Prices
7. Evaluate Model Performance

---

## 🌲 Model Used

**Random Forest Regressor**

The model was trained using Scikit-learn's RandomForestRegressor.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 0.3384 |
| Root Mean Squared Error (RMSE) | 0.5224 |
| R² Score | 0.7936 |

---

## 🔮 Sample Prediction

Example Input

- Median Income: 6.5
- House Age: 25 Years
- Average Rooms: 6.2
- Average Bedrooms: 1.2
- Population: 1800
- Average Occupancy: 3.1
- Latitude: 34.05
- Longitude: -118.25

Predicted House Value:

**$324,847.11**

---

## 📁 Project Structure

```
California-House-Price-Prediction/

│── house_price_prediction.ipynb
│── house_price_prediction.py
│── README.md
│── requirements.txt
│── LICENSE
│── .gitignore
│
└── images/
```

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Importance Visualization
- XGBoost Regression
- Streamlit Web Application

---

## 📚 Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Data Visualization
- Data Preprocessing
- Feature Selection
- Machine Learning
- Random Forest Regression
- Model Evaluation
- Python Programming
- App deployement

---

## 👨‍💻 Author

**Surya Kant Singh**

GitHub: https://github.com/thakursinghsurya300-boop
