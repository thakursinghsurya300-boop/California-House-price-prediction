# 🏠 California House Price Prediction

An end-to-end Machine Learning project that predicts California house prices using **Random Forest Regression**. The project includes model training, evaluation, and a **Streamlit web application** for real-time house price prediction.

## 🌐 Live Demo

**Try the application here:**
https://ca-house-value-predictor.streamlit.app/

---

## 📸 Application Preview

### Home Page

![Home Page](images/home.png)

### Prediction Result

![Prediction Result](images/prediction.png)

---

## 📌 Project Overview

This project uses the **California Housing Dataset** from Scikit-learn to predict the median house value of a California district based on housing and demographic features.

It demonstrates a complete machine learning workflow, including data preprocessing, model training, evaluation, and deployment as an interactive web application.

---

## ✨ Features

* Predict California house prices in real time.
* Interactive and user-friendly Streamlit interface.
* Random Forest Regression model.
* Fast and accurate predictions.
* Deployed on Streamlit Community Cloud.

---

## 📊 Dataset

* **Dataset:** California Housing Dataset
* **Source:** `sklearn.datasets.fetch_california_housing()`
* **Target Variable:** `MedHouseVal`

### Features Used

* Median Income (`MedInc`)
* House Age (`HouseAge`)
* Average Rooms (`AveRooms`)
* Average Bedrooms (`AveBedrms`)
* Population (`Population`)
* Average Occupancy (`AveOccup`)
* Latitude
* Longitude

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit

---

## 🤖 Machine Learning Workflow

* Load California Housing Dataset
* Data Preprocessing
* Train-Test Split
* Train Random Forest Regressor
* Model Evaluation
* House Price Prediction
* Deploy using Streamlit Community Cloud

---

## 📈 Model Performance

| Metric                         |      Score |
| ------------------------------ | ---------: |
| Mean Absolute Error (MAE)      | **0.3384** |
| Root Mean Squared Error (RMSE) | **0.5224** |
| R² Score                       | **0.7936** |

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/thakursinghsurya300-boop/California-House-price-prediction.git
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## 💡 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Feature importance visualization
* Interactive data visualizations
* Enhanced UI/UX
* Model comparison with XGBoost and Gradient Boosting

---

## 👨‍💻 Author

**Surya Kant Singh**

GitHub: https://github.com/thakursinghsurya300-boop
