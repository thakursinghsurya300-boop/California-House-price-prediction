from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Create Figure
plt.figure(figsize=(16, 12))
plt.suptitle("Exploratory Data Analysis of California Housing Dataset",
             fontsize=18,
             fontweight='bold')

# ---------------------------------------------------------
# 1. California Map
# ---------------------------------------------------------
plt.subplot(2, 2, 1)

plt.scatter(
    X["Longitude"],
    X["Latitude"],
    c=y,
    cmap="viridis",
    alpha=0.5
)

plt.colorbar(label="Median House Value (× $100,000)")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.title("House Prices Across California")

# ---------------------------------------------------------
# 2. Median Income vs House Price
# ---------------------------------------------------------
plt.subplot(2, 2, 2)

plt.scatter(
    X["MedInc"],
    y,
    c=X["HouseAge"],
    cmap="viridis",
    alpha=0.5
)

plt.colorbar(label="Median House Age (Years)")
plt.xlabel("Median Income (× $10,000)")
plt.ylabel("Median House Value (× $100,000)")
plt.title("Median Income vs House Value")

# ---------------------------------------------------------
# 3. Average Rooms vs House Price
# ---------------------------------------------------------
plt.subplot(2, 2, 3)

plt.scatter(
    X["AveRooms"],
    y,
    c=X["AveBedrms"],
    cmap="viridis",
    alpha=0.5
)

plt.colorbar(label="Average Bedrooms per Household")
plt.xlabel("Average Rooms per Household")
plt.ylabel("Median House Value (× $100,000)")
plt.title("Average Rooms vs House Value")

# ---------------------------------------------------------
# 4. Population vs House Price
# ---------------------------------------------------------
plt.subplot(2, 2, 4)

plt.scatter(
    X["Population"],
    y,
    alpha=0.5,
)

plt.xlabel("Population (People)")
plt.ylabel("Median House Value (× $100,000)")
plt.title("Population vs House Value")

# Adjust spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

# Using train and test split to divide the data for training and testing purpose

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# Evaluating the model
# mae = mean_absolute_error(y_test, y_pred)
# rmse = root_mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("MAE :", mae)
# print("RMSE:", rmse)
# print("R² :", r2)
# Giving new data to check the price prediction
new_house = pd.DataFrame({
    "MedInc": [6.5],
    "HouseAge": [25],
    "AveRooms": [6.2],
    "AveBedrms": [1.2],
    "Population": [1800],
    "AveOccup": [3.1],
    "Latitude": [34.05],
    "Longitude": [-118.25]
})
prediction=model.predict(new_house)
print(f"Predicted House Value: ${prediction[0] * 100000:,.2f}")
