from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained successfully!")
print("✅ model.pkl has been created.")