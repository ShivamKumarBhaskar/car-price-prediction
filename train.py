import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("dataset/car_data.csv")

# Convert text columns into numbers
df["fuel"] = df["fuel"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2,
    "LPG": 3
})

df["seller_type"] = df["seller_type"].map({
    "Dealer": 0,
    "Individual": 1,
    "Trustmark Dealer": 2
})

df["transmission"] = df["transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

# Drop car name
df["fuel"] = df["fuel"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2,
    "LPG": 3
})

df["seller_type"] = df["seller_type"].map({
    "Dealer": 0,
    "Individual": 1,
    "Trustmark Dealer": 2
})

df["transmission"] = df["transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

df["owner"] = df["owner"].map({
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 0
})

# Train test split
# Drop name column
df.drop("name", axis=1, inplace=True)

# Features and target
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, pred)

print("Accuracy:", score)

# Save model
joblib.dump(model, "model/car_price_model.pkl")

print("Model Saved Successfully")