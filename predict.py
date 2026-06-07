import joblib

model = joblib.load("model/car_price_model.pkl")

data = [[
    2018,   # year
    27000,  # km_driven
    0,      # fuel (Petrol)
    0,      # seller_type (Dealer)
    0,      # transmission (Manual)
    1       # owner (First Owner)
]]

prediction = model.predict(data)

print("Predicted Price:", round(prediction[0], 2), "Lakhs")