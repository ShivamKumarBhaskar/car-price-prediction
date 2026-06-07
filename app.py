from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/car_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

   

    year = int(request.form["year"])
    kms_driven = int(request.form["kms_driven"])
    fuel_type = int(request.form["fuel_type"])
    seller_type = int(request.form["seller_type"])
    transmission = int(request.form["transmission"])
    owner = int(request.form["owner"])

    data = [[
        year,
        kms_driven,
        fuel_type,
        seller_type,
        transmission,
        owner
    ]]

    prediction = model.predict(data)

    result = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Price: ₹{result}"
    )

    prediction = model.predict(data)

    result = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Car Price: {result} Lakhs"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)