from flask import Flask, request, render_template
import numpy as np
import pickle

# ---------------- LOAD OBJECTS ----------------
app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ===== GET INPUTS =====
        age = float(request.form["Age"])
        tenure = float(request.form["Tenure"])
        usage = float(request.form["Usage Frequency"])
        support = float(request.form["Support Calls"])
        payment_delay = float(request.form["Payment Delay"])
        total_calls = float(request.form["Total Calls"])
        gender = request.form["Gender"]

        # ===== ONE HOT ENCODE GENDER =====
        gender_encoded = encoder.transform([[gender]])

        # combine numerical + encoded features
        features = np.concatenate(
            ([age, tenure, usage, support, payment_delay, total_calls],
             gender_encoded[0])
        )

        # reshape for model
        features = np.array(features).reshape(1, -1)

        # ===== SCALE =====
        features = scaler.transform(features)

        # ===== PREDICT =====
        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Sreehari will join ❌"
        else:
            result = "Customer Will Stay ✅"

        return render_template(
            "index.html",
            prediction_text=result
        )

    except Exception as e:
        return str(e)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
