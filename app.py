from flask import Flask,request,render_template
import numpy as np
import pickle
import os
from dotenv import load_dotenv
import mysql.connector

app=Flask(__name__)


model=pickle.load(open("model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))

# MySQL connection
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

cursor = db.cursor()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    CurrentEquipmentDays=float(request.form["CurrentEquipmentDays"])
    MonthsInService=float(request.form["MonthsInService"])
    MonthlyMinutes=float(request.form["MonthlyMinutes"])
    PercChangeMinutes=float(request.form["PercChangeMinutes"])
    TotalRecurringCharge=float(request.form["TotalRecurringCharge"])
    OffPeakCallsInOut=float(request.form["OffPeakCallsInOut"])
    PercChangeRevenues=float(request.form["PercChangeRevenues"])
    MonthlyRevenue=float(request.form["MonthlyRevenue"])
    PeakCallsInOut=float(request.form["PeakCallsInOut"])
    OverageMinutes=float(request.form["OverageMinutes"])
    ReceivedCalls=float(request.form["ReceivedCalls"])
    UnansweredCalls=float(request.form["UnansweredCalls"])

    features=np.array([[CurrentEquipmentDays, MonthsInService,
                          MonthlyMinutes, PercChangeMinutes,
                          TotalRecurringCharge, OffPeakCallsInOut,
                          PercChangeRevenues, MonthlyRevenue,
                          PeakCallsInOut, OverageMinutes,
                          ReceivedCalls, UnansweredCalls]])

    scaled=scaler.transform(features)

    prediction=model.predict(scaled)

    if prediction[0]==1:
        result="Customer Will Churn"
    else:
        result="Customer Will Stay"

    # Insert into MySQL
    query = """
    INSERT INTO predictions
    (CurrentEquipmentDays, MonthsInService, MonthlyMinutes,
    PercChangeMinutes, TotalRecurringCharge, OffPeakCallsInOut,
    PercChangeRevenues, MonthlyRevenue, PeakCallsInOut,
    OverageMinutes, ReceivedCalls, UnansweredCalls, prediction)

    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    values=(
        CurrentEquipmentDays, MonthsInService, MonthlyMinutes,
        PercChangeMinutes, TotalRecurringCharge, OffPeakCallsInOut,
        PercChangeRevenues, MonthlyRevenue, PeakCallsInOut,
        OverageMinutes, ReceivedCalls, UnansweredCalls, result
    )

    cursor.execute(query,values)
    db.commit()

    return render_template("index.html",prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)