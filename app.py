# from flask import Flask, render_template, request
# import pandas as pd
# import joblib

# app = Flask(__name__)
# model = joblib.load("teleco_churn_model.pkl")  # load your trained model

# @app.route("/", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         try:
#             input_data = {
#                 'customerID': request.form['customerID'],
#                 'gender': request.form['gender'],
#                 'SeniorCitizen': 1 if request.form['SeniorCitizen'].lower() == 'yes' else 0,
#                 'Partner': request.form['Partner'],
#                 'Dependents': request.form['Dependents'],
#                 'tenure': int(request.form['tenure']),
#                 'PhoneService': request.form['PhoneService'],
#                 'MultipleLines': request.form['MultipleLines'],
#                 'InternetService': request.form['InternetService'],
#                 'OnlineSecurity': request.form['OnlineSecurity'],
#                 'OnlineBackup': request.form['OnlineBackup'],
#                 'DeviceProtection': request.form['DeviceProtection'],
#                 'TechSupport': request.form['TechSupport'],
#                 'StreamingTV': request.form['StreamingTV'],
#                 'StreamingMovies': request.form['StreamingMovies'],
#                 'Contract': request.form['Contract'],
#                 'PaperlessBilling': request.form['PaperlessBilling'],
#                 'PaymentMethod': request.form['PaymentMethod'],
#                 'MonthlyCharges': float(request.form['MonthlyCharges']),
#                 'TotalCharges': float(request.form['TotalCharges'])  # convert to float if it's numeric
#             }

#             # Create a copy for form data (keeping original string values for form display)
#             form_data = dict(request.form)

#             # Create DataFrame
#             df = pd.DataFrame([input_data])

#             # Drop customerID if not used for prediction
#             df = df.drop(columns=['customerID'])

#             # --- Preprocessing: Apply One-Hot Encoding (get_dummies) ---
#             df = pd.get_dummies(df)

#             # Convert any boolean columns to int (if applicable)
#             bool_cols = df.select_dtypes(include='bool').columns
#             df[bool_cols] = df[bool_cols].astype(int)

#             # --- Align columns with the model's training features ---
#             model_features = model.feature_names_in_  # Only available if saved with sklearn 1.0+
#             for col in model_features:
#                 if col not in df.columns:
#                     df[col] = 0  # Add missing columns with default 0
#             df = df[model_features]  # Reorder columns to match model

#             # Make prediction
#             prediction = model.predict(df)[0]
            
#             # Convert prediction to user-friendly message
#             if prediction == 1:
#                 prediction_message = "⚠️ HIGH RISK: This customer is likely to churn"
#             else:
#                 prediction_message = "✅ LOW RISK: This customer is likely to stay"

#             return render_template("form.html", prediction=prediction_message, form_data=form_data)
            
#         except Exception as e:
#             error_message = f"❌ Error: {str(e)}"
#             # Try to get form data even if there's an error
#             form_data = dict(request.form) if request.form else {}
#             return render_template("form.html", prediction=error_message, form_data=form_data)

#     return render_template("form.html", prediction=None, form_data={})


# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("teleco_churn_model.pkl")  # Load trained model

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            input_data = {
                'customerID': request.form['customerID'],
                'gender': request.form['gender'],
                'SeniorCitizen': 1 if request.form['SeniorCitizen'].lower() == 'yes' else 0,
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'tenure': int(request.form['tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges'])
            }

            form_data = dict(request.form)
            df = pd.DataFrame([input_data])
            df = df.drop(columns=['customerID'])
            df = pd.get_dummies(df)
            bool_cols = df.select_dtypes(include='bool').columns
            df[bool_cols] = df[bool_cols].astype(int)

            model_features = model.feature_names_in_
            for col in model_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[model_features]

            prediction = model.predict(df)[0]
            confidence = model.predict_proba(df)[0][prediction]
            confidence_percentage = round(confidence * 100, 2)

            if prediction == 1:
                prediction_message = "⚠️ HIGH RISK: This customer is likely to churn"
                risk_class = "high-risk"
            else:
                prediction_message = "✅ LOW RISK: This customer is likely to stay"
                risk_class = "low-risk"

            final_output = f"{prediction_message}<br><strong>Confidence:</strong> {confidence_percentage}%"

            return render_template("form.html", prediction=final_output, risk_class=risk_class, form_data=form_data)

        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            form_data = dict(request.form) if request.form else {}
            return render_template("form.html", prediction=error_message, risk_class=None, form_data=form_data)

    return render_template("form.html", prediction=None, risk_class=None, form_data={})


if __name__ == "__main__":
    app.run(debug=True)
