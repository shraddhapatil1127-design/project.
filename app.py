from flask import Flask, render_template, request
import pickle
import numpy as np

# CRITICAL: Tell Flask to look in the current folder for index.html
app = Flask(__name__, template_folder=".")

# Load models safely
try:
    model = pickle.load(open('attrition_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Please run train_model.py first to create the pickle files!")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    inputs = {}
    
    if request.method == 'POST':
        try:
            inputs = {
                'Age': int(request.form['Age']),
                'MonthlyIncome': int(request.form['MonthlyIncome']),
                'JobSatisfaction': int(request.form['JobSatisfaction']),
                'WorkLifeBalance': int(request.form['WorkLifeBalance']),
                'YearsAtCompany': int(request.form['YearsAtCompany']),
                'OverTime': int(request.form['OverTime'])
            }
            
            features = np.array([[inputs['Age'], inputs['MonthlyIncome'], inputs['JobSatisfaction'], 
                                  inputs['WorkLifeBalance'], inputs['YearsAtCompany'], inputs['OverTime']]])
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1] * 100
            
            if prediction == 1:
                prediction_text = f"High Risk: This employee has a {probability:.1f}% chance of leaving."
            else:
                prediction_text = f"Low Risk: This employee has a {probability:.1f}% chance of staying."
        except Exception as e:
            prediction_text = f"Error processing prediction: {str(e)}"
            
        return render_template('index.html', prediction=prediction_text, inputs=inputs)

    # Default GET request
    return render_template('index.html', prediction=None, inputs={})

if __name__ == '__main__':
    app.run(debug=True)