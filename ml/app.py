from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        age = request.form['age']
        gender = request.form['gender']
        marital_status = request.form['marital_status']
        medical_history = request.form['medical_history']
        treatment_type = request.form['treatment_type']
        length_of_stay = request.form['length_of_stay']
        number_of_visits = request.form['number_of_visits']
        diagnosis = request.form['diagnosis']
        discharge_condition = request.form['discharge_condition']
        follow_up_appointment = request.form['follow_up_appointment']
        insurance_type = request.form['insurance_type']
        distance_from_facility = request.form['distance_from_facility']

        # Create input dictionary
        input_data = {
            'Age': [age],
            'Gender': [gender],
            'MaritalStatus': [marital_status],
            'MedicalHistory': [medical_history],
            'TreatmentType': [treatment_type],
            'LengthOfStay': [length_of_stay],
            'NumberOfVisits': [number_of_visits],
            'Diagnosis': [diagnosis],
            'DischargeCondition': [discharge_condition],
            'FollowUpAppointment': [follow_up_appointment],
            'InsuranceType': [insurance_type],
            'DistanceFromFacility': [distance_from_facility]
        }

        # Convert the input data into a pandas DataFrame
        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(input_df)

        # Result interpretation
        result = "Readmitted" if prediction[0] == 'Yes' else "Not Readmitted"

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
