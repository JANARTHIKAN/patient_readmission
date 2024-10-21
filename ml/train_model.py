import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('patient_readmission_dataset.csv')

# Handling missing values
data.fillna(method='ffill', inplace=True)

# Define features (X) and target (y)
categorical_cols = ['Gender', 'MaritalStatus', 'TreatmentType', 'Diagnosis', 'DischargeCondition', 
                    'FollowUpAppointment', 'InsuranceType']
numerical_cols = ['Age', 'MedicalHistory', 'LengthOfStay', 'NumberOfVisits', 'DistanceFromFacility']

X = data[categorical_cols + numerical_cols]
y = data['Readmitted']

# Splitting dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing step
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Building the full pipeline with preprocessing and classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model as model.pkl
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'")
