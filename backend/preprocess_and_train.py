import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import uuid

# Loading the dataset
df = pd.read_excel("VN Student Performance.xlsx")

# Handling missing values and "Passed away"
df['FATHER_JOB'] = df['FATHER_JOB'].replace('Passed away', 'Deceased')
df['MOTHER_JOB'] = df['MOTHER_JOB'].replace('Passed away', 'Deceased')

# Defining features and target variables
features = ['AGE', 'GENDER', 'RESIDENTIAL_AREA', 'FATHER_AGE', 'FATHER_JOB', 'MOTHER_AGE', 'MOTHER_JOB']
subjects = ['CHEMISTRY', 'HISTORY', 'LITERATURE', 'BIOLOGY', 'ENGLISH', 'MATH', 'PHYSICS']
X = df[features]
y = df[subjects]

# Encoding categorical variables
categorical_cols = ['GENDER', 'RESIDENTIAL_AREA', 'FATHER_JOB', 'MOTHER_JOB']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

# Creating DataFrame with encoded features
encoded_cols = encoder.get_feature_names_out(categorical_cols)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_cols, index=X.index)

# Scaling numerical features
numerical_cols = ['AGE', 'FATHER_AGE', 'MOTHER_AGE']
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X[numerical_cols])
X_numerical_df = pd.DataFrame(X_numerical, columns=numerical_cols, index=X.index)

# Combining encoded and numerical features
X_processed = pd.concat([X_numerical_df, X_encoded_df], axis=1)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Training a model for each subject and saving feature importances
models = {}
for subject in subjects:
    print(f"Training model for {subject}...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Performing cross-validation
    cv_scores = cross_val_score(model, X_processed, y[subject], cv=5, scoring='r2')
    print(f"{subject} - Cross-Validation R²: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
    
    # Training on full training set
    model.fit(X_train, y_train[subject])
    
    # Evaluating on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test[subject], y_pred)
    r2 = r2_score(y_test[subject], y_pred)
    print(f"{subject} - Test Mean Squared Error: {mse:.2f}, Test R² Score: {r2:.2f}")
    
    # Saving the model
    models[subject] = model
    joblib.dump(model, f"{subject.lower()}_model.pkl")
    
    # Saving feature importances
    importances = pd.Series(model.feature_importances_, index=X_processed.columns)
    joblib.dump(importances, f"{subject.lower()}_importances.pkl")

# Saving preprocessing objects
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Saving unique values for categorical inputs
unique_values = {
    'GENDER': df['GENDER'].unique().tolist(),
    'RESIDENTIAL_AREA': df['RESIDENTIAL_AREA'].unique().tolist(),
    'FATHER_JOB': df['FATHER_JOB'].unique().tolist(),
    'MOTHER_JOB': df['MOTHER_JOB'].unique().tolist()
}
joblib.dump(unique_values, 'unique_values.pkl')