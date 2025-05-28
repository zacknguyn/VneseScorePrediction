import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import io
import logging
from fuzzywuzzy import process
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

# Setting up logging
logging.basicConfig(level=logging.DEBUG)

# Loading the pre-trained models and feature importances
subjects = ['CHEMISTRY', 'HISTORY', 'LITERATURE', 'BIOLOGY', 'ENGLISH', 'MATH', 'PHYSICS']
models = {subject: joblib.load(f"{subject.lower()}_model.pkl") for subject in subjects}
importances = {subject: joblib.load(f"{subject.lower()}_importances.pkl").to_dict() for subject in subjects}
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Single prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([{
            'AGE': float(data['age']),
            'GENDER': data['gender'],
            'RESIDENTIAL_AREA': data['residentialArea'],
            'FATHER_AGE': float(data['fatherAge']),
            'FATHER_JOB': data['fatherJob'],
            'MOTHER_AGE': float(data['motherAge']),
            'MOTHER_JOB': data['motherJob']
        }])

        categorical_cols = ['GENDER', 'RESIDENTIAL_AREA', 'FATHER_JOB', 'MOTHER_JOB']
        numerical_cols = ['AGE', 'FATHER_AGE', 'MOTHER_AGE']

        X_encoded = encoder.transform(input_data[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_cols)

        X_numerical = scaler.transform(input_data[numerical_cols])
        X_numerical_df = pd.DataFrame(X_numerical, columns=numerical_cols)

        X_processed = pd.concat([X_numerical_df, X_encoded_df], axis=1)
        feature_names = X_processed.columns.tolist()

        predictions = {}
        local_importances = {}
        for subject in subjects:
            model = models[subject]
            prediction = model.predict(X_processed.values)[0]
            predictions[subject.lower()] = float(np.clip(prediction, 0, 10))

            # Local feature importance (tree-based contribution)
            contribs = {col: 0 for col in feature_names}
            for tree in model.estimators_:
                tree_contrib = tree.predict(X_processed.values)[0] - tree.predict(np.zeros_like(X_processed.values))[0]
                for j, col in enumerate(feature_names):
                    contribs[col] += tree_contrib / len(model.estimators_)
            local_importances[subject.lower()] = [
                {'feature': col, 'contribution': float(contribs[col])}
                for col in sorted(contribs, key=lambda x: abs(contribs[x]), reverse=True)[:5]
            ]

        return jsonify({
            'predictions': predictions,
            'importances': {subject.lower(): importances[subject] for subject in subjects},
            'local_importances': local_importances
        }), 200
    except Exception as e:
        logging.error(f"Single prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Batch prediction endpoint
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        input_data = pd.read_csv(file, skipinitialspace=True, comment='#')
        if input_data.empty:
            return jsonify({'error': 'CSV file is empty or contains only comments'}), 400

        logging.debug(f"CSV columns: {list(input_data.columns)}")

        required_cols = ['age', 'gender', 'residentialArea', 'fatherAge', 'fatherJob', 'motherAge', 'motherJob']
        input_data.columns = input_data.columns.str.strip().str.lower()

        column_mapping = {}
        for req_col in required_cols:
            match = process.extractOne(req_col, input_data.columns, score_cutoff=85)
            if match:
                column_mapping[match[0]] = req_col
            else:
                column_mapping[req_col] = req_col

        input_data.rename(columns=column_mapping, inplace=True)

        missing_cols = [col for col in required_cols if col not in input_data.columns]
        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_cols)}. '
                         f'Expected: {", ".join(required_cols)}. Found: {", ".join(input_data.columns)}'
            }), 400

        input_data.columns = [col.upper() for col in input_data.columns]
        input_data = input_data.rename(columns={
            'RESIDENTIALAREA': 'RESIDENTIAL_AREA',
            'FATHERAGE': 'FATHER_AGE',
            'FATHERJOB': 'FATHER_JOB',
            'MOTHERAGE': 'MOTHER_AGE',
            'MOTHERJOB': 'MOTHER_JOB'
        })

        required_upper = ['AGE', 'GENDER', 'RESIDENTIAL_AREA', 'FATHER_AGE', 'FATHER_JOB', 'MOTHER_AGE', 'MOTHER_JOB']
        missing_upper = [col for col in required_upper if col not in input_data.columns]
        if missing_upper:
            return jsonify({'error': f'After renaming, missing columns: {", ".join(missing_upper)}'}), 400

        input_data['AGE'] = pd.to_numeric(input_data['AGE'], errors='coerce')
        input_data['FATHER_AGE'] = pd.to_numeric(input_data['FATHER_AGE'], errors='coerce')
        input_data['MOTHER_AGE'] = pd.to_numeric(input_data['MOTHER_AGE'], errors='coerce')

        if input_data[['AGE', 'FATHER_AGE', 'MOTHER_AGE']].isna().any().any():
            return jsonify({'error': 'Invalid numerical values in age columns'}), 400

        categorical_cols = ['GENDER', 'RESIDENTIAL_AREA', 'FATHER_JOB', 'MOTHER_JOB']
        numerical_cols = ['AGE', 'FATHER_AGE', 'MOTHER_AGE']

        try:
            X_encoded = encoder.transform(input_data[categorical_cols])
        except ValueError as e:
            return jsonify({'error': f'Invalid categorical values: {str(e)}'}), 400
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_cols)

        X_numerical = scaler.transform(input_data[numerical_cols])
        X_numerical_df = pd.DataFrame(X_numerical, columns=numerical_cols)

        X_processed = pd.concat([X_numerical_df, X_encoded_df], axis=1)
        feature_names = X_processed.columns.tolist()

        results = input_data[required_upper].copy()
        summary_stats = {'count': len(input_data)}

        for subject in subjects:
            model = models[subject]
            predictions = model.predict(X_processed.values)
            predictions = np.clip(predictions, 0, 10)

            # Confidence intervals (95%)
            tree_preds = np.array([tree.predict(X_processed.values) for tree in model.estimators_])
            lower_ci = np.percentile(tree_preds, 2.5, axis=0)
            upper_ci = np.percentile(tree_preds, 97.5, axis=0)
            lower_ci = np.clip(lower_ci, 0, 10)
            upper_ci = np.clip(upper_ci, 0, 10)

            # Local feature importance
            feature_contribs = []
            for i in range(len(input_data)):
                contribs = {col: 0 for col in feature_names}
                for tree in model.estimators_:
                    tree_contribs = tree.predict(X_processed.values[i:i+1])[0] - tree.predict(np.zeros_like(X_processed.values[i:i+1]))[0]
                    for j, col in enumerate(feature_names):
                        contribs[col] += tree_contribs / len(model.estimators_)
                top_contrib = max(contribs.items(), key=lambda x: abs(x[1]))
                feature_contribs.append(f"{top_contrib[0]}:{top_contrib[1]:.2f}")

            results[f"{subject.lower()}_score"] = [f"{x:.1f}" for x in predictions]
            results[f"{subject.lower()}_lower_ci"] = [f"{x:.1f}" for x in lower_ci]
            results[f"{subject.lower()}_upper_ci"] = [f"{x:.1f}" for x in upper_ci]
            results[f"{subject.lower()}_top_feature"] = feature_contribs

            # Summary stats
            summary_stats[f"{subject.lower()}_mean"] = predictions.mean()
            summary_stats[f"{subject.lower()}_min"] = predictions.min()
            summary_stats[f"{subject.lower()}_max"] = predictions.max()

        for col in ['AGE', 'FATHER_AGE', 'MOTHER_AGE']:
            results[col] = results[col].apply(lambda x: f"{x:.0f}")

        summary_df = pd.DataFrame([{
            'Metric': f"{subject} Mean Score",
            'Value': f"{summary_stats[f'{subject.lower()}_mean']:.1f}"
        } for subject in subjects] + [
            {'Metric': f"{subject} Min Score", 'Value': f"{summary_stats[f'{subject.lower()}_min']:.1f}"}
            for subject in subjects
        ] + [
            {'Metric': f"{subject} Max Score", 'Value': f"{summary_stats[f'{subject.lower()}_max']:.1f}"}
            for subject in subjects
        ] + [{'Metric': 'Total Students', 'Value': summary_stats['count']}])
        
        summary_output = io.StringIO()
        summary_df.to_csv(summary_output, index=False)
        summary_output.seek(0)

        results_output = io.StringIO()
        results.to_csv(results_output, index=False)
        results_output.seek(0)

        return jsonify({
            'results_csv': results_output.getvalue(),
            'summary_csv': summary_output.getvalue(),
            'headers': results.columns.tolist(),
            'data': results.to_dict(orient='records')
        }), 200
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)