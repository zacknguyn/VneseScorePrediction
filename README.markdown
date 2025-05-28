# Student Score Predictor 🚀

Predict student scores for seven subjects using Random Forest models. Features a modern React frontend with a vibrant UI, enhanced batch predictions, and a Flask backend.

## Features
- 📈 Single predictions with bar and radar charts
- 📊 Batch predictions with confidence intervals, feature importance, and summary report
- 📝 Downloadable results and summary via buttons
- 🔬 5-fold cross-validation
- 🖥️ Responsive UI with dark/light mode, animations
- ⏳ Robust error handling

## Project Structure
- `backend/`:
  - `app.py`: Flask backend.
  - `preprocess_and_train.py`: Model training.
  - `VN Student Performance.xlsx`: Dataset.
  - `batch_template.csv`: Batch template.
  - `*.pkl`: Models and preprocessing objects.
- `frontend/`:
  - `index.html`: React app.

## Setup
1. **Install dependencies**:
   ```bash
   pip install flask flask-cors pandas numpy scikit-learn openpyxl joblib fuzzywuzzy python-Levenshtein
   ```
2. **Ensure files**:
   - `VN Student Performance.xlsx` in `backend/`.
   - Use `batch_template.csv` for batch uploads.

## Running
1. **Train models**:
   ```bash
   cd backend
   python preprocess_and_train.py
   ```
2. **Run backend**:
   ```bash
   python app.py
   ```
3. **Serve frontend**:
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   - Open `http://localhost:3000`.

## Notes
- Scores are decimals (e.g., `5.0`).
- Batch results include confidence intervals and top feature contributions.
- Download results/summary via buttons.
- Requires internet for CDNs.

## Troubleshooting
- **Batch issues**:
  - Verify CSV headers.
  - Check Flask logs (`CSV columns: [...]`).
- **UI issues**:
  - Ensure CDN access.
  - Check browser console.

## Future Enhancements
- User authentication.
- Deployment to Heroku/Netlify.