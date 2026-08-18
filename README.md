# The Appraisal Office — California Housing Price Predictor

**Task 3 · Arch Technologies Machine Learning Internship · Month 2**

A Random Forest Regressor trained on the California Housing dataset, wrapped in a
Streamlit web app. Enter a property's stats (location, structure, community) and get
an estimated market value with a confidence range, a location map, and a feature
importance breakdown.

## Results (held-out test set)
- R² Score: 0.809
- MAE: $32,354
- RMSE: $50,000

## How to run it locally
1. Open a terminal in this folder.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py
   (or `python -m streamlit run app.py` if `streamlit` isn't on your PATH)
4. It opens automatically in your browser (usually http://localhost:8501)

## Files
- app.py                -> the web app (run this)
- train_model.py        -> the training script (already run once)
- housing_model.joblib  -> trained Random Forest model
- feature_cols.joblib   -> exact feature column order the model expects
- metrics.json          -> saved evaluation metrics + feature importances
- housing.csv           -> the California Housing dataset used for training

## Retraining (optional)
   python train_model.py

## Approach
1. Clean: fill missing `total_bedrooms` values with the median.
2. Feature engineering: convert raw totals into per-household ratios
   (rooms_per_household, bedrooms_per_room, population_per_household) —
   these are far more predictive than the raw block totals.
3. One-hot encode the `ocean_proximity` categorical feature.
4. Train a Random Forest Regressor (200 trees) — chosen because housing price
   relationships are non-linear (e.g. price doesn't scale linearly with income),
   and it gives feature importance for free.
5. Evaluate with R², MAE, and RMSE on a 20% held-out test split.

## Note on the dataset
Kaggle requires API credentials to download programmatically, which weren't available
in the development environment. This app uses the California Housing Prices dataset
(20,640 records) from a public GitHub mirror — the same dataset distributed on Kaggle
for this exact task, sourced from the 1990 California census.
