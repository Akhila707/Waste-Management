import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load the original data
try:
    df = pd.read_csv('/Users/anubhavverma/Desktop/waste_management/waste_features_minimal.csv')
    print("Loaded data from CSV")
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features and target
    feature_columns = [
        'hog_0', 'hog_1', 'hog_2', 'hog_3', 'hog_4', 'hog_5', 'hog_6', 'hog_7',
        'mean_blue', 'std_blue', 'mean_green', 'std_green', 'mean_red', 'std_red',
        'area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 'solidity'
    ]
    
    X = df[feature_columns].values
    y = df['label'].values
    
except Exception as e:
    print(f"Could not load CSV: {e}")
    print("Creating sample data instead...")
    
    # Create sample data
    bio_data = np.array([
        [0.24968789517879486, 0.24968789517879486, 0.24968789517879486, 0.24968789517879486, 
         0.24968789517879486, 0.24968789517879486, 0.24968789517879486, 0.24968789517879486,
         107.515380859375, 52.392237409495266, 130.0322265625, 55.97316806720806,
         164.912353515625, 58.79390716107964, 2028.5, 646.4234410524368,
         0.06100292827262692, 0.9838709677419355, 0.5363564251718668, 0.7095138160195873],
        [0.27407708764076233, 0.27407708764076233, 0.27407708764076233, 0.22276535630226135,
         0.27407708764076233, 0.27407708764076233, 0.27407708764076233, 0.27407708764076233,
         30.624267578125, 61.39163460730238, 35.907470703125, 72.05906785821507,
         39.938232421875, 80.31720768790407, 456.0, 116.71067690849304,
         0.4206816133790508, 1.4444444444444444, 0.43304843304843305, 0.8475836431226765]
    ])

    non_bio_data = np.array([
        [0.05446256324648857, 0.04963976517319679, 0.04757774993777275, 0.0694064125418663,
         0.29632413387298584, 0.29632413387298584, 0.29632413387298584, 0.29632413387298584,
         8.024658203125, 24.79082663672984, 10.169921875, 29.45873924598007,
         12.209228515625, 34.216922365441114, 60.5, 106.18376553058624,
         0.06742935320672053, 0.30303030303030304, 0.18333333333333332, 0.3507246376811594],
        [0.25134503841400146, 0.25134503841400146, 0.25134503841400146, 0.25134503841400146,
         0.25134503841400146, 0.25134503841400146, 0.25134503841400146, 0.25134503841400146,
         74.45263671875, 46.759213276645696, 94.273681640625, 51.84199863979937,
         95.57861328125, 44.052810011984604, 898.0, 249.76449930667877,
         0.18089425825300617, 0.9302325581395349, 0.522093023255814, 0.7473990844777362]
    ])

    X = np.vstack([bio_data, non_bio_data])
    y = np.array([0, 0, 1, 1])  # 0: biodegradable, 1: non-biodegradable

print(f"Training data shape: {X.shape}")
print(f"Labels: {y}")
print(f"Biodegradable samples: {sum(y == 0)}")
print(f"Non-biodegradable samples: {sum(y == 1)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model with basic configuration to avoid compatibility issues
model = RandomForestClassifier(
    n_estimators=50,  # Reduced number of trees
    max_depth=10,     # Limited depth
    random_state=42,
    n_jobs=1          # Single thread to avoid multiprocessing issues
)
model.fit(X_scaled, y)

# Save model and scaler with a different method
try:
    # Try saving with joblib first
    joblib.dump(model, 'random_forest_model_fixed.pkl')
    joblib.dump(scaler, 'scaler_fixed.pkl')
    print("Model saved with joblib successfully!")
except Exception as e:
    print(f"Joblib save failed: {e}")
    # Fallback to pickle
    import pickle
    with open('random_forest_model_fixed.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler_fixed.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model saved with pickle successfully!")

print("Model training and saving completed!")