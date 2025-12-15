import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Set global random state
RANDOM_STATE = 42

def load_data(path):
    """Loads the dataset from the CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def clean_data(df):
    """
    Handles missing values and outliers.
    """
    print("Shape before cleaning:", df.shape)
    
    # 1. Handling Missing Values
    # Numerical columns: Fill with Median
    num_cols = ['ram_gb', 'storage_gb', 'battery_mah', 'rear_camera_main_mp', 'display_inches']
    for col in num_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Categorical: Fill with Mode
    cat_cols = ['smartphone_brand', 'processor_brand']
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # 2. Outlier Detection & Handling (IQR Method) for Price
    # We focus on the target variable 'price_inr' to avoid training on extreme anomalies
    Q1 = df['price_inr'].quantile(0.25)
    Q3 = df['price_inr'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of removing to preserve data, or remove if extreme.
    # Let's remove for cleaner training as per instructions "Cap or remove".
    # Removing extreme outliers makes the model more robust for "typical" phones.
    df_clean = df[(df['price_inr'] >= lower_bound) & (df['price_inr'] <= upper_bound)].copy()
    
    print(f"Removed {len(df) - len(df_clean)} outliers based on Price IQR.")
    print("Shape after cleaning:", df_clean.shape)
    
    return df_clean

def train_and_evaluate():
    # Path to dataset
    DATA_PATH = r"d:\price-prediction-project\data\smartphones.csv"
    
    # 1. Load Data
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    # Display info
    print("\nDataset Info:")
    print(df.info())
    print("\nSample Data:")
    print(df.head())
    
    # 2. Data Cleaning
    print("\nCleaning data...")
    df = clean_data(df)
    
    # Select Features and Target
    # Based on API requirements: brand, ram, storage, battery, camera, screen_size, processor
    feature_cols = [
        'smartphone_brand', 
        'ram_gb', 
        'storage_gb', 
        'battery_mah', 
        'rear_camera_main_mp', 
        'display_inches', 
        'processor_brand'
    ]
    target_col = 'price_inr'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 3. Train-Test Split
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # 4. Preprocessing Pipeline
    # Numerical Features: Impute (just in case new nulls introduced or distinct from manual step) + Scale
    numeric_features = ['ram_gb', 'storage_gb', 'battery_mah', 'rear_camera_main_mp', 'display_inches']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Features: Impute + OneHotEncode
    categorical_features = ['smartphone_brand', 'processor_brand']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 5. Model Training
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    best_pipeline = None
    
    print("\nTraining models...")
    for name, model in models.items():
        # Create full pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        
        print(f"\n{name} Results:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = name
            best_pipeline = clf
            
    print(f"\nBest Model: {best_model} with R2: {best_score:.4f}")
    
    # 6. Save Model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "best_model.pkl")
    
    joblib.dump(best_pipeline, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_and_evaluate()
