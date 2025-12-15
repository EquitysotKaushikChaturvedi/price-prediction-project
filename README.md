# Smartphone Price Prediction Project 

Welcome to the **Smartphone Price Prediction Project**! This project is an end-to-end machine learning application designed to predict the price of a smartphone based on its specifications (like Brand, RAM, Storage, etc.). It comes with a complete backend (FastAPI), a stylish black-and-white frontend, and a training pipeline.

##  Project Structure

```
price-prediction-project/
│
├── data/
│   └── smartphones.csv       # The raw dataset
│
├── notebooks/
│   └── analysis.ipynb        # Jupyter Notebook for EDA and visualization
│
├── models/
│   └── best_model.pkl        # The saved trained model (Pipeline)
│
├── app/
│   ├── main.py               # FastAPI application entry point
│   ├── schema.py             # Pydantic models for data validation
│   ├── templates/
│   │   └── index.html        # Frontend UI
│   └── static/
│       └── style.css         # Styling (B&W Theme)
│
├── train.py                  # Script to train and save the model
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Dataset Explanation

I used the `smartphones.csv` dataset found in the "KAGGLE.COM" folder.
It contains columns like:
- `smartphone_brand`: The manufacturer (e.g., Apple, Samsung).
- `price_inr`: The target variable (Price in Rupees).
- `ram_gb`, `storage_gb`, `battery_mah`: Key performace specs.
- `rear_camera_main_mp`, `display_inches`: Hardware specs.
- `processor_brand`: The brain of the phone (e.g., Snapdragon, Apple).

##  How Preprocessing Works

Before feeding data into the model, I perform several cleaning steps (`train.py`):
1. **Missing Values**:
   - Numerical columns (e.g., RAM) are filled with the **median**.
   - Categorical columns (e.g., Brand) are filled with the **most frequent value (mode)**.
2. **Outliers**:
   - I use the IQR (Interquartile Range) method to detect extreme prices and remove those rows to ensure the model learns generic trends rather than anomalies.
3. **Encoding & Scaling**:
   - **OneHotEncoder**: Converts brands and processors into a format the model understands (0s and 1s).
   - **StandardScaler**: Adjusts numerical values like Battery (3000-5000) and RAM (4-12) to the same scale so one doesn't dominate the other.

##  Model Comparison

I trained two models to see which one performs better:
1. **Linear Regression**: A simple model that fits a straight line. Good for understanding trends but struggles with complex patterns.
2. **Random Forest Regressor**: An ensemble of decision trees. Much better at capturing complex, non-linear relationships in data.

**Result**: The **Random Forest** model generally performed better (higher R² score and lower RMSE), so it is selected and saved as `models/best_model.pkl`.

##  How to Train the Model

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python train.py
   ```
   This will load data, clean it, train the models, print evaluation metrics, and save the best model to the `models/` directory.

##  How to Run the API & Frontend

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```
2. Open your browser and go to:
   ```
   http://localhost:8000
   ```
   You will see the **Black & White UI**.
   - **Feature**: If you select "Apple" as the brand, the form automatically adjusts to show only Apple-relevant options (like A-series chips) and hides Android-specific options.

##  API Usage

You can also use the API directly via tools like Postman or `curl`.

**Endpoint**: `POST /predict`

**Sample Request Body**:
```json
{
  "brand": "Samsung",
  "ram": 8,
  "storage": 128,
  "battery": 5000,
  "camera": 64,
  "screen_size": 6.5,
  "processor": "Snapdragon"
}
```

**Response**:
```json
{
  "predicted_price": 24999.0,
  "currency": "INR",
  "details": { ... }
}
```

##  Technologies Used

- **Python**: Core language.
- **Pandas & NumPy**: Data manipulation.
- **Scikit-Learn**: Machine Learning.
- **FastAPI**: Backend API.
- **HTML/CSS/JS**: Frontend UI.
- **Joblib**: Model serialization.
- **Matplotlib/Seaborn**: Visualization.

---

<img width="1920" height="3018" alt="image" src="https://github.com/user-attachments/assets/c86a76ef-830b-4ea4-a9bf-781922cf6735" />

--- 



