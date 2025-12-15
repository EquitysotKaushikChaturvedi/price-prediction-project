from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os
from app.schema import SmartphoneInput

app = FastAPI(title="Smartphone Price Predictor")

# Load Model
MODEL_PATH = os.path.join("models", "best_model.pkl")
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
else:
    model_pipeline = None
    print(f"Warning: Model not found at {MODEL_PATH}")

# Setup Static and Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_price(data: SmartphoneInput):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare input dataframe matching training columns
        # TrainingCols: ['smartphone_brand', 'ram_gb', 'storage_gb', 'battery_mah', 'rear_camera_main_mp', 'display_inches', 'processor_brand']
        
        input_dict = {
            "smartphone_brand": [data.brand],
            "ram_gb": [data.ram],
            "storage_gb": [data.storage],
            "battery_mah": [data.battery],
            "rear_camera_main_mp": [data.camera],
            "display_inches": [data.screen_size],
            "processor_brand": [data.processor]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Predict
        prediction = model_pipeline.predict(input_df)
        predicted_price = float(prediction[0])
        
        return {
            "predicted_price": round(predicted_price, 2),
            "currency": "INR",
            "details": data.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
