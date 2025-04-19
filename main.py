from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

# Load model
model = None
model_path = os.path.join(os.path.dirname(__file__), 'wp.pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print(f"Model not found at: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  p: float = Form(...),
                  tmax: float = Form(...),
                  tmin: float = Form(...),
                  w: float = Form(...)):
    if model is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model not loaded"})

    data = np.array([[p, tmax, tmin, w]])

    try:
        pred = model.predict(data)[0]
        values_mapping = {0: "drizzle", 1: "rain", 2: "sun", 3: "snow", 4: "fog"}
        values = values_mapping.get(pred, str(pred))
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error Making Prediction: {e}"})

    return templates.TemplateResponse("index.html", {"request": request, "values": values})




# API route
class Input(BaseModel):
    p: float
    tmax: float
    tmin: float
    w: float

class Output(BaseModel):
    values: str



@app.post("/api/predict", response_model=Output)
async def api_predict(w: Input):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = [[w.p, w.tmax, w.tmin, w.w]]

    try:
        pred = model.predict(data)[0]
        values_mapping = {0: "drizzle", 1: "rain", 2: "sun", 3: "snow", 4: "fog"}
        values = values_mapping.get(pred, str(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

    return Output(values=values)
