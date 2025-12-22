# app.py

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
from pipeline_utils import model_pipeline
import spacy

from pipeline_utils import combine_text_columns, CountCharacter, SpacyLemmas, SpacyNumericFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder




# Create FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define features
numeric_features = ['Age', 'Positive Feedback Count']
categorical_features = ['Clothing ID', 'Division Name', 'Department Name', 'Class Name']
text_features = ['Title', 'Review Text']

# Load trained pipeline
with open("trained_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Home page with input form
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Predict endpoint
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    positive_feedback_count: int = Form(...),
    clothing_id: str = Form(...),
    division_name: str = Form(...),
    department_name: str = Form(...),
    class_name: str = Form(...),
    title: str = Form(...),
    review: str = Form(...)
):
    # Prepare input as DataFrame
    new_data = pd.DataFrame({
        "Age": [age],
        "Positive Feedback Count": [positive_feedback_count],
        "Clothing ID": [clothing_id],
        "Division Name": [division_name],
        "Department Name": [department_name],
        "Class Name": [class_name],
        "Title": [title],
        "Review Text": [review]
    })

    # Make prediction
    prediction = pipeline.predict(new_data)[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "age": age,
        "positive_feedback_count": positive_feedback_count,
        "clothing_id": clothing_id,
        "division_name": division_name,
        "department_name": department_name,
        "class_name": class_name,
        "title": title,
        "review": review,
        "prediction": prediction
    })
