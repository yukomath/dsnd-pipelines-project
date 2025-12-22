# Fashion Trend Forecasting with ML Pipelines Using SGD Classifier

## Project Overview
This project builds a machine learning pipeline to predict product recommendations from customer reviews for **"StyleSense,"** a rapidly growing online women's clothing store known for its trendy and affordable fashion.

The pipeline analyzes review text, customer age, product category, and other relevant features to predict whether a customer would recommend a product. Automating this process helps:  
- Gain insights into customer satisfaction  
- Identify trending products  
- Support data-driven decisions for the retailer  

**Model:** We use an **SGDClassifier** for classification. 

---


## Requirements

To install dependencies:
pip install -r requirements.txt

Ensure `spaCy` English model is installed:
python -m spacy download en_core_web_sm

***Note:*** To install spaCy, it may be dependent on your machine (especially if using a Mac). You will need to look up your machine type and use the appropriate pip/conda install from spaCy's page: 
https://spacy.io/usage



---

## Repository Structure and File Description
dsnd-pipelines-project
├── README.md # Project description and instructions (this file)
├── data/
│ └── reviews.csv # Customer reviews dataset
│                 # Contains anonymized and cleaned customer reviews. Includes numerical, categorical, and text features.  
├── Pipeline-SGDclassifier.ipynb    # Notebook with ML pipeline using SGDClassifier
├── pipeline_utils.py     #　a module that defines data preprocessing and feature engineering pipelines to transform input data for the machine learning model　(only for App)
├── pipeline_app.ipynb     # This notebook is only for app by importing pipeline_utils.py 
├── trained_pipeline.pkl   # pkl file made by pipeline_app.ipynb
├── app.py              #　a FastAPI app that serves a form, takes user input, runs it through a trained ML pipeline, and returns the prediction.
├── templates/
│   └── index.html                 # HTML form for dashboard input
├── requirements.txt    #Lists all Python libraries and their versions required to run the project. 
│                       #This file allows anyone to install the exact dependencies using pip install -r requirements.txt.
├── check_versions.ipynb　　　# This notebook is only for checking the versions of used libraries 
│　　　　　　　　　　　　　　　　　# and generating a requirements.txt file.
├── LICENSE.txt 
└── CODEOWNERS

---


## Notebook Instructions

The notebook (**Pipeline-SGDclassifier.ipynb**) provides step-by-step instructions, including:

- **Data Exploration:** Checking data types, inspecting top products by recommendation count and rate  
- **Feature Engineering:** Numerical scaling, categorical encoding, and text pipelines  
- **Pipeline Construction:** Using `ColumnTransformer` to combine numerical, categorical, and text pipelines  
- **Model Training:** Fitting **SGDClassifier** within the pipeline  
- **Evaluation:** Computing Accuracy, Precision, Recall, and F1 Score  
- **Hyperparameter Tuning:** Using `RandomizedSearchCV` to optimize **SGDClassifier**
- **Final Evaluation:** Selecting the best model and reporting metrics on test data


---
## Workflow of Main Notebook(Pipeline-SGDclassifier.ipynb)
### 1. Project Overview : short introduction of this project
This project builds a machine learning pipeline to predict product recommendations for "StyleSense" from customer reviews.
It uses review text, customer age, product category, and other features to predict recommendations.
The goal is to gain insights into customer satisfaction and support data-driven decisions.
2. Data Understanding
Data Loading

### 2. Data Understanding
### Data Loading

### Data Structure
The features can be summarized as the following:

Clothing ID: Integer Categorical variable that refers to the specific piece being reviewed.
Age: Positive Integer variable of the reviewers age.
Title: String variable for the title of the review.
Review Text: String variable for the review body.
Positive Feedback Count: Positive Integer documenting the number of other customers who found this review positive.
Division Name: Categorical name of the product high level division.
Department Name: Categorical name of the product department name.
Class Name: Categorical name of the product class name.


The target:

Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.

- Target variable: `Recommended IND` (1 = recommended, 0 = not recommended)  
- Features: 8 columns including numerical, categorical, and text data  

**Feature types used in this model:**  
- **Numerical:** `Age`, `Positive Feedback Count`  
- **Categorical:** `Clothing ID`, `Division Name`, `Department Name`, `Class Name`  
- **Text:** `Title`, `Review Text`  

### Initial Analysis: Top Products by Review Count and Recommendation Rate
We explore recommendation counts and rates per product and visualize the top products by recommendation rate.

### 3. Building Pipeline
### 3.1 Preparing features (X) & target (y)
### Separate Features and Target
We separate the dataset into features (X) and the target variable (y).
- X contains all input features used for prediction.
- y contains the target label, `Recommended IND`, which indicates whether a customer recommends the product.

### Train-Test Split
The dataset is split into training and test sets to evaluate performance on unseen data.
The model is trained on the training set and evaluated on the test set.
Ten percent of the data is reserved for testing, with a fixed random state for reproducibility.
### 3.2 Data Exploration
- Numerical features: 'Age', 'Positive Feedback Count'
- Categorical features: 'Clothing ID', 'Division Name', 'Department Name', 'Class Name']
- Text features: 'Title', 'Review Text'
###  3.3 Numerical Features Pipeline
- Scale numerical features between 0 and 1 using `MinMaxScaler`
### 3.4 Categorical Features Pipeline
- Encode categorical features using `OrdinalEncoder` and `OneHotEncoder`
### 3.5 Text Feature Pipeline and Feature Extraction  
3.5.1 Preparation for using spaCy
- spaCy Model Setup
- Installation
### 3.5.2 Text Feature Pipeline
  - Combine text columns  
  - Extract character counts (`spaces`, `!`, `?`)  
  - Extract spaCy features (lemmas, POS ratios, NER counts) 
  - Convert the lemmas into a sparse matrix of TF-IDF features 
  - Apply TF-IDF to lemmas  
  - Combine all text-based features

### 3.6 Combine Feature Engineering Pipelines
- A `ColumnTransformer` is used to preprocess different feature types separately.
- Numerical features are scaled, categorical features are encoded, and text features (Title and Review Text) are transformed using a custom `FastTextPipeline` with spaCy and TF-IDF.
- All extracted features are combined into a single feature matrix.
- The combined features are scaled using `StandardScaler` with `with_mean=False` to support sparse data.
- A `SGDClassifier`, `LinearSVC` or  `Random Forest classifiers` is applied for efficient classification on high-dimensional text features.

### 4. Training Pipeline and Evaluating Model
### Train Model
Train the full pipeline on the training dataset, applying all preprocessing steps before model fitting.
### Model Evaluation
The trained model is evaluated on the test dataset.
Performance is measured using Accuracy, Precision, Recall, and F1 Score to assess how well the model predicts customer recommendations.
### 5. Fine-Tuning Pipeline
### Hyperparameter Tuning with RandomizedSearchCV
Hyperparameter tuning is performed using RandomizedSearchCV to improve model performance.
Several parameter combinations are tested using 3-fold cross-validation, and the best model is automatically refit on the full training data.
### Best Model Selection
The best model selected by cross-validation is used for final evaluation and prediction.
### Final Model Evaluation
The optimized model (`model_best`) is evaluated on the test dataset.  
Accuracy, Precision, Recall, and F1 Score are used to assess its overall predictive performance.


---

## APP Usage
1. Train and save the ML pipeline
- Run all cells in `pipeline_app.ipynb` to train the model.
- Save the trained pipeline with pickle so you can reload it later without retraining, preserving all preprocessing and the trained model.

2. Run the FastAPI dashboard
- Launch the app with
  ```
  uvicorn app:app --reload.
  ```
- Open your browser at `http://127.0.0.1:8000`.
- The dashboard form lets you input numerical, categorical, and text features.
- Clicking **Predict Recommendation** sends the input through the trained pipeline and outputs whether the customer is predicted to recommend the product.

Form fields in `templates/index.html`:
- Numerical: Age, Positive Feedback Count
- Categorical: Clothing ID, Division Name, Department Name, Class Name
- Text: Title, Review Text

The pipeline handles all preprocessing steps automatically before returning the prediction.

---

## Author

- **Author:** Yuko  
- **GitHub:** [yukomath](https://github.com/yukomath)

---

## Acknowledgements

- **Udacity Data Scientist Nanodegree program**  
- **AI Tool:** ChatGPT


