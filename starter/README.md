# Fashion Trend Forecasting with ML Pipelines Using an SGD Classifier

## 1. Project Overview
This project builds a machine learning pipeline to predict product recommendations from customer reviews for **"StyleSense,"** a rapidly growing online women's clothing store known for its trendy and affordable fashion.

The pipeline analyzes review text, customer age, product category, and other relevant features to predict whether a customer would recommend a product. Automating this process helps:  
- Gain insights into customer satisfaction  
- Identify trending products  
- Support data-driven decisions for the retailer  

**Model:** We use an **SGDClassifier,** **LinearSVC** and **Random Forest classifiers**.

---
# Getting Started

These instructions will help you run the project on your local machine.

### Dependencies
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
spacy
en_core_web_sm


### Installation

1. Clone the repository:
```bash   
git clone <repository_url>

2. Navigate to the project directory:
```bash 
cd <project_folder>

3. Install required packages:
```bash 
pip install -r requirements.txt

4. Download spaCy English model:
```bash 
python -m spacy download en_core_web_sm

5. Place `reviews.csv` in the `data/` folder.

## Testing

To test the model and pipeline, run the Jupyter Notebook included in the repository. The notebook walks through:

1. Loading and exploring the dataset
2. Feature engineering and preprocessing
3. Pipeline construction for numerical, categorical, and text features
4. Training and evaluating the model
5. Hyperparameter tuning
6. Final evaluation of the optimized model

### Break Down Tests

- Dataset splitting: Ensures reproducible train/test sets  
- Feature pipelines: Validate correct scaling, encoding, and text processing  
- Model predictions: Check that metrics (accuracy, precision, recall, F1 score) are computed correctly  
- Hyperparameter tuning: Confirms `RandomizedSearchCV` finds best parameters

## Project Instructions

The notebook provides step-by-step instructions, including:

- **Data Exploration:** Checking data types, inspecting top products by recommendation count and rate  
- **Feature Engineering:** Numerical scaling, categorical encoding, and custom text pipelines (FastTextPipeline)  
- **Pipeline Construction:** Using `ColumnTransformer` to combine numerical, categorical, and text pipelines  
- **Model Training:** Fitting `SGDClassifier` within the pipeline  
- **Evaluation:** Computing Accuracy, Precision, Recall, and F1 Score  
- **Hyperparameter Tuning:** Using `RandomizedSearchCV` to optimize `SGDClassifier`  
- **Final Evaluation:** Selecting the best model and reporting metrics on test data

## Built With

* [Python](https://www.python.org/) - Programming language  
* [pandas](https://pandas.pydata.org/) - Data manipulation  
* [numpy](https://numpy.org/) - Numerical computations  
* [scikit-learn](https://scikit-learn.org/) - Machine learning pipelines and modeling  
* [spaCy](https://spacy.io/) - Natural language processing  
* [matplotlib](https://matplotlib.org/) - Data visualization  

## License

[MIT License](LICENSE.txt)


## 2. Repository Structure

├── data/
│ └── reviews.csv # Customer reviews dataset
├── notebooks/
│ └── recommendation_pipeline.ipynb # Main notebook with ML pipeline
├── src/
│ └── feature_pipelines.py # Custom pipelines for text and categorical features
└── README.md # Project description and instructions


**Notes on files:**  
- `reviews.csv`: Contains anonymized and cleaned customer reviews. Includes numerical, categorical, and text features.  
- `recommendation_pipeline.ipynb`: Step-by-step implementation with detailed comments for feature engineering, model building, and evaluation.  
- `feature_pipelines.py`: Modularized code for numerical, categorical, and text feature pipelines to keep the notebook clean and organized.  

---

## 3. Data Understanding

### Data Structure
- Target variable: `Recommended IND` (1 = recommended, 0 = not recommended)  
- Features: 8 columns including numerical, categorical, and text data  

**Feature types used in this model:**  
- **Numerical:** `Age`, `Positive Feedback Count`  
- **Categorical:** `Clothing ID`, `Division Name`, `Department Name`, `Class Name`  
- **Text:** `Title`, `Review Text`  

### Initial Analysis
We explore recommendation counts and rates per product and visualize the top products by recommendation rate.

---

## 4. Building the Pipeline

### 4.1 Feature Engineering Pipelines
- **Numerical Pipeline:** Scale numerical features between 0 and 1 using `MinMaxScaler`  
- **Categorical Pipeline:** Encode categorical features using `OrdinalEncoder` and `OneHotEncoder`  
- **Text Pipeline (FastTextPipeline):**  
  - Combine text columns  
  - Extract character counts (`spaces`, `!`, `?`)  
  - Extract spaCy features (lemmas, POS ratios, NER counts)  
  - Apply TF-IDF to lemmas  
  - Combine all text-based features  

### 4.2 Full Pipeline
- Use a `ColumnTransformer` to apply each pipeline to the corresponding features  
- Apply `StandardScaler` to ensure compatibility with sparse matrices  
- Use `SGDClassifier` with hinge loss (linear SVM equivalent)  

---

## 5. Installation & Usage

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib spacy
python -m spacy download en_core_web_sm

## 5. Running the Notebook

This section runs the Jupyter Notebook to execute all steps defined in the previous sections.  
It includes feature preparation, pipeline building, model training, and evaluation.

### 5.1 Preparing Features and Target

```python
data = df

# Separate features and target
X = data.drop('Recommended IND', axis=1)
y = data['Recommended IND'].copy()

print('Labels:', y.unique())
print('Features:')
display(X.head())


