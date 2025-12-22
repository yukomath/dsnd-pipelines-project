# pipeline_utils.py

import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


# ------------------------------
# Text Processing Utilities
# ------------------------------

def combine_text_columns(X):
    """Combine 'Title' and 'Review Text' into a single column."""
    title = X['Title'].fillna('')
    review = X['Review Text'].fillna('')
    return (title + ' ' + review).values

combine_text_step = FunctionTransformer(combine_text_columns, validate=False)


class CountCharacter(BaseEstimator, TransformerMixin):
    """Count occurrences of a specific character in text."""
    def __init__(self, character: str):
        self.character = character

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = list(X)
        return np.array([[text.count(self.character)] for text in X])


def character_counts_pipeline():
    """FeatureUnion for character counts: spaces, !, ?"""
    return FeatureUnion([
        ('spaces', CountCharacter(' ')),
        ('exclamations', CountCharacter('!')),
        ('questions', CountCharacter('?')),
    ])


class SpacyNumericFeatures(BaseEstimator, TransformerMixin):
    """Extract POS and NER counts using spaCy."""
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pos_features, ner_features = [], []

        for doc in self.nlp.pipe(X, batch_size=50):
            counts = {"NOUN":0, "VERB":0, "ADJ":0}
            for token in doc:
                if token.pos_ in counts:
                    counts[token.pos_] += 1
            total = len(doc) + 1e-6
            pos_features.append([counts["NOUN"]/total, counts["VERB"]/total, counts["ADJ"]/total])
            ner_features.append([len(doc.ents)])

        return np.hstack([np.array(pos_features), np.array(ner_features)])


class SpacyLemmas(BaseEstimator, TransformerMixin):
    """Extract lemmas from spaCy doc."""
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lemmas = []
        for doc in self.nlp.pipe(X, batch_size=50):
            lemmas.append(' '.join(token.lemma_ for token in doc if not token.is_stop))
        return lemmas


def lemma_tfidf_pipeline(nlp):
    """Pipeline to get lemmas + TF-IDF."""
    return Pipeline([
        ('lemmas', SpacyLemmas(nlp)),
        ('tfidf', TfidfVectorizer(stop_words='english'))
    ])


def text_pipeline(nlp):
    """Full text pipeline: combine text, char counts, POS/NER, TF-IDF."""
    return Pipeline([
        ('combine_text', combine_text_step),
        ('features', FeatureUnion([
            ('char_counts', character_counts_pipeline()),
            ('spacy_numeric', SpacyNumericFeatures(nlp)),
            ('tfidf', lemma_tfidf_pipeline(nlp)),
        ]))
    ])


# ------------------------------
# Numeric Pipeline
# ------------------------------
def num_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])



# ------------------------------
# Categorical Pipeline
# ------------------------------
def cat_pipeline():
    return Pipeline([
        ('ordinal_encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            sparse_output=True,
            handle_unknown='ignore'
        ))
    ])

# ------------------------------
# Full Column Transformer
# ------------------------------
def feature_engineering(numeric_features, categorical_features, text_features, nlp):
    """
    Combines numeric, categorical, and text pipelines into a ColumnTransformer.
    Returns full feature engineering pipeline.
    """
    return ColumnTransformer([
        ('num', num_pipeline(), numeric_features),
        ('cat', cat_pipeline(), categorical_features),
        ('text', text_pipeline(nlp), text_features)
    ])


# ------------------------------
# Complete Pipeline with SGDClassifier
# ------------------------------
def model_pipeline(numeric_features, categorical_features, text_features, nlp):
    fe = feature_engineering(
        numeric_features,
        categorical_features,
        text_features,
        nlp
    )

    return Pipeline([
        ('features', fe),
        ('sgd', SGDClassifier(
            loss='hinge',
            max_iter=1000,
            tol=1e-3,
            random_state=27
        ))
    ])
