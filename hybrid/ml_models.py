"""This module provides a collection of machine learning models
for classification"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def get_ml_models():
    """
    Returns a dictionary of machine learning models with their respective
    implementations from scikit-learn. The models included are:

    - 'Random Forest': RandomForestClassifier with 100 estimators.
    - 'SVM': Support Vector Classifier with probability estimates enabled.
    - 'KNN': K-Nearest Neighbors Classifier.
    - 'Neural Network': Multi-layer Perceptron Classifier with a maximum of
      1000 iterations.

    These models can be used for classification tasks.
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(max_iter=1000)
    }
    return models