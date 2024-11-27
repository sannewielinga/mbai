"""This module provides a collection of machine learning models
for classification"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def get_ml_models(random_state=None):
    
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=random_state)
    }
    return models