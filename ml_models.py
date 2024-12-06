from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def get_ml_models(random_state=None):
    """
    Returns a dictionary of machine learning models to be used for classification.

    Parameters
    ----------
    random_state : int or None (default=None)
        The seed used to shuffle the data. If None, the random number generator
        is the `RandomState` instance used by `np.random`.

    Returns
    -------
    dict
        A dictionary with the keys "Random Forest", "SVM", "KNN", and "Neural Network"
        and the corresponding values being the respective scikit-learn Estimator
        objects.
    """

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=random_state),
    }
    return models
