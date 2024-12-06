import pandas as pd
from sklearn.preprocessing import LabelEncoder
from foldrpp import Foldrpp
from datasets import heart, autism, breastw, ecoli, kidney


def load_dataset(name):
    """
    Load a dataset from a given name.

    Parameters
    ----------
    name : str
        The name of the dataset to load.

    Returns
    -------
    model : Foldrpp
        A Foldrpp model.
    data : list
        A list of tuples containing the data points and their labels.
    """
    if name == "heart":
        model, data = heart()
    elif name == "autism":
        model, data = autism()
    elif name == "breastw":
        model, data = breastw()
    elif name == "ecoli":
        model, data = ecoli()
    elif name == "kidney":
        model, data = kidney()
    else:
        raise ValueError(f"Dataset {name} is not available.")
    return model, data


def preprocess_data_for_ml(data):
    """
    Preprocess a dataset for machine learning.

    Parameters
    ----------
    data : list
        A list of tuples containing the data points and their labels.

    Returns
    -------
    df : pd.DataFrame
        The preprocessed data.
    label_encoders : dict
        A dictionary of LabelEncoder objects, one for each column of the data
        that has been encoded.
    """

    df = pd.DataFrame(data)
    label_encoders = {}
    for column in df.columns:
        if column != "label" and df[column].dtype == object:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le
    return df, label_encoders
