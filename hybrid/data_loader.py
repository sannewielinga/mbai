import pandas as pd
from sklearn.preprocessing import LabelEncoder
from foldrpp import Foldrpp
from datasets import heart, autism, breastw, ecoli, kidney

def load_dataset(name):
    if name == 'heart':
        model, data = heart()
    elif name == 'autism':
        model, data = autism()
    elif name == 'breastw':
        model, data = breastw()
    elif name == 'ecoli':
        model, data = ecoli()
    elif name == 'kidney':
        model, data = kidney()
    else:
        raise ValueError(f"Dataset {name} is not available.")
    return model, data

def preprocess_data_for_ml(data):
    df = pd.DataFrame(data)
    label_encoders = {}
    for column in df.columns:
        if column != 'label' and df[column].dtype == object:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le
    return df, label_encoders