import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from foldrpp import Foldrpp
from datasets import heart

def main():
    # Load the heart dataset using the provided datasets.py module
    model_foldrpp, data = heart()
    
    # Convert the data to a DataFrame for ML models
    df = pd.DataFrame(data)
    
    # Encode categorical variables for ML models
    label_encoders = {}
    for column in df.columns:
        if column != 'label' and df[column].dtype == object:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    # Split data into features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Use the same train-test split for both FOLD-R++ and ML models
    X_train_ml, X_test_ml, y_train_ml, y_test_ml, train_indices, test_indices = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42)
    
    # Prepare data for FOLD-R++ using the same indices
    data_train = [data[i] for i in train_indices]
    data_test = [data[i] for i in test_indices]
    
    # Train FOLD-R++ model
    model_foldrpp.fit(data_train)
    
    # Get FOLD-R++ predictions on the test set
    y_pred_foldrpp = model_foldrpp.predict(data_test)
    y_true = [x['label'] for x in data_test]
    
    # Define ML models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier()
    }
    
    # Initialize a list to store results
    results = []
    
    # Iterate over each ML model to create hybrid models
    for name, ml_model in models.items():
        # Train ML model
        ml_model.fit(X_train_ml, y_train_ml)
        
        # Get ML predictions on the test set
        y_pred_ml = ml_model.predict(X_test_ml)
        
        # Create hybrid predictions
        y_pred_hybrid = []
        for idx in range(len(y_true)):
            fold_pred = y_pred_foldrpp[idx]
            ml_pred = y_pred_ml[idx]
            
            # Hybrid logic: Use FOLD-R++ prediction when it agrees with ML prediction, else trust ML model
            if fold_pred == ml_pred:
                hybrid_pred = fold_pred
            else:
                # You can adjust this logic as per your requirements
                hybrid_pred = ml_pred
            
            y_pred_hybrid.append(hybrid_pred)
        
        # Evaluate performance
        acc_ml = accuracy_score(y_true, y_pred_ml)
        p_ml = precision_score(y_true, y_pred_ml)
        r_ml = recall_score(y_true, y_pred_ml)
        f1_ml = f1_score(y_true, y_pred_ml)
        
        acc_foldrpp = accuracy_score(y_true, y_pred_foldrpp)
        p_foldrpp = precision_score(y_true, y_pred_foldrpp)
        r_foldrpp = recall_score(y_true, y_pred_foldrpp)
        f1_foldrpp = f1_score(y_true, y_pred_foldrpp)
        
        acc_hybrid = accuracy_score(y_true, y_pred_hybrid)
        p_hybrid = precision_score(y_true, y_pred_hybrid)
        r_hybrid = recall_score(y_true, y_pred_hybrid)
        f1_hybrid = f1_score(y_true, y_pred_hybrid)
        
        # Store results
        results.append({
            'Model': name,
            'ML Accuracy': acc_ml,
            'FOLD-R++ Accuracy': acc_foldrpp,
            'Hybrid Accuracy': acc_hybrid,
            'ML Precision': p_ml,
            'FOLD-R++ Precision': p_foldrpp,
            'Hybrid Precision': p_hybrid,
            'ML Recall': r_ml,
            'FOLD-R++ Recall': r_foldrpp,
            'Hybrid Recall': r_hybrid,
            'ML F1 Score': f1_ml,
            'FOLD-R++ F1 Score': f1_foldrpp,
            'Hybrid F1 Score': f1_hybrid,
        })
    
    # Display the results
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == '__main__':
    main()