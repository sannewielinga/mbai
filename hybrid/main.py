import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import load_dataset, preprocess_data_for_ml
from foldrpp_wrapper import train_foldrpp, predict_foldrpp_clingo
from ml_models import get_ml_models
from hybrid_model import create_hybrid_predictions
from error_analysis import generate_confusion_matrices
from explainability import *
import numpy as np

def main():
    """
    Runs experiments to compare the performance of FOLD-R++ and ML models on
    various datasets. For each dataset, it trains a FOLD-R++ model and several
    ML models, evaluates their performance, and creates hybrid predictions by
    combining the predictions of the FOLD-R++ model and each ML model. It also
    performs error analysis and explainability on the results.

    The results are stored in a Pandas DataFrame, which is then saved to a
    CSV file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    datasets = ['heart', 'autism', 'breastw', 'ecoli', 'kidney', 'parkison']
    results = []
    
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        # Load dataset
        model_foldrpp, data = load_dataset(dataset_name)
        df, label_encoders = preprocess_data_for_ml(data)
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Multiple experiments to average results
        num_experiments = 5
        dataset_results = []
        
        for exp_num in range(num_experiments):
            print(f" Experiment {exp_num+1}/{num_experiments}")
            # Use the same train-test split for both FOLD-R++ and ML models
            X_train_ml, X_test_ml, y_train_ml, y_test_ml, train_indices, test_indices = train_test_split(
                X, y, X.index, test_size=0.2, random_state=42 + exp_num)
            
            # Prepare data for FOLD-R++ using the same indices
            data_train = [data[i] for i in train_indices]
            data_test = [data[i] for i in test_indices]
            
            # Train FOLD-R++ model
            model_foldrpp.reset()  # Reset model before training
            model_foldrpp = train_foldrpp(model_foldrpp, data_train)
            model_foldrpp.asp()
            
            # Get FOLD-R++ predictions on the test set
            y_pred_foldrpp, asp_program_filename = predict_foldrpp_clingo(model_foldrpp, data_test, dataset_name=dataset_name, exp_num=exp_num)
            y_true = [x['label'] for x in data_test]
            
            # Train and evaluate ML models
            models = get_ml_models()
            for model_name, ml_model in models.items():
                ml_model.fit(X_train_ml, y_train_ml)
                y_pred_ml = ml_model.predict(X_test_ml)
                if hasattr(ml_model, 'predict_proba'):
                    ml_confidences = ml_model.predict_proba(X_test_ml)
                else:
                    # For models without predict_proba, use decision_function or assign default confidence
                    try:
                        ml_decision = ml_model.decision_function(X_test_ml)
                        ml_confidences = np.vstack((1 - ml_decision, ml_decision)).T
                    except:
                        ml_confidences = np.full((len(X_test_ml), 2), 0.5)
                
                # Create hybrid predictions
                y_pred_hybrid = create_hybrid_predictions(y_true, y_pred_ml, y_pred_foldrpp, ml_confidences)
                
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
                dataset_results.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
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
                
                # Error Analysis
                generate_confusion_matrices(y_true, y_pred_ml, y_pred_foldrpp, y_pred_hybrid, model_name, dataset_name)
                
                # Explainability
                explanations = get_explanations(model_foldrpp, data_test, y_pred_ml, y_pred_hybrid)
                ranked_rules = rank_rules_by_contribution(model_foldrpp, data_test, y_pred_ml, y_pred_hybrid)
                
                save_important_rules(ranked_rules, dataset_name, model_name, exp_num)
                
        # Average results over experiments
        df_results = pd.DataFrame(dataset_results)
        numeric_cols = df_results.select_dtypes(include=['number']).columns
        avg_results = df_results.groupby('Model')[numeric_cols].mean().reset_index()
        avg_results.insert(0, 'Dataset', dataset_name)
        results.append(avg_results)
    
    # Combine results from all datasets
    final_results = pd.concat(results, ignore_index=True)
    print("\nFinal Results:")
    print(final_results)
    
    # Optionally, save the final results to a CSV file
    final_results.to_csv('hybrid_model_results.csv', index=False)
    
if __name__ == '__main__':
    main()