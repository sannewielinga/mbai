# main.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import load_dataset, preprocess_data_for_ml
from foldrpp_wrapper import train_foldrpp, predict_foldrpp_clingo
from ml_models import get_ml_models
from hybrid_model import create_hybrid_predictions
from error_analysis import generate_confusion_matrices
from explainability import get_explanations, rank_rules_by_contribution, save_important_rules

def main():
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
        num_experiments = 10  # Increased number of experiments
        all_experiments_results = []
        stats_data = []
        
        # Initialize dictionaries to store ASP programs and important rules
        asp_programs = {}  # Key: (model_name), Value: list of (exp_num, asp_program)
        important_rules = {}  # Key: (model_name), Value: list of (exp_num, ranked_rules)

        for exp_num in range(num_experiments):
            print(f" Experiment {exp_num+1}/{num_experiments}")
            random_state = 42 + exp_num  # Different seed for each experiment
            
            # Use the same train-test split for both FOLD-R++ and ML models
            X_train_ml, X_test_ml, y_train_ml, y_test_ml, train_indices, test_indices = train_test_split(
                X, y, X.index, test_size=0.2, random_state=random_state)
            
            # Prepare data for FOLD-R++ using the same indices
            data_train = [data[i] for i in train_indices]
            data_test = [data[i] for i in test_indices]
            
            # Train FOLD-R++ model
            model_foldrpp.reset()  # Reset model before training
            model_foldrpp = train_foldrpp(model_foldrpp, data_train)
            model_foldrpp.asp()
            
            # Get FOLD-R++ predictions on the test set
            y_pred_foldrpp, asp_program = predict_foldrpp_clingo(
                model_foldrpp, data_test)
            y_true = [x['label'] for x in data_test]
            
            # Train and evaluate ML models
            models = get_ml_models(random_state=random_state)
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
                p_ml = precision_score(y_true, y_pred_ml, zero_division=0)
                r_ml = recall_score(y_true, y_pred_ml, zero_division=0)
                f1_ml = f1_score(y_true, y_pred_ml, zero_division=0)
                
                acc_foldrpp = accuracy_score(y_true, y_pred_foldrpp)
                p_foldrpp = precision_score(y_true, y_pred_foldrpp, zero_division=0)
                r_foldrpp = recall_score(y_true, y_pred_foldrpp, zero_division=0)
                f1_foldrpp = f1_score(y_true, y_pred_foldrpp, zero_division=0)
                
                acc_hybrid = accuracy_score(y_true, y_pred_hybrid)
                p_hybrid = precision_score(y_true, y_pred_hybrid, zero_division=0)
                r_hybrid = recall_score(y_true, y_pred_hybrid, zero_division=0)
                f1_hybrid = f1_score(y_true, y_pred_hybrid, zero_division=0)
                
                # Store results for this experiment
                experiment_results = {
                    'Dataset': dataset_name,
                    'Experiment': exp_num,
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
                }
                all_experiments_results.append(experiment_results)

                # Append results to stats_data
                stats_data.append({
                    'Dataset': dataset_name,
                    'Experiment': exp_num,
                    'Model': model_name,
                    'ML Accuracy': acc_ml,
                    'Hybrid Accuracy': acc_hybrid,
                })
                
                # Error Analysis
                generate_confusion_matrices(y_true, y_pred_ml, y_pred_foldrpp, y_pred_hybrid, model_name, dataset_name)
                
                # Explainability
                explanations = get_explanations(model_foldrpp, data_test, y_pred_ml, y_pred_hybrid)
                ranked_rules = rank_rules_by_contribution(model_foldrpp, data_test, y_pred_ml, y_pred_hybrid)
                
                # Collect ASP programs and important rules
                # We store all ASP programs and important rules with their experiment numbers
                asp_programs.setdefault((model_name), []).append((exp_num, asp_program))
                important_rules.setdefault((model_name), []).append((exp_num, ranked_rules))
                
        # After all experiments for this dataset
        # Convert to DataFrame
        df_results = pd.DataFrame(all_experiments_results)

        # Calculate mean and standard deviation
        metrics = ['ML Accuracy', 'FOLD-R++ Accuracy', 'Hybrid Accuracy',
                   'ML Precision', 'FOLD-R++ Precision', 'Hybrid Precision',
                   'ML Recall', 'FOLD-R++ Recall', 'Hybrid Recall',
                   'ML F1 Score', 'FOLD-R++ F1 Score', 'Hybrid F1 Score']

        # Group by Model and calculate mean and std
        mean_results = df_results.groupby('Model')[metrics].mean().reset_index()
        std_results = df_results.groupby('Model')[metrics].std().reset_index()

        # Merge mean and std DataFrames
        final_results = mean_results.copy()
        for metric in metrics:
            final_results[f'{metric} Std'] = std_results[metric]

        # Insert Dataset column
        final_results.insert(0, 'Dataset', dataset_name)

        # Append to overall results
        results.append(final_results)

        # Perform statistical significance testing
        stats_df = pd.DataFrame(stats_data)
        models = stats_df['Model'].unique()
        stats_results = []  # Initialize list for statistical test results
        significant_models = []

        for model_name in models:
            model_stats = stats_df[stats_df['Model'] == model_name]
            ml_accuracies = model_stats['ML Accuracy']
            hybrid_accuracies = model_stats['Hybrid Accuracy']

            differences = hybrid_accuracies - ml_accuracies

            if np.all(differences == 0):
                t_stat, p_value = None, None
            else:
                # Paired t-test
                t_stat, p_value = ttest_rel(hybrid_accuracies, ml_accuracies)

            # Collect the results
            stats_results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                't-statistic': t_stat,
                'p-value': p_value
            })
            
            # Check for significance
            if p_value is not None and p_value < 0.05:
                significant_models.append((model_name))

            # Print the results
            print(f"Statistical Significance Test for {model_name} on {dataset_name}:")
            print(f"t-statistic: {t_stat}, p-value: {p_value}\n")

        # Save ASP programs and important rules for significant models
        for model_name in significant_models:
            # Find the experiment with median hybrid accuracy
            model_experiments = df_results[df_results['Model'] == model_name]
            median_exp = model_experiments['Hybrid Accuracy'].median()
            median_experiment = model_experiments.iloc[
                (model_experiments['Hybrid Accuracy'] - median_exp).abs().argsort()[:1]
            ]['Experiment'].values[0]

            # Get the corresponding ASP program and important rules
            asp_list = asp_programs.get((model_name))
            rules_list = important_rules.get((model_name))

            # Find the ASP program and important rules for the median experiment
            asp_program = None
            ranked_rules = None
            for exp_num, asp_prog in asp_list:
                if exp_num == median_experiment:
                    asp_program = asp_prog
                    break

            for exp_num, rules in rules_list:
                if exp_num == median_experiment:
                    ranked_rules = rules
                    break

            # Save the ASP program
            if asp_program:
                asp_program_filename = f'asp_program_{dataset_name}_{model_name}.lp'
                with open(asp_program_filename, 'w') as f:
                    f.write(asp_program)

            # Save the important rules
            if ranked_rules:
                save_important_rules(ranked_rules, dataset_name, model_name)

        # Create DataFrame from stats_results and save
        stats_results_df = pd.DataFrame(stats_results)
        stats_results_df.to_csv('statistical_tests_results.csv', mode='a', index=False)

    # After all datasets
    final_results_df = pd.concat(results, ignore_index=True)
    print("\nFinal Results with Mean and Standard Deviation:")
    print(final_results_df)

    # Save to CSV
    final_results_df.to_csv('hybrid_model_results_with_std.csv', index=False)
    
if __name__ == '__main__':
    main()
