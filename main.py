import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import load_dataset, preprocess_data_for_ml
from foldrpp_wrapper import train_foldrpp, predict_foldrpp_clingo
from ml_models import get_ml_models
from hybrid_model import create_hybrid_predictions
from explainability import (
    get_explanations,
    rank_rules_by_contribution,
    save_explanations,
    save_important_rules,
)


def main():
    """
    Runs multiple experiments on different datasets to evaluate the performance
    of ML models, pure FOLD-R++ models, and hybrid models that combine predictions
    from both. For each dataset, it performs the following steps:

    1. Loads and preprocesses data for ML model training.
    2. Splits the data into train and test sets for multiple experiments.
    3. Trains FOLD-R++ model and makes predictions using ASP.
    4. Evaluates the pure FOLD-R++ predictions.
    5. Trains baseline ML models and makes predictions.
    6. Combines predictions to create hybrid model predictions.
    7. Evaluates and records accuracy, precision, recall, and F1 scores for
       each model type.
    8. Performs statistical significance testing between ML and hybrid models.
    9. Saves ASP programs and important rules for statistically significant models.
    10. Aggregates results across all experiments and datasets, calculates mean
        and standard deviation, and saves the results to CSV files.
    """

    datasets = ["heart", "autism", "breastw", "ecoli", "kidney"]
    results = []

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        model_foldrpp, data = load_dataset(dataset_name)
        df, label_encoders = preprocess_data_for_ml(data)

        X = df.drop("label", axis=1)
        y = df["label"]

        num_experiments = 10
        all_experiments_results = []
        stats_data = []

        asp_programs = {}
        important_rules = {}
        explanations_list = {}

        for exp_num in range(num_experiments):
            print(f" Experiment {exp_num+1}/{num_experiments}")
            random_state = 42 + exp_num

            (
                X_train_ml,
                X_test_ml,
                y_train_ml,
                y_test_ml,
                train_indices,
                test_indices,
            ) = train_test_split(
                X, y, X.index, test_size=0.2, random_state=random_state
            )

            data_train = [data[i] for i in train_indices]
            data_test = [data[i] for i in test_indices]

            model_foldrpp.reset()
            model_foldrpp = train_foldrpp(model_foldrpp, data_train)
            model_foldrpp.asp()

            y_pred_foldrpp, asp_program = predict_foldrpp_clingo(
                model_foldrpp, data_test
            )
            y_true = [x["label"] for x in data_test]

            acc_foldrpp = accuracy_score(y_true, y_pred_foldrpp)
            p_foldrpp = precision_score(y_true, y_pred_foldrpp, zero_division=0)
            r_foldrpp = recall_score(y_true, y_pred_foldrpp, zero_division=0)
            f1_foldrpp = f1_score(y_true, y_pred_foldrpp, zero_division=0)

            # Record the pure FOLD-R++ results separately.
            pure_foldrpp_result = {
                "Dataset": dataset_name,
                "Experiment": exp_num,
                "Model": "FOLD-R++ Pure",
                "ML Accuracy": np.nan,
                "FOLD-R++ Accuracy": acc_foldrpp,
                "Hybrid Accuracy": np.nan,
                "ML Precision": np.nan,
                "FOLD-R++ Precision": p_foldrpp,
                "Hybrid Precision": np.nan,
                "ML Recall": np.nan,
                "FOLD-R++ Recall": r_foldrpp,
                "Hybrid Recall": np.nan,
                "ML F1 Score": np.nan,
                "FOLD-R++ F1 Score": f1_foldrpp,
                "Hybrid F1 Score": np.nan,
            }
            all_experiments_results.append(pure_foldrpp_result)

            models = get_ml_models(random_state=random_state)
            for model_name, ml_model in models.items():
                ml_model.fit(X_train_ml, y_train_ml)
                y_pred_ml = ml_model.predict(X_test_ml)
                if hasattr(ml_model, "predict_proba"):
                    ml_confidences = ml_model.predict_proba(X_test_ml)
                else:
                    try:
                        ml_decision = ml_model.decision_function(X_test_ml)
                        ml_confidences = np.vstack((1 - ml_decision, ml_decision)).T
                    except Exception as e:
                        ml_confidences = np.full((len(X_test_ml), 2), 0.5)

                y_pred_hybrid = create_hybrid_predictions(
                    y_true, y_pred_ml, y_pred_foldrpp, ml_confidences
                )
                acc_ml = accuracy_score(y_true, y_pred_ml)
                p_ml = precision_score(y_true, y_pred_ml, zero_division=0)
                r_ml = recall_score(y_true, y_pred_ml, zero_division=0)
                f1_ml = f1_score(y_true, y_pred_ml, zero_division=0)

                acc_hybrid = accuracy_score(y_true, y_pred_hybrid)
                p_hybrid = precision_score(y_true, y_pred_hybrid, zero_division=0)
                r_hybrid = recall_score(y_true, y_pred_hybrid, zero_division=0)
                f1_hybrid = f1_score(y_true, y_pred_hybrid, zero_division=0)

                experiment_results = {
                    "Dataset": dataset_name,
                    "Experiment": exp_num,
                    "Model": model_name,
                    "ML Accuracy": acc_ml,
                    "FOLD-R++ Accuracy": acc_foldrpp,
                    "Hybrid Accuracy": acc_hybrid,
                    "ML Precision": p_ml,
                    "FOLD-R++ Precision": p_foldrpp,
                    "Hybrid Precision": p_hybrid,
                    "ML Recall": r_ml,
                    "FOLD-R++ Recall": r_foldrpp,
                    "Hybrid Recall": r_hybrid,
                    "ML F1 Score": f1_ml,
                    "FOLD-R++ F1 Score": f1_foldrpp,
                    "Hybrid F1 Score": f1_hybrid,
                }
                all_experiments_results.append(experiment_results)

                stats_data.append(
                    {
                        "Dataset": dataset_name,
                        "Experiment": exp_num,
                        "Model": model_name,
                        "ML Accuracy": acc_ml,
                        "Hybrid Accuracy": acc_hybrid,
                    }
                )

                explanations = get_explanations(
                    model_foldrpp, data_test, y_pred_ml, y_pred_hybrid, y_true
                )
                ranked_rules = rank_rules_by_contribution(
                    model_foldrpp, data_test, y_pred_ml, y_pred_hybrid
                )

                asp_programs.setdefault((model_name), []).append((exp_num, asp_program))
                important_rules.setdefault((model_name), []).append(
                    (exp_num, ranked_rules)
                )
                explanations_list.setdefault((model_name), []).append(
                    (exp_num, explanations)
                )

        df_results = pd.DataFrame(all_experiments_results)
        metrics = [
            "ML Accuracy",
            "FOLD-R++ Accuracy",
            "Hybrid Accuracy",
            "ML Precision",
            "FOLD-R++ Precision",
            "Hybrid Precision",
            "ML Recall",
            "FOLD-R++ Recall",
            "Hybrid Recall",
            "ML F1 Score",
            "FOLD-R++ F1 Score",
            "Hybrid F1 Score",
        ]

        mean_results = df_results.groupby("Model")[metrics].mean().reset_index()
        std_results = df_results.groupby("Model")[metrics].std().reset_index()

        final_results = mean_results.copy()
        for metric in metrics:
            final_results[f"{metric} Std"] = std_results[metric]
        final_results.insert(0, "Dataset", dataset_name)
        results.append(final_results)

        stats_df = pd.DataFrame(stats_data)
        models = stats_df["Model"].unique()
        stats_results = []
        significant_models = []

        for model_name in models:
            model_stats = stats_df[stats_df["Model"] == model_name]
            ml_accuracies = model_stats["ML Accuracy"]
            hybrid_accuracies = model_stats["Hybrid Accuracy"]

            differences = hybrid_accuracies - ml_accuracies

            if np.all(differences == 0):
                t_stat, p_value = None, None
            else:
                t_stat, p_value = ttest_rel(hybrid_accuracies, ml_accuracies)
            stats_results.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "t-statistic": t_stat,
                    "p-value": p_value,
                }
            )

            if p_value is not None and p_value < 0.05:
                significant_models.append((model_name))

            print(f"Statistical Significance Test for {model_name} on {dataset_name}:")
            print(f"t-statistic: {t_stat}, p-value: {p_value}\n")

        for model_name in significant_models:
            model_experiments = df_results[df_results["Model"] == model_name]
            median_exp = model_experiments["Hybrid Accuracy"].median()
            median_experiment = model_experiments.iloc[
                (model_experiments["Hybrid Accuracy"] - median_exp).abs().argsort()[:1]
            ]["Experiment"].values[0]

            asp_list = asp_programs.get((model_name))
            rules_list = important_rules.get((model_name))
            explanations_list_model = explanations_list.get((model_name))

            asp_program = None
            ranked_rules = None
            explanations = None
            for exp_num, asp_prog in asp_list:
                if exp_num == median_experiment:
                    asp_program = asp_prog
                    break

            for exp_num, rules in rules_list:
                if exp_num == median_experiment:
                    ranked_rules = rules
                    break

            for exp_num, expl in explanations_list_model:
                if exp_num == median_experiment:
                    explanations = expl
                    break

            if asp_program:
                asp_program_filename = (
                    f"./results/programs/asp_program_{dataset_name}_{model_name}.lp"
                )
                with open(asp_program_filename, "w") as f:
                    f.write(asp_program)

            if ranked_rules:
                save_important_rules(ranked_rules, dataset_name, model_name)
            if explanations:
                save_explanations(explanations, dataset_name, model_name)

        stats_results_df = pd.DataFrame(stats_results)
        stats_results_df.to_csv(
            "./results/statistical_tests_results.csv", mode="a", index=False
        )

    final_results_df = pd.concat(results, ignore_index=True)
    print("\nFinal Results with Mean and Standard Deviation:")
    print(final_results_df)

    final_results_df.to_csv("./results/hybrid_model_results_with_std.csv", index=False)


if __name__ == "__main__":
    main()
