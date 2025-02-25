import numpy as np
import json


def get_explanations(model_foldrpp, data_test, y_pred_ml, y_pred_hybrid, y_true):
    """
    Get explanations for each instance in the test set, indicating whether
    the hybrid model's prediction led to an improvement, no change, or made
    the prediction worse compared to the ML model, based on the true label.

    Parameters
    ----------
    model_foldrpp: Foldrpp
        The FOLD-R++ model.
    data_test: list
        The test data.
    y_pred_ml: array-like
        The predictions of the ML model.
    y_pred_hybrid: array-like
        The predictions of the hybrid model.
    y_true: array-like
        The ground truth labels for the test data.

    Returns
    -------
    list
        A list of dictionaries where each dictionary contains the instance,
        the prediction of the ML model, the prediction of the FOLD-R++ model,
        the prediction of the hybrid model, a status flag indicating whether
        the hybrid prediction was an improvement, no change, or worse, and
        the explanation (proof tree).
    """
    explanations = []
    for idx, x in enumerate(data_test):
        ml_pred = int(y_pred_ml[idx])
        hybrid_pred = int(y_pred_hybrid[idx])
        fold_pred = int(model_foldrpp.classify(x))
        true_label = int(y_true[idx])

        # Determine status based on comparison with the ground truth
        if hybrid_pred == ml_pred:
            status = "no_change"
        else:
            if ml_pred != true_label and hybrid_pred == true_label:
                status = "improvement"
            elif ml_pred == true_label and hybrid_pred != true_label:
                status = "worsened"
            else:
                status = "changed (no clear improvement)"

        # Convert instance features to serializable types
        instance = {
            k: (v.item() if isinstance(v, np.generic) else v) for k, v in x.items()
        }
        # Generate proof tree and convert to string
        proof_tree = str(model_foldrpp.proof_trees(x))
        explanations.append(
            {
                "instance": instance,
                "ml_prediction": ml_pred,
                "foldrpp_prediction": fold_pred,
                "hybrid_prediction": hybrid_pred,
                "true_label": true_label,
                "status": status,
                "explanation": proof_tree,
            }
        )
    return explanations


def rank_rules_by_contribution(model_foldrpp, data_test, y_pred_ml, y_pred_hybrid):
    """
    Rank rules by the number of times they contributed to corrections of the ML model.

    Parameters
    ----------
    model_foldrpp: Foldrpp
        The FOLD-R++ model
    data_test: list
        The test data
    y_pred_ml: list
        The predictions made by the ML model
    y_pred_hybrid: list
        The predictions made by the Hybrid model

    Returns
    -------
    ranked_rules: list
        A list of tuples, where each tuple contains the rule and the number of times it contributed to a correction
    """
    rule_contributions = {}
    for idx, x in enumerate(data_test):
        ml_pred = y_pred_ml[idx]
        hybrid_pred = y_pred_hybrid[idx]
        if hybrid_pred != ml_pred:
            proofs = model_foldrpp.proof_rules(x)
            for rule in proofs:
                rule_str = str(rule)
                if rule_str not in rule_contributions:
                    rule_contributions[rule_str] = 0
                rule_contributions[rule_str] += 1
    ranked_rules = sorted(
        rule_contributions.items(), key=lambda item: item[1], reverse=True
    )
    return ranked_rules


def save_important_rules(ranked_rules, dataset_name, model_name):
    """
    Save the ranked rules to a file.

    Parameters
    ----------
    ranked_rules: list
        A list of tuples, where each tuple contains the rule and the number of times it contributed to a correction
    dataset_name: str
        The name of the dataset
    model_name: str
        The name of the model

    Returns
    -------
    None

    Notes
    -----
    The file will be saved in the ./results/programs directory with the name important_rules_<dataset_name>_<model_name>.txt
    """
    filename = f"./results/programs/important_rules_{dataset_name}_{model_name}.txt"
    with open(filename, "w") as f:
        for rule, count in ranked_rules:
            f.write(f"Rule used {count} times:\n{rule}\n\n")


def save_explanations(explanations, dataset_name, model_name):
    """
    Save the explanations to a file.

    Parameters
    ----------
    explanations: list
        A list of explanations.
    dataset_name: str
        The name of the dataset.
    model_name: str
        The name of the ML model.

    Returns
    -------
    None
    """
    filename = f"./results/programs/explanations_{dataset_name}_{model_name}.json"
    with open(filename, "w") as f:
        # Custom function to convert NumPy types to native Python types
        def convert_numpy_types(o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, np.bool_):
                return bool(o)
            elif isinstance(o, (np.generic,)):
                return o.item()
            else:
                return str(o)  # Convert other types to string

        json.dump(explanations, f, indent=4, default=convert_numpy_types)
