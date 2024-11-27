# explainability.py

def get_explanations(model_foldrpp, data_test, y_pred_ml, y_pred_hybrid):
    """
    Get explanations for the instances where the hybrid model corrected the ML model.

    Parameters
    ----------
    model_foldrpp: Foldrpp
        The FOLD-R++ model
    data_test: list
        The test data
    y_pred_ml: array-like
        The predictions of the ML model
    y_pred_hybrid: array-like
        The predictions of the hybrid model

    Returns
    -------
    list
        A list of dictionaries where each dictionary contains the instance, the
        prediction of the ML model, the prediction of the FOLD-R++ model, the
        prediction of the hybrid model, and the explanation for the correction
    """
    explanations = []
    for idx, x in enumerate(data_test):
        ml_pred = y_pred_ml[idx]
        hybrid_pred = y_pred_hybrid[idx]
        fold_pred = model_foldrpp.classify(x)
        if hybrid_pred != ml_pred:
            explanation = model_foldrpp.proof_trees(x)
            explanations.append({
                'instance': x,
                'ml_prediction': ml_pred,
                'foldrpp_prediction': fold_pred,
                'hybrid_prediction': hybrid_pred,
                'explanation': explanation
            })

    return explanations

def rank_rules_by_contribution(model_foldrpp, data_test, y_pred_ml, y_pred_hybrid):
    """
    Rank the rules in the FOLD-R++ model by how many times they contributed
    to correcting the predictions of the ML model.

    Parameters
    ----------
    model_foldrpp: Foldrpp
        The FOLD-R++ model
    data_test: list
        The test data
    y_pred_ml: array-like
        The predictions of the ML model
    y_pred_hybrid: array-like
        The predictions of the hybrid model

    Returns
    -------
    ranked_rules: list
        A list of tuples, where each tuple contains a rule and the number of
        times it contributed to corrections
    """
    rule_contributions = {}
    for idx, x in enumerate(data_test):
        ml_pred = y_pred_ml[idx]
        hybrid_pred = y_pred_hybrid[idx]
        if hybrid_pred != ml_pred:
            # Identify which rules contributed to the correction
            proofs = model_foldrpp.proof_rules(x)
            for rule in proofs:
                rule_str = str(rule)
                if rule_str not in rule_contributions:
                    rule_contributions[rule_str] = 0
                rule_contributions[rule_str] += 1
    # Rank rules by the number of times they contributed to corrections
    ranked_rules = sorted(rule_contributions.items(), key=lambda item: item[1], reverse=True)
    return ranked_rules

def save_important_rules(ranked_rules, dataset_name, model_name):
    """
    Save the ranked rules to a file.

    Parameters
    ----------
    ranked_rules: list
        A list of tuples, where each tuple contains a rule and the number of
        times it contributed to corrections
    dataset_name: str
        The name of the dataset
    model_name: str
        The name of the ML model

    Returns
    -------
    None
    """
    filename = f'important_rules_{dataset_name}_{model_name}.txt'
    with open(filename, 'w') as f:
        for rule, count in ranked_rules:
            f.write(f'Rule used {count} times:\n{rule}\n\n')
