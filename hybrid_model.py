def create_hybrid_predictions(
    y_true, y_pred_ml, y_pred_foldrpp, ml_confidences, confidence_threshold=0.6
):

    """
    Create hybrid predictions by combining predictions from ML and FOLD-R++ models based on confidence threshold.

    Parameters
    ----------
    y_true : array-like
        The ground truth target values.
    y_pred_ml : array-like
        The predictions made by the ML model.
    y_pred_foldrpp : array-like
        The predictions made by the FOLD-R++ model.
    ml_confidences : array-like
        The confidence scores associated with the ML model's predictions.
    confidence_threshold : float, optional
        The threshold below which the FOLD-R++ model's prediction is used instead of the ML model's prediction.

    Returns
    -------
    y_pred_hybrid : list
        The list of hybrid predictions combining ML and FOLD-R++ model predictions.
    """
    y_pred_hybrid = []
    for idx in range(len(y_true)):
        fold_pred = y_pred_foldrpp[idx]
        ml_pred = y_pred_ml[idx]
        ml_confidence = ml_confidences[idx][ml_pred]

        if ml_confidence < confidence_threshold:
            hybrid_pred = fold_pred
        else:
            hybrid_pred = ml_pred
        y_pred_hybrid.append(hybrid_pred)

    return y_pred_hybrid
