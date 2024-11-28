import numpy as np

def create_hybrid_predictions(y_true, y_pred_ml, y_pred_foldrpp, ml_confidences, confidence_threshold=0.6):
    
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