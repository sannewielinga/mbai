import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_confusion_matrices(y_true, y_pred_ml, y_pred_foldrpp, y_pred_hybrid, model_name, dataset_name):
    cm_ml = confusion_matrix(y_true, y_pred_ml)
    disp_ml = ConfusionMatrixDisplay(confusion_matrix=cm_ml)
    disp_ml.plot()
    plt.title(f'{model_name} Confusion Matrix on {dataset_name} (ML Model)')
    plt.savefig(f'confusion_matrices_{model_name}_{dataset_name}_ml.png')
    plt.close()

    # FOLD-R++ Confusion Matrix
    cm_foldrpp = confusion_matrix(y_true, y_pred_foldrpp)
    disp_foldrpp = ConfusionMatrixDisplay(confusion_matrix=cm_foldrpp)
    disp_foldrpp.plot()
    plt.title(f'FOLD-R++ Confusion Matrix on {dataset_name}')
    plt.savefig(f'confusion_matrix_foldrpp_{dataset_name}.png')
    plt.close()

    # Hybrid Confusion Matrix
    cm_hybrid = confusion_matrix(y_true, y_pred_hybrid)
    disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid)
    disp_hybrid.plot()
    plt.title(f'Hybrid Confusion Matrix on {dataset_name}')
    plt.savefig(f'confusion_matrix_hybrid_{dataset_name}.png')
    plt.close()