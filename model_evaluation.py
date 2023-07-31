from sklearn.metrics import confusion_matrix


def calculate_uar(y_true, y_pred):
    '''
    calculates the unweighted average recall across all the classes present
    in the true labels (y_true)
    and predicted labels (y_pred) using the confusion matrix.
    '''
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(cm)
    recalls = []
    for i in range(num_classes):
        true_positives = cm[i, i]
        actual_positives = sum(cm[i, :])
        recall = true_positives / actual_positives
        recalls.append(recall)
    average_recall = sum(recalls) / num_classes
    return average_recall
