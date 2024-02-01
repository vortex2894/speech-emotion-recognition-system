from tqdm import tqdm
import pandas as pd
import model_evaluation
import numpy as np


def estimate_model(model, X_k_folds, y_k_folds):
    y_pred = []
    y_true = []

    for k in X_k_folds.keys(): #tqdm(X_k_folds.keys()):
        # Prepare dataset
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()
        for i in X_k_folds.keys():
            if (i != k):
                X_train = pd.concat([X_train, X_k_folds[i]])
                y_train = pd.concat([y_train, y_k_folds[i]])
            else:
                X_test = X_k_folds[i]
                y_test = y_k_folds[i]
        model.fit(X_train.values, y_train.values.ravel())
        y_pred_k_fold = model.predict(X_test.values)

        y_pred = np.concatenate((y_pred, y_pred_k_fold), axis=None)
        y_true = np.concatenate((y_true, y_test.values), axis=None)

    UAR = model_evaluation.calculate_uar(y_true, y_pred)
    # print(f'UAR = {UAR:.3f}')
    return UAR, y_pred, y_true
