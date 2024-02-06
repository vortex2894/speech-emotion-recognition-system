import copy
import numpy as np
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
import model_training
import data_loader

# from tqdm import tqdm


def lasso_rank(X, y, lambda_range=[-9, -0.25]):
    y_reg = copy.copy(y)
    y_reg[y_reg == 0] = -1
    Lasso_model = linear_model.Lasso()
    Npt = 200
    coeffs = np.zeros((X.shape[1], Npt))
    # lambda_ = np.logspace(-4.5,0.9,Npt)
    lambda_ = np.logspace(lambda_range[0], lambda_range[1], Npt)
    i = 0
    for lamb in lambda_:
        Lasso_model = linear_model.Lasso(alpha=lamb, max_iter=180000)
        Lasso_model.fit(X, y_reg)
        print(f'Lasso iter #{i}')
        coeffs[:, i] = Lasso_model.coef_
        i = i + 1

    Rank_lasso = []
    for i in range(coeffs.shape[1] - 1, -1, -1):
        withdraw_features = np.nonzero(coeffs[:, i])[0];
        something_new = np.setdiff1d(withdraw_features, Rank_lasso)
        if len(something_new):
            for ii in range(len(something_new)):
                Rank_lasso.append(something_new[ii])
    return Rank_lasso


def SVM_eval(X, y, feature_ind, subj_IDs):
    # Choose C parameter
    C = np.logspace(-4, 1, 12, endpoint=True)
    auc_list = []
    auc_best = 0
    C_best = []
    ind_half = X.shape[0] // 2
    for val_C in C:
        model = SVC(kernel='linear', C=val_C, class_weight='balanced')
        model.fit(X[np.ix_(range(ind_half), feature_ind)], y[:ind_half])
        y_pred = model.predict(X[np.ix_(range(ind_half, X.shape[0]), feature_ind)])
        y_true = copy.copy(y[ind_half:])
        y_true[y_true == -1] = 0
        y_pred[y_pred == -1] = 0
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        sensetivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = (sensetivity + specificity) / 2
        # auc = roc_auc_score(y[ind_half:], y_pred)      # Calculate the AUC score using the predictions and actual class labels
        if auc_best < auc:
            C_best = val_C
            auc_best = auc
            # print(f'C = {val_C:0.8f}', f' AUC = {auc:.4f}' )

    # print('C_best = ',C_best)

    # Doing LOSO-CV
    IDs = subj_IDs['0'].unique()
    ID_only = np.squeeze(subj_IDs.values)

    y_pred = np.zeros(X.shape[0])

    for i in range(len(IDs)):
        train_index, test_index = ID_only != (i + 1), ID_only == (i + 1)

        X_train = copy.copy(X[np.ix_(train_index, feature_ind)])
        if len(feature_ind) == 1:
            X_train = X_train.reshape(-1, 1)

        model = SVC(kernel='linear', C=C_best, class_weight='balanced')

        model.fit(X_train, y[train_index])

        X_test = copy.copy(X[np.ix_(test_index, feature_ind)])
        if len(feature_ind) == 1:
            X_test = X_test.reshape(-1, 1)

        y_pred[test_index] = model.predict(X_test)

    # print('y_true = ',y_true)
    # print('y_pred = ',y_pred)
    auc = balanced_accuracy_score(y, y_pred)
    # acc = metrics.accuracy_score(y_true, y_pred)
    # tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    # sensetivity = tp / (tp+fn)
    # specificity = tn / (tn+fp)
    return auc

# 
def SVM_RBF_eval(X, y, feature_ind, subj_IDs):
    '''
    X -- pd.DataFrame with extracted features
    y -- pd.DataFrame with labels
    feature_ind -- indeces of features, that will be used
    subj_IDs -- IDs of the actors
    '''    
    
    X_selected = X.iloc[list(range(X.shape[0])),feature_ind]
    
    group_members= data_loader.get_k_fold_group_member()

    X_k_folds, y_k_folds = data_loader.get_custom_k_folds(X_selected, y, subj_IDs, group_members)

    # Choose C and gamma params
    C = np.logspace(-4, 1, 12, endpoint=True)
    gamma = np.logspace(-3.5, -1.5, 10, endpoint=True)
    uar_best = 0
    C_best,gamma_best = None, None

    for val_C in C:
        for val_gamma in gamma:
            model = SVC(kernel='rbf', C=val_C, gamma=val_gamma, random_state=42)
            UAR,y_pr,y_tr = model_training.estimate_model(model, X_k_folds, y_k_folds)

            if uar_best < UAR:
                C_best = val_C
                gamma_best = val_gamma
                y_pred = copy.copy(y_pr)
                y_true = copy.copy(y_tr)
                uar_best = UAR
                # print(f'UAR = {UAR:.3f}')

    return uar_best, C_best, gamma_best, y_pred, y_true


def LDA_eval(X, y, feature_ind, subj_IDs):
    '''
    X -- pd.DataFrame with extracted features
    y -- pd.DataFrame with labels
    feature_ind -- indeces of features, that will be used
    subj_IDs -- IDs of the actors
    '''
    X_selected = X.iloc[list(range(X.shape[0])),feature_ind]
    
    group_members= data_loader.get_k_fold_group_member()

    X_k_folds, y_k_folds = data_loader.get_custom_k_folds(X_selected, y, subj_IDs, group_members)

    model = LinearDiscriminantAnalysis()  # solver='eigen'

    UAR,y_pr,y_tr = model_training.estimate_model(model, X_k_folds, y_k_folds)


    # kf = KFold(n_splits=X.shape[0])
    # y_true = np.zeros(X.shape[0])
    # y_pred = np.zeros(X.shape[0])

    # for i, (train_index, test_index) in enumerate(kf.split(X)):
    #     X_tmp = copy.copy(X[np.ix_(train_index, feature_ind)])
    #     if len(feature_ind) == 1:
    #         X_tmp = X_tmp.reshape(-1, 1)

    #     LDA_model.fit(X_tmp, y[train_index])
    #     y_true[i] = y[test_index]

    #     X_test_tmp = copy.copy(X[np.ix_(test_index, feature_ind)])
    #     if len(feature_ind) == 1:
    #         X_test_tmp = X_test_tmp.reshape(-1, 1)
    #     y_pred[i] = LDA_model.predict(X_test_tmp)

    # # print('y_true = ',y_true)
    # # print('y_pred = ',y_pred)
    # acc = metrics.accuracy_score(y_true, y_pred)
    # # tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    # # sensetivity = tp / (tp+fn)
    # # specificity = tn / (tn+fp)
    
    return UAR,y_pr,y_tr


def LDA_LOSO_eval(X, y, feature_ind, subj_IDs):
    IDs = subj_IDs['0'].unique()
    ID_only = np.squeeze(subj_IDs.values)

    y_pred = np.zeros(X.shape[0])

    for i in range(len(IDs)):
        train_index, test_index = ID_only != (i + 1), ID_only == (i + 1)

        X_train = copy.copy(X[np.ix_(train_index, feature_ind)])
        if len(feature_ind) == 1:
            X_train = X_train.reshape(-1, 1)

        model = LinearDiscriminantAnalysis()  # solver='eigen'        

        model.fit(X_train, y[train_index])

        X_test = copy.copy(X[np.ix_(test_index, feature_ind)])
        if len(feature_ind) == 1:
            X_test = X_test.reshape(-1, 1)

        y_pred[test_index] = model.predict(X_test)

    # print('y_true = ',y_true)
    # print('y_pred = ',y_pred)
    auc = balanced_accuracy_score(y, y_pred)
    # acc = metrics.accuracy_score(y_true, y_pred)
    # tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    # sensetivity = tp / (tp+fn)
    # specificity = tn / (tn+fp)
    return auc
