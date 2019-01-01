import code_for_modeling as an
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':

    'Running several different Decision Tree models'
    'Note that * unpacks the tuple of dfs created by dt_models'
    'MODEL 1: Downsampled and without LDA topics'
    an.dt_modeling(*an.dt_models(1))
    'MODEL 2: Downsampled and with LDA topics'
    an.dt_modeling(*an.dt_models(2))
    'MODEL 3: Upsampled without LDA topics'
    an.dt_modeling(*an.dt_models(3))
    'MODEL 4:Upsampled with LDA topics'
    an.dt_modeling(*an.dt_models(4))

    '''Model 3 is best; run model through other algorithms'''

    '''Run lines 22-51 to compare algorithms and create lists of items for ROC plot'''

    '''Lists used to generate ROC curve plot'''
    X_train, y_train, X_test, y_test = an.dt_models(3)
    algo_name_list = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'AdaBoosting']
    y_predict_list = []
    algo_for_predict_proba_list = []

    '''Decision Tree'''
    y_predict, dt = an.d_tree(*an.dt_models(3))
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(dt)

    '''Logistic Regression'''
    y_predict, lgt = an.logit(*an.dt_models(3))
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(lgt)

    '''Random Forests'''
    rf_cnf_matrix, accuracy, y_predict, importances, rfc = an.rf(*an.dt_models(3))
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(rfc)

    '''Gradient Boosting'''
    y_predict, gdbc = an.gdb(*an.dt_models(3))
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(gdbc)

    '''AdaBoosting'''
    y_predict, ada = an.ada(*an.dt_models(3))
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(ada)

    '''Generate ROC plot'''
    an.roc_plot(algo_name_list, y_predict_list, algo_for_predict_proba_list, y_test, X_test)

    '''Return obects for plotting confusion matrix and feature importances, derived from RF model'''
    rf_cnf_matrix, accuracy, y_predict, importances, rfc = an.rf(X_train, y_train, X_test, y_test)
    class_names = ['Not suicide', 'Suicide']

    '''Plot normalized and not normalized confusion matrices'''
    an.plot_confusion_matrix(rf_cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix using Random Forests')
    an.plot_confusion_matrix(rf_cnf_matrix, classes=class_names, title='Confusion Matrix, Without Normalization, Using Random Forests')
    #
    '''Plot feature importances'''
    an.feat_imp(importances)
