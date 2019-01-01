from pprint import pprint

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import math
from scipy import stats

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_auc_score

def dt_models(model):
    '''Run several Decision Tree Models to find best fit'''
    if model <3:
        '''DOWNSAMPLED'''
        df_train = pd.read_csv('data/downsample_LDA_Train_forDT.csv', low_memory=False)
        df_train.dropna(inplace = True)
        df_train=df_train.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)

        df_test = pd.read_csv('data/downsample_LDA_Test_forDT.csv', low_memory=False)
        df_test.dropna(inplace = True)
        df_test=df_test.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)
    if model >2:
        df_train = pd.read_csv('data/upsample_LDA_Train_forDT.csv', low_memory=False)
        df_train.dropna(inplace = True)
        df_train=df_train.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)
        df_test = pd.read_csv('data/upsample_LDA_Test_forDT.csv', low_memory=False)
        df_test.dropna(inplace = True)
        df_test=df_test.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)

    if model==1:
        print('MODEL 1: Downsampled and without LDA topics')
        df_train = df_train[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]
        df_test = df_test[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]
    if model==2:
        print('MODEL 2: Downsampled and with LDA topics')
    if model==3:
        print('MODEL 3: Upsampled without LDA topics')
        df_train = df_train[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]
        df_test = df_test[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]
    if model==4:
        print('MODEL 7:Upsampled with LDA topics')
    return make_train_test(df_train, df_test)

def make_train_test(df_train, df_test):
    y_train = df_train[['suicide']]
    y_train = df_train.suicide.values
    X_train = df_train.drop(['suicide'], axis=1, inplace=True)
    X_train = df_train.values

    y_test = df_test[['suicide']]
    y_test = df_test.suicide.values
    X_test = df_test.drop(['suicide'], axis=1, inplace=True)
    X_test = df_test.values

    return X_train, y_train, X_test, y_test

'''Test Decision Tree model with count vs binary features'''
def dt_modeling(X_train, y_train, X_test, y_test):
    '''Model the Decision Tree'''
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(X_train, y_train)
    y_predict = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)
    accuracy = np.around(accuracy,3)
    precision = precision_score(y_test, y_predict)
    precision = np.around(precision,3)
    recall = recall_score(y_test, y_predict)
    recall = np.around(recall,3)
    print('Accuracy of logistics regression classifer on test set:', (accuracy))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print("Recall: (TP/TP + FN), lower = we failed to predict some positives", (recall))
    print("Precision (TP/TP+FP), lower = we over-predicted positives: ", (precision))
    print(y_test.shape)

'''Decision tree modeling that outputs info for ROC'''
def d_tree(X_train, y_train, X_test, y_test):
    print('DECISION TREE')
    y_test = y_test
    X_test = X_test
    dt = DecisionTreeClassifier()
    result = dt.fit(X_train, y_train)
    y_predict = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)
    accuracy = np.around(accuracy,3)
    precision = precision_score(y_test, y_predict)
    precision = np.around(precision,3)
    recall = recall_score(y_test, y_predict)
    recall = np.around(recall,3)
    print('Accuracy of logistics regression classifer on test set:', (accuracy))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print("Recall: (TP/TP + FN), lower = we failed to predict some positives", (recall))
    print("Precision (TP/TP+FP), lower = we over-predicted positives: ", (precision))
    return y_predict, dt

'''Logistic Regregression modeling that outputs info for ROC'''
def logit(X_train, y_train, X_test, y_test):
    print('LOGISTIC REGRESSION')
    lgt = LogisticRegression()
    result = lgt.fit(X_train, y_train)
    y_predict = lgt.predict(X_test)
    accuracy = lgt.score(X_test, y_test)
    accuracy = np.around(accuracy,3)
    precision = precision_score(y_test, y_predict)
    precision = np.around(precision,3)
    recall = recall_score(y_test, y_predict)
    recall = np.around(recall,3)
    print('Accuracy of logistics regression classifer on test set:', (accuracy))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print("Recall: (TP/TP + FN), lower = we failed to predict some positives", (recall))
    print("Precision (TP/TP+FP), lower = we over-predicted positives: ", (precision))
    return y_predict, lgt

'''Random Forests modeling that outputs info for ROC, CFM, and feature importances'''
def rf(X_train, y_train, X_test, y_test):
    print('RANDOM FOREST')
    rfc = RandomForestClassifier(n_estimators=100)
    result = rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)
    accuracy =rfc.score(X_test, y_test)
    accuracy = np.around(accuracy,3)
    precision = precision_score(y_test, y_predict)
    precision = np.around(precision,3)
    recall = recall_score(y_test, y_predict)
    recall = np.around(recall,3)
    print('Accuracy of logistics regression classifer on test set:', (accuracy))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print("Recall: (TP/TP + FN), lower = we failed to predict some positives", (recall))
    print("Precision (TP/TP+FP), lower = we over-predicted positives: ", (precision))
    importances = rfc.feature_importances_
    columns = rename_features()
    zipped = (zip(columns, importances))
    importance_zipped = sorted(zipped, key = lambda t: t[1])
    return cnf_matrix, accuracy, y_predict, importance_zipped, rfc

def rename_features():
    renamed = []
    columns_dict = {'claimed':'Claimed Responsibility', 'explo_vehicle': 'Explosive Vehicle', 'explo_unknown': 'Unknown Explosive', 'firearm_unknown':'Unknown Firearm',  'explo_project':'Projectile Explosive', 'explo_other':'Other Explosive', 'ishostkid':'Hostage Kidnapping', 'Iraq':'Iraq', 'Afghanistan':'Afghanistan', 'India':'India', 'religion':'Religious Service', 'infrastructure':'Infrastructural Service', 'health':'Health Service', 'education':'Education Service', 'finance':'Financial Service', 'security':'Security Service', 'social':'Social Service'}
    for key, value in columns_dict.items():
        renamed.append(value)
    return renamed

'''Gradient Boosting modeling that outputs info for ROC'''
def gdb(X_train, y_train, X_test, y_test):
    print('GRADIENT BOOSTING')
    gdbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=1)
    result = gdbc.fit(X_train, y_train)
    y_predict = gdbc.predict(X_test)
    accuracy =gdbc.score(X_test, y_test)
    accuracy = np.around(accuracy,3)
    precision = precision_score(y_test, y_predict)
    precision = np.around(precision,3)
    recall = recall_score(y_test, y_predict)
    recall = np.around(recall,3)
    print('Accuracy of logistics regression classifer on test set:', (accuracy))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print("Recall: (TP/TP + FN), lower = we failed to predict some positives", (recall))
    print("Precision (TP/TP+FP), lower = we over-predicted positives: ", (precision))
    return y_predict, gdbc

'''AdaBoost modeling that outputs info for ROC'''
def ada(X_train, y_train, X_test, y_test):
    print('ADABOOSTING')
    ada = AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.1, n_estimators=100, random_state=1)
    result = ada.fit(X_train, y_train)
    y_predict = ada.predict(X_test)
    accuracy =ada.score(X_test, y_test)
    accuracy = np.around(accuracy,3)
    precision = precision_score(y_test, y_predict)
    precision = np.around(precision,3)
    recall = recall_score(y_test, y_predict)
    recall = np.around(recall,3)
    print('Accuracy of logistics regression classifer on test set:', (accuracy))
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print("Recall: (TP/TP + FN), lower = we failed to predict some positives", (recall))
    print("Precision (TP/TP+FP), lower = we over-predicted positives: ", (precision))
    return y_predict, ada

'''Plot ROC for all algorithms'''
def roc_plot(algo_name_list, y_predict_list, algo_for_predict_proba_list, y_test, X_test):
    plt.figure()
    for name, y_pred, instantiation in zip(algo_name_list, y_predict_list, algo_for_predict_proba_list):
        roc_auc = roc_auc_score(y_test, y_pred)
        roc_auc = np.around(roc_auc,2)
        fpr, tpr, thresolds = roc_curve(y_test, instantiation.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label='{} (area = {})'.format(name, roc_auc))
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1.1])
    plt.xlabel('False Positive Rate', weight='bold')
    plt.ylabel('True Positive Rate', weight='bold')
    plt.title('ROC Curve', weight='bold')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig('images/ROC')

'''Plot confusion matrices for Random Forests model'''
def plot_confusion_matrix(cnf_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cnf_matrix)

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title, weight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label', weight='bold')
    plt.xlabel('Predicted label', weight='bold')
    plt.tight_layout()
    # plt.show()
    if normalize:
        plt.savefig('images/CFM_normed')
    else:
        plt.savefig('images/CFM_notnormed')

'''Plot feature importances for Random Forests model'''
def feat_imp(importances):

    feature_names = [i[0] for i in importances]
    feature_importances = [i[1] for i in importances]

    plt.figure(figsize = (15,10))
    plt.bar(feature_names, feature_importances)
    plt.title('Feature Importance for Random Forest Model', weight='bold', fontsize=15)
    plt.ylabel("Feature Importance", weight='bold', fontsize=12)
    plt.xlabel("Features", weight='bold', fontsize=12)
    plt.xticks(rotation=60, horizontalalignment='right')
    plt.tight_layout()
    # plt.show()
    plt.savefig('images/feature_importance')

'''Plot ROC for algorithms'''
def roc(algo_name_list, y_predict_list, algo_for_predict_proba_list, y_test, X_test):
    plt.figure()
    for name, y_pred, instantiation in zip(algo_name_list, y_predict_list, algo_for_predict_proba_list):
        roc_auc = roc_auc_score(y_test, y_pred)
        roc_auc = np.around(roc_auc,2)
        fpr, tpr, thresolds = roc_curve(y_test, instantiation.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label='{} (area = {})'.format(name, roc_auc))
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1.1])
    plt.xlabel('False Positive Rate', weight='bold')
    plt.ylabel('True Positive Rate', weight='bold')
    plt.title('ROC Curvey for Model 3', weight='bold')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig('images/ROC')

'''Use for ttests'''
def bonferonni(n,k):
    x = math.factorial(n)/(math.factorial(n-k)*math.factorial(k))
    adjust = .05/x
    return adjust

'''Conduct ttests for diff features by groups that engage in cs. do not engage in suicide bombings'''
def ttests():
    df = pd.read_csv('data/df_merged_count.csv', low_memory=False)
    df_suicide = df[df.suicide==1]
    df_not_suicide = df[df.suicide==0]
    features = ['education_count', 'explo_vehicle_count', 'claimed_count', 'security_count', 'health_count', 'religion_count', 'social_count', 'explo_unknown_count', 'firearm_unknown_count', 'finance_count', 'infrastructure_count', 'explo_project_count', 'ishostkid_count', 'explo_other_count', 'India_count', 'Iraq_count', 'Afghanistan_count']
    correction = bonferonni(len(features),2)
    print('BONFERONNI CORRECTION: {}'.format(correction))
    pd.options.display.max_columns = 200

    for i in features:
        t, p = stats.ttest_ind(df_suicide[i], df_not_suicide[i], equal_var=True)
        if p <= correction:
            pprint('***{} is SIGNIFICANT: {}'.format(i, stats.ttest_ind(df_suicide[i], df_not_suicide[i], equal_var=True)))
            pprint('Suicide mean: {}, Not suicide mean: {}'.format(df_suicide[i].mean(), df_not_suicide[i].mean()))
            print('\n')
        else:
            pprint('{} is not significnat: {}'.format(i, stats.ttest_ind(df_suicide[i], df_not_suicide[i], equal_var=True)))
            print('\n')
