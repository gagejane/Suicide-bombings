import pandas as pd
import numpy as np
from pprint import pprint

#----Dealing with imbalanced class------------------
from sklearn.utils import resample

#----Model selection--------------------------------
from sklearn.model_selection import train_test_split
#----------------------------------------------------

#----Models--------------------------------
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant

#-----Visualization-------------------------------
import matplotlib.pyplot as plt
import itertools



def rename(df):
    df=df.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)
    return df

def model3(df):
    df = df[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]
    return df

algo_name_list = []
y_predict_list = []
algo_for_predict_proba_list = []

def DT(X_train, y_train, X_test, y_test):
    print('DECISION TREE')
    dt = DecisionTreeClassifier()
    result = dt.fit(X_train, y_train)
    y_predict = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)
    algo_name_list.append('Decision Tree')
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(dt)
    return CFM(accuracy, y_test, y_predict)

def LGT(X_train, y_train, X_test, y_test):
    print('LOGISTIC REGRESSION')
    lgt = LogisticRegression()
    result = lgt.fit(X_train, y_train)
    y_predict = lgt.predict(X_test)
    accuracy = lgt.score(X_test, y_test)
    algo_name_list.append('Logistic Regression')
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(lgt)
    return CFM(accuracy, y_test, y_predict)

def RFD(columns, X_train, y_train, X_test, y_test):
    print('RANDOM FOREST')
    rfc = RandomForestClassifier(n_estimators=100)
    result = rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)
    accuracy = rfc.score(X_test, y_test)
    algo_name_list.append('Random Forest')
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(rfc)
    cnf_matrix = confusion_matrix(y_test, y_predict)
    class_names = ['Not suicide', 'Suicide']
    importances = rfc.feature_importances_
    return CFM(accuracy, y_test, y_predict), plot_confusion_matrix(cnf_matrix, class_names, normalize=True,
                          title='Normalized Confusion Matrix for Model 3 using Random Forests'), plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion Matrix, Without Normalization, for Model 3 Using Random Forests'), feat_imp(columns, importances)

def GDBC(X_train, y_train, X_test, y_test):
    print('GRADIENT BOOSTING')
    gdbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=1)
    result = gdbc.fit(X_train, y_train)
    y_predict = gdbc.predict(X_test)
    accuracy = gdbc.score(X_test, y_test)
    algo_name_list.append('Gradient Boosting')
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(gdbc)
    return CFM(accuracy, y_test, y_predict)

def ADA(parX_train, y_train, X_test, y_test):
    print('ADABOOSTING')
    ada = AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.1, n_estimators=100, random_state=1)
    result = ada.fit(X_train, y_train)
    y_predict = ada.predict(X_test)
    accuracy = ada.score(X_test, y_test)
    algo_name_list.append('AdaBoosting')
    y_predict_list.append(y_predict)
    algo_for_predict_proba_list.append(ada)
    return CFM(accuracy, y_test, y_predict)

def CFM(accuracy, y_test, y_predict):
    accuracy = np.around(accuracy,3)
    precision = precision_score(y_test, y_predict)
    precision = np.around(precision,3)
    recall = recall_score(y_test, y_predict)
    recall = np.around(recall,3)
    pprint('Accuracy of logistics regression classifer on test set: {}'.format(accuracy))
    #:.2f
    print("\nConfusion matrix:")
    print("  TN    FP")
    print("  FN    TP")
    cnf_matrix = confusion_matrix(y_test, y_predict)
    print(cnf_matrix)
    # print(classification_report(y_test, y_predict))
    print("Recall: (TP/TP + FALSE NEGATIVE; bottom column), i.e., lower = we failed to predict some positives", (recall))
    print("\nPrecision (TP/TP+FALSE POSITIVE; right hand column), i..e, lower = we over-predicted positives: ", (precision))

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
    plt.savefig('ROC_Model3')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, weight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', weight='bold')
    plt.xlabel('Predicted label', weight='bold')
    plt.tight_layout()
    # plt.show()
    if normalize:
        plt.savefig('CFM_Model3_normed')
    else:
        plt.savefig('CFM_Model3_notnormed')

def feat_imp(columns, importances):
    zipped = (zip(columns, importances))
    importance = sorted(zipped, key = lambda t: t[1])
    # pprint (importance)

    feature_names = [i[0] for i in importance]
    feature_importances = [i[1] for i in importance]

    plt.figure(figsize = (15,10))
    plt.bar(feature_names, feature_importances)
    plt.title('Feature Importance for Model 3', weight='bold')
    plt.ylabel("Feature Importance", weight='bold')
    plt.xlabel("Features", weight='bold')
    plt.xticks(rotation=70, horizontalalignment='right')
    plt.tight_layout()
    # plt.show()
    plt.savefig('feature_importance_Model3')

if __name__ == '__main__':
    print('THESE FOLLOWING TESTS MODEL 3')
    df_train = pd.read_csv('data/upsample_LDA_Train_forDT.csv', low_memory=False)

    df_train = rename(df_train)
    df_train = model3(df_train)

    y_train = df_train[['suicide']]
    y_train = df_train.suicide.values
    X_train = df_train.drop(['suicide'], axis=1, inplace=True)
    X_train = df_train.values
    columns = df_train.columns

    df_test = pd.read_csv('data/upsample_LDA_Train_forDT.csv', low_memory=False)
    df_test = rename(df_test)
    df_test = model3(df_test)

    y_test = df_test[['suicide']]
    y_test = df_test.suicide.values
    X_test = df_test.drop(['suicide'], axis=1, inplace=True)
    X_test = df_test.values

    print(DT(X_train, y_train, X_test, y_test))
    print(LGT(X_train, y_train, X_test, y_test))
    print(RFD(columns, X_train, y_train, X_test, y_test))
    print(GDBC(X_train, y_train, X_test, y_test))
    print(ADA(X_train, y_train, X_test, y_test))

    print(roc(algo_name_list, y_predict_list, algo_for_predict_proba_list, y_test, X_test))
