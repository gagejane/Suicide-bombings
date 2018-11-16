import pandas as pd
import numpy as np
from pprint import pprint
# np.set_printoptions(threshold='nan')

#----Dealing with imbalanced class------------------
from sklearn.utils import resample

#----Model selection--------------------------------
from sklearn.model_selection import train_test_split
# from sklearn import linear_model
#----------------------------------------------------

#----Models--------------------------------
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#----Tree visualization--------------------------------
import matplotlib.pyplot as plt
import itertools
import graphviz
import pydotplus
from IPython.display import Image


from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
# from sklearn.tree import export_graphviz
# from IPython.display import Image
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from scipy import interp
#from sklearn.cross_validation import KFold
# from sklearn.preprocessing import StandardScaler
# from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer
#
# from sklearn.tree import export_graphviz

'''DOWNSAMPLED'''
# #
# df = pd.read_csv('data/downsample_LDA_Train_forDT.csv', low_memory=False)
# df.dropna(inplace = True)
# # #
# # # print('MODEL 2')
# df=df.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)
# # # #
# # print('MODEL 1')
# #
# # # # '''use for models that omit topics'''
# # df = df[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]
#
# y_train = df[['suicide']]
# y_train = df.suicide.values
# X_train = df.drop(['suicide'], axis=1, inplace=True)
# X_train = df.values
# #
# df2 = pd.read_csv('data/downsample_LDA_Test_forDT.csv', low_memory=False)
# df2.dropna(inplace = True)
# # #
# # print('MODEL 2')
# df2=df2.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)
# # #
# # print('MODEL 1')
#
# # # # '''use for models that omit topics'''
# # df2 = df2[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]
#
# y_test = df2[['suicide']]
# y_test = df2.suicide.values
# X_test = df2.drop(['suicide'], axis=1, inplace=True)
# X_test = df2.values
# #




'''UPSAMPLED'''


# #
df = pd.read_csv('data/upsample_LDA_Train_forDT.csv', low_memory=False)
df.dropna(inplace = True)
# # print('MODEL 7 IS FULL MODEL')
#
df=df.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)
# # #
# #
# print('MODEL 3')

# ###WINNOWING DOWN MODELS
# ##WITHOUT TOPICS
# #full model, omitting topics'''
# df = df[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]
#
# print('MODEL 4')
#
#revised model that uses features with importance >= .02
# df = df[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'suicide']]
#
# print('MODEL 5')
# #
# # #revised model that uses features with importance >= .03
# df = df[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Projectile', 'Other_Explosive', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'suicide']]
#
# print('MODEL 6')
#
# #revised model that uses features with importance >= .08
# df = df[['Claimed_responsibility', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'suicide']]
# #
#print('MODEL 8')
# #WITH TOPICS
# #model that use features with importance >= .02 and includes topics
# df = df[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'topic1', 'topic2', 'suicide']]
#
# print('MODEL 9')
# #model that use features with importance >= .03 and includes topics
# df = df[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Projectile', 'Other_Explosive', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'topic1', 'topic2', 'suicide']]
#
# print('MODEL 10')
# #model that use features with importance >= .08 and includes topics
df = df[['Claimed_responsibility', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'topic1', 'topic2', 'suicide']]

y_train = df[['suicide']]
y_train = df.suicide.values
X_train = df.drop(['suicide'], axis=1, inplace=True)
X_train = df.values

df2 = pd.read_csv('data/upsample_LDA_Test_forDT.csv', low_memory=False)
df2.dropna(inplace = True)


df2=df2.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)
# #
# # #

# # ###WINNOWING DOWN MODELS
# # ##WITHOUT TOPICS
# # #full model, omitting topics'''
# df2 = df2[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_PlaceOfWorship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'suicide']]

#revised model that uses features with importance >= .02
# df2 = df2[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'suicide']]

# #revised model that uses features with importance >= .03
# df2 = df2[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Projectile', 'Other_Explosive', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'suicide']]
#
# #revised model that uses features with importance >= .08
# df2 = df2[['Claimed_responsibility', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'suicide']]
#
#
# #WITH TOPICS
# #model that use features with importance >= .02 and includes topics
# df2 = df2[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'topic1', 'topic2', 'suicide']]
#
# #model that use features with importance >= .03 and includes topics
# df2 = df2[['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Projectile', 'Other_Explosive', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'topic1', 'topic2', 'suicide']]
#
# #model that use features with importance >= .08 and includes topics
df2 = df2[['Claimed_responsibility', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'topic1', 'topic2', 'suicide']]

y_test = df2[['suicide']]
y_test = df2.suicide.values
X_test = df2.drop(['suicide'], axis=1, inplace=True)
X_test = df2.values

'''DECISION TREE'''
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
y_predict = dt.predict(X_test)
accuracy = dt.score(X_test, y_test)
#
# '''LOGISTIC REGRESSION'''
#
# model = LogisticRegression()
# lgt = model.fit(X_train, y_train)
# y_predict = lgt.predict(X_test)
# accuracy = lgt.score(X_test, y_test)
#
# '''RANDOM FOREST'''
# rfc = RandomForestClassifier(n_estimators=100)
# result = rfc.fit(X_train, y_train)
# y_predict = rfc.predict(X_test)
# accuracy = rfc.score(X_test, y_test)
#
# '''GRADIENT BOOSTING'''
# gdbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=1)
# result = gdbc.fit(X_train, y_train)
# y_predict = gdbc.predict(X_test)
# accuracy = gdbc.score(X_test, y_test)
#
# '''ADABOOSTING'''
# abc = AdaBoostClassifier(DecisionTreeClassifier(learning_rate=0.1, n_estimators=100, random_state=1)
# result = abd.fit(X_train, y_train)
# y_predict = abc.predict(X_test)
# accuracy = abc.score(X_test, y_test)


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
print('n test: {}'.format(len(y_test)))
'''PLOTTING THE TREE ON TEST DATA'''

#pip install graphviz
# #pip install pydot
#
# features = ['MiddleEast_NorthAfrican', 'Claimed_responsibility', 'Happened_After_2002', 'Military_Barracks', 'Politicial_Checkpoint', 'Political_Building', 'Religious_Place_Of_Worship', 'Utility_Location', 'Government_Politician', 'NonStateMilitia', 'Military_Checkpoint', 'Explosive_Vehicle', 'Unknown_Explosive', 'Unknown_Firearm', 'Rifle', 'Projectile', 'Other_Explosive', 'Handgun', 'Claim_via_Internet', 'Claim_via_Note', 'Personal_Claim', 'Hostage_Kidnapping', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria']
#
# class_names = ['Not suicide', 'Suicide']
#
# dot_data = tree.export_graphviz(dt, out_file=None, feature_names=features, class_names=class_names, filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)
# graph

'''ROC'''
# roc_auc = roc_auc_score(y_test, rfc.predict(X_test))
# roc_auc = np.around(roc_auc,2)
# # roc_auc = roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1])
# '''figure this shit out'''
# fpr, tpr, thresolds = roc_curve(y_test, rfc.predict_proba(X_test)[:, 1])
# plt.figure()
# plt.plot(fpr, tpr, label='AUC score (area = {})'.format(roc_auc))
# plt.plot([0,1],[0,1], 'r--')
# plt.xlim([0,1])
# plt.ylim([0,1.1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curvey for Model 3')
# plt.legend(loc='lower right')
# # plt.show()
# plt.savefig('roc_Model3')
#
#
'''CONFUSION MATRIX'''
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#
# class_names = ['Not suicide', 'Suicide']
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization for Model 3')
# plt.savefig('CFM_Model3_notnormed')
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix for Model 3')
# plt.savefig('CFM_Model3_normed')
# # plt.show()
# #
# # #
'''FEATURE IMPORTANCES'''
# zipped = (zip(df.columns, rfc.feature_importances_))
# importance = sorted(zipped, key = lambda t: t[1])
# pprint (importance)
#
# feature_names = [i[0] for i in importance]
# feature_importances = [i[1] for i in importance]
#
# plt.figure(figsize = (15,10))
# plt.bar(feature_names, feature_importances)
# plt.title('Feature Importance for Model 3')
# plt.ylabel("Feature Importance")
# plt.xlabel("Features")
# plt.xticks(rotation=70, horizontalalignment='right')
# plt.tight_layout()
# # plt.show()
# plt.savefig('feature_importance_Model3')
#
# #
