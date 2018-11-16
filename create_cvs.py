import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

'''THIS CREATES TRAIN AND TEST DATAFRAMES'''

def column_clean(df, filename):
    df['ME_NA'] = df['ME_NA'] == True
    df['claimed'] = df['claimed'] == True
    df['suicide'] = df['suicide'] == True
    df['year_2003'] = df['year_2003'] == True

    df['mil_check'] = df['mil_check'] == True
    df['mil_barr'] = df['mil_barr'] == True
    df['pol_check'] = df['pol_check'] == True
    df['pol_build'] = df['pol_build'] == True
    df['rel_place'] = df['rel_place'] == True
    df['util_elec'] = df['util_elec'] == True
    df['gov_polit'] = df['gov_polit'] == True
    df['terr_nonstate'] = df['terr_nonstate'] == True
    df['mil_check'] = df['mil_check'] == True

    df['explo_vehicle'] = df['explo_vehicle'] == True
    df['explo_unknown'] = df['explo_unknown'] == True
    df['firearm_unknown'] = df['firearm_unknown'] == True
    df['firearm_rifle'] = df['firearm_rifle'] == True
    df['explo_project'] = df['explo_project'] == True
    df['explo_other'] = df['explo_other'] == True
    df['firearm_handgun'] = df['firearm_handgun'] == True
    df['claim_internet'] = df['claim_internet'] == True
    df['claim_note'] = df['claim_note'] == True
    df['claim_personal'] = df['claim_personal'] == True

    df['ishostkid'] = df['ishostkid'] == True

    df['Iraq'] = df['Iraq'] == True
    df['Afghanistan'] = df['Afghanistan'] == True
    df['India'] = df['India'] == True
    df['Columbia'] = df['Columbia'] == True
    df['Syria'] = df['Syria'] == True

    df=df.rename({'ME_NA':'MiddleEast_NorthAfrican', 'claimed':'Claimed_responsibility', 'year_2003':'Happened_After_2002', 'mil_check':'Military_Checkpoint', 'mil_barr':'Military_Barracks', 'pol_check':'Politicial_Checkpoint', 'pol_build':'Political_Building', 'rel_place':'Religious_PlaceOfWorship', 'util_elec':'Utility_Location', 'gov_polit':'Government_Politician', 'terr_nonstate': 'NonStateMilitia', 'explo_vehicle': 'Explosive_Vehicle', 'explo_unknown': 'Unknown_Explosive', 'firearm_unknown':'Unknown_Firearm', 'firearm_rifle':'Rifle', 'explo_project':'Projectile', 'explo_other':'Other_Explosive', 'firearm_handgun':'Handgun', 'claim_internet':'Claim_via_Internet', 'claim_note':'Claim_via_Note', 'claim_personal':'Personal_Claim', 'ishostkid':'Hostage_Kidnapping'}, axis=1)

    # df.to_csv(filename, index=False)
    return df

def create_test_train(df, filename1, filename2):
    y = df.pop('suicide').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)

    X_train = pd.DataFrame(data=X_train, index = None, columns = ['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'motive'])
    # print('X_train.shape {}'.format(X_train.shape))
    X_test = pd.DataFrame(data=X_test, index = None, columns = ['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'motive'])
    # print('X_test.shape {}'.format(X_test.shape))
    y_train = pd.DataFrame(data=y_train, index = None, columns = ['suicide'])
    # print('y_train.shape {}'.format(y_train.shape))
    y_test = pd.DataFrame(data=y_test, index = None, columns = ['suicide'])
    # print('y_test.shape {}'.format(y_test.shape))

    Train = pd.concat([X_train, y_train], axis=1)
    Test = pd.concat([X_test,y_test], axis=1)
    # print ('Train.shape {}, Test.shape {}'.format(Train.shape, Test.shape   ))
    Train.to_csv(filename1, index=False)
    Test.to_csv(filename2, index=False)
    print (Train.shape, Test.shape)
    # y_train.to_csv(filename3, index=False)
    # y_test.to_csv(filename4, index=False)


if __name__ == '__main__':

    df = pd.read_csv('data/df_suicide_DT.csv', low_memory=False)
    pd.options.display.max_columns = 200

    df.dropna(inplace = True)

    '''the data are imbalanced -- only 5% of suicide == 1'''
    '''discussion: https://stats.stackexchange.com/questions/28029/training-a-decision-tree-against-unbalanced-data'''
    '''here is how to deal with it: https://elitedatascience.com/imbalanced-classes'''
    df_majority = df[df.suicide==0]
    df_minority = df[df.suicide==1]
    #
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=2567, random_state=123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    print(df_downsampled['suicide'].value_counts())
    # # #
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=46277, random_state=123)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    print(df_upsampled['suicide'].value_counts())

    # print(column_clean(df_upsampled, 'data/df_suicide_upsampled.csv'))
    create_test_train(column_clean(df_downsampled, 'data/df_suicide_downsampled.csv'), filename1='data/df_suicide_downsampled_Train.csv', filename2='data/df_suicide_downsampled_Test.csv')
    create_test_train(column_clean(df_upsampled, 'data/df_suicide_upsampled.csv'), filename1='data/df_suicide_upsampled_Train.csv', filename2='data/df_suicide_upsampled_Test.csv')
