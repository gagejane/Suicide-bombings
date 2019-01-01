import code_for_prepare_data as prep
import pandas as pd

if __name__ == '__main__':

    '''Turn excel file into csv, which makes the code run faster'''
    # prep.make_csv()

    '''Get the GTD data ready for merging; create df that contains variables for modeling'''
    # prep.clean_gtd()

    '''Deal with class imbalance; suicide bombings are rare, so make their occurance equal to no-suicide bombing occurence'''
    '''Then, create Train and Test dfs, which will be run through LDA before modeling'''
    # prep.up_down_sample()

    '''Run LDA and merge with up/down train/test datasets'''
    # df_down_train = pd.read_csv('data/df_suicide_downsampled_Train.csv', low_memory=False)
    # filename = 'data/downsample_LDA_Train_forDT.csv'
    # prep.make_LDA(df_down_train, filename)
    # df_down_test = pd.read_csv('data/df_suicide_downsampled_Test.csv', low_memory=False)
    # filename = 'data/downsample_LDA_Test_forDT.csv'
    # prep.make_LDA(df_down_test, filename)
    # df_up_train = pd.read_csv('data/df_suicide_upsampled_Train.csv', low_memory=False)
    # filename = 'data/upsample_LDA_Train_forDT.csv'
    # prep.make_LDA(df_up_train, filename)
    # df_up_test = pd.read_csv('data/df_suicide_upsampled_Test.csv', low_memory=False)
    # filename = 'data/upsample_LDA_Test_forDT.csv'
    # prep.make_LDA(df_up_test, filename)
