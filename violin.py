import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import resample

df = pd.read_csv('data/df_2017.csv', low_memory=False)
df_majority = df[df.suicide==0]
df_minority = df[df.suicide==1]
#
df_majority_downsampled = resample(df_majority, replace=False, n_samples=844, random_state=123)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df = df_downsampled.copy()
# print(df['suicide'].value_counts())
df['suicideR'] = np.where((df['suicide'] == 1), 1, 0)
# print(df['suicideR'].value_counts())
#
# df = df[:10000]
# df_suicide = df[df.suicide==1]
# df_suicide_kill = df_suicide[['nkill']]
# df_suicide_wound = df_suicide[['nwound']]
# # df_suicide_topic1 = df_suicide[['topic1']]
# # df_suicide_topic2 = df_suicide[['topic2']]
# # df_suicide_topic1.dropna(inplace = True)'
#
#
# # df_suicide_wound.dropna(inplace = True)
# # # df_suicide = df_suicide[df_suicide['nkill'] < 1300]
# # df_suicide = df_suicide.values
# # df_suicide = np.ravel(df_suicide)
# # # df_suicide=df_suicide[:100]
#
# df_not_suicide = df[df.suicide==0]
# df_not_suicide_kill = df_not_suicide[['nkill']]
# df_not_suicide_wound = df_not_suicide[['nwound']]
# # df_not_suicide_topic1 = df_not_suicide[['topic1']]
# # df_not_suicide_topic2 = df_not_suicide[['topic2']]
#
# # df_not_suicide_kill.dropna(inplace = True)
# # df_not_suicide_wound.dropna(inplace = True)
# # #
# # df_not_suicide_kill = df_not_suicide_kill.sample(n=6525)
# # df_not_suicide_wound = df_not_suicide_wound.sample(n=6525)
# #
# # print(stats.ttest_ind(df_suicide_topic1,df_not_suicide_topic1, equal_var=False))
# # print(stats.ttest_ind(df_suicide_topic2,df_not_suicide_topic2, equal_var=False))
#
'''create object version of suicide variable'''
dict_suicide = {1: 'Suicide', 0: 'Not suicide'}
df['suicide_text'] = df['suicideR'].replace(dict_suicide)
# print(df['suicide_text'].value_counts())
# print(df['suicideR'].value_counts())

df = df[['suicide_text','nkill','nwound']]
df.dropna(inplace = True)

numeric_cols = [col for col in df if df[col].dtype.kind != 'O']
df[numeric_cols] += 1

df['log_wound'] = np.log(df['nwound'])
df['log_kill'] = np.log(df['nkill'])



#plot it
# sns.violinplot( x=df["species"], y=df["sepal_length"], palette=my_pal)

df_plot = df[['suicide_text','log_kill', 'log_wound']]
# Make a dictionary with one specific color per group:
# my_pal = {"Not suicide": "o", "Suicide": "b"}
ax = sns.violinplot(x = 'suicide_text', y = 'log_kill', data = df_plot)
ax.set(xlabel='Type of attack', ylabel='Log(number killed)', title ='Log Number of People Killed by Type of Attack in 2017')
plt.savefig('killed')
# ax = sns.violinplot(x = 'suicide_text', y = 'log_wound', data = df_plot)
# ax.set(xlabel='Type of attack', ylabel='Log(number wounded)', title ='Log Number of People Wounded by Type of Attack')
# plt.savefig('wounded')


# plt.show()
# # x1 = np.random.normal(0, 0.8, 1000)
# # # x2 = np.random.normal(-2, 1, 1000)
# # # x3 = np.random.normal(3, 2, 1000)
# # plt.figure(figsize = (15,10))
# # # kwargs = dict(histtype='stepfilled', alpha=0.3)
# # #, bins=40, , normed=True
# plt.hist(df_suicide_wound, density=True)
# # # plt.ylim(0, 1500)
# #
# # # plt.hist(df_not_suicide, **kwargs)
# # # plt.hist(x3, **kwargs);
# plt.show()
# # # plt.savefig(hist)
