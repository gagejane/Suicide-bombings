import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import geopandas as gpd
import re
# from geopandas import GeoSeries, GeoDataFrame
#website for colors https://www.w3schools.com/colors/colors_picker.asp

# for df, item in zip(df_sum_list, legend_list):
#     line = df.plot(kind = 'area', label=item, alpha=0.25, linewidth=2.0)
#     line.set_label(item)
# plt.legend(title='legend_title', loc='upper left')
# plt.title('title', weight='bold')
# # if xrestrict_upper > 0:
# #     plt.xlim(xrestrict_lower, xrestrict_upper)
# # if yrestrict_upper > 0:
#     # plt.ylim(0, yrestrict_upper)
# plt.xlabel('xlab', weight='bold')
# plt.ylabel('ylab', weight='bold')
# # if save_bool == 1:
# # plt.savefig(save_as)
# # if plot_bool == 1:
# plt.show()

# def prev_over_time(df, save_bool, plot_bool):
'''
Plot prevalence of groups over time
save_bool: 1 == save the plot, 0 == don't save it
plot_bool: 1 == display the plot, 0 == don't display it
'''
# df_year_count = df.groupby('iyear').count()['suicide']
# df_year_count.plot(kind='area', color='m', alpha=0.25, linewidth=3.0)
# plt.title('Prevalence of Suicide Bombing Over Time', weight='bold', size=14)
# plt.xlabel('Year', weight='bold')
# plt.ylabel('Event Count', weight='bold')
# # if save_bool == 1:
# # plt.savefig('Suicide_over_time')
# # if plot_bool == 1:
# plt.show()

def prov_over_time(save_bool, plot_bool):
    '''
    Plot count of provisions over time
    save_bool: 1 == save the plot, 0 == don't save it
    plot_bool: 1 == display the plot, 0 == don't display it
    '''
    df_year_count_prov = df.groupby('year').sum()['total']
    df_year_count_prov.plot(kind='area', color='g', alpha=0.25, linewidth=3.0)
    plt.title('Total Provisions Over Time', weight='bold', size=14)
    plt.xlabel('Year', weight='bold')
    plt.ylabel('Provision Count', weight='bold')
    plt.tight_layout()
    if save_bool == 1:
        plt.savefig('Provisions_over_time')
    if plot_bool == 1:
        plt.show()

def make_pie(save_bool, plot_bool):
    '''
    Plot pie chart of different provision types, across time, and across groups
    save_bool: 1 == save the plot, 0 == don't save it
    plot_bool: 1 == display the plot, 0 == don't display it
    '''
    df_sum = df.sum()
    df_prov_pie = df_sum[['religion','infrastructure', 'health', 'education', 'finance', 'security', 'society']]
    title = plt.title('Sum of Provisions in Full Dataset, By Type', weight='bold', size=14)
    plt.gca().axis('equal')
    pie = plt.pie(df_prov_pie, startangle=0, autopct='%1.0f%%')
    labels = ['religion','infrastructure', 'health', 'education', 'finance', 'security', 'society']
    plt.legend(pie[0], labels, bbox_to_anchor=(1,0.5), loc='center right', fontsize=10,
    bbox_transform=plt.gcf().transFigure, title='Provision Type')
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.8)
    plt.ylabel('')
    if save_bool == 1:
        plt.savefig('Pie_prov')
    if plot_bool == 1:
        plt.show()

def make_world(save_bool, plot_bool):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        ax = world.plot(figsize=(20,5))
        ax.axis('off')
        world.head()
        if save_bool == 1:
            plt.savefig('World')
            world.head().to_csv('world_head.csv')
        if plot_bool == 1:
            plt.show()

# def clean_then_plot_heatmap(df):
#     '''
#     Stacking country names when there are multiple names per cell that are separated by a comma. This results in a df that has rows of countries and years, where each row is representative of a non-state group. e.g., if there are 20 different groups in one country in a given year, there will be 20 country entries for that year
#     df: original pandas df
#     '''
#     df_base = df[['country']]
#     df_base_disaggr = pd.DataFrame(df_base.base.str.split(',').tolist(), index=df_base.year.stack()
#     df_base_disaggr = df_base_disaggr.reset_index()[[0, 'iyear']] # var1 variable is currently labeled 0
#     df_base_disaggr.columns = ['base', 'year'] # renaming var1
#     df_base_disaggr['base'] = df_base_disaggr['base'].apply(remove_spaces) # for .apply don't have to pass in a parameter; it knows to check row by row
#     # return df_base_disaggr.head()
#     count_countries(df_base_disaggr)

# def remove_spaces(row):
#     '''
#     A function to remove remaining weird spaces in country names:
#     Remove spaces from column of type string
#     row: don't have to pass in parameter here, because .apply automatically knows to check data row by row
#     '''
#     country = row.split()
#     cleaned_lst = []
#     for name in country:
#         name = name.replace(" ","")
#         if len(name) > 1:
#             cleaned_lst.append(name)
#     return " ".join(cleaned_lst)

def count_countries(df):
    '''
    Counting the number of country instances, across years
    df: original pandas df
    '''
    df_base_count = df.groupby('name').count()
    df_base_count = df_base_count.reset_index()
    df_base_count.rename(columns ={'id':'count'}, inplace = True)
    # # df_base_count.rename(columns ={'base':'name'}, inplace = True)
    df_base_count = df_base_count.sort_values('count', ascending = False)
    df_base_count.reset_index(drop=True, inplace = True)
    final_clean(df_base_count)
    # print(df_base_count.info())
    # print(df_base_count.head())

def final_clean(df):
    '''
    Making sure spelling/wording for countries in original df is consistent with that of the world dataset
    df: df from count_countries
    '''
    # dict_cleaned = {'West Bank/Gaza':'Israel',
    # 'Northern Ireland (UK)':'Ireland',
    # 'Kashmir':'India',
    # 'Democratic Republic of the Congo':'Congo',
    # 'Burma (Myanmar)':'Myanmar',
    # 'Northern Ireland':'Ireland',
    # 'United Kingdom)':'United Kingdom',
    # 'Chechnya':'Russia',
    # 'Federal Republic of Germany':'Germany',
    # 'German Democratic Republic':'Germany',
    # 'Western Sahara':'W. Sahara',
    # 'Namibia (South West Africa)':'Namibia',
    # 'Republic of Macedonia':'Macedonia',
    # 'Burma (myanmar)':'Myanmar',
    # 'Rhodesia':'Zimbabwe',
    # 'chile':'Chile',
    # 'Gaza/Westbank':'Israel',
    # "Cote d'Ivoire":"Côte d'Ivoire",
    # 'FRY (Kosovo)':'Kosovo',
    # 'Bosnia':'Bosnia and Herz.',
    # 'Bahrain':'Saudi Arabia',
    # 'Serbia and Montenegro':'Serbia',
    # 'Kyrgyztan':'Kyrgyzstan',
    # 'Singapore':'Malaysia',
    # 'Guadeloupe':'Dominican Rep.',
    # 'Corsica':'Italy',
    # 'Colmbia':'Colombia',
    # 'Laos':'Lao PDR'}
    # df['name'] = df['name'].replace(dict_cleaned)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # print(world.info())
    # print(world.head())
    country_names = world[['name', 'continent', 'geometry', 'iso_a3']]
    world_merged = country_names.merge(df, how='left', on='name')
    world_merged.fillna(value=0,inplace=True)
    # print(world_merged['count'])
    plot_heatmap(world_merged)

def plot_heatmap(df):
    '''
    Plot the heatmap
    df: df from final_clean()
    '''
    df.plot(column='count', cmap='Oranges', linewidth=0.5, edgecolor='black', legend=True,
     figsize=(20,20), scheme='fisher_jenks')
    plt.title('Suicide Bombing Frequency in 2017', weight='bold', fontsize=25)
    plt.savefig('Heatmap')
    # plt.show()

def US_groups(df):
    '''list of all organizations in the U.S.'''
    df_US = df[df['base'].str.contains("United States")]
    df_US['name'].value_counts().to_csv('US_group_list.csv')

def sep_dfs(df, group_list):
    '''
    This function will take in an original dataframe, and create separate dataframes for all groups in group_list
    df: original dataframe
    group_list: list of groups to observe
    base_str: country of interest, defaults to United States
    '''
    df_list = []
    for str in group_list:
        df_for_merge = df[(df['suicide_text'].str.contains(str))]
        df_list.append(df_for_merge)
    return df_list
    # (df_list, group_list, groupby, to_sum)

def sum_one_var(df_list, groupby, to_sum):
    '''
    This function will take in a list of dataframes, each for a different group, group by some variable (groupby),
    and sum up another variable (to_sum), and return the grouped and summed dfs
    df_list: list of dfs created in sep_dfs
    groupby: variable to group on
    to_sum: variable to sum up
    '''
    sum_list = []
    for df in df_list:
        grouped = df.groupby(groupby).count()[to_sum]
        sum_list.append(grouped)
        # print(grouped)
    return sum_list

def sum_mult_vars(df_list, groupby, to_sum_list):
    '''
    This function will take in a list of dataframes, each for a different group (can also be a list containing only one group),
    group by some variable (groupby), and sum up a group of variables, separately (to_sum_list), and return the grouped and summed dfs
    df_list: list of dfs created in sep_dfs
    groupby: variable to group on
    to_sum: list of variables to sum up
    '''
    sum_list = []
    for df in df_list:
        for item in to_sum_list:
            grouped = df.groupby(groupby).count()[item]
            sum_list.append(grouped)
    return sum_list

def multi_line_plot(df_sum_list, legend_list, title, xlab, ylab, save_bool, plot_bool, save_as, legend_title, xrestrict_lower=0, xrestrict_upper=0, yrestrict_upper=0):  ### May need to add save_as = 0 , plot_bool = 0 to make sure you dont get an error
    '''
    This function will create a set of line graphs in a single space.
    df_sum_list: list of dfs that have been grouped and summed on some variable in function sum_var
    group_list: list of groups/categories to observe (this is the multi-category variable in the legend)
    title: title for plot
    xlab: x axis label for plot
    ylab: y axis label for plot
    save_bool: 1 == save the plot, 0 == don't save it
    plot_bool: 1 == display the plot, 0 == don't display it
    '''
    colors = ['#cc6600', '#336699']
    alphas = [.6, .2]
    for df, item, color, alpha in zip(df_sum_list, legend_list, colors, alphas):
        line = df.plot(kind = 'area', label=item, alpha=alpha, linewidth=2.0, color=color)
        line.set_label(item)
    plt.legend(title=legend_title, loc='upper left')
    plt.title(title, weight='bold')
    if xrestrict_upper > 0:
        plt.xlim(xrestrict_lower, xrestrict_upper)
    if yrestrict_upper > 0:
        plt.ylim(0, yrestrict_upper)
    plt.xlabel(xlab, weight='bold')
    plt.ylabel(ylab, weight='bold')
    if save_bool == 1:
        plt.savefig(save_as)
    if plot_bool == 1:
        plt.show()

def merge_dfs(list_of_dfs):
    merged = pd.DataFrame()
    for df in list_of_dfs:
        merged = merged.append(df, ignore_index=True)
    return merged



if __name__ == '__main__':
    # df = pd.read_csv("data/df_suicide.csv")
    # df = pd.read_csv('data/globalterrorismdb_0718dist.csv', low_memory=False)
    df2017 = pd.read_csv('data/df_2017.csv', low_memory=False)
    df2017_suicide = df2017.loc[(df2017.suicide==1)]
    '''making a heatmap'''
    s = "4: Afghanistan 5: Albania 6: Algeria 7: Andorra 8: Angola 10: Antigua and Barbuda 11: Argentina 12: Armenia 14: Australia 15: Austria 16: Azerbaijan 17: Bahamas 18: Bahrain 19: Bangladesh 20: Barbados 21: Belgium 22: Belize 23: Benin 24: Bermuda 25: Bhutan 26: Bolivia 28: Bosnia-Herzegovina 29: Botswana 30: Brazil 31: Brunei 32: Bulgaria 33: Burkina Faso 34: Burundi 35: Belarus 36: Cambodia 37: Cameroon 38: Canada 40: Cayman Islands 41: Central African Republic 42: Chad 43: Chile 44: China 45: Colombia 46: Comoros 47: Republic of the Congo 49: Costa Rica 50: Croatia 51: Cuba 53: Cyprus 54: Czech Republic 55: Denmark 56: Djibouti 57: Dominica 58: Dominican Republic 59: Ecuador 60: Egypt 61: El Salvador 62: Equatorial Guinea 63: Eritrea 64: Estonia 65: Ethiopia 66: Falkland Islands 67: Fiji 68: Finland 69: France 70: French Guiana 71: French Polynesia 72: Gabon 73: Gambia 74: Georgia 75: Germany 76: Ghana 77: Gibraltar 78: Greece 79: Greenland 80: Grenada 81: Guadeloupe 83: Guatemala 84: Guinea 85: Guinea-Bissau 86: Guyana 87: Haiti 88: Honduras 89: Hong Kong 90: Hungary 91: Iceland 92: India 93: Indonesia 94: Iran 95: Iraq 96: Ireland 97: Israel 98: Italy 99: Ivory Coast 100: Jamaica 101: Japan 102: Jordan 103: Kazakhstan 104: Kenya 106: Kuwait 107: Kyrgyzstan 108: Laos 109: Latvia 110: Lebanon 111: Lesotho 112: Liberia 113: Libya 114: Liechtenstein 115: Lithuania 116: Luxembourg 117: Macau 118: Macedonia 119: Madagascar 120: Malawi 121: Malaysia 122: Maldives 123: Mali 124: Malta 125: Man, Isle of 127: Martinique 128: Mauritania 129: Mauritius 130: Mexico 132: Moldova 134: Mongolia 136: Morocco 137: Mozambique 138: Myanmar 139: Namibia 141: Nepal 142: Netherlands 143: New Caledonia 144: New Zealand 145: Nicaragua 146: Niger 147: Nigeria 149: North Korea 151: Norway 152: Oman 153: Pakistan 155: West Bank and Gaza Strip 156: Panama 157: Papua New Guinea 158: Paraguay 159: Peru 160: Philippines 161: Poland 162: Portugal 163: Puerto Rico 164: Qatar 166: Romania 167: Russia 168: Rwanda 173: Saudi Arabia 174: Senegal 175: Serbia-Montenegro 176: Seychelles 177: Sierra Leone 178: Singapore 179: Slovak Republic 180: Slovenia 181: Solomon Islands 182: Somalia 183: South Africa 184: South Korea 185: Spain 186: Sri Lanka 189: St. Kitts and Nevis 190: St. Lucia 192: St. Martin 195: Sudan 196: Suriname 197: Swaziland 198: Sweden 199: Switzerland 200: Syria 201: Taiwan 202: Tajikistan 203: Tanzania 204: Togo 205: Thailand 206: Tonga 207: Trinidad and Tobago 208: Tunisia 209: Turkey 210: Turkmenistan 213: Uganda 214: Ukraine 215: United Arab Emirates 216: Great Britain 217: United States 218: Uruguay 219: Uzbekistan 220: Vanuatu 221: Vatican City 222: Venezuela 223: Vietnam 226: Wallis and Futuna 228: Yemen 229: Democratic Republic of the Congo 230: Zambia 231: Zimbabwe 233: Northern Ireland 235: Yugoslavia 236: Czechoslovakia 238: Corsica 334: Asian 347: East Timor 349: Western Sahara 351: Commonwealth of Independent States 359: Soviet Union 362: West Germany (FRG) 377: North Yemen 403: Rhodesia 406: South Yemen 422: International 428: South Vietnam 499: East Germany (GDR) 520: Sinhalese 532: New Hebrides 603: United Kingdom 604: Zaire 605: People's Republic of the Congo 999: Multinational 1001: Serbia 1002: Montenegro 1003: Kosovo 1004: South Sudan"

    country_list = re.split('(\d+)',s)
    length = len(country_list)
    country_list_len = range(length)
    numbers = country_list[1::2]
    countries = country_list[2::2]

    new_countries = []
    for country in countries:
        # print(type(country))
        new = country[2::]
        new2 = new[:-1]
        new_countries.append(new2)

    new_numbers = []
    for number in numbers:
        new = int(number)
        # print(new)
        new_numbers.append(new)

    dict_country = dict(zip(new_numbers, new_countries))
    df2017_suicide['country_names'] = df2017_suicide['country'].replace(dict_country)
    df_country_names = df2017_suicide[['country','country_names']]
    df_country_names.columns=['id', 'name']


    # df = pd.read_csv('data/globalterrorismdb_0718dist.csv')
    # df_suicide = df[df.suicide==1]
    # df_suicide = df_suicide[['iyear', 'gname']]
    # df_not_suicide = df[df.suicide==0]
    # df_not_suicide = df_not_suicide[['iyear', 'gname']]
    # df_year_suicide = df_suicide.groupby('iyear').count()
    # df_year_not_suicide = df_not_suicide.groupby('iyear').count()
    # df_sum_list = [df_year_suicide, df_year_not_suicide]
    # legend_list = ['Suicide bombing', 'Not suicide bombing']
    #install ConvertToUTF8 package in ATOM for editing and saving files in order to open

    # print(prev_over_time(df, 1,0))
    # print(prov_over_time(1,1))
    # print(make_pie(1,0))
    # print(make_world(1,0))
    print(count_countries(df_country_names))
    # print(US_groups(df))

    '''Plotting two line graphs in one space'''
    # df_lines = df[['suicide', 'iyear']]
    # dict_suicide = {1: 'Suicide', 0: 'Not suicide'}
    # df_lines['suicide_text'] = df_lines['suicide'].replace(dict_suicide)
    # suicide_text = ['Suicide', 'Not suicide']
    # # print(sep_dfs(df_lines, suicide_text))
    # make_separate_dfs = sep_dfs(df_lines, suicide_text)
    # # print(sum_one_var(make_separate_dfs, 'iyear', 'suicide_text'))
    # make_sums_one_var = sum_one_var(make_separate_dfs, 'iyear', 'suicide_text')
    # print(multi_line_plot(make_sums_one_var, suicide_text, title='Count of Suicide Bombings over time', xlab='Year', ylab='Event count', save_bool=1, plot_bool=0, save_as='Suicide_over_time', legend_title='Type of Attack', xrestrict_lower=0, xrestrict_upper=0, yrestrict_upper=17000))

    '''Plot provisions for a single group over time and by type'''
    # group_list = ['KKK']
    # prov_list = ['religion','infrastructure', 'health', 'education', 'finance', 'security', 'society']
    # prov_list_no_soc = ['religion','infrastructure', 'health', 'education', 'finance', 'security']
    # make_separate_dfs = sep_dfs(df, group_list)

    '''Plot with Society'''
    # make_sums_mult_vars = sum_mult_vars(make_separate_dfs, 'year', prov_list)
    # print(multi_line_plot(make_sums_mult_vars, prov_list, 'Ku Klux Klan Provisions Over Time', 'Year', 'Provision Count', 1, 0, 'KKK_prov_type_over_time', 'Provision Type', 0, 0, yrestrict_upper=25000))

    '''Plot without Society'''
    # make_sums_mult_vars_no_soc = sum_mult_vars(make_separate_dfs, 'year', prov_list_no_soc)
    # print(multi_line_plot(make_sums_mult_vars_no_soc, prov_list_no_soc, 'Ku Klux Klan Provisions Over Time (without Society)', 'Year', 'Provision Count', 1, 0, 'KKK_prov_type_over_time_no_soc', 'Provision Type'))
    # print(multi_line_plot(make_sums_mult_vars_no_soc, prov_list_no_soc, 'Ku Klux Klan Provisions During the 1990s (without Society)', 'Year', 'Provision Count', 1, 0, 'KKK_prov_type_over_time_no_soc_90s', 'Provision Type', xrestrict_lower=1990, xrestrict_upper=1999, yrestrict_upper=0))
    # # def multi_line_plot(df_sum_list, legend_list, title, xlab, ylab, save_bool, plot_bool, save_as, legend_title, xrestrict_lower=0, xrestrict_upper=0, yrestrict_upper=0):  ### May need to add save_as = 0 , plot_bool = 0 to make sure you dont get an error
    # grouped = df.groupby('gtd_gname')
