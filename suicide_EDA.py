import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy import stats
import re

def multipletests(pvals, alpha=0.05, method='bonferonni', is_sorted=False, returnsorted=False):

    '''from: https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html'''
    pass

def crosstabs(df, new_df_name_list, feature_name_list, df_suicide):
    for df_name,feature in zip(new_df_name_list, feature_name_list):
        df_name = df[feature]
# df_crosstabs_targ = pd.crosstab(df_targtype1,df_suicide, normalize='columns')
        print('feature {}'.format(pd.crosstab(df_name,df_suicide, normalize='columns')))

if __name__ == '__main__':
    # df = pd.read_excel('globalterrorismdb_0718dist.xlsx')
    # df.to_csv('globalterrorismdb_0718dist.csv', encoding='utf-8', index=False)
    df = pd.read_csv('data/globalterrorismdb_0718dist.csv', low_memory=False)
    # df = pd.read_csv('data/downsample_LDA.csv', low_memory=False)
    # df = pd.read_csv('data/upsample_LDA.csv', low_memory=False)
    pd.options.display.max_columns = 200

    '''recode variables so that labels are visible'''
    dict_targtype1 = {1: 'Business', 2: 'Government', 3: 'Police', 4: 'Military', 5: 'Abortion related', 6: 'Airports and Aircraft', 7: 'Government',
    8: 'Education Institutions', 9: 'Food or water supply', 10: 'Journalists and media', 11: 'Maritime', 12: 'NGO', 13: 'Other',
    14: 'Private citizens and property', 15: 'Religious figures/institutions', 16: 'Telecommunication', 17: 'Terrorists/Non-state militias',
    18: 'Tourists', 19: 'Transportation (non-aviation)', 20: 'Unknown', 21: 'Utilities', 22: 'Violent political parties'}
    df['targtype1_names'] = df['targtype1'].replace(dict_targtype1)
    df_suicide = df['suicide']
    df_targtype1 = df['targtype1_names']
    df_crosstabs_targ = pd.crosstab(df_targtype1,df_suicide, normalize='columns')
    # df_crosstabs_targ.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()
    '''military and police and primary target types for suicide bombings'''
    '''create binary variable where military/police == 1 and all else == 0'''
    df['mil_pol_targ'] = np.where(((df['targtype1'] == 3) | (df['targtype1'] == 4)), 1, 0)


    dict_targsub = {1: 'Business:Gas/Oil/Electric', 2: 'Business: Restaurant/Bar/Café', 3: 'Business: Bank/Commerce', 4: 'Business: Multinational Corporation', 5: 'Business: Industrial/Textiles/Factory', 6: 'Business: Medical/Pharmaceutical', 7: 'Business: Retail/Grocery/Bakery', 8: 'Business: Hotel/Resort', 9: 'Business: Farm/Ranch',  10: 'Business: Mining', 11: 'Business: Entertainment/Cultural/Stadium/Casino', 12: 'Business: Construction', 13: 'Business: Private Security Company/Firm', 112: 'Business: Legal Services', 14: 'Government: Judges/Attorneys/Courts', 15: 'Government: Politician or Political Party Movement/Meeting/Rally', 16: 'Government: Royalty', 17: 'Government: Head of State', 18: 'Government: Government Personnel (excluding police, military)', 19: 'Government: Election-related', 20: 'Government: Intelligence', 21: 'Government: Government Buildings/Facility/Office',  22: 'Police: Police Buildings', 23: 'Police: Police Patrol', 24: 'Police: Police Checkpoint', 25: 'Police: Police Security Forces/Officers', 26: 'Police: Prison/Jail', 27: 'Military: Military Barracks/Base/Headquarters/Checkpost', 28: 'Military: Military Recruiting Station/Academy', 29: 'Military: Military Unit/Patrol/Convoy', 30: 'Military: Military Weaponry', 31: 'Military: Military Aircraft', 32: 'Military: Military Maritime', 33: 'Military: Non-combatant Personnel', 34: 'Military: Military Personnel', 35: 'Military: Military Transportation/Vehicle', 36: 'Military: Military Checkpoint', 37: 'Military: NATO Related', 39: 'Military: Paramilitary', 40: 'Abortion Related: Clinics', 41: 'Abortion Related: Personnel', 42: 'Airports & Aircraft: Aircraft',  43: 'Airports & Aircraft: Airline Officer/Personnel', 44: 'Airports & Aircraft: Airport', 45: 'Government (Diplomatic): Diplomatic Personnel', 46: 'Government (Diplomatic): Embassy/Consulate', 47: 'Government (Diplomatic): International Organization', 48: 'Educational Institution: Teacher/Professor/Instructor', 49: 'Educational Institution: School/University/Educational Building', 50: 'Educational Institution: Other Personnel', 51: 'Food and Water Supply: Food Supply', 52: 'Food and Water Supply: Water Supply', 53: 'Journalists & Media: Newspaper Journalist/Staff/Facility', 54: 'Journalists & Media: Radio Journalist/Staff/Facility', 55: 'Journalists & Media: Television Journalist/Staff/Facility', 56: 'Journalists & Media: Other', 57: 'Maritime: Civilian Maritime', 58: 'Maritime: Commercial Maritime', 59: 'Maritime: Oil Tanker', 60: 'Maritime: Port', 61: 'NGO: Domestic NGO', 62: 'NGO: International NGO', 63: 'Other: Ambulance', 64: 'Other: Fire Fighter/Truck', 66: 'Other: Demilitarized Zone', 65: 'Private Citizens & Property: Refugee', 67: 'Private Citizens & Property: Unnamed Civilian/Unspecified', 68: 'Private Citizens & Property: Named Civilian', 69: 'Private Citizens & Property: Religion Identified', 70: 'Private Citizens & Property: Student', 71: 'Private Citizens & Property: Race/Ethnicity Identified', 72: 'Private Citizens & Property: Farmer', 73: 'Private Citizens & Property: Vehicles/Transportation', 74: 'Private Citizens & Property: Marketplace/Plaza/Square', 75: 'Private Citizens & Property: Village/City/Town/Suburb', 76: 'Private Citizens & Property: House/Apartment/Residence', 77: 'Private Citizens & Property: Laborer (General)/Occupation Identified', 78: 'Private Citizens & Property: Procession/Gathering', 79: 'Private Citizens & Property: Public Areas', 80: 'Private Citizens & Property: Memorial/Cemetery/Monument', 81: 'Private Citizens & Property: Museum/Cultural Center/Cultural House', 82: 'Private Citizens & Property: Labor Union Related', 83: 'Private Citizens & Property: Protester', 84: 'Private Citizens & Property: Political Party Member/Rally',  113: 'Private Citizens & Property: Alleged Informant', 85: 'Religious Figures/Institutions: Religious Figure', 86: 'Religious Figures/Institutions: Place of Worship', 87: 'Religious Figures/Institutions: Affiliated Institution', 88: 'Telecommunication: Radio', 89: 'Telecommunication: Television', 90: 'Telecommunication: Telephone/Telegraph', 91: 'Telecommunication: Internet Infrastructure', 92: 'Telecommunication: Multiple Telecommunication Targets', 93: 'Terrorist/Non-State Militia: Terrorist Organization', 94: 'Terrorist/Non-State Militia: Non-State Militia', 95: 'Tourists: Tourism Travel Agency', 96: 'Tourists: Tour Bus/Van/Vehicle', 97: 'Tourists: Tourist', 98: 'Tourists: Other Facility', 99: 'Transportation: Bus', 100: 'Transportation: Train/Train Tracks/ Trolley', 101: 'Transportation: Bus Station/Stop', 102: 'Transportation: Subway', 103: 'Transportation: Bridge/Car Tunnel', 104: 'Transportation: Highway/Road/Toll/Traffic Signal', 105: 'Transportation: Taxi/Rickshaw', 106: 'Utilities: Gas', 107: 'Utilities: Electricity', 108: 'Utilities: Oil', 109: 'Violent Political Parties: Party Official/Candidate/Other Personnel', 110: 'Violent Political Parties: Party Office/Facility', 111: 'Violent Political Parties: Rally'}
    df['targsub_names'] = df['targsubtype1'].replace(dict_targsub)
    df_suicide = df['suicide']
    df_targsub = df['targsub_names']
    df_crosstabs_targ = pd.crosstab(df_targsub,df_suicide, normalize='columns')
    df_crosstabs_targ.columns=['not_suicide', 'suicide']
    # df_crosstabs_targ = df_crosstabs_targ(names=colnames)
    df_crosstabs_targ['diff'] = df_crosstabs_targ.apply(lambda row: row.not_suicide - row.suicide, axis=1).abs()
    df_crosstabs_targ.sort_values(by =['diff'], inplace=True, ascending=False)
    # df_crosstabs_targ.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()
    '''military and police and primary target types for suicide bombings'''
    '''create binary variable where military/police == 1 and all else == 0'''
    # Military: Military Checkpoint                       0.053962
# Military: Military Barracks/Base/Headquarters/C...  0.047095
# Military: Military Personnel                        0.041502
# Police: Police Checkpoint                           0.035572
# Police: Police Buildings                            0.033521
# Religious Figures/Institutions: Place of Worship    0.026803
# Utilities: Electricity                              0.024046
# Government: Politician or Political Party Movem...  0.021316
# Terrorist/Non-State Militia: Non-State Militia      0.021170
# Military: Military Unit/Patrol/Convoy               0.020650
    df['mil_check'] = np.where((df['targsubtype1'] == 36), 1, 0)
    df['mil_barr'] = np.where((df['targsubtype1'] == 27), 1, 0)
    df['pol_check'] = np.where((df['targsubtype1'] == 24), 1, 0)
    df['pol_build'] = np.where((df['targsubtype1'] == 22), 1, 0)
    df['rel_place'] = np.where((df['targsubtype1'] == 86), 1, 0)
    df['util_elec'] = np.where((df['targsubtype1'] == 107), 1, 0)
    df['gov_polit'] = np.where((df['targsubtype1'] == 15), 1, 0)
    df['terr_nonstate'] = np.where((df['targsubtype1'] == 94), 1, 0)
    df['mil_check'] = np.where((df['targsubtype1'] == 36), 1, 0)

    dict_region = {1:'North America', 2:'Central American and Caribbean', 3:'South America', 4:'East Asian', 5:'Southeast Asia', 6:'South Asia', 7:'Central Asia', 8:'Western Europe', 9:'Eastern Europe', 10:'Middle East and North Africa', 11:'Sub-Saharan Africa', 12:'Australasia & Oceania'}
    df['region_names'] = df['region'].replace(dict_region)
    df_region_names = df['region_names']
    df_crosstabs_reg = pd.crosstab(df_region_names, df_suicide, normalize='columns')
    # df_crosstabs_reg.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()
    '''Middle East + NA are primary region of suicide bombings'''
    '''create binary variable where ME and NA == 1 and all else == 0'''
    df['ME_NA'] = np.where((df['region'] == 10), 1, 0)

    dict_attacktype1 = {1:'Assassination', 2:'Armed Assault', 3:'Bombing/Explosion', 4:'Hijacking', 5:'Hostage Taking (Barricade Incident)', 6:'Histage Taking (Kidnapping)', 7:'Facility/Infrastructure Attack', 8:'Unarmed Assault', 9:'Unknown'}
    df['attacktype1_names'] = df['attacktype1'].replace(dict_attacktype1)
    df_attacktype1_names = df['attacktype1_names']
    df_crosstabs_attack = pd.crosstab(df_attacktype1_names, df_suicide, normalize='columns')
    # df_crosstabs_attack.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()
    '''Bombing/Explosion is primary means of attack for suicide bombings'''
    '''create binary variable where B/E == 1 and all else == 0'''
    df['bomb_explo'] = np.where((df['attacktype1'] == 3), 1, 0)

    dict_weapsub1 = {1: 'Chem: Poisoning', 30: 'Chem: Explosive', 2: 'Firearms: Automatic/Semi-automatic rifle', 3:'Firearms: Handgun', 4: 'Firearms: Rifle/shotgun (non-autmoatic)', 5: 'Firearms: Unknown gun type', 6: 'Firearms: Other gun type', 7: 'Explo: Grenate (not RPGs)', 8: 'Explo: Landmine', 9: 'Explo: Mail bomb', 10: 'Explo: Pressure Trigger', 11: 'Explo: Projectile', 12: 'Explo: Remote device', 13: 'Explo: Suicide', 14: 'Explo: Time fuse', 15: 'Explo: Vehicle', 16: 'Explo: Unknown explosive type', 17: 'Explo: Other explosive type', 28: 'Explo: Dynamite/TNT', 29: 'Explo: Sticky bomb', 31: 'Explo: Pipe bomb', 18: 'Incendiary: Arson/fire', 19: 'Incendiary: Molotov cocktail/petrol bomb', 20: 'Incendiary: Gasoline or alcohol', 21: 'Melee: blunt object', 22: 'Melee: hands, feet, fists', 23: 'Melee: knife or other sharp object', 24: 'Melee: Rope or othe strangling device', 26: 'Melee: suffocation', 27: 'Melee: unknown weapon type'}
    df['weapsub1_names'] = df['weapsubtype1'].replace(dict_weapsub1)
    weapsub1_names = df['weapsub1_names']
    weapsubtype1 = df['weapsubtype1']
    df_crosstabs_weapsub = pd.crosstab(weapsub1_names, df_suicide, normalize='columns')
    df_crosstabs_weapsub.columns=['not_suicide', 'suicide']
    # df_crosstabs_targ = df_crosstabs_targ(names=colnames)
    df_crosstabs_weapsub['diff'] = df_crosstabs_weapsub.apply(lambda row: row.not_suicide - row.suicide, axis=1).abs()
    df_crosstabs_weapsub.sort_values(by =['diff'], inplace=True, ascending=False)
    # df_crosstabs_weapsub.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()
    '''Different types of attacks'''
    '''create binary variable where attack subtype == 1 and all else == 0'''
    # Explo: Suicide                                0.000000    0.489959  0.489959
    # Explo: Vehicle                                0.042767    0.498415  0.455647
    # Explo: Unknown explosive type                 0.291491    0.000453  0.291038
    # Firearms: Unknown gun type                    0.214725    0.000755  0.213970
    # Firearms: Automatic/Semi-automatic rifle      0.112774    0.001661  0.111113
    # Explo: Projectile                             0.063811    0.000302  0.063509
    # Explo: Other explosive type                   0.047323    0.000302  0.047021
    # Firearms: Handgun                             0.043409    0.000906  0.042503
    df['explo_vehicle'] = np.where((df['weapsubtype1'] == 15), 1, 0)
    df['explo_unknown'] = np.where((df['weapsubtype1'] == 16), 1, 0)
    df['firearm_unknown'] = np.where((df['weapsubtype1'] == 5), 1, 0)
    df['firearm_rifle'] = np.where((df['weapsubtype1'] == 2), 1, 0)
    df['explo_project'] = np.where((df['weapsubtype1'] == 11), 1, 0)
    df['explo_other'] = np.where((df['weapsubtype1'] == 17), 1, 0)
    df['firearm_handgun'] = np.where((df['weapsubtype1'] == 3), 1, 0)





    dict_weaptype1 = {1:'Biological', 2:'Chemical', 3:'Radiological', 4:'Nuclear', 5:'Firearms', 6:'Explosives', 7:'Fake Weapons', 8:'Incendiary', 9:'Melee', 10:'Vehicle', 11:'Sabotage Equipment', 12:'Other', 13:'Unknown'}
    df['weaptype1_names'] = df['weaptype1'].replace(dict_weaptype1)
    weaptype1_names = df['weaptype1_names']
    df_crosstabs_weapon = pd.crosstab(weaptype1_names, df_suicide, normalize='columns')
    # df_crosstabs_weapon.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()
    '''Explosives are the means of attack for suicide bombings'''
    '''create binary variable where Explosives == 1 and all else == 0'''
    df['explosives'] = np.where((df['weaptype1'] == 6), 1, 0)
    df['firearms'] = np.where((df['weaptype1'] == 5), 1, 0)

    '''code -9 values as missing so that there are only 1 and 0 in this variable'''
    df['claimed'].replace(to_replace=[-9],value=np.NaN, inplace=True)

    dict_claimmode = {1: 'letter', 2: 'call (post-incident)', 3: 'Call (pre-incident)', 4: 'email', 5: 'note left at scene', 6: 'video', 7: 'posted on internet', 8: 'personal claim', 9: 'other', 10: 'unknown'}
    df['claimmode_names'] = df['claimmode'].replace(dict_claimmode)
    claimmode_names = df['claimmode_names']
    df_crosstabs_claimmode = pd.crosstab(claimmode_names, df_suicide, normalize='columns')
    df_crosstabs_claimmode.columns=['not_suicide', 'suicide']
    # df_crosstabs_targ = df_crosstabs_targ(names=colnames)
    df_crosstabs_claimmode['diff'] = df_crosstabs_claimmode.apply(lambda row: row.not_suicide - row.suicide, axis=1).abs()
    df_crosstabs_claimmode.sort_values(by =['diff'], inplace=True, ascending=False)
    # crit1 = df['crit1']
    # df_crosstabs_crit1 = pd.crosstab(crit1, df_suicide, normalize='columns')
    # df_crosstabs_crit1.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()
    df['claim_internet'] = np.where((df['claimmode'] == 7), 1, 0)
    df['claim_note'] = np.where((df['claimmode'] == 5), 1, 0)
    df['claim_personal'] = np.where((df['claimmode'] == 8), 1, 0)

    '''competing claims of responsibility'''
    # dict_compclaim = {1: 'Yes', 0: 'No'}
    # df['compclaim_names'] = df['compclaim'].replace(dict_compclaim)
    # compclaim_names = df['compclaim_names']
    # df_crosstabs_compclaim = pd.crosstab(compclaim_names, df_suicide, normalize='columns')
    # df_crosstabs_compclaim.columns=['not_suicide', 'suicide']
    # # df_crosstabs_targ = df_crosstabs_targ(names=colnames)
    # df_crosstabs_compclaim['diff'] = df_crosstabs_compclaim.apply(lambda row: row.not_suicide - row.suicide, axis=1).abs()
    # df_crosstabs_compclaim.sort_values(by =['diff'], inplace=True, ascending=False)

    '''property damage'''
    # dict_property = {1: 'Yes', 0: 'No'}
    # df['property'].replace(to_replace=[-9],value=np.NaN, inplace=True)
    # df['property_names'] = df['property'].replace(dict_property)
    # property_names = df['property_names']
    # df_crosstabs_property = pd.crosstab(property_names, df_suicide, normalize='columns')
    # df_crosstabs_property.columns=['not_suicide', 'suicide']
    # df_crosstabs_property['diff'] = df_crosstabs_property.apply(lambda row: row.not_suicide - row.suicide, axis=1).abs()
    # df_crosstabs_property.sort_values(by =['diff'], inplace=True, ascending=False)

    '''kidnapping or hostage taking'''
    dict_kidhost = {1: 'Yes', 0: 'No'}
    df['ishostkid'].replace(to_replace=[-9],value=np.NaN, inplace=True)
    df['ishostkid_names'] = df['ishostkid'].replace(dict_kidhost)
    ishostkid_names = df['ishostkid_names']
    df_crosstabs_hostkid = pd.crosstab(ishostkid_names, df_suicide, normalize='columns')
    df_crosstabs_hostkid.columns=['not_suicide', 'suicide']
    df_crosstabs_hostkid['diff'] = df_crosstabs_hostkid.apply(lambda row: row.not_suicide - row.suicide, axis=1).abs()
    df_crosstabs_hostkid.sort_values(by =['diff'], inplace=True, ascending=False)

    # crit1 = df['crit2']
    # df_crosstabs_crit2 = pd.crosstab(crit1, df_suicide, normalize='columns')
    # df_crosstabs_crit2.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()

    # '''suicide bombings more likley to fall outside of humanitarian law'''
    # crit3 = df['crit3']
    # df_crosstabs_crit3 = pd.crosstab(crit3, df_suicide, normalize='columns')
    # df_crosstabs_crit3.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()

    # multiple = df['multiple']
    # df_crosstabs_multiple = pd.crosstab(multiple, df_suicide, normalize='columns')
    # df_crosstabs_multiple.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # plt.show()

    '''create object version of suicide variable'''
    dict_suicide = {1: 'suicide bombing', 0: 'not suicide bombing'}
    df['suicide_text'] = df['suicide'].replace(dict_suicide)

    '''look at suicide bombing occurance over time'''
    # df_year = df['iyear']
    # df_crosstabs_year = pd.crosstab(df_year,df_suicide, normalize='columns')
    # df_crosstabs_year.plot(kind='bar')
    # plt.xticks(rotation=60, horizontalalignment='right')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('suicide_time')
    '''create binary variable where year >= 2003 == 1 and all else == 0'''
    df['year_2003'] = np.where((df['iyear'] >=2003), 1, 0)

    '''month of incident'''
    # dict_month = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'June', 7: 'July', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    # df['imonth'].replace(to_replace=[0],value=np.NaN, inplace=True)
    # df['imonth_name'] = df['imonth'].replace(dict_month)
    # imonth_name = df['imonth_name']
    # df_crosstabs_month = pd.crosstab(imonth_name, df_suicide, normalize='columns')
    # df_crosstabs_month.columns=['not_suicide', 'suicide']
    # df_crosstabs_month['diff'] = df_crosstabs_month.apply(lambda row: row.not_suicide - row.suicide, axis=1).abs()
    # df_crosstabs_month.sort_values(by =['diff'], inplace=True, ascending=False)

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
    df['country_names'] = df['country'].replace(dict_country)
    df_country_names = df['country_names']
    df_crosstab_country_names = pd.crosstab(df_country_names,df_suicide, normalize='columns')
    df_crosstab_country_names.columns=['not_suicide', 'suicide']
    df_crosstab_country_names['diff'] = df_crosstab_country_names.apply(lambda row: row.not_suicide - row.suicide, axis=1).abs()
    df_crosstab_country_names.sort_values(by =['diff'], inplace=True, ascending=False)
    # Iraq                               0.125867  0.392281  0.266414
    # Afghanistan                        0.065710  0.185135  0.119425
    # India                              0.067977  0.009046  0.058932
    # Colombia                           0.047424  0.000603  0.046821
    # Syria                              0.010608  0.051862  0.041254
    df['Iraq'] = np.where((df['country'] == 95), 1, 0)
    df['Afghanistan'] = np.where((df['country'] == 4), 1, 0)
    df['India'] = np.where((df['country'] == 92), 1, 0)
    df['Columbia'] = np.where((df['country'] == 45), 1, 0)
    df['Syria'] = np.where((df['country'] == 200), 1, 0)

    #
    df_suicide_DT=df[['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'motive', 'suicide']]
    # # #
    # #
    df_suicide_DT.to_csv('data/df_suicide_DT.csv', index=False)

    # df_suicide_DT_withLDA_downsample=df[['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria','topic1', 'topic2', 'suicide']]
    # df_suicide_DT_withLDA_upsample=df[['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria','topic1', 'topic2', 'suicide']]
    # # # 'mil_pol_targ',
    # # # , 'crit3'
    # # # ,'bomb_explo'
    # # # , 'firearms'
    # # # df_suicide = df_suicide[suicide_cols]

# [('Utility_Location', 0.0015750212170650157),
#  ('NonStateMilitia', 0.004039472358080833),
#  ('Military_Checkpoint', 0.004212109309459604),
#  ('Government_Politician', 0.004264872668015723),
#  ('Syria', 0.004582629756375149),
#  ('Columbia', 0.005020840261969052),
#  ('Handgun', 0.0076042620318038205),
#  ('Personal_Claim', 0.009197075736071396),
#  ('Political_Building', 0.010011141292099044),
#  ('Politicial_Checkpoint', 0.010629598368992601),
#  ('Claim_via_Note', 0.012687814612350573),
#  ('Military_Barracks', 0.013231894633328958),
#  ('Religious_PlaceOfWorship', 0.016056917379832746),
#  ('Rifle', 0.0222467931519469),
#  ('Claim_via_Internet', 0.02362597720196808),
#  ('Happened_After_2002', 0.02480187902613347),
#  ('MiddleEast_NorthAfrican', 0.03295403502280479),
#  ('India', 0.034934990593468315),
#  ('Other_Explosive', 0.03807041759034671),
#  ('Afghanistan', 0.03808827671248003),
#  ('Iraq', 0.04393304549304783),
#  ('Projectile', 0.047624081292954486),
#  ('Hostage_Kidnapping', 0.04766174778297396),
#  ('Claimed_responsibility', 0.09130356053407562),
#  ('Unknown_Explosive', 0.09494648180988571),
#  ('Unknown_Firearm', 0.1476135287453559),
#  ('Explosive_Vehicle', 0.20908153541711363)]
#     # df_suicide_DT_withLDA_upsample.to_csv('data/df_suicide_DT_withLDA_upsample.csv', index=False)
    # df_suicide_DT_withLDA_downsample.to_csv('data/df_suicide_DT_withLDA_downsample.csv', index=False)

    # #
    # ####
    # # '''Crosstabs'''
    # # feature_name_list = ['ME_NA', 'claimed', 'year_2003', 'mil_barr', 'pol_check', 'pol_build', 'rel_place', 'util_elec', 'gov_polit', 'terr_nonstate', 'mil_check', 'explo_vehicle', 'explo_unknown', 'firearm_unknown', 'firearm_rifle', 'explo_project', 'explo_other', 'firearm_handgun', 'claim_internet', 'claim_note', 'claim_personal', 'ishostkid', 'Iraq', 'Afghanistan', 'India', 'Columbia', 'Syria', 'motive', 'suicide']
    # feature_name_list = ['util_elec','terr_nonstate', 'mil_check', 'Syria','gov_polit',  'Columbia', 'firearm_handgun', 'claim_personal',  'pol_build', 'pol_check', 'claim_note', 'mil_barr', 'rel_place','firearm_rifle',  'claim_internet', 'year_2003', 'ME_NA','India',   'Afghanistan', 'explo_other','Iraq', 'explo_project', 'ishostkid', 'claimed', 'explo_unknown', 'firearm_unknown', 'explo_vehicle']
    #
    # not_suicide_higher = ['util_elec', 'gov_polit', 'terr_nonstate', 'Columbia', 'firearm_handgun', 'claim_personal', 'claim_note', 'claim_internet', 'firearm_rifle', 'India', 'explo_other', 'explo_project', 'ishostkid', 'explo_unknown', 'firearm_unknown']
    #
    # suicide_higher = ['mil_check', 'Syria', 'pol_build', 'pol_check', 'mil_barr', 'rel_place', 'year_2003', 'ME_NA', 'Afghanistan', 'Iraq', 'claimed', 'explo_vehicle']
    #
    # df_suicide = df['suicide']
    #
    #
    # # print(crosstabs(df, new_df_name_list, feature_name_list, df_suicide))
    # # def crosstabs(df, new_df_name_list, feature_name_list, df_suicide):
    #     # for df_name,feature in zip(new_df_name_list, feature_name_list):
    # # df_Utility_Location = df['util_elec']
    # # df_crosstabs_targ = pd.crosstab(df_targtype1,df_suicide, normalize='columns')
    # for feature in feature_name_list:
    #     print('{}: {}'.format(feature, pd.crosstab(df[feature],df['suicide_text'], normalize='columns')))
