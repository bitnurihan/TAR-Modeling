import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def GRPs(x, grp_const, grp_slope):
    g = np.around(grp_const + grp_slope * np.log(x), decimals=100)
    GRP = np.exp(g)
    return GRP


def reach1(GRP, r1_const, r1_slope):
    r = np.around(r1_const + r1_slope * np.log(GRP), decimals=10)
    reach = np.exp(r)
    return (reach/(1+reach))*100


def modeling():
    find_dgt = df_2s_ad_data_final[df_2s_ad_data_final['screen'] == 'DGT']
    a = df_2s_ad_data_final['Cost'].values
    if find_dgt is True:
        b = df_2s_ad_data_final['GRP'].values
    else:
        b = df_2s_ad_data_final['GRP'].values
    c = df_2s_ad_data_final['r1'].values

    pocg, pcov = curve_fit(GRPs, a, b, R_guess_grp)
    pocr, pcov = curve_fit(reach1, b, c, R_guess_r1)

    constant_coefficiant.append(['%s' % age, '%s' % i, pocg[0],pocg[1], pocr[0],pocr[1]])


df_2s_ad_data = pd.read_csv('modeling_data_2S.csv', encoding='ms949') # raw data
df_2s_max_reach = pd.read_excel('modeling_data_2S.xlsx') # raw data
df_2s_modeling = pd.read_excel('Cheil3A_DS2_190813.xlsx', sheet_name='Coef_3A')
# preprocessing
df_2s_modeling['AGE_CD'] = df_2s_modeling['AGE_CD'].astype(str)
df_2s_modeling['Target'] = df_2s_modeling['GENDER_CD'].astype(str) + df_2s_modeling['AGE_CD'].str.zfill(4)
df_2s_modeling = df_2s_modeling.set_index('Target')
df_2s_ad_data['screen'] = df_2s_ad_data['screen'].apply(lambda x: 'DGT' if x == 'PC∪MO' else x)
df_2s_ad_data['CPRP'] = df_2s_ad_data['Cost'] / df_2s_ad_data['GRP']
df_2s_max_reach['CPRP'] = df_2s_max_reach['Cost'] / df_2s_max_reach['GRP']
df_2s_ad_data['cpm'] = (df_2s_ad_data['Cost'] / df_2s_ad_data['impression'])*1000
df_2s_max_reach['cpm'] = (df_2s_max_reach['Cost'] / df_2s_max_reach['impression'])*1000

ages = (df_2s_ad_data['age'].drop_duplicates()).values
screen = (df_2s_ad_data['screen'].drop_duplicates()).values
constant_coefficiant = []
for age in ages:
    df_2s_ad_data_age = df_2s_ad_data[df_2s_ad_data['age'] == age].copy()
    df_2s_max_reach_age = df_2s_max_reach[df_2s_max_reach['age'] == age].copy()
    df_2s_max_reach_age = pd.concat([pd.DataFrame(df_2s_max_reach_age) for i in range(10)], ignore_index=True)
    for s in screen:
        df_2s_ad_data_screen = df_2s_ad_data_age[df_2s_ad_data_age['screen'] == s].copy()
        find_constant_grp = list(df_2s_modeling.loc['%s' % age, ['GRP_INTERCEPT_%s' % s, 'GRP_SLOP_%s' % s]])
        find_constant_r1 = list(df_2s_modeling.loc['%s' % age, ['R1_INTERCEPT_%s' % s,'R1_SLOP_%s' % s]])
        if s.endswith('TV'):
            df_2s_ad_data_screen['CPRP'] = df_2s_ad_data_screen['Cost'] / df_2s_ad_data_screen['GRP']
            df_2s_ad_data_screen = df_2s_ad_data_screen[df_2s_ad_data_screen['CPRP'] <= 15000000]
        elif s.endswith('DGT'):
            df_2s_ad_data_screen = df_2s_ad_data_screen[df_2s_ad_data_screen['CPRP'] <= df_2s_ad_data_screen['CPRP'].quantile(.75)]
            df_2s_ad_data_screen = pd.concat([df_2s_ad_data_screen, df_2s_max_reach_age], sort=True)
        df_2s_ad_data_final = df_2s_ad_data_screen.sort_values(['Cost'], ascending=True)
        df_2s_ad_data_final = df_2s_ad_data_final[['age', 'screen', 'r1', 'GRP', 'impression', 'Cost']]
        modeling()
    df_constant_coefficiant = pd.DataFrame(constant_coefficiant, columns=['age', 'screen', 'GRP_INTERCEPT', 'GRP_SLOP', 'R1_INTERCEPT', 'R1_SLOP'])
    pivot_constant_coefficiant = pd.pivot_table(df_constant_coefficiant, index=['age'], columns=['screen'])

excel_writer = pd.ExcelWriter('constant_slop.xlsx', engine='xlsxwriter')
pivot_constant_coefficiant.to_excel(excel_writer, sheet_name='Python_constant_slop')
excel_writer.save()
