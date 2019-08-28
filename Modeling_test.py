import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages


def GRPs(x, grp_const, grp_slope):
    g = np.around(grp_const + grp_slope * np.log(x), decimals=100)
    GRP = np.exp(g)
    return GRP


def reach1(GRP, r1_const, r1_slope):
    r = np.around(r1_const + r1_slope * np.log(GRP), decimals=100)
    reach = np.exp(r)
    return (reach/(1+reach))*100


def grp_to_reach(x, grp_const, grp_slope):
    g = np.around(grp_const + grp_slope * np.log(x), decimals=100)
    reach = np.exp(g)
    return (reach/(1+reach))*100


def non_linear_regression():
    n = 500
    cost_grp = np.empty(n)
    grp_r1 = np.empty(n)
    data = []
    # cost = np.linspace(1, max(df_2s_ad_data_age['Cost']), 500)
    cost = np.linspace(1,5000000000,500)
    # [cost to grp, grp to reach 계산]
    screen = (df_2s_ad_data['screen'].drop_duplicates()).values
    for s in screen:
        R_guess_grp = list(df_2s_modeling.loc['%s' % age, ['GRP_INTERCEPT_%s' % s, 'GRP_SLOP_%s' % s]])
        R_guess_r1 = list(df_2s_modeling.loc['%s' % age, ['R1_INTERCEPT_%s' % s, 'R1_SLOP_%s' % s]])
        for x in range(n):
            cost_grp[x] = GRPs(cost[x], R_guess_grp[0], R_guess_grp[1])
            grp_r1[x] = reach1(cost_grp[x], R_guess_r1[0], R_guess_r1[1])
            data.append([s, cost[x], grp_r1[x]])

    # [cost to reach 계산]
    # screen = ['TV', 'DIGITAL']
    # for s in screen:
    #     R_guess_r1 = list(df_2s_modeling.loc['%s' % age, ['R1_INTERCEPT_%s' % s, 'R1_SLOP_%s' % s]])
    #     for x in range(n):
    #         grp_r1[x] = grp_to_reach(cost[x], R_guess_r1[0], R_guess_r1[1])
    #         data.append([s, cost[x], grp_r1[x]])
    r_r1 = pd.DataFrame(data, columns=['screen', 'cost', 'reach1+'])
    regression = sns.lineplot(x='cost', y='reach1+', hue='screen', hue_order=['CATV','JSPTV','DGT'], data=r_r1)
    # regression = sns.lineplot(x='cost', y='reach1+', hue='screen', data=r_r1)
    sns.scatterplot(x=df_2s_ad_data_dgt['Cost'], y=df_2s_ad_data_dgt['r1'], data=df_2s_ad_data_dgt, color='g', alpha=.4)  # 디지털 raw data
    sns.scatterplot(x=df_2s_max_reach_age['Cost'], y=df_2s_max_reach_age['r1'], data=df_2s_max_reach_age, color='g',
                    alpha=.4)
    # sns.scatterplot(x=df_2s_ad_data_age['Cost'], y=df_2s_ad_data_age['r1'], hue=df_2s_ad_data_age['screen'],
    #                 hue_order=['CATV','JSPTV','DGT'], data=df_2s_ad_data_age, style=df_2s_ad_data_age['screen'], alpha=.7)
    regression.set(xlabel='Cost', ylabel='reach1+')
    title_age = [df_2s_ad_data_age['age'].iloc[1]]
    regression.set_title('%s' % title_age[0])
    plt.legend()
    regression.set_xlim(0, 5000000000)
    regression.set_ylim(0,100)
    regression.grid()
    fig = plt.gcf()
    plt.show()
    pdf_pages.savefig(fig)
    plt.close()


def non_linear_regression_py():
    n = 500
    data = []
    screen = (df_2s_ad_data['screen'].drop_duplicates()).values
    # cost = np.linspace(1, max(df_2s_ad_data_age['Cost']), 500)
    cost = np.linspace(1,5000000000,500)

    for s in screen:
        R_guess_grp = list(df_2s_modeling.loc['%s' % age, ['GRP_INTERCEPT_%s' % s, 'GRP_SLOP_%s' % s]])
        R_guess_r1 = list(df_2s_modeling.loc['%s' % age, ['R1_INTERCEPT_%s' % s, 'R1_SLOP_%s' % s]])
        df_2s_ad_data_screen = df_2s_ad_data_age[df_2s_ad_data_age['screen'] == s].copy()
        if s.endswith('TV'):
            df_2s_ad_data_screen = df_2s_ad_data_screen[df_2s_ad_data_screen['CPRP'] <= 15000000]
        elif s.endswith('DGT'):
            df_2s_ad_data_screen = df_2s_ad_data_screen[df_2s_ad_data_screen['CPRP'] <= df_2s_ad_data_screen['CPRP'].quantile(.75)]
            df_2s_ad_data_screen = pd.concat([df_2s_ad_data_screen, df_2s_max_reach_age, df_2s_max_reach_semi], sort=True)

        a = df_2s_ad_data_screen['Cost'].values
        if s == 'DGT':
            b = df_2s_ad_data_screen['impression'].values
        else:
            b = df_2s_ad_data_screen['GRP'].values
        c = df_2s_ad_data_screen['r1'].values
        p_cost_grp = np.empty(n)
        p_grp_r1 = np.empty(n)
        pocg, pcov = curve_fit(GRPs, a, b, R_guess_grp)
        pocr, pcov = curve_fit(reach1, b, c, R_guess_r1)
        for x in range(n):
            p_cost_grp[x] = GRPs(cost[x], *pocg)
            p_grp_r1[x] = reach1(p_cost_grp[x], *pocr)
            data.append([s, cost[x], p_grp_r1[x]])

    r_r1 = pd.DataFrame(data, columns=['screen', 'cost', 'reach1+'])
    regression = sns.lineplot(x='cost', y='reach1+', hue='screen', hue_order=['CATV','JSPTV','DGT'], data=r_r1)

    # sns.scatterplot(x=df_2s_ad_data_age['Cost'], y=df_2s_ad_data_age['r1'], hue=df_2s_ad_data_age['screen'],
    #                 hue_order=['CATV','JSPTV','DGT'], data=df_2s_ad_data_age, style=df_2s_ad_data_age['screen'], alpha=.7)
    sns.scatterplot(x=df_2s_ad_data_dgt['Cost'], y=df_2s_ad_data_dgt['r1'], data=df_2s_ad_data_dgt, color='g', alpha=.4)
    sns.scatterplot(x=df_2s_max_reach_semi['Cost'], y=df_2s_max_reach_semi['r1'], data=df_2s_max_reach_semi, color='g', alpha=.4)
    sns.scatterplot(x=df_2s_max_reach_age['Cost'], y=df_2s_max_reach_age['r1'], data=df_2s_max_reach_age, color='g', alpha=.4)
    regression.set(xlabel='Cost', ylabel='reach1+')
    title_age = [df_2s_ad_data_age['age'].iloc[1]]
    regression.set_title('%s' % title_age[0])
    plt.legend()
    regression.set_ylim(0,100)
    regression.set_xlim(0, 5000000000)
    regression.grid()
    fig = plt.gcf()
    plt.show()
    pdf_pages.savefig(fig)
    plt.close()


df_2s_ad_data = pd.read_csv(r'C:\Users\hanbi01\Desktop\한빛누리\제일기획\TAR\modeling_data\modeling_data_2S.csv', encoding='ms949')
df_2s_max_reach = pd.read_excel(r'C:\Users\hanbi01\Desktop\한빛누리\제일기획\TAR\modeling_data\modeling_data_2S.xlsx')
df_2s_modeling = pd.read_excel(r'C:\Users\hanbi01\Desktop\한빛누리\제일기획\TAR\modeling_data\Cheil3A_DS2_190826 - Copy.xlsx')
df_2s_modeling['AGE_CD'] = df_2s_modeling['AGE_CD'].astype(str)
df_2s_modeling['Target'] = df_2s_modeling['GENDER_CD'].astype(str) + df_2s_modeling['AGE_CD'].str.zfill(4)
df_2s_modeling = df_2s_modeling.set_index('Target')
df_2s_ad_data['screen'] = df_2s_ad_data['screen'].apply(lambda x: 'DGT' if x == 'PC∪MO' else x)
df_2s_max_reach['screen'] = df_2s_max_reach['screen'].apply(lambda x: 'DGT' if x == 'PC∪MO' else x)
df_2s_ad_data['CPRP'] = df_2s_ad_data['Cost'] / df_2s_ad_data['GRP']
df_2s_max_reach['CPRP'] = df_2s_max_reach['Cost'] / df_2s_max_reach['GRP']
df_2s_ad_data = df_2s_ad_data[['Month','age', 'screen', 'r1', 'GRP', 'impression', 'Cost', 'CPRP']]
df_2s_max_reach = df_2s_max_reach[['Month','age', 'screen', 'r1', 'GRP', 'impression', 'Cost', 'CPRP']]
df_2s_ad_data['cpm'] = (df_2s_ad_data['Cost'] / df_2s_ad_data['impression'])*1000
df_2s_max_reach['cpm'] = (df_2s_max_reach['Cost'] / df_2s_max_reach['impression'])*1000
find_cost = df_2s_ad_data[df_2s_ad_data['age'] == 'MF0769'][
                df_2s_ad_data[df_2s_ad_data['age'] == 'MF0769']['screen'] == 'DGT']
find_cost = find_cost[find_cost['CPRP'] <= find_cost['CPRP'].quantile(.75)]
# ages = (df_2s_ad_data['age'].drop_duplicates()).values
ages = ['MF0769','F0769','M0769','MF1318','MF1929', 'MF3039', 'MF4049','MF5059','MF6069', 'MF1949', 'MF3059']
# ages = ['MF1359', 'M1359', 'F1359', 'MF1318', 'MF1929','MF3039', 'MF4049','MF5059']

with PdfPages('1_test.pdf') as pdf_pages:
    for age in ages:
        df_2s_ad_data_age = df_2s_ad_data[df_2s_ad_data['age'] == age].copy()
        df_2s_max_reach_age = df_2s_max_reach[df_2s_max_reach['age'] == age].copy()
        df_semi_data = df_2s_max_reach[df_2s_max_reach['age'] == age].copy()
        df_semi_data['Cost'] = df_semi_data['Cost']/1.1
        df_semi_data['r1'] = df_semi_data['r1']*0.8
        df_2s_ad_data_dgt = df_2s_ad_data_age[df_2s_ad_data_age['screen'] == 'DGT']
        df_2s_max_reach_age = pd.concat([pd.DataFrame(df_2s_max_reach_age) for _ in range(8)],
                                        ignore_index=True)
        df_2s_max_reach_semi = pd.concat([pd.DataFrame(df_semi_data) for _ in range(18)],
                                        ignore_index=True)

        # non_linear_regression()
        non_linear_regression_py()
