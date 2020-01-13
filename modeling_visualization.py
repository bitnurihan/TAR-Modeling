import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages


def GRPs(x, grp_const, grp_slope):  # calculating grps by cost
    g = np.around(grp_const + grp_slope * np.log(x), decimals=100)
    GRP = np.exp(g)
    return GRP


def reach1(GRP, r1_const, r1_slope):  # calculating reach1+% by grps(impression)
    r = np.around(r1_const + r1_slope * np.log(GRP), decimals=100)
    reach = np.exp(r)
    return (reach/(1+reach))*100


def visualization_of_final_model():
    n = 500
    cost_grp = np.empty(n)
    grp_r1 = np.empty(n)
    modeling_data = []
    cost = np.linspace(1,5000000000,n)
    # [cost to grp, grp to reach]
    screen = (df_2s_ad_data['screen'].drop_duplicates()).values
    for s in screen:
        find_constant_for_grp = list(df_2s_modeling.loc['%s' % age, ['GRP_INTERCEPT_%s' % s, 'GRP_SLOP_%s' % s]])
        find_constant_for_r1 = list(df_2s_modeling.loc['%s' % age, ['R1_INTERCEPT_%s' % s, 'R1_SLOP_%s' % s]])
        for x in range(n):
            cost_grp[x] = GRPs(cost[x], find_constant_for_grp[0], find_constant_for_grp[1])
            grp_r1[x] = reach1(cost_grp[x], find_constant_for_r1[0], find_constant_for_r1[1])
            modeling_data.append([s, cost[x], grp_r1[x]])

    # Visualization
    df_modeling = pd.DataFrame(modeling_data, columns=['screen', 'cost', 'reach1+'])
    regression = sns.lineplot(x='cost', y='reach1+', hue='screen', hue_order=['CATV','JSPTV','DGT'], data=df_modeling)
    sns.scatterplot(x=df_2s_ad_data_total['Cost'], y=df_2s_ad_data_total['r1'], hue=df_2s_ad_data_total['screen'],
                    hue_order=['CATV','JSPTV','DGT'], data=df_2s_ad_data_total, style=df_2s_ad_data_total['screen'], alpha=.7)
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


df_2s_ad_data = pd.read_csv('modeling_data_2S.csv', encoding='ms949')  # raw data for modeling
df_2s_max_reach = pd.read_excel('modeling_data_2S.xlsx') # raw data for modeling
df_2s_modeling = pd.read_excel('Cheil3A_DS2_190826.xlsx') # constanat & slop of each target and screen

# preprocessing
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
targets = (df_2s_ad_data['age'].drop_duplicates()).values

with PdfPages('modeling_visualization.pdf') as pdf_pages:
    for target in targets: # modeling graph is showed by target
        df_2s_ad_data_age = df_2s_ad_data[df_2s_ad_data['age'] == age].copy()
        df_2s_max_reach_age = df_2s_max_reach[df_2s_max_reach['age'] == age].copy()
        df_2s_ad_data_total = pd.concat([df_2s_ad_data_age, df_2s_max_reach_age])
        visualization_of_final_model()
