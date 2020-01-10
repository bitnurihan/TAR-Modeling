import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages


def GRPs(x, grp_const, grp_slope):  # Calculating GRPs (if the screen is 'digital', it's not GRPs but Impression)
    g = np.around(grp_const + grp_slope * np.log(x), decimals=100)
    GRP = np.exp(g)
    return GRP


def reach1(GRP, r1_const, r1_slope):  # Calculating reach 1+%
    r = np.around(r1_const + r1_slope * np.log(GRP), decimals=100)
    reach = np.exp(r)
    return (reach/(1+reach))*100


def comparing_modeling_with_r_and_python():
    n = 500
    cost = np.linspace(1, 10000000000, n)
    
    # R Modeling
    cost_grp = np.empty(n)
    grp_r1 = np.empty(n)

    for i in range(n):
        cost_grp[i] = GRPs(cost[i], R_guess_grp[0], R_guess_grp[1])  # cost to GRPs
        grp_r1[i] = reach1(cost_grp[i], R_guess_r1[0], R_guess_r1[1])  # GRPs to reach1+%

    # Python Modeling
    a = df_2s_ad_data_final['Cost'].values
    if find_dgt is True:
        b = df_2s_ad_data_final['impression'].values
    else:
        b = df_2s_ad_data_final['GRP'].values
    c = df_2s_ad_data_final['r1'].values

    pocg, pcov = curve_fit(GRPs, a, b, R_guess_grp)
    pocr, pcov = curve_fit(reach1, b, c, R_guess_r1)

    p_cost_grp = np.empty(n)
    p_grp_r1 = np.empty(n)

    for i in range(n):
        p_cost_grp[i] = GRPs(cost[i], *pocg)
        p_grp_r1[i] = reach1(p_cost_grp[i], *pocr)
        
    # Visualization
    plt.plot(cost, grp_r1, 'r.', label='R_guess')  # Line plot by R
    plt.plot(cost, p_grp_r1, 'r.', color='b', label='Py_guess') # Line plot by Python
    plt.scatter(df_2s_ad_data_final['Cost'], df_2s_ad_data_final['r1'], color='y', label='Real data') # Scatter plot for real data
    plt.xlabel('Cost')
    plt.ylabel('reach1+')
    plt.xlim(1, 10000000000)
    plt.ylim(1, 100)
    plt.legend(loc='lower right')
    title_age = [df_2s_ad_data_final['age'].iloc[1]]
    title_screen = [df_2s_ad_data_final['screen'].iloc[1]]
    plt.title('%s / %s' % (title_age[0], title_screen[0]))
    fig = plt.gcf()
    plt.grid()
    plt.show()
    pdf_pages.savefig(fig)

    
df_2s_ad_data = pd.read_csv(r'C:\Users\hanbi01\Desktop\한빛누리\제일기획\TAR\modeling_data\modeling_data_2S.csv', encoding='ms949')
df_2s_max_reach = pd.read_excel(r'C:\Users\hanbi01\Desktop\한빛누리\제일기획\TAR\modeling_data\modeling_data_2S.xlsx')
df_2s_modeling = pd.read_excel(r'C:\Users\hanbi01\Desktop\한빛누리\제일기획\TAR\modeling_data\Cheil3A_DS2_190814.xlsx', sheet_name='Coef_3A')
df_2s_modeling['AGE_CD'] = df_2s_modeling['AGE_CD'].astype(str)
df_2s_modeling['Target'] = df_2s_modeling['GENDER_CD'].astype(str) + df_2s_modeling['AGE_CD'].str.zfill(4)
df_2s_modeling = df_2s_modeling.set_index('Target')
df_2s_ad_data['screen'] = df_2s_ad_data['screen'].apply(lambda x: 'DGT' if x == 'PC∪MO' else x)
df_2s_max_reach['screen'] = df_2s_max_reach['screen'].apply(lambda x: 'DGT' if x == 'PC∪MO' else x)
df_2s_ad_data['CPRP'] = df_2s_ad_data['Cost'] / df_2s_ad_data['GRP']
df_2s_max_reach['CPRP'] = df_2s_max_reach['Cost'] / df_2s_max_reach['GRP']
df_2s_ad_data['cpm'] = (df_2s_ad_data['Cost'] / df_2s_ad_data['impression'])*1000
df_2s_max_reach['cpm'] = (df_2s_max_reach['Cost'] / df_2s_max_reach['impression'])*1000

ages = (df_2s_ad_data['age'].drop_duplicates()).values
screen = (df_2s_ad_data['screen'].drop_duplicates()).values
with PdfPages('python_modeling.pdf') as pdf_pages:
    for age in ages:
        df_2s_ad_data_age = df_2s_ad_data[df_2s_ad_data['age'] == age].copy()
        df_2s_max_reach_age = df_2s_max_reach[df_2s_max_reach['age'] == age].copy()
        df_2s_max_reach_age = pd.concat([pd.DataFrame(df_2s_max_reach_age) for i in range(10)], ignore_index=True)
        for i in screen:
            df_2s_ad_data_screen = df_2s_ad_data_age[df_2s_ad_data_age['screen'] == i].copy()
            R_guess_grp = list(df_2s_modeling.loc['%s' % age, ['GRP_INTERCEPT_%s' % i, 'GRP_SLOP_%s' % i]])
            R_guess_r1 = list(df_2s_modeling.loc['%s' % age, ['R1_INTERCEPT_%s' % i,'R1_SLOP_%s' % i]])
            if i.endswith('TV'): # Outlier 
                df_2s_ad_data_screen = df_2s_ad_data_screen[df_2s_ad_data_screen['CPRP'] <= 15000000]  
            elif i.endswith('DGT'):
                df_2s_ad_data_screen = df_2s_ad_data_screen[df_2s_ad_data_screen['CPRP'] <= df_2s_ad_data_screen['CPRP'].quantile(.75)]
                # df_2s_ad_data_screen = pd.concat([df_2s_ad_data_screen, df_2s_max_reach_age], sort=True)
            df_2s_ad_data_final = df_2s_ad_data_screen.sort_values(['Cost'], ascending=True)
            df_2s_ad_data_final = df_2s_ad_data_final[['age', 'screen', 'r1', 'GRP', 'impression', 'Cost']]
            non_linear_regression()
  
