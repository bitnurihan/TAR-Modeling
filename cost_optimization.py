import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl import load_workbook


def GRPs(a, grp_const, grp_slope):
    g = np.around(grp_const + grp_slope * np.log(a), decimals=100)
    GRP = np.exp(g)
    return GRP


def reach1(GRP, r1_const, r1_slope):
    r = np.around(r1_const + r1_slope * np.log(GRP), decimals=100)
    reach = np.exp(r)
    return reach/(1+reach)


def cprp_to_grp(cost, cprp):  # it uses when there is specific cprp
    grp = cost/cprp/100
    return grp


def cpm(cost, cpm):  # it uses when there is specific cpm
    grp = cost/cpm*1000
    return grp


def cal_reach1(s, cost):
    find_constant_grp = list(df_2s_modeling.loc['%s' % age, ['GRP_INTERCEPT_%s' % s, 'GRP_SLOP_%s' % s]])
    find_constant_r1 = list(df_2s_modeling.loc['%s' % age, ['R1_INTERCEPT_%s' % s, 'R1_SLOP_%s' % s]])
    cost_grp = GRPs(cost, find_constant_grp[0], find_constant_grp[1])
    # screen = ['JSPTV', 'CATV', 'DGT']
    # if s == 'DGT':
    #     cost_grp = cpm(cost, 15000)
    # else:
    #     cost_grp = cprp_to_grp(cost, 2000000)
    grp_r1 = reach1(cost_grp, find_constant_r1[0], find_constant_r1[1])
    return grp_r1


def f(x):
    x1 = x[0]*100000000
    x2 = x[1]*100000000
    x3 = x[2]*100000000
    max_reach1 \
        = cal_reach1('JSPTV', x1) + cal_reach1('CATV', x2) + cal_reach1('DGT', x3) \
          - DupRate_jsp_ca[0] * (cal_reach1('JSPTV', x1) * cal_reach1('CATV', x2)) \
          - DupRate_jsp_dgt[0] * (cal_reach1('JSPTV', x1) * cal_reach1('DGT', x3)) \
          - DupRate_ca_dgt[0] * (cal_reach1('CATV', x2) * cal_reach1('DGT', x3)) \
          + DupRate_jsp_ca_dgt[0] * (cal_reach1('JSPTV', x1) * cal_reach1('CATV', x2) * cal_reach1('DGT', x3))
    reach1 = max_reach1*100
    return reach1


excel_writer = pd.ExcelWriter('optimization_result.xlsx', engine='openpyxl') # budget optimizing result file
df_2s_ad_data = pd.read_csv('modeling_data_2S.csv', encoding='ms949')
df_2s_modeling = pd.read_excel('Cheil3A_DS2_190904.xlsx', sheet_name='Coef_3A(6안)') # constant & slop data
df_3s_dup = pd.read_excel('Cheil3A_DS2_190904.xlsx', sheet_name='Simul_3A')

# preprocessing
df_2s_ad_data['screen'] = df_2s_ad_data['screen'].apply(lambda x: 'DGT' if x == 'PC∪MO' else x)
df_2s_modeling['AGE_CD'] = df_2s_modeling['AGE_CD'].astype(str)
df_2s_modeling['Target'] = df_2s_modeling['GENDER_CD'].astype(str) + df_2s_modeling['AGE_CD'].str.zfill(4)
df_2s_modeling = df_2s_modeling.set_index('Target')
df_3s_dup['AGE_CD'] = df_3s_dup['AGE_CD'].astype(str)
df_3s_dup['Target'] = df_3s_dup['GENDER_CD'].astype(str) + df_3s_dup['AGE_CD'].str.zfill(4)
ages = ['MF0769','MF1318','MF1929', 'MF3039', 'MF4049','MF5059','MF6069', 'MF1949', 'MF3059']


total_cost = np.linspace(1, 10, 10)  # KRW 1 billion ~ 10 billion, 10개 구간
a = [0.25, 10]  # TV 최소 2500만~ 최대 50억
b = [0.1, 10]  # DGT 최소 1000만 ~ 최대 50억
bnds = (a, a, b)
con = {'type': 'eq', 'fun': lambda x: (x[0] + x[1] + x[2] - total_cost[i])}
with PdfPages('opti_none.pdf') as pdf_pages:
    for age in ages:
        # screen duplicate
        df_3s_dup_age = df_3s_dup[df_3s_dup['Target'] == age].copy()
        DupRate_jsp_ca = list(df_3s_dup_age['JSPTV&CATV'])
        DupRate_ca_dgt = list(df_3s_dup_age['CATV&DGT'])
        DupRate_jsp_dgt = list(df_3s_dup_age['JSPTV&DGT'])
        DupRate_jsp_ca_dgt = list(df_3s_dup_age['JSPTV&CATV&DGT'])

        optimizing_x = []
        result = []
        for i in range(10):
            x0 = np.array([total_cost[i]-0.35, 0.25, 0.1]) # starting cost
            sol = minimize(lambda x: -f(x), x0, method='SLSQP', bounds=bnds, constraints=con)
            optimizing_x.append(list(sol.x))  # optimized result(cost) of each screen
            result.append(f(list(sol.x)))  # total reach1+%
        x = total_cost*100000000  # total cost
        y = (optimizing_x[j] for j in range(len(total_cost)))
        df = pd.DataFrame(y, index=x, columns=['JSPTV', 'CATV','DGT'])
        
        # visualization
        df_graph = df.divide(df.sum(axis=1), axis=0)
        ax = df_graph.plot(kind='area', stacked=True,
                           color=('lightcyan','skyblue', 'steelblue'),
                           title='%s' % age)
        plt.grid()
        plt.ylim(0, 1)
        fig = plt.gcf()
        plt.show()
        pdf_pages.savefig(fig)
        df_excel = df
        z = (result[i] for i in range(len(total_cost))) # total reach1%
        result_excel = pd.DataFrame(z, index=x, columns=['total_reach1'])
        df['JSPTV'] = df['JSPTV'] * 100000000
        df['CATV'] = df['CATV'] * 100000000
        df['DGT'] = df['DGT'] * 100000000
        df_excel['JSPTV reach1'] = cal_reach1('JSPTV', df['JSPTV']) * 100
        df_excel['CATV reach1'] = cal_reach1('CATV', df['CATV']) * 100
        df_excel['DGT reach1'] = cal_reach1('DGT', df['DGT']) * 100
        df_excel['total reach1'] = result_excel['total_reach1']
        df_excel.to_excel(excel_writer, sheet_name='%s' % age)
    excel_writer.save()
