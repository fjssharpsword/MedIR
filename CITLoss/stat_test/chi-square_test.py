import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency



def cal_chi(gender_data):
    '''
    function:卡方检验,检验两个属性是否与患病有关系
    :param gender_data:[[a,b],[c,d]]
    return (statistic,p-value,dof,expected)
    Example:
               male   female
    growth      a       b
    nongrowth   c       d
    '''
    kt= chi2_contingency(gender_data)

    return kt



if __name__ == '__main__':
    df = pd.read_csv('example.csv')

    #### gender example ####
    growth_male_num = len(df[(df['gender']=='M') & (df['Seg_label']==1)])
    growth_female_num = len(df[(df['gender']=='F') & (df['Seg_label']==1)])
    print ('growth: male_count: {}, female_count: {}'.format(growth_male_num, growth_female_num))

    nongrowth_male_num = len(df[(df['gender']=='M') & (df['Seg_label']==0)])
    nongrowth_female_num = len(df[(df['gender']=='F') & (df['Seg_label']==0)])
    print ('nongrowth: male_count: {}, female_count: {}'.format(nongrowth_male_num, nongrowth_female_num))

    gender_data = np.array([[growth_male_num, growth_female_num], [nongrowth_male_num, nongrowth_female_num]])

    # 卡方检验
    kt = cal_chi(gender_data)

    print ('Gender:  Chi-suqared: %.4f, p-value:  %.4f, freedom:  %i, expected_frep:  %s'% kt)