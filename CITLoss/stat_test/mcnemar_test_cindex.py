from statsmodels.sandbox.stats.runs import mcnemar
import pandas as pd
import numpy as np

def cal_mcnemar(obs,exact=False):
    '''
    function:McNemar检验
    params:obs  np.array([[1, 4],[0, 2]])
    params:exact  Bool. 默认False.The test statistic is the chisquare statistic if exact is false.
    return statistic,P值
    Example:

          |    算法A
    算法B  ———————————
          |正确    错误
    ————————————————————       
    正确  |e00,    e01
    错误  |e10,    e11

    obs = ([[e00, e01],[e10, e11]])
    cal_mcnemar(obs,exact=False)
    '''

    (statistic, pVal) = mcnemar(obs,exact=exact)

    return statistic, pVal


def convert_risk_to_label(origin_df):
    '''
    function:根据cindex计算原理，把预测的结果转换成0或1的值
    return:DataFrame
    '''

    id = origin_df['id'].values
    survival_time = origin_df['time'].values
    label = origin_df['state'].values
    risk = origin_df['risk'].values

    idx = np.argsort(survival_time) # 升序
    id = id[idx]
    risk = risk[idx]
    label = label[idx]
    survival_time = survival_time[idx]

    df = pd.DataFrame()
    df['id'] = id
    df.index = id
    df['time'] = survival_time
    df['state'] = label
    df['risk'] = risk

    # 根据cindex计算原理，把预测的结果生成0或1的值
    fenmu = 0 # cindex 分母
    fenzi = 0 # cindex 分子
    for r in range(len(risk)-1):
        if label[r] == 1:
            num = risk[r]>risk[r+1:]
            fenmu += len(num)
            fenzi += np.sum(num)
            df.loc[id[r+1]:,id[r]] = num

    # cindex = fenzi/fenmu

    return df


def pair_label(refer_df,df):
    '''
    function:配对两个模型的label
    '''
    refer_df.index = refer_df['id']
    refer_columns = refer_df.columns
   
    df.index = df['id']
    df_columns = df.columns

    assert refer_columns.values.all() == df_columns.values.all()

    labels = []
    refer_labels = []
    for i,c in enumerate(refer_columns[4:]): # 前四列是id,state,time,risk
        names = df.loc[c:,'id']
        tf = df.loc[names,c] # 读取c样本配对的情况，值是True和False
        labels.extend([1 if x else 0 for x in tf[1:]]) # tf[1:]表示不计算自身

        refer_tf = refer_df.loc[names,c]
        refer_labels.extend([1 if x else 0 for x in refer_tf[1:]])

    labels = np.array(labels)
    refer_labels = np.array(refer_labels)

    e00 = np.sum(labels*refer_labels) # 共同正确的数量
    e10 =  np.sum(refer_labels) - e00 # 方法A正确且方法B错误的数量
    e01 = np.sum(labels) - e00 # 方法B正确且方法A错误的数量
    e11 = np.sum((1-labels)*(1-refer_labels)) # 共同错误的数量

    return [[e00, e01],[e10, e11]]


if __name__ == "__main__":
    refer_origin_csv = r'/data_local/data_local_project/Surv_on_HJW/roc_comparison/csv/c-index loss/DVH_Cli_CT_RD_RS/test_sysucc.csv'
    refer_origin_df = pd.read_csv(refer_origin_csv)

    origin_csv = r'/data_local/data_local_project/Surv_on_HJW/roc_comparison/csv/cce loss/DVH_Cli_CT_RD_RS/test_sysucc.csv'
    origin_df = pd.read_csv(origin_csv)

    # 根据cindex计算原理，把预测的结果转换成0或1的值
    refer_df = convert_risk_to_label(refer_origin_df)
    df = convert_risk_to_label(origin_df)

    # 配对两个模型的label
    obs = pair_label(refer_df,df)

    # McNemar检验
    obs = np.array(obs)
    statistic, pVal = cal_mcnemar(obs)
    print('p-Value = {}, statistic = {}'.format(pVal,statistic))