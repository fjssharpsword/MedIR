from scipy import stats

def cal_ttest_ind(sample1, sample2):
    '''
    function:T-test,用于检验两个独立样本的均值差异是否显著。
    params:sample1 array_like 
    params:sample2 array_like
    Return statistic,P值

    Examples:
    sample1 = stats.norm.rvs(loc=5,scale=10,size=500)
    sample2 = stats.norm.rvs(loc=5,scale=10,size=500)
    statistic,pVal = cal_ttest_ind(sample1, sample2)

    Notes:
    步骤：
    1.判断方差齐性（当两样本的方差相等时，即具有“方差齐性”，可以直接检验）;
      当不确定两总体方差是否相等时，应先利用levene检验，检验两总体是否具有方差齐性。
    2.方差齐性：则执行假设总体方差相等的标准独立两样本检验
      方差非齐性：执行 Welch 的T-test
    如果两总体具有方差齐性，错将equal_var设为False，p值变大
    '''
    assert sample1.shape == sample2.shape
    equal_var = True  # True(默认)，则执行假设总体方差相等的标准独立两样本检验 [1] 。如果为 False，执行 Welch 的 t-test
    _,p = stats.levene(sample1, sample2) # p值远大于0.05，认为两总体具有方差齐性。
    if p<0.05:
        equal_var = False

    statistic,pVal = stats.ttest_ind(sample1,sample2,equal_var=equal_var)


    return statistic,pVal


if __name__ == "__main__":
    sample1 = stats.norm.rvs(loc=5,scale=10,size=500)
    sample2 = stats.norm.rvs(loc=5,scale=10,size=500)
    statistic,pVal = cal_ttest_ind(sample1, sample2)

    print('p-Value = {}, statistic = {}'.format(pVal,statistic))