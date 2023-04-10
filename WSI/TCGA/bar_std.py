import matplotlib.pyplot as plt
import numpy as np

def plt_var_std():

    fig, axes = plt.subplots(3,5,constrained_layout=True, figsize=(15,5))

    #----LUSC dataset---------#
    #LUSC-DeepAttnMISL
    x=[1,2,3,4,5]
    y=[72.29,70.90,72.24,72.48,72.97]
    std_err=[5.01,5.19,4.20,5.82,4.78]
    axes[0,0].set_ylim(69.0,74.0)
    std_err = np.array(std_err)*0.05#scale factor
    axes[0,0].set_yticks(np.arange(69.0, 74.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[0,0].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    #for a, b in zip(x,y):
    #    axes[0,0].text(a, b+0.05, '%.2f'%b, ha='center', va='bottom', fontsize=10)
    axes[0,0].set_ylabel('LUSC')
    axes[0,0].set_title('DAMISL')
    axes[0,0].set_xticks([])
    axes[0,0].grid(True, axis='y',ls=':',color='gray',alpha=0.3)
    
    #LUSC-BDOCOX
    y=[70.18,69.28,73.29,72.55,72.01]
    std_err=[5.89,7.81,2.91,5.92,4.24]
    axes[0,1].set_ylim(65.0,75.0)
    std_err = np.array(std_err)*0.1 
    axes[0,1].set_yticks(np.arange(65.0, 75.0, 2))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[0,1].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[0,1].set_title('BDOCOX')
    axes[0,1].set_xticks([])
    axes[0,1].grid(True, axis='y',ls=':',color='gray',alpha=0.3)
    axes[0,1].spines['bottom'].set_color('red')
    axes[0,1].spines['top'].set_color('red')
    axes[0,1].spines['left'].set_color('red')
    axes[0,1].spines['right'].set_color('red')

    #LUSC-TransMIL
    y=[69.29,68.35,65.82,65.71,70.98]
    std_err=[3.58,3.24,4.73,3.82,2.84]
    axes[0,2].set_ylim(63.0,73.0)
    std_err = np.array(std_err)*0.1 
    axes[0,2].set_yticks(np.arange(63.0, 73.0, 2))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[0,2].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[0,2].set_title('TransMIL')
    axes[0,2].set_xticks([])
    axes[0,2].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #LUSC-PathGCN
    y=[68.01,68.49,66.42,64.29,69.82]
    std_err=[8.92,2.38,6.79,4.08,4.52]
    axes[0,3].set_ylim(62.0,72.0)
    std_err = np.array(std_err)*0.1 
    axes[0,3].set_yticks(np.arange(62.0, 72.0, 2))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[0,3].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[0,3].set_title('PathGCN')
    axes[0,3].set_xticks([])
    axes[0,3].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #LUSC-TMCNet
    y=[71.12,70.44,71.78,70.13,73.09]
    std_err=[6.90,5.78,6.27,8.13,6.39]
    axes[0,4].set_ylim(69.0,74.0)
    std_err = np.array(std_err)*0.05
    axes[0,4].set_yticks(np.arange(69.0, 74.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[0,4].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[0,4].text(5, 73.09+0.05, '%.2f'%73.09, ha='center', va='bottom', fontsize=10)
    axes[0,4].set_title('TMCNet')
    axes[0,4].set_xticks([])
    axes[0,4].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #----GBM dataset---------#

    #GBM-DeepAttnMISL
    y=[82.97,82.03,81.81,81.28,84.22]
    std_err=[6.74,8.01,9.74,7.96,7.43]
    axes[1,0].set_ylim(80.0,85.0)
    std_err = np.array(std_err)*0.05
    axes[1,0].set_yticks(np.arange(80.0, 85.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[1,0].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[1,0].text(5, 84.22+0.05, '%.2f'%84.22, ha='center', va='bottom', fontsize=10)
    axes[1,0].set_ylabel('GBM')
    axes[1,0].set_xticks([])
    axes[1,0].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #GBM-BDOCOX
    y=[80.19,79.30,81.22,79.04,82.17]
    std_err=[7.34,4.92,8.39,7.28,6.73]
    axes[1,1].set_ylim(78.0,83.0)
    std_err = np.array(std_err)*0.05
    axes[1,1].set_yticks(np.arange(78.0, 83.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[1,1].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[1,1].set_xticks([])
    axes[1,1].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #GBM-TransMIL
    y=[82.20,82.12,79.01,81.99,81.30]
    std_err=[8.63,7.62,6.77,6.27,3.82]
    axes[1,2].set_ylim(78.0,83.0)
    std_err = np.array(std_err)*0.05
    axes[1,2].set_yticks(np.arange(78.0, 83.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[1,2].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[1,2].set_xticks([])
    axes[1,2].grid(True, axis='y',ls=':',color='gray',alpha=0.3)
    axes[1,2].spines['bottom'].set_color('red')
    axes[1,2].spines['top'].set_color('red')
    axes[1,2].spines['left'].set_color('red')
    axes[1,2].spines['right'].set_color('red')

    #GBM-PathGCN
    y=[77.90,76.56,74.68,78.09,78.77]
    std_err=[5.39,4.11,6.72,5.93,3.66]
    axes[1,3].set_ylim(74.0,80.0)
    std_err = np.array(std_err)*0.06
    axes[1,3].set_yticks(np.arange(74.0, 80.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[1,3].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[1,3].set_xticks([])
    axes[1,3].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #GBM-TMCNet
    y=[81.01,80.28,82.18,79.57,83.56]
    std_err=[8.02,7.29,7.94,8.24,7.24]
    axes[1,4].set_ylim(78.0,85.0)
    std_err = np.array(std_err)*0.07
    axes[1,4].set_yticks(np.arange(78.0, 85.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[1,4].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params)
    axes[1,4].set_xticks([])
    axes[1,4].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #----BRCA dataset---------#

    #BRCA-DeepAttnMISL
    y=[73.22,73.01,71.49,72.83,73.91]
    std_err=[7.92,7.91,6.72,7.81,5.78]
    axes[2,0].set_ylim(70.0,75.0)
    std_err = np.array(std_err)*0.05
    axes[2,0].set_yticks(np.arange(70.0, 75.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[2,0].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params, tick_label=['Cox','Cox+','CE','CE+','TMC'])
    axes[2,0].set_ylabel('BRCA')
    axes[2,0].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #BRCA-BDOCOX
    y=[69.98,68.75,71.48,73.21,74.76]
    std_err=[4.38,5.83,5.60,7.38,4.27]
    axes[2,1].set_ylim(67.0,76.0)
    std_err = np.array(std_err)*0.09
    axes[2,1].set_yticks(np.arange(67.0, 76.0, 2))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[2,1].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params, tick_label=['Cox','Cox+','CE','CE+','TMC'])
    axes[2,1].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #BRCA-TransMIL
    y=[70.90,68.44,68.76,69.02,72.14]
    std_err=[2.35,2.78,2.81,2.19,2.38]
    axes[2,2].set_ylim(67.0,73.0)
    std_err = np.array(std_err)*0.06
    axes[2,2].set_yticks(np.arange(67.0, 73.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[2,2].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params, tick_label=['Cox','Cox+','CE','CE+','TMC'])
    axes[2,2].grid(True, axis='y',ls=':',color='gray',alpha=0.3)

    #BRCA-PathGCN
    y=[71.17,66.90,66.87,71.87,69.80]
    std_err=[1.83,3.64,5.38,1.40,3.24]
    axes[2,3].set_ylim(65.0,73.0)
    std_err = np.array(std_err)*0.08
    axes[2,3].set_yticks(np.arange(65.0, 73.0, 2))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[2,3].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params, tick_label=['Cox','Cox+','CE','CE+','TMC'])
    axes[2,3].grid(True, axis='y',ls=':',color='gray',alpha=0.3)
    axes[2,3].spines['bottom'].set_color('red')
    axes[2,3].spines['top'].set_color('red')
    axes[2,3].spines['left'].set_color('red')
    axes[2,3].spines['right'].set_color('red')

    #BRCA-TMCNet
    y=[73.20,74.30,72.19,73.21,74.98]
    std_err=[9.74,7.84,8.49,7.49,5.94]
    axes[2,4].set_ylim(71.0,76.0)
    std_err = np.array(std_err)*0.05
    axes[2,4].set_yticks(np.arange(71.0, 76.0, 1))
    error_params=dict(elinewidth=4, ecolor='coral',capsize=5)
    axes[2,4].bar(x,y,color=['b','g','y','c','m'], yerr=std_err, error_kw=error_params, tick_label=['Cox','Cox+','CE','CE+','TMC'])
    axes[2,4].text(5, 74.98+0.05, '%.2f'%74.98, ha='center', va='bottom', fontsize=10)
    axes[2,4].grid(True, axis='y',ls=':',color='gray',alpha=0.3)
 

    fig.savefig('/data/pycode/MedIR/WSI/imgs/tmc_bar_std.png', dpi=300, bbox_inches='tight')

def ref_test():
    plt.figure()
    N = 7
    # 包含每个柱子对应值的序列
    x = np.array(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'])
    # 绘制柱状图的均值和方差
    means_men = np.array([53.38, 56.83, 59.60, 63.99, 64.28, 67.58, 72.43])
    std_men = np.array([1.38, 0.78, 0.60, 3.41, 0.86, 1.42, 1.75])

    # 包含每个柱子下标的序列
    index = np.arange(N)
    error_config=dict(elinewidth=4,ecolor='coral',capsize=6)
    plt.rcParams['font.family'] = "Times New Roman"

    # 柱子的宽度
    width = 0.78
    # 绘制柱状图
    p2 = plt.bar(index, means_men, width,
                    alpha=1, color='#3498db',align='center',
                    yerr=std_men,error_kw=error_config,
                    label='AUC')

    # 添加标题
    # plt.title('Monthly average rainfall')

    # y轴刻度范围
    #plt.ylim((50, 78))

    # 添加横坐标、纵横轴的刻度
    plt.xticks(index, x, fontsize=15, rotation=30)
    #plt.yticks(np.arange(50, 80, 5), fontsize=15)  # 纵坐标刻度是5

    # 添加图例
    plt.legend(loc="upper left", fontsize=15)
    # 添加网格线
    plt.grid(axis="y", linestyle='-.')

    # 保存
    plt.savefig('/data/pycode/MedIR/WSI/imgs/sigf_auc.png', bbox_inches='tight', pad_inches=0.1) #format="svg", 
    #plt.show()

def main():
    plt_var_std()
    #ref_test()

if __name__ == '__main__':
    main()