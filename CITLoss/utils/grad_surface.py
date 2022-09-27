import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D
def plot_3d():
    fig = plt.figure(figsize=(12,8))
    ax = Axes3D(fig)
    x = np.arange(-2,2,0.05)
    y = np.arange(-2,2,0.05)
    ##对x,y数据执行网格化
    x,y = np.meshgrid(x,y)
    z1 = np.exp(-x**2-y**2)
    z2 = np.exp(-(x-1)**2-(y-1)**2)
    z = -(z1-z2)*2
    ax.plot_surface(x,y,z,    ##x,y,z二维矩阵（坐标矩阵xv，yv,zv）
                    rstride=1,##retride(row)指定行的跨度
                    cstride=1,##retride(column)指定列的跨度
                    cmap='rainbow')  ##设置颜色映射
    ##设置z轴范围
    ax.set_zlim(-2,2)
    ##设置标题
    plt.title('优化设计之梯度下降--目标函数',fontproperties = 'SimHei',fontsize = 20)
    plt.savefig('/data/pycode/MedIR/CITLoss/imgs/grad_surface.png', dpi=300)

plot_3d()
if __name__ == '__main__':
    plot_3d()