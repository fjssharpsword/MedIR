import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    fig, axes = plt.subplots(1, 1, constrained_layout=True) 

    x = np.linspace(-2.5, 2.5, 10)
    y = np.exp(x)

    axes.plot(x, y, 'b-')
    axes.set_xlabel('$x$')
    axes.set_ylabel('$\exp(x)$')
    axes.set_title('Exponential function')

    #axes[1].plot(x, -np.exp(-x))
    #axes[1].set_xlabel('$x$')
    #axes[1].set_ylabel('$-\exp(-x)$')
    #axes[1].set_title('Negative exponential\nfunction')

    fig.savefig('/data/pycode/MedIR/CITLoss/imgs/citloss_expgraph.png', dpi=300, bbox_inches='tight')
    """
    """
    # 100 linearly spaced numbers
    x = np.linspace(-2,2)

    # the function, which is y = e^x here
    y = np.exp(x)

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid()

    # plot the function
    plt.plot(x,y, 'y-', label='$y=e^x$')
    #plot right zone
    xz =  np.linspace(0.0,1.0)
    yz = np.exp(xz)
    plt.fill_between(xz, 1.0, yz, color='c', alpha=.75)
    plt.plot([0.0, 1.0], [np.exp(1.0), np.exp(1.0)], 'c--')
    plt.plot([1.0, 1.0], [1.0, np.exp(1.0)], 'c--')
    plt.text(1.0,np.exp(1.0), 'e', color='c')
    plt.text(0.755,np.exp(0.4), 'A', color='r')

    xz =  np.linspace(0.0,2.0)
    yz = np.exp(xz)
    plt.fill_between(xz, 1.0, yz, color='g', alpha=.25)
    plt.text(1.5,np.exp(1.5)/2, 'C', color='r')
    #plot right zone
    xz =  np.linspace(-1.0, 0.0)
    yz = np.exp(xz)
    plt.fill_between(xz, 0, yz, color='b', alpha=.75)
    plt.plot([-1.0, -1.0], [0.0, np.exp(-1.0)], 'b--')
    #plt.plot([-1.0, 0.0], [np.exp(0.0), np.exp(0.0)], 'b--')
    plt.text(-0.5, np.exp(-0.5)/2, 'B', color='r')

    xz =  np.linspace(-2.0,0.0)
    yz = np.exp(xz)
    plt.fill_between(xz, 0, yz, color='g', alpha=.25)
    plt.text(-1.5, np.exp(-1.5)/2, 'D', color='r')

    plt.text(-2.0, 5.5, 'Zone A: $\Vert \mathbf{x}_{a}-\mathbf{x}_{p} \Vert_{2} > \Vert \mathbf{x}_{a}-\mathbf{x}_{n} \Vert_{2} $')
    plt.text(-2.0, 4.5, 'Zone B: $\Vert \mathbf{x}_{a}-\mathbf{x}_{p} \Vert_{2} < \Vert \mathbf{x}_{a}-\mathbf{x}_{n} \Vert_{2} $')
    plt.text(-2.0, 3.5, 'Zone C: Zone A with $\gamma$')
    plt.text(-2.0, 2.5, 'Zone D: Zone B with $\gamma$')
    #plt.plot([-2.0, 2.0], [1.0, 1.0], 'g--')
    #plt.title('Distribution zone of our CIT loss')
    plt.legend(loc='upper left')

    # show the plot
    plt.savefig('/data/pycode/MedIR/CITLoss/imgs/cit_exp.png', dpi=300, bbox_inches='tight')
    """

    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12,6) )

    #plot exp
    axes[0].spines['left'].set_position('center')
    axes[0].spines['bottom'].set_position('zero')
    axes[0].spines['right'].set_color('none')
    axes[0].spines['top'].set_color('none')
    axes[0].xaxis.set_ticks_position('bottom')
    axes[0].yaxis.set_ticks_position('left')
    axes[0].grid()

    # plot the function
    # 100 linearly spaced numbers
    x = np.linspace(-2,2)
    # the function, which is y = e^x here
    y = np.exp(x)
    axes[0].plot(x,y, 'y-', label='$y=e^x$')
    #plot right zone
    xz =  np.linspace(0.0,1.0)
    yz = np.exp(xz)
    axes[0].fill_between(xz, 1.0, yz, color='c', alpha=.75)
    axes[0].plot([0.0, 1.0], [np.exp(1.0), np.exp(1.0)], 'c--')
    axes[0].plot([1.0, 1.0], [1.0, np.exp(1.0)], 'c--')
    axes[0].text(1.0,np.exp(1.0), 'e', color='c')
    axes[0].text(0.755,np.exp(0.4), 'A', color='r')

    xz =  np.linspace(0.0,2.0)
    yz = np.exp(xz)
    axes[0].fill_between(xz, 1.0, yz, color='g', alpha=.25)
    axes[0].text(1.5,np.exp(1.5)/2, 'C', color='r')
    #plot right zone
    xz =  np.linspace(-1.0, 0.0)
    yz = np.exp(xz)
    axes[0].fill_between(xz, 0, yz, color='b', alpha=.75)
    axes[0].plot([-1.0, -1.0], [0.0, np.exp(-1.0)], 'b--')
    #plt.plot([-1.0, 0.0], [np.exp(0.0), np.exp(0.0)], 'b--')
    axes[0].text(-0.5, np.exp(-0.5)/2, 'B', color='r')

    xz =  np.linspace(-2.0,0.0)
    yz = np.exp(xz)
    axes[0].fill_between(xz, 0, yz, color='g', alpha=.25)
    axes[0].text(-1.5, np.exp(-1.5)/2, 'D', color='r')

    axes[0].text(-2.0, 5.5, 'Zone A: $\Vert \mathbf{x}_{a}-\mathbf{x}_{p} \Vert_{2} > \Vert \mathbf{x}_{a}-\mathbf{x}_{n} \Vert_{2} $')
    axes[0].text(-2.0, 4.5, 'Zone B: $\Vert \mathbf{x}_{a}-\mathbf{x}_{p} \Vert_{2} < \Vert \mathbf{x}_{a}-\mathbf{x}_{n} \Vert_{2} $')
    axes[0].text(-2.0, 3.5, 'Zone C: Zone A with $\gamma=0.5$')
    axes[0].text(-2.0, 2.5, 'Zone D: Zone B with $\gamma=0.5$')
    #plt.plot([-2.0, 2.0], [1.0, 1.0], 'g--')
    #plt.title('Distribution zone of our CIT loss')
    axes[0].legend(loc='upper left')

    #plot loss
    y_gamma_b = pd.read_csv('/data/pycode/MedIR/CITLoss/imgs/loss/market_cnn_gamma5.csv', sep=',')#loss value, gamma=0.5
    y_gamma_s = pd.read_csv('/data/pycode/MedIR/CITLoss/imgs/loss/market_cnn_gamma10.csv', sep=',')#loss value, gamma=1.0

    y_gamma_b = y_gamma_b['Value'].to_numpy().flatten()
    y_gamma_s = y_gamma_s['Value'].to_numpy().flatten()
    x_epochs = np.arange(len(y_gamma_b))
    axes[1].plot(x_epochs, y_gamma_s,'b',label='$\gamma$=1.0, Zone A and B')
    axes[1].plot(x_epochs, y_gamma_b,'g',label='$\gamma$=0.5, Zone C and D')
    axes[1].set_ylabel('CIT Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid()

    # show the plot
    fig.savefig('/data/pycode/MedIR/CITLoss/imgs/cit_exp_loss.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()