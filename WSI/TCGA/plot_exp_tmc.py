import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    fig, axes = plt.subplots(1, 1, constrained_layout=True) 

    x = np.arange(0, 2.0, 0.2) 
    y = np.exp(-x)

    axes.plot(x, y, 'b-')
    axes.set_xlabel('Time intervals')
    axes.set_ylabel('Survival risk')
    axes.set_title('Hazard function $\lambda_{0}(t)=e^{-x}$')
    axes.set_xticks(x, ['1', '2', '$\cdots$', '$t_{i}$', '$\cdots$', '$t_{j}$', '$\cdots$', '$t_{n}$', '$\cdots$', 'k'])  

    axes.plot([0.6, 0.6], [0, np.exp(-0.6)], 'g--d')
    axes.plot([1.0, 1.0], [0, np.exp(-1.0)], 'g--d')
    axes.plot([1.4, 1.4], [0, np.exp(-1.4)], 'g-->')
    axes.text(0.7, np.exp(-0.6), '$f_{\Theta}(t\leq t_{i}|x_{i}) > f_{\Theta}(t\leq t_{i}|x_{j})$')
    axes.text(0.9, np.exp(-0.8), '$f_{\Theta}(t\leq t_{j}|x_{j}) > f_{\Theta}(t\leq t_{j}|x_{n})$')
    axes.text(1.1, np.exp(-1.0), '$f_{\Theta}(t\leq t_{i}|x_{i}) > f_{\Theta}(t\leq t_{i}|x_{n})$')
    axes.text(0.4, np.exp(-0.2), 'Classification and ranking learning', color='r')

    axes.grid()

    fig.savefig('/data/pycode/MedIR/WSI/imgs/tmc_exp.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('/data/pycode/MedIR/WSI/imgs/tmc_exp.png', dpi=300, bbox_inches='tight')
    """


if __name__ == '__main__':
    main()