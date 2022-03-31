import numpy as np
import matplotlib.pyplot as plt

def main():
    fig, axes = plt.subplots(1, 2, constrained_layout=True) 

    x = np.linspace(-1, 2, 100)
    y = np.exp(x)

    axes[0].plot(x, y)
    axes[0].xlabel('$x$')
    axes[0].ylabel('$\exp(x)$')
    axes[0].title('Exponential function')

    axes[0].plot(x, -np.exp(-x))
    axes[0].xlabel('$x$')
    axes[0].ylabel('$-\exp(-x)$')
    axes[0].title('Negative exponential\nfunction')

    fig.savefig('/data/pycode/MedIR/CIndex/imgs/citloss_expgraph.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()