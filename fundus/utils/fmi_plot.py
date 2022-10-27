import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_fmi():

    x_axies = ['MT', 'Circle', 'CCT']
    fig, axes = plt.subplots(2,3, constrained_layout=True,figsize=(12,6)) 

    axes[0,0].set_ylabel('OIA-DDR')
    axes[0,0].set_title('ResNet50')
    y_ln = [61.75, 60.28, 62.59]
    y_sn = [62.00, 62.04, 63.18]
    axes[0,0].plot(x_axies, y_ln,'bo-',label='LNWD')
    axes[0,0].text(x_axies[0], y_ln[0], y_ln[0], ha='left', va='top', color='g')
    axes[0,0].text(x_axies[1], y_ln[1], y_ln[1], ha='left', va='bottom', color='g')
    axes[0,0].text(x_axies[2], y_ln[2], y_ln[2], ha='right', va='top', color='g')
    axes[0,0].plot(x_axies, y_sn,'r^-',label='SNWD')
    axes[0,0].text(x_axies[0], y_sn[0], y_sn[0], ha='left', va='top', color='g')
    axes[0,0].text(x_axies[1], y_sn[1], y_sn[1], ha='left', va='top', color='g')
    axes[0,0].text(x_axies[2], y_sn[2], y_sn[2], ha='right', va='top', color='g')
    axes[0,0].grid()
    axes[0,0].legend()

    axes[0,1].set_title('DenseNet121')
    y_ln = [60.37, 59.17, 62.10]
    y_sn = [61.03, 60.28, 63.04]
    axes[0,1].plot(x_axies, y_ln,'bo-',label='LNWD')
    axes[0,1].text(x_axies[0], y_ln[0], y_ln[0], ha='left', va='top', color='g')
    axes[0,1].text(x_axies[1], y_ln[1], y_ln[1], ha='left', va='bottom', color='g')
    axes[0,1].text(x_axies[2], y_ln[2], y_ln[2], ha='right', va='top', color='g')
    axes[0,1].plot(x_axies, y_sn,'r^-',label='SNWD')
    axes[0,1].text(x_axies[0], y_sn[0], y_sn[0], ha='left', va='top', color='g')
    axes[0,1].text(x_axies[1], y_sn[1], y_sn[1], ha='left', va='top', color='g')
    axes[0,1].text(x_axies[2], y_sn[2], y_sn[2], ha='right', va='top', color='g')
    axes[0,1].grid()
    axes[0,1].legend()

    axes[0,2].set_title('MobileNet_V2')
    y_ln = [57.22, 60.41, 64.11]
    y_sn = [58.41, 61.28, 65.91]
    axes[0,2].plot(x_axies, y_ln,'bo-',label='LNWD')
    axes[0,2].text(x_axies[0], y_ln[0], y_ln[0], ha='left', va='bottom', color='g')
    axes[0,2].text(x_axies[1], y_ln[1], y_ln[1], ha='left', va='top', color='g')
    axes[0,2].text(x_axies[2], y_ln[2], y_ln[2], ha='right', va='top', color='g')
    axes[0,2].plot(x_axies, y_sn,'r^-',label='SNWD')
    axes[0,2].text(x_axies[0], y_sn[0], y_sn[0], ha='left', va='bottom', color='g')
    axes[0,2].text(x_axies[1], y_sn[1], y_sn[1], ha='left', va='top', color='g')
    axes[0,2].text(x_axies[2], y_sn[2], y_sn[2], ha='right', va='top', color='g')
    axes[0,2].grid()
    axes[0,2].legend()

    axes[1,0].set_ylabel('IDRiD')
    #axes[1,0].set_title('ResNet50')
    y_ln = [46.48, 41.26, 49.69]
    y_sn = [47.10, 42.91, 50.22]
    axes[1,0].plot(x_axies, y_ln,'bo-',label='LNWD')
    axes[1,0].text(x_axies[0], y_ln[0], y_ln[0], ha='left', va='top', color='g')
    axes[1,0].text(x_axies[1], y_ln[1], y_ln[1], ha='left', va='bottom', color='g')
    axes[1,0].text(x_axies[2], y_ln[2], y_ln[2], ha='right', va='top', color='g')
    axes[1,0].plot(x_axies, y_sn,'r^-',label='SNWD')
    axes[1,0].text(x_axies[0], y_sn[0], y_sn[0], ha='left', va='top', color='g')
    axes[1,0].text(x_axies[1], y_sn[1], y_sn[1], ha='left', va='top', color='g')
    axes[1,0].text(x_axies[2], y_sn[2], y_sn[2], ha='right', va='top', color='g')
    axes[1,0].grid()
    axes[1,0].legend()

    #axes[1,1].set_title('DenseNet121')
    y_ln = [38.04, 36.81, 42.63]
    y_sn = [44.47, 40.77, 47.22]
    axes[1,1].plot(x_axies, y_ln,'bo-',label='LNWD')
    axes[1,1].text(x_axies[0], y_ln[0], y_ln[0], ha='left', va='top', color='g')
    axes[1,1].text(x_axies[1], y_ln[1], y_ln[1], ha='left', va='bottom', color='g')
    axes[1,1].text(x_axies[2], y_ln[2], y_ln[2], ha='right', va='top', color='g')
    axes[1,1].plot(x_axies, y_sn,'r^-',label='SNWD')
    axes[1,1].text(x_axies[0], y_sn[0], y_sn[0], ha='left', va='top', color='g')
    axes[1,1].text(x_axies[1], y_sn[1], y_sn[1], ha='left', va='top', color='g')
    axes[1,1].text(x_axies[2], y_sn[2], y_sn[2], ha='right', va='top', color='g')
    axes[1,1].grid()
    axes[1,1].legend()

    #axes[1,2].set_title('MobileNet_V2')
    y_ln = [42.00, 46.24, 51.25]
    y_sn = [42.90, 46.80, 52.82]
    axes[1,2].plot(x_axies, y_ln,'bo-',label='LNWD')
    axes[1,2].text(x_axies[0], y_ln[0], y_ln[0], ha='left', va='bottom', color='g')
    axes[1,2].text(x_axies[1], y_ln[1], y_ln[1], ha='left', va='top', color='g')
    axes[1,2].text(x_axies[2], y_ln[2], y_ln[2], ha='right', va='top', color='g')
    axes[1,2].plot(x_axies, y_sn,'r^-',label='SNWD')
    axes[1,2].text(x_axies[0], y_sn[0], y_sn[0], ha='left', va='bottom', color='g')
    axes[1,2].text(x_axies[1], y_sn[1], y_sn[1], ha='left', va='top', color='g')
    axes[1,2].text(x_axies[2], y_sn[2], y_sn[2], ha='right', va='top', color='g')
    axes[1,2].grid()
    axes[1,2].legend()

    fig.savefig('/data/pycode/MedIR/fundus/imgs/fmi_dr.png', dpi=300, bbox_inches='tight')

def main():
    plot_fmi()
    


if __name__ == '__main__':
    main()