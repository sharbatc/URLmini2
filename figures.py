"""
helper file for figures (escape time)
Authors: Andras Ecker, Sharbatanu Chatterjee
"""

import numpy as np
import matplotlib.pyplot as plt

mplPars = { #'text.usetex'       :    True,
            'axes.labelsize'    :   'large',
            'font.family'       :   'serif',
            'font.sans-serif'   :   'computer modern roman',
            'font.size'         :    14,
            'xtick.labelsize'   :    12,
            'ytick.labelsize'   :    12
            }
for key, val in mplPars.items():
    plt.rcParams[key] = val


def plot_escape_time(esc_ts, tau, lambda_, x_size, xdot_size, w_ini):
    """Plots escape time"""
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(esc_ts, linewidth=2)
    ax.set_title("Escape time (tau:%.3f, lambda:%.3f, x_size:%s, xdot_size:%s, w_ini:%s)"%(tau, lambda_, x_size, xdot_size, w_ini))
    ax.set_xlabel("#{episodes}")
    ax.set_ylabel("Escape time: #{iteration}")
       
    figName = "figures/esc_time__tau%.3f_lambda%.3f_x_size%s_xdot_size%s_wini%s.jpg"%(tau, lambda_, x_size, xdot_size, w_ini)
    fig.savefig(figName)
    print("figure saved to:", figName)
    plt.close()
    
    


