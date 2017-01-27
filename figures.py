"""
helper file for figures (escape times)
Authors: Andras Ecker, Sharbatanu Chatterjee
"""

import numpy as np
import matplotlib.pyplot as plt

mplPars = { #'text.usetex'       :    True,
            'axes.labelsize'    :   'large',
            'font.family'       :   'serif',
            'font.sans-serif'   :   'computer modern roman',
            'font.size'         :    12,
            'xtick.labelsize'   :    10,
            'ytick.labelsize'   :    10
            }
for key, val in mplPars.items():
    plt.rcParams[key] = val


def plot_escape_time(esc_ts, tau, lambda_, x_size, xd_size, w_ini):
    """Plots escape time"""
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(esc_ts, linewidth=2)
    ax.set_title("Escape time (tau:%.5f, lambda:%.2f, x_size:%s, xd_size:%s, w_ini:%s)"%(tau, lambda_, x_size, xd_size, w_ini))
    ax.set_xlabel("#{episodes}")
    ax.set_ylabel("Escape time: #{iteration}")
       
    figName = "figures/esc_time__tau%.5f_lambda%.2f_x_size%s_xd_size%s_wini%s.jpg"%(tau, lambda_, x_size, xd_size, w_ini)
    fig.savefig(figName)
    print("figure saved to:", figName)
    plt.close()
    
    
def plot_mult_escape_times(mult_esc_ts, tau, lambda_, x_size, xd_size, w_ini):
    """Plots escape time"""
    
    # convert multiple escape times to array (from list of lists)
    X = np.zeros((len(mult_esc_ts), len(mult_esc_ts[0])))
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    
    for i, esc_ts in enumerate(mult_esc_ts):
        ax.plot(esc_ts, linewidth=0.5, color='0.5')
        X[i,:] = esc_ts
    avg_esc_t = np.mean(X, axis=0)
    ax.plot(avg_esc_t, linewidth=2)
    ax.set_title("Avg. escape time (tau:%.5f, lambda:%.2f, x_size:%s, xd_size:%s, w_ini:%s)"%(tau, lambda_, x_size, xd_size, w_ini))
    ax.set_xlabel("#{episodes}")
    ax.set_ylabel("Escape time (10 averaged): #{iteration}")
       
    figName = "figures/avg_esc_time__tau%.5f_lambda%.2f_x_size%s_xd_size%s_wini%s.jpg"%(tau, lambda_, x_size, xd_size, w_ini)
    fig.savefig(figName)
    print("figure saved to:", figName)
    plt.close()
    
    


