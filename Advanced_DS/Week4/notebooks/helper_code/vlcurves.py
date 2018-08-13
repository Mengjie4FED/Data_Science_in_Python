import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_range(trn_szs, trn_scr, tst_scr):

    trn_scr_mu = np.mean(trn_scr, axis=1)
    trn_scr_sig = np.std(trn_scr, axis=1)

    trn_high = np.array(trn_scr_mu) + np.array(trn_scr_sig)
    trn_low = np.array(trn_scr_mu) - np.array(trn_scr_sig)

    tst_scr_mu = np.mean(tst_scr, axis=1)
    tst_scr_sig = np.std(tst_scr, axis=1)

    tst_high = np.array(tst_scr_mu) + np.array(tst_scr_sig)
    tst_low = np.array(tst_scr_mu) - np.array(tst_scr_sig)
    
    return(trn_scr_mu, trn_high, trn_low, tst_scr_mu, tst_high, tst_low)


def lv_plot(ttle, x, y1, y1h, y1l, y2, y2h, y2l, xlbl, ylbl, ylim=None):

    # Plot the results
    fig, ax = plt.subplots(figsize=(10,8))

    trn_color = color=sns.xkcd_rgb["denim blue"]
    ax.plot(x, y1, label="Training Score", marker='d', lw=3, color=trn_color)
    ax.fill_between(x, y1h, y1l, alpha=0.25, color=trn_color)

    tst_color = color=sns.xkcd_rgb["medium green"]
    ax.plot(x, y2, label="CV Score", marker='d', lw=3, color=tst_color)
    ax.fill_between(x, y2h, y2l, alpha=0.25, color=tst_color)

    # Decorate plot
    ax.set(title=ttle, xlabel=xlbl, ylabel=ylbl)

    if ylim is not None:
        ax.set_ylim(*ylim)
        
    ax.legend(loc='best', borderaxespad=1.5)
    sns.despine(offset=10, trim=True)