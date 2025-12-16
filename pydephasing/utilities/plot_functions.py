import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import matplotlib.colors as mcol
from matplotlib.collections import LineCollection
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p

#
#  module to plot functions
#

def plot_elec_struct(Ew, Eks, mu, n_interp=10):
    plt.figure(figsize=(8,5))
    for ib in range(Eks.shape[1]):
        plt.plot(range(Eks.shape[0]), Eks[:, ib]-mu, 'k.', markersize=2, label='DFT (KS)' if ib==0 else "")
    # Wannier lines (slightly shifted on x)
    xw = np.arange(Ew.shape[0]) / n_interp
    for ib in range(Ew.shape[1]):
        plt.plot(xw, Ew[:, ib]-mu, 'r-', color='r', lw=1, label='Wannier' if ib==0 else "")
    plt.axhline(0, color='gray', lw=1.5)
    plt.ylim(-0.3, 0.5)
    plt.xlim(0., Eks.shape[0])
    plt.ylabel("E - VBM [eV]")
    plt.xlabel("k-point index")
    plt.legend()
    plt.title("Wannier vs KS Band Structure (shifted to VBM)")
    plt.tight_layout()
    plt.savefig(f"{p.write_dir}/Electronic_bandstructure.png",dpi=300,bbox_inches="tight" )
    plt.show()

def plot_ph_band_struct(wq, nQ, n_interp=10):
    # Plot phonon dispersion
    plt.figure(figsize=(8,6))
    for ib in range(wq.shape[1]):
        plt.plot(wq[:, ib], color='b')
    plt.xlim([0, nQ-1])
    plt.ylim([0, None])
    plt.ylabel("Phonon energy [meV]")
    plt.xlabel("K-point index")
    # Extract k-point labels from bandstruct.plot if available
    try:
        kpathLabels = sp.check_output(['awk', '/set xtics/ {print}', 'bandstruct.plot']).decode().split()
        kpathLabelText = [label.split('"')[1] for label in kpathLabels[3:-2:2]]
        kpathLabelPos = [n_interp * int(pos.split(',')[0]) for pos in kpathLabels[4:-1:2]]
        plt.xticks(kpathLabelPos, kpathLabelText)
    except Exception as e:
        log.warning('Warning: could not extract labels from bandstruct.plot')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{p.write_dir}/Phonon_bandstructure.png",dpi=300,bbox_inches="tight" )
    plt.show()

def plot_lph_struct(wq, lph, nQ, n_interp=10):
    cm1 = mcol.LinearSegmentedColormap.from_list("MyColorMap", ['r', 'k', 'b'])
    norm_in = np.abs(lph).max()
    norm = plt.Normalize(-norm_in, norm_in)
    fig, axs = plt.subplots(1,3,figsize=(14,4),sharey=True)
    lin_x = np.arange(nQ)
    ylims = [wq.min(),wq.max() * 1.05]
    # plots
    for i, (d,ax) in enumerate(zip(['x','y','z'], axs)):
        PAM = lph[i].real
        for ib in range(wq.shape[1]):
            y = wq[:,ib]
            points = np.array([lin_x, y]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cm1, norm=norm)
            lc.set_array(PAM[ib])
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
        ax.set_title(r"$l_{ph}^{" + "xyz"[i] + "}$",fontsize=14)
        ax.set_ylim(ylims)
        ax.set_xlim(0,nQ)
        ax.set_xlabel("K-point index")
    cbar = fig.colorbar(line, ax=axs, ticks=np.linspace(-norm_in,norm_in,5), shrink=0.8)
    cbar.ax.set_title(r"$l_{ph}$")
    axs[0].set_ylabel("Phonon energy [meV]")
    #plt.tight_layout()# tight_layout moves colorbar in the middle of axs[2]
    plt.savefig(f"{p.write_dir}/Phonon_PAM_bandstructure.png",dpi=300,bbox_inches="tight" )
    plt.show()

def plot_Mph_heatmap(Mph):
    cm1 = mcol.LinearSegmentedColormap.from_list("MyColorMap", ['r', 'k', 'b'])
    norm_in = np.nanmax(np.abs(Mph))
    norm = plt.Normalize(-norm_in, norm_in)
    fig, axs = plt.subplots(1,3,figsize=(14,4))
    for i,ax in enumerate(axs):
        cax = ax.matshow(Mph[i],norm=norm,cmap=cm1)
        ax.set_title(r"$\rm M_{ph}^{" + "xyz"[i] + r"}$($\rm q= \Gamma $)",fontsize=14)
        ax.set_ylabel('Mode Index')
        ax.set_xlabel('Mode Index')
        ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    cbar = fig.colorbar(cax, ax=axs, ticks=np.linspace(-norm_in,norm_in,5), shrink=0.8)
    cbar.ax.set_title(r"$\rm M_{ph}$")
    #plt.tight_layout()# tight_layout moves colorbar in the middle of axs[2]
    plt.savefig(f"{p.write_dir}/Mph_heatmap.png",dpi=300,bbox_inches="tight" )
    plt.show()
