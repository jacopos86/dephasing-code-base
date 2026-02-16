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

def plot_elec_struct(Eks, mu, Ylim=[-0.3, 0.5]):
    plt.figure(figsize=(8,5))
    for isp in range(Eks.shape[2]):
        for ib in range(Eks.shape[0]):
            plt.plot(range(Eks.shape[1]), Eks[ib, :, isp]-mu, 'k.', markersize=2, label='DFT (KS)' if ib==0 else "")
    plt.axhline(0, color='gray', lw=1.5)
    if Ylim is not None:
        plt.ylim(Ylim[0], Ylim[1])
    if Eks.shape[1] == 1:
        plt.xlim([-1, 1])
    else:
        plt.xlim(0., Eks.shape[1])
    plt.ylabel("E - VBM [eV]")
    plt.xlabel("k-point index")
    plt.legend()
    plt.title("KS Band Structure (shifted to VBM)")
    plt.tight_layout()
    plt.savefig(f"{p.write_dir}/Electronic_bandstructure.png",dpi=300,bbox_inches="tight" )
    plt.show()

def plot_wan_struct(Ew, Eks, mu, n_interp=10, Ylim=[-0.3, 0.5]):
    plt.figure(figsize=(8,5))
    for ib in range(Eks.shape[1]):
        plt.plot(range(Eks.shape[0]), Eks[:, ib]-mu, 'k.', markersize=2, label='DFT (KS)' if ib==0 else "")
    # Wannier lines (slightly shifted on x)
    xw = np.arange(Ew.shape[0]) / n_interp
    for ib in range(Ew.shape[1]):
        plt.plot(xw, Ew[:, ib]-mu, 'r-', color='r', lw=1, label='Wannier' if ib==0 else "")
    plt.axhline(0, color='gray', lw=1.5)
    if Ylim is not None:
        plt.ylim(Ylim[0], Ylim[1])
    plt.xlim(0., Eks.shape[0])
    plt.ylabel("E - VBM [Ha]")
    plt.xlabel("k-point index")
    plt.legend()
    plt.title("Wannier vs KS Band Structure (shifted to VBM)")
    plt.tight_layout()
    plt.savefig(f"{p.write_dir}/Wannier_bandstructure.png",dpi=300,bbox_inches="tight" )
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

def plot_ph_dos(w, dos):
    # Plot phonon DOS
    plt.figure(figsize=(8,6))
    plt.plot(w, dos)
    plt.ylabel("Phonon DOS")
    plt.xlabel("phonon freq. [THz]")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{p.write_dir}/Phonon_DOS.png",dpi=300,bbox_inches="tight" )
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
    cbar.ax.set_title(r"$l_{ph} (\hbar)$")
    axs[0].set_ylabel("Phonon energy [meV]")
    #plt.tight_layout()# tight_layout moves colorbar in the middle of axs[2]
    plt.savefig(f"{p.write_dir}/Phonon_PAM_bandstructure.png",dpi=300,bbox_inches="tight" )
    plt.show()

def plot_Mph_heatmap(Mph):
    cm1 = mcol.LinearSegmentedColormap.from_list("MyColorMap", ['r', (1,1,1), 'b'])
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
    cbar.ax.set_title(r"$\rm M_{ph} (\hbar)$")
    #plt.tight_layout()# tight_layout moves colorbar in the middle of axs[2]
    plt.savefig(f"{p.write_dir}/Mph_heatmap.png",dpi=300,bbox_inches="tight" )
    plt.show()

def plot_td_Bq(evol_params, Bq_t):
    # input parameters
    dt = evol_params.get("time_step")                # ps
    save_every = evol_params.get("save_every")
    # dimensions
    nq, nmd, nt_save = Bq_t.shape
    # --- correct physical time axis ---
    time = dt * save_every * np.arange(nt_save)
    # subplots
    fig, ax = plt.subplots()
    # q pts.
    q_indices = range(nq)
    for iq in q_indices:
        ax.plot(time, Bq_t[iq, nmd-1, :], label=f"q={iq}")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"$|B_q|$")
    ax.legend()
    plt.savefig(f"{p.write_dir}/Bq_t.png",dpi=300,bbox_inches="tight" )
    plt.show()
    # --- save data ---
    np.savez(
        f"{p.write_dir}/Bq_t.npz",
        time=time,
        Bq=Bq_t
    )

def plot_td_trace(evol_params, tr_t, ylim=[-2, 2]):
    # input parameters
    dt = evol_params.get("time_step")                # ps
    save_every = evol_params.get("save_every")
    # dimensions
    nk, nsp, nt_save = tr_t.shape
    # --- correct physical time axis ---
    time = dt * save_every * np.arange(nt_save)
    wk = np.ones(nk) / nk
    Nel = np.zeros(nt_save)
    for ik in range(nk):
        for isp in range(nsp):
            Nel[:] += wk[ik] * tr_t[ik, isp, :]
    # subplots
    fig, ax = plt.subplots()
    ax.plot(time, Nel)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"$N_e$")
    ax.set_ylim(ylim)
    ax.legend()
    plt.savefig(f"{p.write_dir}/Ne_t.png",dpi=300,bbox_inches="tight" )
    plt.show()

def plot_td_occup(evol_params, rho_t):
    # input parameters
    dt = evol_params.get("time_step")                # ps
    save_every = evol_params.get("save_every")
    # dimensions
    nk, nsp, nt_save, nb, nb = rho_t.shape
    # --- correct physical time axis ---
    time = dt * save_every * np.arange(nt_save)
    wk = np.ones(nk) / nk
    rh = np.zeros((nb,nb,nt_save))
    for ib1 in range(nb):
        for ib2 in range(nb):
            for ik in range(nk):
                for isp in range(nsp):
                    rh[ib1,ib2,:] += wk[ik] * rho_t[ik, isp, :, ib1, ib2].real
    # subplots
    fig, ax = plt.subplots()
    for ib1 in range(nb):
        for ib2 in range(ib1, nb):
            ax.plot(time, rh[ib1,ib2,:]-rh[ib1,ib2,0], label=r"$\rho_{"+f"{ib1}{ib2}"+"}$")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"$\rho$")
    ax.legend()
    plt.savefig(f"{p.write_dir}/rho_t.png",dpi=300,bbox_inches="tight" )
    plt.show()
    # --- save data ---
    np.savez(
        f"{p.write_dir}/rho_t.npz",
        time=time,
        rho=rh
    )

def plot_ph_pulse(tgr, Fq):
    nq = Fq.shape[0]
    fig, ax = plt.subplots()
    for iq in range(nq):
        ax.plot(tgr, Fq[iq,0,:], label=f"pulse{iq}")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("F_q(t)")
    ax.legend()
    plt.savefig(f"{p.write_dir}/Fq_t.png",dpi=300,bbox_inches="tight")
    plt.show()
    # --- save data ---
    np.savez(
        f"{p.write_dir}/Fq_t.npz",
        time=tgr,
        Fq=Fq
    )

def plot_A_pulse(tgr, A_t):
    fig, ax = plt.subplots()
    ax.plot(tgr, A_t, label="pulse")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("A(t)")
    ax.legend()
    plt.savefig(f"{p.write_dir}/A_t.png",dpi=300,bbox_inches="tight")
    plt.show()
    # --- save data ---
    np.savez(
        f"{p.write_dir}/A_t.npz",
        time=tgr,
        A_t=A_t
    )

def plot_total_energy(evol_params, Eph_t, Ee_t, Eeph_t):
    # account for absent dynamics
    Eph_t = Eph_t if Eph_t is not None else np.array([0])
    Ee_t = Ee_t if Ee_t is not None else np.array([0])
    Eeph_t= Eeph_t if Eeph_t is not None else np.array([0])
    # input parameters
    dt = evol_params.get("time_step")                # ps
    save_every = evol_params.get("save_every")
    # get longest length of plotting (should be 1 if no dynamics)
    nt_save = 0
    for i in [Eph_t, Ee_t, Eeph_t]:
        nt_save = len(i) if len(i) > nt_save else nt_save
    # --- correct physical time axis ---
    time = dt * save_every * np.arange(nt_save)
    # subplots
    fig, ax = plt.subplots()
    
    Etot_t0 = Eph_t[0]+Ee_t[0]+Eeph_t[0]
    Etot_t = Eph_t + Ee_t + Eeph_t
    if Eph_t.shape != (1,):
        ax.plot(time, Eph_t-Eph_t[0], label=r"ph. energy")
    if Ee_t.shape != (1,):
        ax.plot(time, Ee_t-Ee_t[0], label=r"electronic energy")
    if Eeph_t.shape != (1,):
        ax.plot(time, Eeph_t-Eph_t[0], label=r"e-ph energy")
    ax.plot(time, Etot_t-Etot_t0, ls='--', label=r"total energy")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"Energy-Energy(0) (eV)")
    ax.legend()
    plt.savefig(f"{p.write_dir}/Energy_t.png",dpi=300,bbox_inches="tight" )
    plt.show()
    # --- save data ---
    np.savez(
        f"{p.write_dir}/energy_t.npz",
        time=time,
        Eph=Eph_t,
        Ee=Ee_t,
        Eeph=Eeph_t,
        Etot=Eph_t + Ee_t + Eeph_t
    )
