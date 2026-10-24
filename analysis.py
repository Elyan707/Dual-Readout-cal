import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

output_dir = "pics_2"
os.makedirs(output_dir, exist_ok=True)

def Gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


#branches with scintillating energy. totSiPMSene is the total energy from the central module
sipm_s_cols = [
           "totSiPMSene", "SPMT1", "SPMT2", "SPMT3", "SPMT4",
            "SPMT5", "SPMT6", "SPMT7", "SPMT8"
        ]
#branches with cherenkov energy. totSiPMCene is the total energy from the central module
sipm_c_cols = [
    "totSiPMCene" ,"CPMT1", "CPMT2", "CPMT3", "CPMT4",
    "CPMT5", "CPMT6", "CPMT7", "CPMT8"
]

df=pd.read_parquet('data.parquet', engine='pyarrow')

def fit_peak(ene, beam_energy):
    bin_width = 0.5
    x_max = 2 * beam_energy
    nbins = int(x_max / bin_width)

    if nbins < 10:  # need comfortably more bins than fit parameters
        raise ValueError(
            f"Too few bins ({nbins}) (beam_energy={beam_energy}); skipping fit"
        )
    bins = np.linspace(0, x_max, nbins + 1)
    h, edges = np.histogram(ene, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:]) #center of each bin
    p0 = [h.max(), np.mean(ene), np.std(ene, ddof=1)]

    popt, pcov = curve_fit(
            Gauss,
            centers,
            h,
            p0=p0,
            maxfev=10000
        )

    perr = np.sqrt(np.diag(pcov))

    return {
        "h": h, "centers": centers, "bin_width": bin_width, "x_max": x_max,
        "popt": popt, "perr": perr,
        "mean": popt[1], "err_mean": perr[1],
        "std": popt[2], "err_std": perr[2],
    }


def plot(ax, results, x_min, color, title):

    popt, perr = results["popt"],results["perr"]
    x_fit = np.linspace(x_min, results["x_max"], 500)

    ax.bar(
            results["centers"],
            results["h"],
            width=results["bin_width"],
            alpha=0.5,   #transparency of the bar
            color= color,
            label="data"
        )
    ax.plot(
        x_fit,
        Gauss(x_fit, *results["popt"]),
        color,
        label="Gauss fit"
    )
    ax.set_title(title)
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Counts")
    
    ax.text(
    0.05, 0.95,
    f"A = {popt[0]:.2f} ± {perr[0]:.2f}\n"
    f"μ = {popt[1]:.2f} ± {perr[1]:.2f}\n"
    f"σ = {popt[2]:.2f} ± {perr[2]:.2f}",
    transform=ax.transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
)
    ax.legend()
     
     

def print_plot(run, cal_s=1, cal_c=1):
    df_run = df.loc[(df["run"] == run) & (df["is_positron"] == 1)].copy()

    beam_energies = df_run["energy"].dropna().unique()

    if len(beam_energies) != 1:
        raise ValueError(
            f"Expected one beam energy for run {run}, "
            f"found {beam_energies}"
        )

    beam_energy = float(beam_energies[0])  

    ene_s = cal_s * df_run[sipm_s_cols].sum(axis=1)
    ene_c = cal_c * df_run[sipm_c_cols].sum(axis=1)

    if ene_s.size ==0 or ene_c.size ==0:
        raise ValueError(f"No events survive selection cuts for run {run}")

    results_s = fit_peak(ene_s, beam_energy)
    results_c = fit_peak(ene_c, beam_energy)

    mean_s = results_s["mean"]
    std_s = results_s["std"]
    mean_c = results_c["mean"]
    std_c = results_c["std"]

    x_min_s = beam_energy-std_s
    x_min_c = beam_energy-std_c

    err_mean_s = results_s["err_mean"]
    err_std_s = results_s["err_std"]

    err_mean_c = results_c["err_mean"]
    err_std_c = results_c["err_std"]
    
    x_fit_s = np.linspace(x_min_s, results_s["x_max"], 500)
    x_fit_c = np.linspace(x_min_c, results_c["x_max"], 500)

    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(12, 5))

    plot(ax_s, results_s, x_min_s, "red", "Scintillating")
    plot(ax_c, results_c, x_min_c, "blue", "Cherenkov")

    fig.suptitle(f"Run {run}, Energy {beam_energy}: Cherenkov and Scintillating energy distributions")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"histo_run_{run}.png"))
    plt.close(fig)

    return mean_s, err_mean_s, mean_c, err_mean_c,  std_s, err_std_s, std_c, err_std_c 


def calibrate(run):

    df_run = df.loc[(df["run"] == run) & (df["is_positron"] == 1)].copy()
    
    beam_energies = df_run["energy"].dropna().unique()
    if len(beam_energies) != 1:
                raise ValueError(
                    f"Expected one beam energy for run {run}, "
                    f"found {beam_energies}"
                )
    beam_energy = float(beam_energies[0])

    mean_s, err_mean_s, mean_c, err_mean_c, std_s, err_std_s, std_c, err_std_c = print_plot(run,1,1)

    cal_c = beam_energy / mean_c
    cal_s = beam_energy / mean_s

    return cal_s, cal_c


#run 185 = 20
cal_s, cal_c = calibrate(185)
print_plot(192, cal_s, cal_c)
     
