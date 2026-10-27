import os
import re
import uproot
import numpy as np
import pandas as pd
import energy


#This script builds a database from the .root files and place it in a parquet file for later analysis
#Needs pyarrow for parquet

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

#branches needed to perform particles selection
selection_branches = [
    "XDWC1", "YDWC1", "PShower", "MCounter", "TriggerMask",
    "C1", "C2", "C1_ped", "C2_ped"
] 

all_cols = sipm_s_cols + sipm_c_cols + selection_branches

#html file containing the nominal energy values
html_path = 'twiki.html'
energy_map = energy.get_energy(html_path)


#defines the conditions to have positrons
def positron_conditions(df):

    sigma_ped_1 = df["C1_ped"].std()
    sigma_ped_2 = df["C2_ped"].std()

    mip_events = df.loc[
        df["MCounter"] > 200,
        "PShower"
    ]

    if mip_events.empty:
        raise ValueError("No MIP events found (MCounter > 200); cannot compute MIP threshold")

    mip_val = mip_events.mean()

    conditions = (
        #geometrical conditions to take events fully contained in the central tower
        (df["XDWC1"] > -4)
        & (df["XDWC1"] < 4)
        & (df["YDWC1"] > -4)
        & (df["YDWC1"] < 4)
        #set-up conditions
        & (df["PShower"] > 3 * mip_val)
        & (df["TriggerMask"] == 5)
        & (df["C1"] > 3 * sigma_ped_1)
        & (df["C2"] > 3 * sigma_ped_2)
    )

    return conditions


directory = 'run'

#extract the run number from the file name
def get_run(filename):
    return int(re.search(r'run(\d+)', filename).group(1))

#sorts files by run
files = sorted(
    [
        entry for entry in os.scandir(directory)
        if entry.is_file() and entry.name.endswith(".root")
    ],
    key=lambda entry: get_run(entry.name)
)

all_runs_df = []
run_index=0

for entry in files :
    filepath = entry.path if hasattr(entry, 'path') else entry
    tree = uproot.open(entry)["Ftree"]

    arrays= tree.arrays(all_cols, library="np") # returns a dictionary / or awkward array 
    df = pd.DataFrame(arrays) #one row per event, events for a single run

    positron_cond = positron_conditions(df)
    muon_cond = df["MCounter"] > 200

    #defines the labels to identify positrons and muons
    df['is_positron'] = np.where(positron_cond, 1, 0)
    df['is_muon'] = np.where(muon_cond, 1, 0)
    df['particle_id'] = np.select([positron_cond, muon_cond], [1,2], default=0) #positron=1, muons=2, other particles=0

    run = get_run(filepath)
    run_index += 1
    energy = energy_map.get(run)

    # Skip the run if energy is missing, NaN, or not numeric
    if not isinstance(energy, (int, float, np.number)) or not np.isfinite(energy):
        print(f"Skipping run {run}: invalid energy = {energy}")
        continue

    df.insert(0, "run_index", run_index) # first column is run_index
    df.insert(1, "run", run) # second column is run number
    df.insert(2, "energy", energy) # third column is beam energy

    all_runs_df.append(df)

data= pd.concat(all_runs_df, ignore_index=True) #single df for all runs

data.to_parquet("data.parquet")   
print(data.shape)

data.to_csv("data.csv", index=False)
data = pd.read_csv("data.csv") 
       


