import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize

#pip install html5lib
table_MN = pd.read_html('twiki.html', match='Energy')
for df in table_MN:
 print(df.head())

def get_energy(html_file):
    table_MN = pd.read_html(html_file, match='Energy')
    run_energy_map = {} 
    for df in table_MN:
        run_energy_map.update( dict(zip(df.iloc[:, 1], df.iloc[:, 0])))
    return run_energy_map    
    
energy_map = get_energy('twiki.html')
print(energy_map)