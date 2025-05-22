import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycountry

def iso3_to_country_name(iso3_code):
    try:
        country = pycountry.countries.get(alpha_3=iso3_code)
        return country.name if country else None
    except Exception as e:
        return None

def draw_line_chart(Chronos_with_NFCI, Chronos_without_NFCI, WEO, truth, years, save_path, country_code):
    # Define color mapping with new style requirements
    color_map = {
        'Chronos_with_NFCI': 'red',           # Solid red with marker (wider line)
        'Chronos_without_NFCI': 'black',      # Solid black with marker (thin line)
        'WEO': '#1F77B4',                     # Light blue, short dash, no marker
        'Truth': 'grey'                       # Grey, thin solid, no marker
    }

    # Plotting
    plt.figure(figsize=(20, 10))

    plt.plot(years, Chronos_with_NFCI, label='Chronos with NFCI',
             color=color_map['Chronos_with_NFCI'], linewidth=3, marker='o', markersize=7)

    plt.plot(years, Chronos_without_NFCI, label='Chronos without NFCI',
             color=color_map['Chronos_without_NFCI'], linewidth=1.5, marker='o', markersize=7)

    plt.plot(years, WEO, label='WEO',
             color=color_map['WEO'], linestyle=(0, (3, 2)), linewidth=1.5)

    plt.plot(years, truth, label='Ground Truth',
             color=color_map['Truth'], linestyle='solid', linewidth=1.5)

    plt.xticks(np.arange(min(years), max(years) + 1, step=2))
    plt.title(f"{iso3_to_country_name(country_code)} - NGDP_PCH", fontsize=18)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot for {iso3_to_country_name(country_code)} to {save_path}")