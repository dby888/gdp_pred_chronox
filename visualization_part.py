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

def draw_line_chart(Chronos_with_NFCI, Chronos_without_NFCI, WEO, truth, years, save_path, country_code,title):
    # Define color mapping with new style requirements
    color_map = {
        'Chronos_with_NFCI': 'red',           # Solid red with marker (wider line)
        'Chronos_without_NFCI': 'black',      # Solid black with marker (thin line)
        'WEO': '#1F77B4',                     # Light blue, short dash, no marker
        'NFCI': 'yellow',  # Solid red with marker (wider line)
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
    plt.title(f"{iso3_to_country_name(country_code)} - {title}", fontsize=18)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot for {iso3_to_country_name(country_code)} to {save_path}")

def draw_line_chart2(Chronos_with_NFCI, Chronos_without_NFCI, WEO, truth, years, save_path, country_code, title):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Define color mapping
    color_map = {
        'Chronos_with_NFCI': 'red',
        'Chronos_without_NFCI': 'black',
        'WEO': '#1F77B4',
        'NFCI': 'yellow',
        'Truth': 'grey'
    }

    # Load and compute NFCI (Q1 avg pct change)
    nfci_df = pd.read_csv('NFCI.csv', parse_dates=['observation_date'])
    nfci_df['Year'] = nfci_df['observation_date'].dt.year
    nfci_df['Month'] = nfci_df['observation_date'].dt.month
    nfci_q1 = nfci_df[
        (nfci_df['Month'] <= 3) & (nfci_df['Year'] >= 1979) & (nfci_df['Year'] <= years[-1] + 1)]
    nfci_yearly = nfci_q1.groupby('Year')['NFCI'].apply(lambda s: s.pct_change().dropna().mean())
    nfci_vals = [nfci_yearly.get(y, np.nan) for y in years]

    # Plotting with dual y-axis
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()

    # Left Y-axis: GDP related
    ax1.plot(years, Chronos_with_NFCI, label='Chronos with NFCI',
             color=color_map['Chronos_with_NFCI'], linewidth=3, marker='o', markersize=7)

    ax1.plot(years, Chronos_without_NFCI, label='Chronos without NFCI',
             color=color_map['Chronos_without_NFCI'], linewidth=1.5, marker='o', markersize=7)

    ax1.plot(years, WEO, label='WEO',
             color=color_map['WEO'], linestyle=(0, (3, 2)), linewidth=1.5)

    ax1.plot(years, truth, label='Ground Truth',
             color=color_map['Truth'], linestyle='solid', linewidth=1.5)

    ax1.set_ylabel("GDP Value", fontsize=14)
    ax1.set_xlabel("Year", fontsize=14)

    # Right Y-axis: NFCI
    ax2.plot(years, nfci_vals, label='NFCI', color=color_map['NFCI'], linestyle='solid', linewidth=2)
    ax2.set_ylabel("NFCI Change", fontsize=14)

    # Ticks, labels, legend
    ax1.set_xticks(np.arange(min(years), max(years) + 1, step=2))
    plt.title(f"{iso3_to_country_name(country_code)} - {title}", fontsize=18)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=14)

    ax1.grid(True)
    fig.tight_layout()

    # Save plot
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved dual-axis plot for {iso3_to_country_name(country_code)} to {save_path}")
