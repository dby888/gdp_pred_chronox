# -*- coding: utf-8 -*-

# !pip install git+https://github.com/amazon-science/chronos-forecasting.git
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from chronos import BaseChronosPipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize the Chronos pipeline using Amazon's Chronos T5-small model
# chronos_pipeline = BaseChronosPipeline.from_pretrained(
#     "amazon/chronos-t5-large",
#     device_map="cpu",  # use "cpu" for CPU inference (or "mps" for Apple Silicon)
#     torch_dtype=torch.bfloat16,
# )
local_path = r"D:\huggingFaceModels\chronos-t5-large"
chronos_pipeline = BaseChronosPipeline.from_pretrained(
    local_path,
    device_map="cuda",  # use GPU (CUDA)
    torch_dtype=torch.float16,
)

# ============================
# 1. User settings & parameters
# ----------------------------
# Fill in your list of country codes here.
# countries = ['USA','DEU','GBR','FRA','AUS','ITA','CAN','BRA']
countries = ['USA']
target_subject = 'NGDP_RPCH'         # Change target subject if needed
start_year = 2007
end_year = 2010

# Option: Use nothing but NGDP_RPCH, so leave selected_subject_codes empty.
selected_subject_codes = []

# Truth file (assumed constant for all forecast years)
truth_file = 'copieofWEO2024.xlsx'
truth_df = pd.read_excel(truth_file, skiprows=1)
truth_df.columns = truth_df.columns.astype(str)

# Global dictionaries to store errors for each forecast year across all countries.
# For each forecast year, these lists will accumulate the error from each country.
global_chronos_errors = {}
global_organism_errors = {}

for file_year in range(start_year, end_year):
    global_chronos_errors[file_year] = []
    global_organism_errors[file_year] = []

# Optionally, store each country's forecast DataFrame for later use.
country_forecasts = {}

# ============================
# Loop over each country in the list
# ----------------------------
for country in countries:
    print(f"\nProcessing country: {country}")

    # Local dictionaries for the current country
    chronos_errors = {}             # keys: forecast year, values: error for Chronos prediction
    organism_errors = {}            # keys: forecast year, values: error for Organism prediction
    organism_predictions_all = {}   # keys: forecast year, values: Organism prediction
    all_forecasts = pd.DataFrame()   # to collect forecasts over all forecast years

    covariate_series = None
    if country == "USA":
        # read NFCI
        # Read NFCI.csv
        nfci_df = pd.read_csv('NFCI.csv', parse_dates=['observation_date'])
        nfci_df['Year'] = nfci_df['observation_date'].dt.year
        nfci_df['Month'] = nfci_df['observation_date'].dt.month

        # Filter for Jan-Mar and from 1980 to 2024
        nfci_q1 = nfci_df[(nfci_df['Month'] <= 3) & (nfci_df['Year'] >= 1980) & (nfci_df['Year'] <= 2024)]


        # Group by year, compute mean growth within Jan-Mar, and take negative
        def mean_growth(series):
            return series.pct_change().dropna().mean()


        nfci_yearly = nfci_q1.groupby('Year')['NFCI'].apply(mean_growth).apply(lambda x: 30 * x)
        covariate_series = nfci_yearly
    # Loop over forecast years (2010 to 2023)
    for file_year in range(start_year, end_year):
        file_name = f"WEOApr{file_year}all.xlsx"
        print(f"\nProcessing file: {file_name} for country: {country}")

        # ============================
        # 2. Load and preprocess the data from the current file
        # ----------------------------
        df = pd.read_excel(file_name)
        year_cols = df.columns[2:]
        df_long = df.melt(id_vars=[df.columns[0], df.columns[1]],
                          value_vars=year_cols,
                          var_name='Year',
                          value_name='Value')
        df_long.columns = ['ISO', 'WEO Subject Code', 'Year', 'Value']
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

        # Use data only up to file_year for nowcasting
        stopping_year = file_year
        df_long_filtered = df_long[df_long['Year'] <= stopping_year]

        # ============================
        # 3. Filter by country and subjects, then pivot the data
        # ----------------------------
        if selected_subject_codes:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'].isin(selected_subject_codes))
        else:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'] == target_subject)

        df_filtered = df_long_filtered[filter_mask].copy()
        df_filtered['variable'] = df_filtered['ISO'] + "_" + df_filtered['WEO Subject Code'].astype(str)
        df_pivot = df_filtered.pivot(index='Year', columns='variable', values='Value').dropna()
        df_pivot = df_pivot.astype(float)

        # ============================
        # 4. Forecast using ChronosPipeline
        forecasts = {}
        # Dictionary to store the forecast standard deviation for the target subject (for plotting error bars)
        if 'forecast_stds' not in locals():
            forecast_stds = {}

        for col in df_pivot.columns:
            series = df_pivot[col].tolist()
            # Convert the historical series into a torch tensor (1D)
            context = torch.tensor(series, dtype=torch.float32)
            pred_length = 1  # forecast one step ahead

            cov_tensor = None
            if covariate_series is not None:
                # Align covariates to context years
                context_years = df_pivot.index.tolist()
                cov_values = []
                for year in context_years:
                    cov_value = covariate_series.get(year, np.nan)
                    cov_values.append(cov_value)
                cov_tensor = torch.tensor(cov_values, dtype=torch.float32).unsqueeze(-1)

                # Set num_samples to 50 for multiple forecast samples
            forecast_tensor = chronos_pipeline.predict(
                context,
                pred_length,
                num_samples=20,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                covariates=cov_tensor
            )
            # forecast_tensor_x = chronos_pipeline.predict(
            #     context,
            #     pred_length,
            #     num_samples=20,
            #     temperature=1.0,
            #     top_k=50,
            #     top_p=1.0,
            #
            # )
            # Extract all 50 forecast samples (assuming tensor shape [50, 1, 1])
            forecast_samples = [forecast_tensor[i][0][0].item() for i in range(forecast_tensor.size(0))]
            mean_forecast = np.mean(forecast_samples)
            std_forecast = np.std(forecast_samples)

            forecasts[col] = mean_forecast

            # If this is the target subject column, save its std for plotting
            if col == f"{country}_{target_subject}":
                forecast_stds[file_year] = std_forecast

        forecast_df = pd.DataFrame(forecasts, index=[file_year])
        all_forecasts = pd.concat([all_forecasts, forecast_df])

        # ============================
        # 5. Error analysis for the target subject
        # ----------------------------
        col_key = f"{country}_{target_subject}"
        var_prediction = forecast_df.loc[file_year, col_key]

        # Organism's prediction is taken from the current file (column for file_year)
        organism_df = pd.read_excel(file_name)
        organism_df.columns = organism_df.columns.astype(str)
        organism_row = organism_df[(organism_df['ISO'] == country) &
                                   (organism_df['WEO Subject Code'] == target_subject)]
        organism_prediction = organism_row[str(file_year)].values[0]
        organism_predictions_all[file_year] = float(organism_prediction)

        # Extract the true value from the truth file for the same forecast year
        truth_row = truth_df[(truth_df['ISO'] == country) &
                             (truth_df['WEO Subject Code'] == target_subject)]
        truth_value = truth_row[str(file_year)].values[0]

        truth_value = float(truth_value)
        organism_prediction = float(organism_prediction)
        var_prediction = float(var_prediction)

        error_var = abs(truth_value - var_prediction)
        error_organism = abs(truth_value - organism_prediction)

        chronos_errors[file_year] = error_var
        organism_errors[file_year] = error_organism

        # Update the global dictionaries (to later compute RMSE across countries per year)
        global_chronos_errors[file_year].append(error_var)
        global_organism_errors[file_year].append(error_organism)

    # ============================
    # Plotting for the current country
    # ----------------------------

    # Combined Bar Plot: Prediction Errors Over the Years for this country
    years = sorted(chronos_errors.keys())
    chronos_error_values = [chronos_errors[yr] for yr in years]
    organism_error_values = [organism_errors[yr] for yr in years]


    x = np.arange(len(years))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, chronos_error_values, width, label='Chronos Prediction Error', color='blue')
    plt.bar(x + width/2, organism_error_values, width, label='Organism Prediction Error', color='orange')
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.xticks(x, years)
    plt.title(f'Prediction Errors for {target_subject} ({country}) Over the Years')
    plt.legend()
    # 保存图像
    save_path = f"figure/{country}_{target_subject}_error.png"
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

    # Overall RMSE: Compute RMSE over all forecast years for this country
    rmse_chronos = np.sqrt(np.mean(np.square(chronos_error_values)))
    rmse_organism = np.sqrt(np.mean(np.square(organism_error_values)))

    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos RMSE', 'Organism RMSE'], [rmse_chronos, rmse_organism], color=['blue', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE for {target_subject} Predictions ({country})')
    # 保存图像
    save_path = f"figure/{country}_{target_subject}_rmse.png"
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")


    # Additional Plot: NGDP_RPCH Prediction vs. Organism vs. Truth Over the Years for this country
    target_col = f"{country}_{target_subject}"
    years = sorted(all_forecasts.index)

    # Forecast mean values from Chronos and corresponding uncertainty (std dev)
    forecast_values = [all_forecasts.loc[year, target_col] for year in years]
    forecast_std_values = [forecast_stds[year] for year in years]

    # Organism predictions and Truth values
    organism_values = [organism_predictions_all[year] for year in years]
    truth_values = []
    for year in years:
        truth_row = truth_df[(truth_df['ISO'] == country) &
                              (truth_df['WEO Subject Code'] == target_subject)]
        truth_val = float(truth_row[str(year)].values[0])
        truth_values.append(truth_val)

    plt.figure(figsize=(10, 6))

    # Plot Chronos predictions as a line with markers.
    plt.plot(years, forecast_values, marker='o', linestyle='-', color='blue',
            label='Chronos Prediction (mean)')

    # Create a shaded band for the uncertainty (mean ± std)
    forecast_arr = np.array(forecast_values)
    std_arr = np.array(forecast_std_values)
    lower_bound = forecast_arr - std_arr
    upper_bound = forecast_arr + std_arr
    plt.fill_between(years, lower_bound, upper_bound, color='blue', alpha=0.2,
                    label='Chronos Uncertainty (±1 std)')

    # Plot Organism prediction and Truth as continuous lines with markers.
    plt.plot(years, organism_values, marker='x', linestyle='-', color='orange',
            label='Organism Prediction')
    plt.plot(years, truth_values, marker='s', linestyle='-', color='green',
            label='Truth')

    plt.xlabel('Forecast Year')
    plt.ylabel('NGDP_RPCH Value')
    plt.title(f'NGDP_RPCH Prediction vs. Organism vs. Truth for {country} Through the Years')
    plt.legend()
    plt.grid(True)
    # 保存图像
    save_path = f"figure/{country}_{target_subject}pred_vs_true.png"
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

    print(f"All forecasts over years for {country}:")
    print(all_forecasts)

    # Optionally store this country's forecast data
    country_forecasts[country] = all_forecasts

# ============================
# Additional Plot: RMSE Over All Countries for Each Forecast Year
# ----------------------------
years = sorted(global_chronos_errors.keys())
rmse_chronos_all = []
rmse_organism_all = []

for year in years:
    # For each forecast year, compute RMSE across the countries (i.e. mean squared error across countries)
    errors_chronos = global_chronos_errors[year]
    errors_organism = global_organism_errors[year]
    rmse_year_chronos = np.sqrt(np.mean(np.square(errors_chronos)))
    rmse_year_organism = np.sqrt(np.mean(np.square(errors_organism)))
    rmse_chronos_all.append(rmse_year_chronos)
    rmse_organism_all.append(rmse_year_organism)

plt.figure(figsize=(10, 6))
plt.plot(years, rmse_chronos_all, marker='o', label='Chronos RMSE (All Countries)')
plt.plot(years, rmse_organism_all, marker='x', label='Organism RMSE (All Countries)')
plt.xlabel('Forecast Year')
plt.ylabel('RMSE Across Countries')
plt.title('RMSE Over All Countries for Each Forecast Year')
plt.legend()
plt.grid(True)
plt.show()



"""NEXT ONE IS FOR NGDP AND DOES NOT COUNT 2020 BECAUSE MISSING FROM THE SOURCE FILE"""

''' THIS ONE DISCARDS 2020 '''



# ============================
# 1. User settings & parameters
# ----------------------------
# Fill in your list of country codes here.
countries = ['CHN','USA','DEU','FRA','AUS']
target_subject = 'NGDP'         # Change target subject if needed

# Option: Use nothing but NGDP, so leave selected_subject_codes empty.
selected_subject_codes = []

# Truth file (assumed constant for all forecast years)
truth_file = 'copieofWEO2024.xlsx'
truth_df = pd.read_excel(truth_file, skiprows=1)
truth_df.columns = truth_df.columns.astype(str)

# Global dictionaries to store errors for each forecast year across all countries.
# We now consider forecast years from 2013 to 2023 (inclusive).
global_chronos_errors = {}
global_organism_errors = {}
for file_year in range(2013, 2024):
    global_chronos_errors[file_year] = []
    global_organism_errors[file_year] = []

# Optionally, store each country's forecast DataFrame for later use.
country_forecasts = {}

# ============================
# Loop over each country in the list
# ----------------------------
for country in countries:
    print(f"\nProcessing country: {country}")

    # Local dictionaries for the current country
    chronos_errors = {}             # keys: forecast year, values: error for Chronos prediction
    organism_errors = {}            # keys: forecast year, values: error for Organism prediction
    organism_predictions_all = {}   # keys: forecast year, values: Organism prediction
    all_forecasts = pd.DataFrame()   # to collect forecasts over all forecast years
    forecast_stds = {}              # to store forecast uncertainty (std) for each year

    # Loop over forecast years (2013 to 2023)
    for file_year in range(2017, 2024):
        # If the year is 2020, skip file reading and insert placeholders
        if file_year == 2020:
            print(f"\nSkipping file for year {file_year} (missing row). Inserting placeholder values.")
            # Create a placeholder forecast row with NaN for the target column
            placeholder_forecast = {f"{country}_{target_subject}": np.nan}
            forecast_df = pd.DataFrame(placeholder_forecast, index=[file_year])
            all_forecasts = pd.concat([all_forecasts, forecast_df])

            # Set placeholder values in error and prediction dictionaries
            chronos_errors[file_year] = np.nan
            organism_errors[file_year] = np.nan
            forecast_stds[file_year] = np.nan
            organism_predictions_all[file_year] = np.nan

            # Update global error dictionaries with a NaN placeholder
            global_chronos_errors[file_year].append(np.nan)
            global_organism_errors[file_year].append(np.nan)
            continue  # Skip the rest of the loop for 2020

        file_name = f"WEOApr{file_year}sentiment.xlsx"
        print(f"\nProcessing file: {file_name} for country: {country}")

        # ============================
        # 2. Load and preprocess the data from the current file
        # ----------------------------
        df = pd.read_excel(file_name)
        year_cols = df.columns[2:]
        df_long = df.melt(id_vars=[df.columns[0], df.columns[1]],
                          value_vars=year_cols,
                          var_name='Year',
                          value_name='Value')
        df_long.columns = ['ISO', 'WEO Subject Code', 'Year', 'Value']
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

        # Use data only up to file_year for nowcasting
        stopping_year = file_year
        df_long_filtered = df_long[df_long['Year'] <= stopping_year]

        # ============================
        # 3. Filter by country and subjects, then pivot the data
        # ----------------------------
        if selected_subject_codes:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'].isin(selected_subject_codes))
        else:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'] == target_subject)

        df_filtered = df_long_filtered[filter_mask].copy()
        df_filtered['variable'] = df_filtered['ISO'] + "_" + df_filtered['WEO Subject Code'].astype(str)
        df_pivot = df_filtered.pivot(index='Year', columns='variable', values='Value').dropna()
        df_pivot = df_pivot.astype(float)

        # ============================
        # 4. Forecast using ChronosPipeline
        forecasts = {}
        for col in df_pivot.columns:
            series = df_pivot[col].tolist()
            # Convert the historical series into a torch tensor (1D)
            context = torch.tensor(series, dtype=torch.float32)
            pred_length = 1  # forecast one step ahead

            # Set num_samples to 20 for multiple forecast samples
            forecast_tensor = chronos_pipeline.predict(
                context,
                pred_length,
                num_samples=50,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )

            # Extract all forecast samples (assuming tensor shape [num_samples, 1, 1])
            forecast_samples = [forecast_tensor[i][0][0].item() for i in range(forecast_tensor.size(0))]
            median_forecast = np.median(forecast_samples)  # Using the median to reduce outlier impact
            std_forecast = np.std(forecast_samples)

            forecasts[col] = median_forecast

            # If this is the target subject column, save its std for plotting
            if col == f"{country}_{target_subject}":
                forecast_stds[file_year] = std_forecast

        forecast_df = pd.DataFrame(forecasts, index=[file_year])
        all_forecasts = pd.concat([all_forecasts, forecast_df])

        # ============================
        # 5. Error analysis for the target subject
        # ----------------------------
        col_key = f"{country}_{target_subject}"
        var_prediction = forecast_df.loc[file_year, col_key]

        # Organism's prediction is taken from the current file (column for file_year)
        organism_df = pd.read_excel(file_name)
        organism_df.columns = organism_df.columns.astype(str)
        organism_row = organism_df[(organism_df['ISO'] == country) &
                                   (organism_df['WEO Subject Code'] == target_subject)]
        organism_prediction = organism_row[str(file_year)].values[0]
        organism_predictions_all[file_year] = float(organism_prediction)

        # Extract the true value from the truth file for the same forecast year
        truth_row = truth_df[(truth_df['ISO'] == country) &
                             (truth_df['WEO Subject Code'] == target_subject)]
        truth_value = truth_row[str(file_year)].values[0]

        truth_value = float(truth_value)
        organism_prediction = float(organism_prediction)
        var_prediction = float(var_prediction)

        error_var = abs(truth_value - var_prediction)
        error_organism = abs(truth_value - organism_prediction)

        chronos_errors[file_year] = error_var
        organism_errors[file_year] = error_organism

        # Update the global dictionaries (to later compute RMSE across countries per year)
        global_chronos_errors[file_year].append(error_var)
        global_organism_errors[file_year].append(error_organism)

    # ============================
    # Plotting for the current country
    # ----------------------------
    # For the error bar plots, update x-axis labels to show that 2020 is missing.
    years = sorted(chronos_errors.keys())
    chronos_error_values = [chronos_errors[yr] for yr in years]
    organism_error_values = [organism_errors[yr] for yr in years]

    x = np.arange(len(years))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, chronos_error_values, width, label='Chronos Prediction Error', color='blue')
    plt.bar(x + width/2, organism_error_values, width, label='Organism Prediction Error', color='orange')
    # Update tick labels to indicate missing 2020
    x_labels = [f"{yr}" if yr != 2020 else "2020\n(Missing)" for yr in years]
    plt.xticks(x, x_labels)
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.title(f'Prediction Errors for {target_subject} ({country}) Over the Years')
    plt.legend()
    plt.show()

    # Overall RMSE: Compute RMSE over all forecast years for this country
    # (NaN values will be ignored in the RMSE calculation by using np.nanmean)
    rmse_chronos = np.sqrt(np.nanmean(np.square(chronos_error_values)))
    rmse_organism = np.sqrt(np.nanmean(np.square(organism_error_values)))

    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos RMSE', 'Organism RMSE'], [rmse_chronos, rmse_organism], color=['blue', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE for {target_subject} Predictions ({country})')
    plt.show()

    # Additional Plot: NGDP Prediction vs. Organism vs. Truth Over the Years for this country
    target_col = f"{country}_{target_subject}"
    years = sorted(all_forecasts.index)

    # Forecast mean values from Chronos and corresponding uncertainty (std dev)
    forecast_values = [all_forecasts.loc[year, target_col] for year in years]
    forecast_std_values = [forecast_stds.get(year, np.nan) for year in years]

    # Organism predictions and Truth values
    organism_values = [organism_predictions_all.get(year, np.nan) for year in years]
    truth_values = []
    for year in years:
        # For 2020 we insert a placeholder label if needed
        if year == 2020:
            truth_values.append(np.nan)
        else:
            truth_row = truth_df[(truth_df['ISO'] == country) &
                                  (truth_df['WEO Subject Code'] == target_subject)]
            truth_val = float(truth_row[str(year)].values[0])
            truth_values.append(truth_val)

    plt.figure(figsize=(10, 6))

    # Plot Chronos predictions as a line with markers.
    plt.plot(years, forecast_values, marker='o', linestyle='-', color='blue',
             label='Chronos Prediction (mean)')

    # Create a shaded band for the uncertainty (mean ± std)
    forecast_arr = np.array(forecast_values, dtype=np.float64)
    std_arr = np.array(forecast_std_values, dtype=np.float64)
    lower_bound = forecast_arr - std_arr
    upper_bound = forecast_arr + std_arr
    plt.fill_between(years, lower_bound, upper_bound, color='blue', alpha=0.2,
                     label='Chronos Uncertainty (±1 std)')

    # Plot Organism prediction and Truth as continuous lines with markers.
    plt.plot(years, organism_values, marker='x', linestyle='-', color='orange',
             label='Organism Prediction')
    plt.plot(years, truth_values, marker='s', linestyle='-', color='green',
             label='Truth')

    # Annotate the missing 2020 value if desired
    if 2020 in years:
        idx_2020 = years.index(2020)
        plt.text(2020, np.nanmean([v for v in forecast_values if not np.isnan(v)]),
                 "Missing", color='red', ha='center', va='bottom')

    plt.xlabel('Forecast Year')
    plt.ylabel('NGDP Value')
    plt.title(f'NGDP Prediction vs. Organism vs. Truth for {country} Through the Years')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"All forecasts over years for {country}:")
    print(all_forecasts)

    # Optionally store this country's forecast data
    country_forecasts[country] = all_forecasts

# ============================
# Additional Plot: RMSE Over All Countries for Each Forecast Year
# ----------------------------
years = sorted(global_chronos_errors.keys())
rmse_chronos_all = []
rmse_organism_all = []

for year in years:
    # For each forecast year, compute RMSE across the countries (ignoring NaN values)
    errors_chronos = np.array(global_chronos_errors[year], dtype=np.float64)
    errors_organism = np.array(global_organism_errors[year], dtype=np.float64)
    rmse_year_chronos = np.sqrt(np.nanmean(np.square(errors_chronos)))
    rmse_year_organism = np.sqrt(np.nanmean(np.square(errors_organism)))
    rmse_chronos_all.append(rmse_year_chronos)
    rmse_organism_all.append(rmse_year_organism)

plt.figure(figsize=(10, 6))
plt.plot(years, rmse_chronos_all, marker='o', label='Chronos RMSE (All Countries)')
plt.plot(years, rmse_organism_all, marker='x', label='Organism RMSE (All Countries)')
plt.xlabel('Forecast Year')
plt.ylabel('RMSE Across Countries')
plt.title('RMSE Over All Countries for Each Forecast Year')
plt.legend()
plt.grid(True)
plt.show()





"""NEXT ONE APR->OCT NGDP




"""

# ============================
# 1. User settings & parameters
# ----------------------------
countries = ['CHN','USA','DEU','FRA','AUS','ESP','ITA','NLD','GBR','CAN','TUR','RUS','JPN','IND','IDN','BRA','SAU']
target_subject = 'NGDP'
selected_subject_codes = []

truth_file = 'copieofWEO2024.xlsx'
truth_df = pd.read_excel(truth_file, skiprows=1)
truth_df.columns = truth_df.columns.astype(str)

global_chronos_errors = {}
global_organism_errors = {}
for file_year in range(2007, 2024):
    global_chronos_errors[file_year] = []
    global_organism_errors[file_year] = []

country_forecasts = {}

# ============================
# Loop over each country
# ----------------------------
for country in countries:
    print(f"\nProcessing country: {country}")

    chronos_errors = {}
    organism_errors = {}
    organism_predictions_all = {}
    all_forecasts = pd.DataFrame()
    forecast_stds = {}

    # Loop over forecast years
    for file_year in range(2018, 2024):
        if file_year == 2020:
            print(f"\nSkipping file for year {file_year} (missing row). Inserting placeholder values.")
            placeholder_forecast = {f"{country}_{target_subject}": np.nan}
            forecast_df = pd.DataFrame(placeholder_forecast, index=[file_year])
            all_forecasts = pd.concat([all_forecasts, forecast_df])

            chronos_errors[file_year] = np.nan
            organism_errors[file_year] = np.nan
            forecast_stds[file_year] = np.nan
            organism_predictions_all[file_year] = np.nan

            global_chronos_errors[file_year].append(np.nan)
            global_organism_errors[file_year].append(np.nan)
            continue

        # ========= Modification: Use October file from the previous year for forecasting =========
        source_file_name = f"WEOOct{file_year - 1}all.xlsx"
        print(f"\nProcessing source file: {source_file_name} for country: {country}")

        df = pd.read_excel(source_file_name)
        year_cols = df.columns[2:]
        df_long = df.melt(id_vars=[df.columns[0], df.columns[1]],
                          value_vars=year_cols,
                          var_name='Year',
                          value_name='Value')
        df_long.columns = ['ISO', 'WEO Subject Code', 'Year', 'Value']
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

        # Use data only up to file_year (e.g. 2019)
        stopping_year = file_year
        df_long_filtered = df_long[df_long['Year'] <= stopping_year]

        if selected_subject_codes:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'].isin(selected_subject_codes))
        else:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'] == target_subject)

        df_filtered = df_long_filtered[filter_mask].copy()
        df_filtered['variable'] = df_filtered['ISO'] + "_" + df_filtered['WEO Subject Code'].astype(str)
        df_pivot = df_filtered.pivot(index='Year', columns='variable', values='Value').dropna()
        df_pivot = df_pivot.astype(float)

        # ========= Forecast using ChronosPipeline =========
        forecasts = {}
        for col in df_pivot.columns:
            series = df_pivot[col].tolist()
            context = torch.tensor(series, dtype=torch.float32)
            pred_length = 1

            forecast_tensor = chronos_pipeline.predict(
                context,
                pred_length,
                num_samples=50,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )

            forecast_samples = [forecast_tensor[i][0][0].item() for i in range(forecast_tensor.size(0))]
            median_forecast = np.median(forecast_samples)
            std_forecast = np.std(forecast_samples)

            forecasts[col] = median_forecast

            if col == f"{country}_{target_subject}":
                forecast_stds[file_year] = std_forecast

        forecast_df = pd.DataFrame(forecasts, index=[file_year])
        all_forecasts = pd.concat([all_forecasts, forecast_df])

        # ========= Error analysis =========
        col_key = f"{country}_{target_subject}"
        var_prediction = forecast_df.loc[file_year, col_key]

        # ---------- Extract Organism forecast from the original April file ----------
        organism_file_name = f"WEOOct{file_year}all.xlsx"
        organism_df = pd.read_excel(organism_file_name)
        organism_df.columns = organism_df.columns.astype(str)
        organism_row = organism_df[(organism_df['ISO'] == country) &
                                   (organism_df['WEO Subject Code'] == target_subject)]
        organism_prediction = organism_row[str(file_year)].values[0]
        organism_predictions_all[file_year] = float(organism_prediction)

        # ---------- Extract the true value from the truth file ----------
        truth_row = truth_df[(truth_df['ISO'] == country) &
                             (truth_df['WEO Subject Code'] == target_subject)]
        truth_value = truth_row[str(file_year)].values[0]

        truth_value = float(truth_value)
        organism_prediction = float(organism_prediction)
        var_prediction = float(var_prediction)

        error_var = abs(truth_value - var_prediction)
        error_organism = abs(truth_value - organism_prediction)

        chronos_errors[file_year] = error_var
        organism_errors[file_year] = error_organism

        global_chronos_errors[file_year].append(error_var)
        global_organism_errors[file_year].append(error_organism)

    # ============================
    # Plotting for the current country
    # ----------------------------
    # For the error bar plots, update x-axis labels to show that 2020 is missing.
    years = sorted(chronos_errors.keys())
    chronos_error_values = [chronos_errors[yr] for yr in years]
    organism_error_values = [organism_errors[yr] for yr in years]

    x = np.arange(len(years))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, chronos_error_values, width, label='Chronos Prediction Error', color='blue')
    plt.bar(x + width/2, organism_error_values, width, label='Organism Prediction Error', color='orange')
    # Update tick labels to indicate missing 2020
    x_labels = [f"{yr}" if yr != 2020 else "2020\n(Missing)" for yr in years]
    plt.xticks(x, x_labels)
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.title(f'Prediction Errors for {target_subject} ({country}) Over the Years')
    plt.legend()
    plt.show()

    # Overall RMSE: Compute RMSE over all forecast years for this country
    # (NaN values will be ignored in the RMSE calculation by using np.nanmean)
    rmse_chronos = np.sqrt(np.nanmean(np.square(chronos_error_values)))
    rmse_organism = np.sqrt(np.nanmean(np.square(organism_error_values)))

    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos RMSE', 'Organism RMSE'], [rmse_chronos, rmse_organism], color=['blue', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE for {target_subject} Predictions ({country})')
    plt.show()

    # Additional Plot: NGDP Prediction vs. Organism vs. Truth Over the Years for this country
    target_col = f"{country}_{target_subject}"
    years = sorted(all_forecasts.index)

    # Forecast mean values from Chronos and corresponding uncertainty (std dev)
    forecast_values = [all_forecasts.loc[year, target_col] for year in years]
    forecast_std_values = [forecast_stds.get(year, np.nan) for year in years]

    # Organism predictions and Truth values
    organism_values = [organism_predictions_all.get(year, np.nan) for year in years]
    truth_values = []
    for year in years:
        # For 2020 we insert a placeholder label if needed
        if year == 2020:
            truth_values.append(np.nan)
        else:
            truth_row = truth_df[(truth_df['ISO'] == country) &
                                  (truth_df['WEO Subject Code'] == target_subject)]
            truth_val = float(truth_row[str(year)].values[0])
            truth_values.append(truth_val)

    plt.figure(figsize=(10, 6))

    # Plot Chronos predictions as a line with markers.
    plt.plot(years, forecast_values, marker='o', linestyle='-', color='blue',
             label='Chronos Prediction (mean)')

    # Create a shaded band for the uncertainty (mean ± std)
    forecast_arr = np.array(forecast_values, dtype=np.float64)
    std_arr = np.array(forecast_std_values, dtype=np.float64)
    lower_bound = forecast_arr - std_arr
    upper_bound = forecast_arr + std_arr
    plt.fill_between(years, lower_bound, upper_bound, color='blue', alpha=0.2,
                     label='Chronos Uncertainty (±1 std)')

    # Plot Organism prediction and Truth as continuous lines with markers.
    plt.plot(years, organism_values, marker='x', linestyle='-', color='orange',
             label='Organism Prediction')
    plt.plot(years, truth_values, marker='s', linestyle='-', color='green',
             label='Truth')

    # Annotate the missing 2020 value if desired
    if 2020 in years:
        idx_2020 = years.index(2020)
        plt.text(2020, np.nanmean([v for v in forecast_values if not np.isnan(v)]),
                 "Missing", color='red', ha='center', va='bottom')

    plt.xlabel('Forecast Year')
    plt.ylabel('NGDP Value')
    plt.title(f'NGDP Prediction vs. Organism vs. Truth for {country} Through the Years')
    plt.legend()
    plt.grid(True)
    plt.show()# (Plotting and further processing follow here unchanged.)

    print(f"All forecasts over years for {country}:")
    print(all_forecasts)
    country_forecasts[country] = all_forecasts

# (Additional global RMSE plotting code follows unchanged.)

# ============================
# Additional Plot: RMSE Over All Countries for Each Forecast Year
# ----------------------------
years = sorted(global_chronos_errors.keys())
rmse_chronos_all = []
rmse_organism_all = []

for year in years:
    # For each forecast year, compute RMSE across the countries (ignoring NaN values)
    errors_chronos = np.array(global_chronos_errors[year], dtype=np.float64)
    errors_organism = np.array(global_organism_errors[year], dtype=np.float64)
    rmse_year_chronos = np.sqrt(np.nanmean(np.square(errors_chronos)))
    rmse_year_organism = np.sqrt(np.nanmean(np.square(errors_organism)))
    rmse_chronos_all.append(rmse_year_chronos)
    rmse_organism_all.append(rmse_year_organism)

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(years))  # x locations for the groups
width = 0.35  # width of the bars

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, rmse_chronos_all, width, label='Chronos RMSE (All Countries)')
plt.bar(x + width/2, rmse_organism_all, width, label='Organism RMSE (All Countries)')
plt.xlabel('Forecast Year')
plt.ylabel('RMSE Across Countries')
plt.title('RMSE Over All Countries for Each Forecast Year')
plt.xticks(x, years)
plt.legend()
plt.grid(True)
plt.show()

"""NEXT ONE OCT->APR NGDP_RPCH



"""


# ============================
# 1. User settings & parameters
# ----------------------------
countries = ['CHN','USA','DEU','FRA','SGP','ITA','GBR','CAN','JPN']
target_subject = 'NGDP_RPCH'
selected_subject_codes = []

# Set this flag to True to process 2020 normally, or False to skip 2020.
include_2020 = True

truth_file = 'copieofWEO2024.xlsx'
truth_df = pd.read_excel(truth_file, skiprows=1)
truth_df.columns = truth_df.columns.astype(str)

# Prepare global error dictionaries based on forecast years.
# Here we use years from 2018 to 2023 and decide on including 2020.
all_years = range(2007, 2024)
if include_2020:
    forecast_years = list(all_years)
else:
    forecast_years = [year for year in all_years if year != 2020]

global_chronos_errors = {}
global_organism_errors = {}
for file_year in forecast_years:
    global_chronos_errors[file_year] = []
    global_organism_errors[file_year] = []

country_forecasts = {}

# === ADD: Initialize a list to store RMSE results per country ===
rmse_results = []

# ============================
# Loop over each country
# ----------------------------
for country in countries:
    print(f"\nProcessing country: {country}")

    chronos_errors = {}
    organism_errors = {}
    organism_predictions_all = {}
    all_forecasts = pd.DataFrame()
    forecast_stds = {}

    # Loop over forecast years (2018 to 2023)
    for file_year in range(2007, 2024):
        # If file_year is 2020 and we are skipping it, insert placeholders.
        if file_year == 2020 and not include_2020:
            print(f"\nSkipping file for year {file_year} (missing row). Inserting placeholder values.")
            placeholder_forecast = {f"{country}_{target_subject}": np.nan}
            forecast_df = pd.DataFrame(placeholder_forecast, index=[file_year])
            all_forecasts = pd.concat([all_forecasts, forecast_df])

            chronos_errors[file_year] = np.nan
            organism_errors[file_year] = np.nan
            forecast_stds[file_year] = np.nan
            organism_predictions_all[file_year] = np.nan

            global_chronos_errors[file_year].append(np.nan)
            global_organism_errors[file_year].append(np.nan)
            continue

        # ========= Use October file from the previous year for forecasting =========
        source_file_name = f"WEOOct{file_year - 1}all.xlsx"
        print(f"\nProcessing source file: {source_file_name} for country: {country}")

        df = pd.read_excel(source_file_name)
        year_cols = df.columns[2:]
        df_long = df.melt(id_vars=[df.columns[0], df.columns[1]],
                          value_vars=year_cols,
                          var_name='Year',
                          value_name='Value')
        df_long.columns = ['ISO', 'WEO Subject Code', 'Year', 'Value']
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

        # Use data only up to file_year+1 (e.g. for 2019 use data up to 2020)
        stopping_year = file_year
        df_long_filtered = df_long[df_long['Year'] <= stopping_year]

        if selected_subject_codes:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'].isin(selected_subject_codes))
        else:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'] == target_subject)

        df_filtered = df_long_filtered[filter_mask].copy()
        df_filtered['variable'] = df_filtered['ISO'] + "_" + df_filtered['WEO Subject Code'].astype(str)
        df_pivot = df_filtered.pivot(index='Year', columns='variable', values='Value').dropna()
        df_pivot = df_pivot.astype(float)

        # ========= Forecast using ChronosPipeline =========
        forecasts = {}
        for col in df_pivot.columns:
            series = df_pivot[col].tolist()
            context = torch.tensor(series, dtype=torch.float32)
            pred_length = 1

            forecast_tensor = chronos_pipeline.predict(
                context,
                pred_length,
                num_samples=50,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )

            forecast_samples = [forecast_tensor[i][0][0].item() for i in range(forecast_tensor.size(0))]
            median_forecast = np.median(forecast_samples)
            std_forecast = np.std(forecast_samples)

            forecasts[col] = median_forecast

            if col == f"{country}_{target_subject}":
                forecast_stds[file_year] = std_forecast

        forecast_df = pd.DataFrame(forecasts, index=[file_year])
        all_forecasts = pd.concat([all_forecasts, forecast_df])

        # ========= Error analysis =========
        col_key = f"{country}_{target_subject}"
        var_prediction = forecast_df.loc[file_year, col_key]

        # ---------- Extract Organism forecast from the original April file ----------
        organism_file_name = f"WEOApr{file_year}sentiment.xlsx"
        organism_df = pd.read_excel(organism_file_name)
        organism_df.columns = organism_df.columns.astype(str)
        organism_row = organism_df[(organism_df['ISO'] == country) &
                                   (organism_df['WEO Subject Code'] == target_subject)]
        organism_prediction = organism_row[str(file_year)].values[0]
        organism_predictions_all[file_year] = float(organism_prediction)

        # ---------- Extract the true value from the truth file ----------
        truth_row = truth_df[(truth_df['ISO'] == country) &
                             (truth_df['WEO Subject Code'] == target_subject)]
        truth_value = truth_row[str(file_year)].values[0]

        truth_value = float(truth_value)
        organism_prediction = float(organism_prediction)
        var_prediction = float(var_prediction)

        error_var = abs(truth_value - var_prediction)
        error_organism = abs(truth_value - organism_prediction)

        chronos_errors[file_year] = error_var
        organism_errors[file_year] = error_organism

        global_chronos_errors[file_year].append(error_var)
        global_organism_errors[file_year].append(error_organism)

    # ============================
    # Plotting for the current country
    # ----------------------------
    years_list = sorted(chronos_errors.keys())
    chronos_error_values = [chronos_errors[yr] for yr in years_list]
    organism_error_values = [organism_errors[yr] for yr in years_list]

    x = np.arange(len(years_list))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, chronos_error_values, width, label='Chronos Prediction Error', color='blue')
    plt.bar(x + width/2, organism_error_values, width, label='Organism Prediction Error', color='orange')
    x_labels = [f"{yr}" if yr != 2020 else "2020\n(Missing)" for yr in years_list]
    plt.xticks(x, x_labels)
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.title(f'Prediction Errors for {target_subject} ({country}) Over the Years')
    plt.legend()
    plt.show()

    rmse_chronos = np.sqrt(np.nanmean(np.square(chronos_error_values)))
    rmse_organism = np.sqrt(np.nanmean(np.square(organism_error_values)))

    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos RMSE', 'Organism RMSE'], [rmse_chronos, rmse_organism], color=['blue', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE for {target_subject} Predictions ({country})')
    plt.show()

    # === ADD: Store the RMSE for the current country ===
    rmse_results.append({
        "Country": country,
        "Chronos RMSE": rmse_chronos,
        "Organism RMSE": rmse_organism
    })

    # Additional Plot: NGDP Prediction vs. Organism vs. Truth Over the Years for this country
    target_col = f"{country}_{target_subject}"
    years_list = sorted(all_forecasts.index)

    # Forecast mean values from Chronos and corresponding uncertainty (std dev)
    forecast_values = [all_forecasts.loc[year, target_col] for year in years_list]
    forecast_std_values = [forecast_stds.get(year, np.nan) for year in years_list]

    # Organism predictions and Truth values
    organism_values = [organism_predictions_all.get(year, np.nan) for year in years_list]
    truth_values = []
    for year in years_list:
        # If we're skipping 2020, set truth value to nan; otherwise, fetch from the truth file.
        if year == 2020 and not include_2020:
            truth_values.append(np.nan)
        else:
            truth_row = truth_df[(truth_df['ISO'] == country) &
                                  (truth_df['WEO Subject Code'] == target_subject)]
            truth_val = float(truth_row[str(year)].values[0])
            truth_values.append(truth_val)

    plt.figure(figsize=(10, 6))

    # Plot Chronos predictions as a line with markers.
    plt.plot(years_list, forecast_values, marker='o', linestyle='-', color='blue',
            label='Chronos Prediction (mean)')

    # Create a shaded band for the uncertainty (mean ± std)
    forecast_arr = np.array(forecast_values, dtype=np.float64)
    std_arr = np.array(forecast_std_values, dtype=np.float64)
    lower_bound = forecast_arr - std_arr
    upper_bound = forecast_arr + std_arr
    plt.fill_between(years_list, lower_bound, upper_bound, color='blue', alpha=0.2,
                    label='Chronos Uncertainty (±1 std)')

    # Plot Organism prediction and Truth as continuous lines with markers.
    plt.plot(years_list, organism_values, marker='x', linestyle='-', color='orange',
            label='Organism Prediction')
    plt.plot(years_list, truth_values, marker='s', linestyle='-', color='green',
            label='Truth')

    # Annotate 2020 as "Missing" only if it is not being processed
    if 2020 in years_list and not include_2020:
        plt.text(2020, np.nanmean([v for v in forecast_values if not np.isnan(v)]),
                "Missing", color='red', ha='center', va='bottom')

    plt.xlabel('Forecast Year')
    plt.ylabel('NGDP Value')
    plt.title(f'NGDP Prediction vs. Organism vs. Truth for {country} Through the Years')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"All forecasts over years for {country}:")
    print(all_forecasts)
    country_forecasts[country] = all_forecasts

# ============================
# AFTER PROCESSING ALL COUNTRIES: Save RMSE results to an Excel file
# ----------------------------
rmse_df = pd.DataFrame(rmse_results)
output_file = "rmse_summary_chronos.xlsx"
rmse_df.to_excel(output_file, index=False)
print(f"\nCountry RMSE summary saved to {output_file}")

# (Additional global RMSE plotting code follows unchanged.)

"""SAVES THE FORECASTS

"""



# 1) Gather all years (assumes every country has the same index)
all_years = sorted(next(iter(country_forecasts.values())).index)

# 2) Build a DataFrame: rows = years, columns = country codes
first_step_df = pd.DataFrame(index=all_years)
for country, df in country_forecasts.items():
    col_name = f"{country}_{target_subject}"
    # extract the single‐step forecast series
    first_step_df[country] = df[col_name]

first_step_df.index.name = 'Year'

# 3) Save to Excel
output_path = 'chronos_first_step_forecasts.xlsx'
first_step_df.to_excel(output_path)
print(f"First‐step forecasts saved to '{output_path}'")

# ============================
# Additional Plot: RMSE Over All Countries for Each Forecast Year
# ----------------------------
years = sorted(global_chronos_errors.keys())
rmse_chronos_all = []
rmse_organism_all = []

for year in years:
    # For each forecast year, compute RMSE across the countries (ignoring NaN values)
    errors_chronos = np.array(global_chronos_errors[year], dtype=np.float64)
    errors_organism = np.array(global_organism_errors[year], dtype=np.float64)
    rmse_year_chronos = np.sqrt(np.nanmean(np.square(errors_chronos)))
    rmse_year_organism = np.sqrt(np.nanmean(np.square(errors_organism)))
    rmse_chronos_all.append(rmse_year_chronos)
    rmse_organism_all.append(rmse_year_organism)

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(years))  # x locations for the groups
width = 0.35  # width of the bars

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, rmse_chronos_all, width, label='Chronos RMSE (All Countries)')
plt.bar(x + width/2, rmse_organism_all, width, label='Organism RMSE (All Countries)')
plt.xlabel('Forecast Year')
plt.ylabel('RMSE Across Countries')
plt.title('RMSE Over All Countries for Each Forecast Year')
plt.xticks(x, years)
plt.legend()
plt.grid(True)
plt.show()





"""NEXT ONE IS FOR RPCH AND COUNTS 2020"""

# ============================
# 1. User settings & parameters
# ----------------------------
# Fill in your list of country codes here.
countries = ['RUS','TUR','NZL','IRL','SWE','NOR','DNK','IND']
target_subject = 'NGDP_RPCH'         # Change target subject if needed

# Option: Use nothing but NGDP, so leave selected_subject_codes empty.
selected_subject_codes = []

# Truth file (assumed constant for all forecast years)
truth_file = 'copieofWEO2024.xlsx'
truth_df = pd.read_excel(truth_file, skiprows=1)
truth_df.columns = truth_df.columns.astype(str)

# Global dictionaries to store errors for each forecast year across all countries.
# We now consider forecast years from 2013 to 2023 (inclusive).
global_chronos_errors = {}
global_organism_errors = {}
for file_year in range(2013, 2024):
    global_chronos_errors[file_year] = []
    global_organism_errors[file_year] = []

# Optionally, store each country's forecast DataFrame for later use.
country_forecasts = {}

# ============================
# Loop over each country in the list
# ----------------------------
for country in countries:
    print(f"\nProcessing country: {country}")

    # Local dictionaries for the current country
    chronos_errors = {}             # keys: forecast year, values: error for Chronos prediction
    organism_errors = {}            # keys: forecast year, values: error for Organism prediction
    organism_predictions_all = {}   # keys: forecast year, values: Organism prediction
    all_forecasts = pd.DataFrame()   # to collect forecasts over all forecast years
    forecast_stds = {}              # to store forecast uncertainty (std) for each year

    # Loop over forecast years (2013 to 2023)
    for file_year in range(2013, 2024):

        file_name = f"WEOApr{file_year}sentiment.xlsx"
        print(f"\nProcessing file: {file_name} for country: {country}")

        # ============================
        # 2. Load and preprocess the data from the current file
        # ----------------------------
        df = pd.read_excel(file_name)
        year_cols = df.columns[2:]
        df_long = df.melt(id_vars=[df.columns[0], df.columns[1]],
                          value_vars=year_cols,
                          var_name='Year',
                          value_name='Value')
        df_long.columns = ['ISO', 'WEO Subject Code', 'Year', 'Value']
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

        # Use data only up to file_year for nowcasting
        stopping_year = file_year
        df_long_filtered = df_long[df_long['Year'] <= stopping_year]

        # ============================
        # 3. Filter by country and subjects, then pivot the data
        # ----------------------------
        if selected_subject_codes:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'].isin(selected_subject_codes))
        else:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'] == target_subject)

        df_filtered = df_long_filtered[filter_mask].copy()
        df_filtered['variable'] = df_filtered['ISO'] + "_" + df_filtered['WEO Subject Code'].astype(str)
        df_pivot = df_filtered.pivot(index='Year', columns='variable', values='Value').dropna()
        df_pivot = df_pivot.astype(float)

        # ============================
        # 4. Forecast using ChronosPipeline
        forecasts = {}
        for col in df_pivot.columns:
            series = df_pivot[col].tolist()
            # Convert the historical series into a torch tensor (1D)
            context = torch.tensor(series, dtype=torch.float32)
            pred_length = 1  # forecast one step ahead

            # Set num_samples to 20 for multiple forecast samples
            forecast_tensor = chronos_pipeline.predict(
                context,
                pred_length,
                num_samples=20,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )

            # Extract all forecast samples (assuming tensor shape [num_samples, 1, 1])
            forecast_samples = [forecast_tensor[i][0][0].item() for i in range(forecast_tensor.size(0))]
            median_forecast = np.median(forecast_samples)  # Using the median to reduce outlier impact
            std_forecast = np.std(forecast_samples)

            forecasts[col] = median_forecast

            # If this is the target subject column, save its std for plotting
            if col == f"{country}_{target_subject}":
                forecast_stds[file_year] = std_forecast

        forecast_df = pd.DataFrame(forecasts, index=[file_year])
        all_forecasts = pd.concat([all_forecasts, forecast_df])

        # ============================
        # 5. Error analysis for the target subject
        # ----------------------------
        col_key = f"{country}_{target_subject}"
        var_prediction = forecast_df.loc[file_year, col_key]

        # Organism's prediction is taken from the current file (column for file_year)
        organism_df = pd.read_excel(file_name)
        organism_df.columns = organism_df.columns.astype(str)
        organism_row = organism_df[(organism_df['ISO'] == country) &
                                   (organism_df['WEO Subject Code'] == target_subject)]
        organism_prediction = organism_row[str(file_year)].values[0]
        organism_predictions_all[file_year] = float(organism_prediction)

        # Extract the true value from the truth file for the same forecast year
        truth_row = truth_df[(truth_df['ISO'] == country) &
                             (truth_df['WEO Subject Code'] == target_subject)]
        truth_value = truth_row[str(file_year)].values[0]

        truth_value = float(truth_value)
        organism_prediction = float(organism_prediction)
        var_prediction = float(var_prediction)

        error_var = abs(truth_value - var_prediction)
        error_organism = abs(truth_value - organism_prediction)

        chronos_errors[file_year] = error_var
        organism_errors[file_year] = error_organism

        # Update the global dictionaries (to later compute RMSE across countries per year)
        global_chronos_errors[file_year].append(error_var)
        global_organism_errors[file_year].append(error_organism)

    # ============================
    # Plotting for the current country
    # ----------------------------
    # For the error bar plots, update x-axis labels to show that 2020 is missing.
    years = sorted(chronos_errors.keys())
    chronos_error_values = [chronos_errors[yr] for yr in years]
    organism_error_values = [organism_errors[yr] for yr in years]

    x = np.arange(len(years))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, chronos_error_values, width, label='Chronos Prediction Error', color='blue')
    plt.bar(x + width/2, organism_error_values, width, label='Organism Prediction Error', color='orange')
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.xticks(x, years)
    plt.title(f'Prediction Errors for {target_subject} ({country}) Over the Years')
    plt.legend()
    plt.show()

    # Overall RMSE: Compute RMSE over all forecast years for this country
    # (NaN values will be ignored in the RMSE calculation by using np.nanmean)
    rmse_chronos = np.sqrt(np.nanmean(np.square(chronos_error_values)))
    rmse_organism = np.sqrt(np.nanmean(np.square(organism_error_values)))

    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos RMSE', 'Organism RMSE'], [rmse_chronos, rmse_organism], color=['blue', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE for {target_subject} Predictions ({country})')
    plt.show()

    # Additional Plot: NGDP Prediction vs. Organism vs. Truth Over the Years for this country
    target_col = f"{country}_{target_subject}"
    years = sorted(all_forecasts.index)

    # Forecast mean values from Chronos and corresponding uncertainty (std dev)
    forecast_values = [all_forecasts.loc[year, target_col] for year in years]
    forecast_std_values = [forecast_stds.get(year, np.nan) for year in years]

    # Organism predictions and Truth values
    organism_values = [organism_predictions_all.get(year, np.nan) for year in years]
    truth_values = []
    for year in years:
        truth_row = truth_df[(truth_df['ISO'] == country) &
                             (truth_df['WEO Subject Code'] == target_subject)]
        truth_val = float(truth_row[str(year)].values[0])
        truth_values.append(truth_val)

    plt.figure(figsize=(10, 6))

    # Plot Chronos predictions as a line with markers.
    plt.plot(years, forecast_values, marker='o', linestyle='-', color='blue',
             label='Chronos Prediction (mean)')

    # Create a shaded band for the uncertainty (mean ± std)
    forecast_arr = np.array(forecast_values, dtype=np.float64)
    std_arr = np.array(forecast_std_values, dtype=np.float64)
    lower_bound = forecast_arr - std_arr
    upper_bound = forecast_arr + std_arr
    plt.fill_between(years, lower_bound, upper_bound, color='blue', alpha=0.2,
                     label='Chronos Uncertainty (±1 std)')

    # Plot Organism prediction and Truth as continuous lines with markers.
    plt.plot(years, organism_values, marker='x', linestyle='-', color='orange',
             label='Organism Prediction')
    plt.plot(years, truth_values, marker='s', linestyle='-', color='green',
             label='Truth')

    plt.xlabel('Forecast Year')
    plt.ylabel('NGDP Value')
    plt.title(f'NGDP Prediction vs. Organism vs. Truth for {country} Through the Years')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"All forecasts over years for {country}:")
    print(all_forecasts)

    # Optionally store this country's forecast data
    country_forecasts[country] = all_forecasts

# ============================
# Additional Plot: RMSE Over All Countries for Each Forecast Year
# ----------------------------
years = sorted(global_chronos_errors.keys())
rmse_chronos_all = []
rmse_organism_all = []

for year in years:
    # For each forecast year, compute RMSE across the countries (ignoring NaN values)
    errors_chronos = np.array(global_chronos_errors[year], dtype=np.float64)
    errors_organism = np.array(global_organism_errors[year], dtype=np.float64)
    rmse_year_chronos = np.sqrt(np.nanmean(np.square(errors_chronos)))
    rmse_year_organism = np.sqrt(np.nanmean(np.square(errors_organism)))
    rmse_chronos_all.append(rmse_year_chronos)
    rmse_organism_all.append(rmse_year_organism)

plt.figure(figsize=(10, 6))
plt.plot(years, rmse_chronos_all, marker='o', label='Chronos RMSE (All Countries)')
plt.plot(years, rmse_organism_all, marker='x', label='Organism RMSE (All Countries)')
plt.xlabel('Forecast Year')
plt.ylabel('RMSE Across Countries')
plt.title('RMSE Over All Countries for Each Forecast Year')
plt.legend()
plt.grid(True)
plt.show()




# ============================
# 1. User settings & parameters
# ----------------------------
# Fill in your list of country codes here.
countries = ['RUS','TUR','NZL','IRL','SWE','NOR','DNK','IND']
target_subject = 'NGDP_RPCH'         # Change target subject if needed

# Option: Use nothing but NGDP_RPCH, so leave selected_subject_codes empty.
selected_subject_codes = []

# Truth file (assumed constant for all forecast years)
truth_file = 'copieofWEO2024.xlsx'
truth_df = pd.read_excel(truth_file, skiprows=1)
truth_df.columns = truth_df.columns.astype(str)

# Global dictionaries to store errors for each forecast year across all countries.
# For each forecast year, these lists will accumulate the error from each country.
global_chronos_errors = {}
global_organism_errors = {}

for file_year in range(2008, 2024):
    global_chronos_errors[file_year] = []
    global_organism_errors[file_year] = []

# Optionally, store each country's forecast DataFrame for later use.
country_forecasts = {}

# ============================
# Loop over each country in the list
# ----------------------------


for country in countries:
    print(f"\nProcessing country: {country}")

    # Local dictionaries for the current country
    chronos_errors = {}             # keys: forecast year, values: error for Chronos prediction
    organism_errors = {}            # keys: forecast year, values: error for Organism prediction
    organism_predictions_all = {}   # keys: forecast year, values: Organism prediction
    all_forecasts = pd.DataFrame()   # to collect forecasts over all forecast years

    # Loop over forecast years (2010 to 2023)
    for file_year in range(2008, 2024):
        file_name = f"WEOApr{file_year}sentiment.xlsx"
        print(f"\nProcessing file: {file_name} for country: {country}")

        # ============================
        # 2. Load and preprocess the data from the current file
        # ----------------------------
        df = pd.read_excel(file_name)
        year_cols = df.columns[2:]
        df_long = df.melt(id_vars=[df.columns[0], df.columns[1]],
                          value_vars=year_cols,
                          var_name='Year',
                          value_name='Value')
        df_long.columns = ['ISO', 'WEO Subject Code', 'Year', 'Value']
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

        # Use data only up to file_year for nowcasting
        stopping_year = file_year
        df_long_filtered = df_long[df_long['Year'] <= stopping_year]

        # ============================
        # 3. Filter by country and subjects, then pivot the data
        # ----------------------------
        if selected_subject_codes:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'].isin(selected_subject_codes))
        else:
            filter_mask = (df_long_filtered['ISO'] == country) & \
                          (df_long_filtered['WEO Subject Code'] == target_subject)

        df_filtered = df_long_filtered[filter_mask].copy()
        df_filtered['variable'] = df_filtered['ISO'] + "_" + df_filtered['WEO Subject Code'].astype(str)
        df_pivot = df_filtered.pivot(index='Year', columns='variable', values='Value').dropna()
        df_pivot = df_pivot.astype(float)

        # ============================
        # 4. Forecast using ChronosPipeline
        # ----------------------------
        forecasts = {}
        for col in df_pivot.columns:
            series = df_pivot[col].tolist()
            # Convert the historical series into a torch tensor (1D)
            context = torch.tensor(series, dtype=torch.float32)
            pred_length = 1  # forecast one step ahead
            forecast_tensor = chronos_pipeline.predict(
                context,
                pred_length,
                num_samples=1,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )
            # Extract forecast value (tensor shape: [1, 1, 1])
            predicted_value = forecast_tensor[0][0][0].item()
            forecasts[col] = predicted_value

        forecast_df = pd.DataFrame(forecasts, index=[file_year])
        all_forecasts = pd.concat([all_forecasts, forecast_df])

        # ============================
        # 5. Error analysis for the target subject
        # ----------------------------
        col_key = f"{country}_{target_subject}"
        var_prediction = forecast_df.loc[file_year, col_key]

        # Organism's prediction is taken from the current file (column for file_year)
        organism_df = pd.read_excel(file_name)
        organism_df.columns = organism_df.columns.astype(str)
        organism_row = organism_df[(organism_df['ISO'] == country) &
                                   (organism_df['WEO Subject Code'] == target_subject)]
        organism_prediction = organism_row[str(file_year)].values[0]
        organism_predictions_all[file_year] = float(organism_prediction)

        # Extract the true value from the truth file for the same forecast year
        truth_row = truth_df[(truth_df['ISO'] == country) &
                             (truth_df['WEO Subject Code'] == target_subject)]
        truth_value = truth_row[str(file_year)].values[0]

        truth_value = float(truth_value)
        organism_prediction = float(organism_prediction)
        var_prediction = float(var_prediction)

        error_var = abs(truth_value - var_prediction)
        error_organism = abs(truth_value - organism_prediction)

        chronos_errors[file_year] = error_var
        organism_errors[file_year] = error_organism

        # Update the global dictionaries (to later compute RMSE across countries per year)
        global_chronos_errors[file_year].append(error_var)
        global_organism_errors[file_year].append(error_organism)

    # ============================
    # Plotting for the current country
    # ----------------------------

    # Combined Bar Plot: Prediction Errors Over the Years for this country
    years = sorted(chronos_errors.keys())
    chronos_error_values = [chronos_errors[yr] for yr in years]
    organism_error_values = [organism_errors[yr] for yr in years]

    x = np.arange(len(years))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, chronos_error_values, width, label='Chronos Prediction Error', color='blue')
    plt.bar(x + width/2, organism_error_values, width, label='Organism Prediction Error', color='orange')
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.xticks(x, years)
    plt.title(f'Prediction Errors for {target_subject} ({country}) Over the Years')
    plt.legend()
    plt.show()

    # Overall RMSE: Compute RMSE over all forecast years for this country
    rmse_chronos = np.sqrt(np.mean(np.square(chronos_error_values)))
    rmse_organism = np.sqrt(np.mean(np.square(organism_error_values)))

    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos RMSE', 'Organism RMSE'], [rmse_chronos, rmse_organism], color=['blue', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE for {target_subject} Predictions ({country})')
    plt.show()

    # --- New Code: Compute and display RMSE excluding the forecast year 2020 ---
    # Filter out errors for 2020
    chronos_errors_excl2020 = [error for yr, error in chronos_errors.items() if yr != 2020]
    organism_errors_excl2020 = [error for yr, error in organism_errors.items() if yr != 2020]
    rmse_chronos_excl2020 = np.sqrt(np.mean(np.square(chronos_errors_excl2020)))
    rmse_organism_excl2020 = np.sqrt(np.mean(np.square(organism_errors_excl2020)))

    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos RMSE (excl 2020)', 'Organism RMSE (excl 2020)'],
            [rmse_chronos_excl2020, rmse_organism_excl2020],
            color=['blue', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE (Excluding 2020) for {target_subject} Predictions ({country})')
    plt.show()

    # Additional Plot: NGDP_RPCH Prediction vs. Organism vs. Truth Over the Years for this country
    target_col = f"{country}_{target_subject}"
    years = sorted(all_forecasts.index)
    forecast_values = [all_forecasts.loc[year, target_col] for year in years]
    organism_values = [organism_predictions_all[year] for year in years]
    truth_values = []
    for year in years:
        truth_row = truth_df[(truth_df['ISO'] == country) &
                             (truth_df['WEO Subject Code'] == target_subject)]
        truth_val = float(truth_row[str(year)].values[0])
        truth_values.append(truth_val)

    plt.figure(figsize=(10, 6))
    plt.plot(years, forecast_values, marker='o', label='Chronos Prediction')
    plt.plot(years, organism_values, marker='x', label='Organism Prediction')
    plt.plot(years, truth_values, marker='s', label='Truth')
    plt.xlabel('Forecast Year')
    plt.ylabel('NGDP_RPCH Value')
    plt.title(f'NGDP_RPCH Prediction vs. Organism vs. Truth for {country} Through the Years')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"All forecasts over years for {country}:")
    print(all_forecasts)

    # Optionally store this country's forecast data
    country_forecasts[country] = all_forecasts

# ============================
# Additional Plot: RMSE Over All Countries for Each Forecast Year
# ----------------------------
years = sorted(global_chronos_errors.keys())
rmse_chronos_all = []
rmse_organism_all = []

for year in years:
    # For each forecast year, compute RMSE across the countries (i.e. mean squared error across countries)
    errors_chronos = global_chronos_errors[year]
    errors_organism = global_organism_errors[year]
    rmse_year_chronos = np.sqrt(np.mean(np.square(errors_chronos)))
    rmse_year_organism = np.sqrt(np.mean(np.square(errors_organism)))
    rmse_chronos_all.append(rmse_year_chronos)
    rmse_organism_all.append(rmse_year_organism)

plt.figure(figsize=(10, 6))
plt.plot(years, rmse_chronos_all, marker='o', label='Chronos RMSE (All Countries)')
plt.plot(years, rmse_organism_all, marker='x', label='Organism RMSE (All Countries)')
plt.xlabel('Forecast Year')
plt.ylabel('RMSE Across Countries')
plt.title('RMSE Over All Countries for Each Forecast Year')
plt.legend()
plt.grid(True)
plt.show()