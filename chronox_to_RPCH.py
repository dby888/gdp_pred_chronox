import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from chronos import BaseChronosPipeline

os.makedirs("figure", exist_ok=True)

local_path = r"D:\huggingFaceModels\chronos-t5-large"
chronos_pipeline = BaseChronosPipeline.from_pretrained(
    local_path,
    device_map="cuda",  # use GPU (CUDA)
    torch_dtype=torch.float16,
)

countries = ['USA']
target_subject = 'NGDP'
output_subject = 'NGDP_RPCH'
start_year = 2007
end_year = 2024
truth_file = 'copieofWEO2024.xlsx'
truth_df = pd.read_excel(truth_file, skiprows=1)
truth_df.columns = truth_df.columns.map(str)

for country in countries:
    print(f"\nProcessing country: {country}")
    all_forecasts = pd.DataFrame()
    organism_predictions_all = {}

    nfci_df = pd.read_csv('NFCI.csv', parse_dates=['observation_date'])
    nfci_df['Year'] = nfci_df['observation_date'].dt.year
    nfci_df['Month'] = nfci_df['observation_date'].dt.month
    nfci_q1 = nfci_df[(nfci_df['Month'] <= 3) & (nfci_df['Year'] >= 1980) & (nfci_df['Year'] <= 2024)]
    nfci_yearly = nfci_q1.groupby('Year')['NFCI'].apply(lambda s: s.pct_change().dropna().mean())

    years = []

    for file_year in range(start_year, end_year):
        years.append(file_year)
        file_name = f"WEOApr{file_year}all.xlsx"
        print(f"\nProcessing file: {file_name}")

        df = pd.read_excel(file_name)
        df.columns = df.columns.map(str)
        df_long = df.melt(id_vars=[df.columns[0], df.columns[1]], value_vars=df.columns[2:],
                          var_name='Year', value_name='Value')
        df_long.columns = ['ISO', 'WEO Subject Code', 'Year', 'Value']
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
        df_long_filtered = df_long[df_long['Year'] <= file_year]
        filter_mask = (df_long_filtered['ISO'] == country) & (df_long_filtered['WEO Subject Code'] == target_subject)
        df_filtered = df_long_filtered[filter_mask].copy()
        df_filtered['variable'] = df_filtered['ISO'] + "_" + df_filtered['WEO Subject Code'].astype(str)
        df_pivot = df_filtered.pivot(index='Year', columns='variable', values='Value').dropna()
        df_pivot = df_pivot.astype(float)

        for col in df_pivot.columns:
            series = df_pivot[col].tolist()
            context = torch.tensor(series, dtype=torch.float32)
            pred_length = 1

            context_years = df_pivot.index.tolist()
            cov_values = [nfci_yearly.get(y, np.nan) for y in context_years]
            cov_tensor = torch.tensor(cov_values, dtype=torch.float32).unsqueeze(-1)

            forecast_with = chronos_pipeline.predict(context, pred_length, 20, 1.0, 50, 1.0, covariates=cov_tensor)
            forecast_samples_with = [forecast_with[i][0][0].item() for i in range(forecast_with.size(0))]
            mean_with = np.mean(forecast_samples_with)

            forecast_without = chronos_pipeline.predict(context, pred_length, 20, 1.0, 50, 1.0)
            forecast_samples_without = [forecast_without[i][0][0].item() for i in range(forecast_without.size(0))]
            mean_without = np.mean(forecast_samples_without)

            all_forecasts.loc[file_year, f'{col}_with_NFCI'] = mean_with
            all_forecasts.loc[file_year, f'{col}_without_NFCI'] = mean_without

        organism_df = pd.read_excel(file_name)
        organism_df.columns = organism_df.columns.map(str)
        organism_row = organism_df[(organism_df['ISO'] == country) & (organism_df['WEO Subject Code'] == target_subject)]
        if str(file_year) in organism_row.columns and not organism_row.empty:
            organism_predictions_all[file_year] = float(organism_row[str(file_year)].values[0])
        else:
            organism_predictions_all[file_year] = np.nan
            print(f"Warning: Year {file_year} not found in organism_row, setting as NaN.")

    target_col = f"{country}_{target_subject}"
    truth_vals = [float(truth_df[(truth_df['ISO'] == country) & (truth_df['WEO Subject Code'] == target_subject)][str(y)].values[0])
                  for y in years]
    forecast_with_vals = all_forecasts[f'{target_col}_with_NFCI'].values
    forecast_without_vals = all_forecasts[f'{target_col}_without_NFCI'].values
    organism_vals = [organism_predictions_all[y] for y in years]

    truth_series = pd.Series(truth_vals)
    forecast_with_series = pd.Series(forecast_with_vals)
    forecast_without_series = pd.Series(forecast_without_vals)
    organism_series = pd.Series(organism_vals)

    forecast_with_growth = (forecast_with_series - truth_series.shift()) / truth_series.shift() * 100
    forecast_without_growth = (forecast_without_series - truth_series.shift()) / truth_series.shift() * 100
    organism_growth = (organism_series - truth_series.shift()) / truth_series.shift() * 100
    truth_growth = truth_series.diff() / truth_series.shift() * 100

    years = years[1:]
    forecast_with_vals = forecast_with_growth[1:].values
    forecast_without_vals = forecast_without_growth[1:].values
    organism_vals = organism_growth[1:].values
    truth_vals = truth_growth[1:].values

    error_with = np.abs(np.array(truth_vals) - forecast_with_vals)
    error_without = np.abs(np.array(truth_vals) - forecast_without_vals)
    error_organism = np.abs(np.array(truth_vals) - np.array(organism_vals))

    rmse_with = np.sqrt(np.mean(error_with ** 2))
    rmse_without = np.sqrt(np.mean(error_without ** 2))
    rmse_organism = np.sqrt(np.mean(error_organism ** 2))

    plt.figure(figsize=(12, 6))
    x_ticks = np.arange(len(years))
    width = 0.25
    plt.bar(x_ticks - width, error_with, width, label='Chronos with NFCI', color='blue')
    plt.bar(x_ticks, error_without, width, label='Chronos without NFCI', color='cyan')
    plt.bar(x_ticks + width, error_organism, width, label='Organism', color='orange')
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.xticks(x_ticks, years)
    plt.title(f'Prediction Errors Comparison for {output_subject} ({country})')
    plt.legend()
    plt.savefig(f'figure/{country}_{output_subject}_error_compare.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos with NFCI', 'Chronos without NFCI', 'Organism'],
            [rmse_with, rmse_without, rmse_organism],
            color=['blue', 'cyan', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE Comparison for {output_subject} Predictions ({country})')
    plt.savefig(f'figure/{country}_{output_subject}_rmse_compare.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(years, forecast_with_vals, marker='o', linestyle='-', color='blue', label='Chronos with NFCI')
    plt.plot(years, forecast_without_vals, marker='o', linestyle='--', color='cyan', label='Chronos without NFCI')
    plt.plot(years, organism_vals, marker='x', linestyle='-', color='orange', label='Organism')
    plt.plot(years, truth_vals, marker='s', linestyle='-', color='green', label='Truth')
    plt.xlabel('Forecast Year')
    plt.ylabel(f'{output_subject} Value')
    plt.title(f'{output_subject} Prediction Comparison for {country}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figure/{country}_{output_subject}_pred_vs_true_compare.png')
    plt.close()

    rmse_with_all, rmse_without_all, rmse_organism_all = [], [], []
    for idx in range(len(years)):
        rmse_with_all.append(np.sqrt(np.mean(error_with[:idx + 1] ** 2)))
        rmse_without_all.append(np.sqrt(np.mean(error_without[:idx + 1] ** 2)))
        rmse_organism_all.append(np.sqrt(np.mean(error_organism[:idx + 1] ** 2)))

    plt.figure(figsize=(10, 6))
    plt.plot(years, rmse_with_all, marker='o', linestyle='-', color='blue', label='Chronos with NFCI')
    plt.plot(years, rmse_without_all, marker='o', linestyle='--', color='cyan', label='Chronos without NFCI')
    plt.plot(years, rmse_organism_all, marker='x', linestyle='-', color='orange', label='Organism')
    plt.xlabel('Forecast Year')
    plt.ylabel('RMSE Across Years')
    plt.title('RMSE Over All Countries for Each Forecast Year')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figure/{country}_{output_subject}_rmse_all_years_compare.png')
    plt.close()

    error_df = pd.DataFrame({
        'Year': years,
        'Chronos_with_NFCI_Error': error_with,
        'Chronos_without_NFCI_Error': error_without,
        'Organism_Error': error_organism
    })

    rmse_df = pd.DataFrame({
        'Metric': ['RMSE'],
        'Chronos_with_NFCI': [rmse_with],
        'Chronos_without_NFCI': [rmse_without],
        'Organism': [rmse_organism]
    })

    rmse_years_df = pd.DataFrame({
        'Year': years,
        'Chronos_with_NFCI_RMSE': rmse_with_all,
        'Chronos_without_NFCI_RMSE': rmse_without_all,
        'Organism_RMSE': rmse_organism_all
    })

    excel_path = f'figure/{country}_{output_subject}_results_summary.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        error_df.to_excel(writer, sheet_name='Absolute_Errors', index=False)
        rmse_df.to_excel(writer, sheet_name='Overall_RMSE', index=False)
        rmse_years_df.to_excel(writer, sheet_name='RMSE_Over_Years', index=False)

    print(f"Excel summary saved to {excel_path}")
