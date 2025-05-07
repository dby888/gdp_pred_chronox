import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set_theme(style='whitegrid')

def run_forecast(country, target_subject, start_year, end_year, truth_file, cov_tensor_dict):
    local_path = r"D:\huggingFaceModels\chronos-t5-large"
    chronos_pipeline = BaseChronosPipeline.from_pretrained(
        local_path,
        device_map="cuda",
        torch_dtype=torch.float16,
    )

    truth_df = pd.read_excel(truth_file, skiprows=1)
    truth_df.columns = truth_df.columns.map(str)
    os.makedirs("result_csv", exist_ok=True)

    all_forecasts = pd.DataFrame()
    organism_predictions_all = {}

    years = []
    for file_year in range(start_year, end_year+1):
        years.append(file_year)
        file_name = f"WEOApr{file_year}all.xlsx"
        print(f"\nProcessing file: {file_name} for country: {country}")

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

            cov_tensor = None
            if cov_tensor_dict:
                cov_tensor = cov_tensor_dict[file_year]

            if cov_tensor is not None:
                forecast_with = chronos_pipeline.predict(context, pred_length, 20, 1.0, 50, 1.0, covariates=cov_tensor)
                forecast_samples_with = [forecast_with[i][0][0].item() for i in range(forecast_with.size(0))]
                mean_with = np.mean(forecast_samples_with)
                all_forecasts.loc[file_year, f'{col}_with_NFCI'] = mean_with

            forecast_without = chronos_pipeline.predict(context, pred_length, 20, 1.0, 50, 1.0)
            forecast_samples_without = [forecast_without[i][0][0].item() for i in range(forecast_without.size(0))]
            mean_without = np.mean(forecast_samples_without)
            all_forecasts.loc[file_year, f'{col}_without_NFCI'] = mean_without

        organism_df = pd.read_excel(file_name)
        organism_df.columns = organism_df.columns.map(str)
        organism_row = organism_df[(organism_df['ISO'] == country) & (organism_df['WEO Subject Code'] == target_subject)]
        if str(file_year) in organism_row.columns and not organism_row.empty:
            organism_predictions_all[file_year] = float(organism_row[str(file_year)].values[0])
        else:
            organism_predictions_all[file_year] = np.nan
            print(f"Warning: Year {file_year} not found in organism_row, setting as NaN.")

    all_forecasts.to_csv(f'result_csv/{country}_{target_subject}_forecast_results.csv')
    pd.Series(organism_predictions_all).to_csv(f'result_csv/{country}_{target_subject}_WEO_predictions.csv')
    print(f"Prediction results for {country} saved to result_csv/ folder.")


def prepare_covariates(start_year, end_year, country):
    # if country == "USA":
    nfci_df = pd.read_csv('NFCI.csv', parse_dates=['observation_date'])
    nfci_df['Year'] = nfci_df['observation_date'].dt.year
    nfci_df['Month'] = nfci_df['observation_date'].dt.month
    nfci_q1 = nfci_df[(nfci_df['Month'] <= 3) & (nfci_df['Year'] >= 1980) & (nfci_df['Year'] <= 2024)]
    nfci_yearly = nfci_q1.groupby('Year')['NFCI'].apply(lambda s: s.pct_change().dropna().mean())

    # ⚠️ 这里可以扩展：为每个国家自定义不同的 covariate 来源
    # 比如未来你可以用 country-specific covariates 替换这里的 nfci_yearly

    cov_tensor_dict = {}
    for year in range(start_year, end_year+1):
        cov_values = [nfci_yearly.get(y, np.nan) for y in range(start_year, year + 1)]
        cov_tensor = torch.tensor(cov_values, dtype=torch.float32).unsqueeze(-1)
        cov_tensor_dict[year] = cov_tensor
    return cov_tensor_dict
    # return None
def plot_results(country, target_subject, start_year, end_year, truth_file):
    os.makedirs("figure", exist_ok=True)

    truth_df = pd.read_excel(truth_file, skiprows=1)
    truth_df.columns = truth_df.columns.map(str)
    years = list(range(start_year, end_year+1))

    target_col_with = f"{country}_{target_subject}_with_NFCI"
    target_col_without = f"{country}_{target_subject}_without_NFCI"

    all_forecasts = pd.read_csv(f'result_csv/{country}_{target_subject}_forecast_results.csv', index_col=0)
    all_forecasts.index = all_forecasts.index.astype(int)
    all_forecasts = all_forecasts.loc[start_year:end_year]

    forecast_with_vals = None
    if target_col_with in all_forecasts.columns:
        forecast_with_vals = all_forecasts[target_col_with].values

    forecast_without_vals = all_forecasts[target_col_without].values if target_col_without in all_forecasts.columns else None

    organism_df = pd.read_csv(f'result_csv/{country}_{target_subject}_WEO_predictions.csv', index_col=0)
    organism_df.columns = ['prediction']
    organism_df.index = organism_df.index.astype(int)
    organism_df = organism_df.loc[start_year:end_year]
    organism_vals = organism_df['prediction'].values

    truth_vals = []
    for y in years:
        if str(y) in truth_df.columns:
            row = truth_df[(truth_df['ISO'] == country) & (truth_df['WEO Subject Code'] == target_subject)]
            if y > row["Estimates Start After"].values[0]:
                truth_vals.append(np.nan)
                continue
            if not row.empty and not pd.isna(row[str(y)].values[0]):
                truth_vals.append(float(row[str(y)].values[0]))
                continue

        truth_vals.append(np.nan)

    save_dir = os.path.join("figure", country)
    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ 折线图
    plt.figure(figsize=(10, 6))

    # Chronos with NFCI （深蓝实线，圆点）
    if forecast_with_vals is not None:
        plt.plot(years, forecast_with_vals,
                 marker='o', markersize=10, linewidth=2, linestyle='-', color='blue',
                 label='Chronos with NFCI')

    # Chronos without NFCI （天蓝实线，圆点）
    if forecast_without_vals is not None:
        plt.plot(years, forecast_without_vals,
                 marker='o', markersize=10, linewidth=2, linestyle='-', color='cyan',
                 label='Chronos without NFCI')

    # WEO （灰色实线，x 标记）
    plt.plot(years, organism_vals,
             marker='x', markersize=10, linewidth=2, linestyle='-', color='grey',
             label='WEO')

    # Truth （绿色实线，方块）
    plt.plot(years, truth_vals,
             marker='s', markersize=10, linewidth=2, linestyle='-', color='green',
             label='Truth')

    plt.xlabel('Forecast Year', fontsize=14)
    plt.ylabel(f'{target_subject} Value', fontsize=14)
    plt.title(f'{target_subject} Prediction Comparison for {country}', fontsize=16)
    plt.legend(frameon=True, fontsize=12, loc='best')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f'figure/{country}/{country}_{target_subject}_pred_vs_true_compare.png', dpi=400)
    plt.close()

    # ✅ 保存 CSV 数据
    data = {
        'Year': years,
        'Chronos_with_NFCI': forecast_with_vals if forecast_with_vals is not None else [None] * len(years),
        'Chronos_without_NFCI': forecast_without_vals if forecast_without_vals is not None else [None] * len(years),
        'WEO': organism_vals,
        'Truth': truth_vals
    }

    df_out = pd.DataFrame(data)
    csv_path = os.path.join(save_dir, f'{country}_{target_subject}_pred_vs_true_compare.csv')
    df_out.to_csv(csv_path, index=False)
    print(f'Saved CSV to {csv_path}')

    # 2️⃣ 绝对误差柱状图
    error_with = np.abs(np.array(truth_vals) - forecast_with_vals) if forecast_with_vals is not None else None
    error_without = np.abs(np.array(truth_vals) - forecast_without_vals) if forecast_without_vals is not None else None
    error_organism = np.abs(np.array(truth_vals) - organism_vals)

    plt.figure(figsize=(12, 6))
    x_ticks = np.arange(len(years))
    width = 0.25
    if error_with is not None:
        plt.bar(x_ticks - width, error_with, width, label='Chronos with NFCI', color='#1f77b4')
    if error_without is not None:
        plt.bar(x_ticks, error_without, width, label='Chronos without NFCI', color='#17becf')
    plt.bar(x_ticks + width, error_organism, width, label='WEO', color='grey')
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.xticks(x_ticks, years)
    plt.title(f'Prediction Errors Comparison for {target_subject} ({country})')
    plt.legend()
    plt.savefig(f'figure/{country}/{country}_{target_subject}_error_compare.png')
    plt.close()

    # 3️⃣ RMSE柱状图
    rmse_with = np.sqrt(np.nanmean(error_with ** 2)) if error_with is not None else None
    rmse_without = np.sqrt(np.nanmean(error_without ** 2)) if error_without is not None else None
    rmse_organism = np.sqrt(np.nanmean(error_organism ** 2))

    plt.figure(figsize=(8, 5))
    labels = []
    rmse_values = []
    colors = []
    if rmse_with is not None:
        labels.append('Chronos with NFCI')
        rmse_values.append(rmse_with)
        colors.append('#1f77b4')
    if rmse_without is not None:
        labels.append('Chronos without NFCI')
        rmse_values.append(rmse_without)
        colors.append('#17becf')
    labels.append('WEO')
    rmse_values.append(rmse_organism)
    colors.append('grey')
    plt.bar(labels, rmse_values, color=colors)
    plt.ylabel('RMSE')
    plt.title(f'{country} Overall RMSE Comparison for {target_subject} Predictions ({country})')
    plt.savefig(f'figure/{country}/{country}_{target_subject}_rmse_compare.png')
    plt.close()

    print(f"Plots for {country} saved to figure/ folder.")


if __name__ == "__main__":
    countries = ['CHN', 'CAN', 'FRA', 'ITA', 'DEU', 'JPN', 'GBR', 'USA', 'SGP']
    # countries = ['SGP']
    target_subject = 'NGDP'
    start_year = 2007
    end_year = 2025
    truth_file = 'copieofWEO2025.xlsx'

    # for country in countries:
    #     cov_tensor_dict = prepare_covariates(start_year, end_year, country)
    #     run_forecast(country, target_subject, start_year, end_year, truth_file, cov_tensor_dict)

    # 绘图部分
    for country in countries:
        plot_results(country, target_subject, start_year, end_year, truth_file)
