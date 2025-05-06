import os
import pandas as pd
import numpy as np

def load_country_data(country, target_subject, start_year, end_year, truth_file):
    # 确保 figure 目录存在
    save_dir = os.path.join("figure", country)
    os.makedirs(save_dir, exist_ok=True)

    # === 读取 truth file ===
    truth_df = pd.read_excel(truth_file, skiprows=1)
    truth_df.columns = truth_df.columns.map(str)
    years = list(range(start_year, end_year))

    truth_vals = [float(truth_df[(truth_df['ISO'] == country) &
                                 (truth_df['WEO Subject Code'] == target_subject)][str(y)].values[0])
                  for y in list(range(start_year-1, end_year))]

    # === 读取 forecast_results.csv ===
    forecast_path = f'result_csv/{country}_{target_subject}_forecast_results.csv'
    all_forecasts = pd.read_csv(forecast_path, index_col=0)
    all_forecasts.index = all_forecasts.index.astype(int)
    all_forecasts = all_forecasts.loc[start_year:end_year - 1]

    target_col_with = f"{country}_{target_subject}_with_NFCI"
    target_col_without = f"{country}_{target_subject}_without_NFCI"
    forecast_with_vals = all_forecasts[target_col_with].values
    forecast_without_vals = all_forecasts[target_col_without].values

    # === 读取 organism_predictions.csv ===
    organism_path = f'result_csv/{country}_{target_subject}_organism_predictions.csv'
    organism_df = pd.read_csv(organism_path, index_col=0)
    organism_df.columns = ['prediction']
    organism_df.index = organism_df.index.astype(int)
    organism_df = organism_df.loc[start_year:end_year - 1]
    organism_vals = organism_df['prediction'].values

    # === 构造 Pandas Series
    truth_series = pd.Series(truth_vals, index= list(range(start_year-1, end_year)))
    forecast_with_series = pd.Series(forecast_with_vals, index=years)
    forecast_without_series = pd.Series(forecast_without_vals, index=years)
    organism_series = pd.Series(organism_vals, index=years)

    return truth_series, forecast_with_series, forecast_without_series, organism_series

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_results(country, target_subject, start_year, end_year, truth_file):
    # 确保 figure/{country}/ 文件夹存在
    save_dir = os.path.join("figure", country)
    os.makedirs(save_dir, exist_ok=True)

    # === 读取 truth file ===
    truth_df = pd.read_excel(truth_file, skiprows=1)
    truth_df.columns = truth_df.columns.map(str)
    years = list(range(start_year, end_year))
    base_years = [y - 1 for y in years]

    # 提取 truth series（包含2006基数年）
    truth_base_series = pd.Series(
        [float(
            truth_df[(truth_df['ISO'] == country) & (truth_df['WEO Subject Code'] == target_subject)][str(y)].values[0])
         for y in base_years],
        index=base_years
    )

    truth_target_series = pd.Series(
        [float(
            truth_df[(truth_df['ISO'] == country) & (truth_df['WEO Subject Code'] == target_subject)][str(y)].values[0])
         for y in years],
        index=years
    )

    # === 读取 forecast_results.csv ===
    all_forecasts = pd.read_csv(f'result_csv/{country}_{target_subject}_forecast_results.csv', index_col=0)
    all_forecasts.index = all_forecasts.index.astype(str).str.extract(r'(\d+)')[0].astype(int)
    all_forecasts = all_forecasts.loc[years]
    target_col_with = f"{country}_{target_subject}_with_NFCI"
    target_col_without = f"{country}_{target_subject}_without_NFCI"
    forecast_with_series = pd.Series(all_forecasts[target_col_with].values, index=years)
    forecast_without_series = pd.Series(all_forecasts[target_col_without].values, index=years)

    # === 读取 organism_predictions.csv ===
    organism_df = pd.read_csv(f'result_csv/{country}_{target_subject}_organism_predictions.csv', index_col=0)
    organism_df.columns = ['prediction']
    organism_df.index = organism_df.index.map(str).str.extract(r'(\d+)')[0].astype(int)
    organism_df = organism_df.loc[years]
    organism_series = pd.Series(organism_df['prediction'].values, index=years)

    # === 计算增长率（严格按年份对齐）
    truth_base_vals = truth_series.loc[base_years].values
    truth_target_vals = truth_series.loc[years].values
    forecast_with_vals = forecast_with_series.loc[years].values
    forecast_without_vals = forecast_without_series.loc[years].values
    organism_vals = organism_series.loc[years].values

    forecast_with_growth = (forecast_with_vals - truth_base_vals) / truth_base_vals * 100
    forecast_without_growth = (forecast_without_vals - truth_base_vals) / truth_base_vals * 100
    organism_growth = (organism_vals - truth_base_vals) / truth_base_vals * 100
    truth_growth = (truth_target_vals - truth_base_vals) / truth_base_vals * 100

    # === 误差计算
    error_with = np.abs(truth_growth - forecast_with_growth)
    error_without = np.abs(truth_growth - forecast_without_growth)
    error_organism = np.abs(truth_growth - organism_growth)
    rmse_with = np.sqrt(np.mean(error_with ** 2))
    rmse_without = np.sqrt(np.mean(error_without ** 2))
    rmse_organism = np.sqrt(np.mean(error_organism ** 2))

    # 1️⃣ 误差柱状图
    plt.figure(figsize=(12, 6))
    x_ticks = np.arange(len(years))
    width = 0.25
    plt.bar(x_ticks - width, error_with, width, label='Chronos with NFCI', color='blue')
    plt.bar(x_ticks, error_without, width, label='Chronos without NFCI', color='cyan')
    plt.bar(x_ticks + width, error_organism, width, label='Organism', color='orange')
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.xticks(x_ticks, years)
    plt.title(f'Prediction Errors Comparison for {target_subject}_RPCH ({country})')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{target_subject}_RPCH_error_compare.png'))
    plt.close()

    # 2️⃣ 总体 RMSE
    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos with NFCI', 'Chronos without NFCI', 'Organism'],
            [rmse_with, rmse_without, rmse_organism],
            color=['blue', 'cyan', 'orange'])
    plt.ylabel('RMSE')
    plt.title(f'Overall RMSE Comparison for {target_subject}_RPCH Predictions ({country})')
    plt.savefig(os.path.join(save_dir, f'{target_subject}_RPCH_rmse_compare.png'))
    plt.close()

    # 3️⃣ 预测 vs 真实值
    plt.figure(figsize=(10, 6))
    plt.plot(years, forecast_with_growth, marker='o', linestyle='-', color='blue', label='Chronos with NFCI')
    plt.plot(years, forecast_without_growth, marker='o', linestyle='--', color='cyan', label='Chronos without NFCI')
    plt.plot(years, organism_growth, marker='x', linestyle='-', color='orange', label='Organism')
    plt.plot(years, truth_growth, marker='s', linestyle='-', color='green', label='Truth')
    plt.xlabel('Forecast Year')
    plt.ylabel(f'Growth Rate (%)')
    plt.title(f'{target_subject}_RPCH Prediction Comparison for {country}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{target_subject}_RPCH_pred_vs_true_compare.png'))
    plt.close()

    # 4️⃣ 累积 RMSE
    rmse_with_all = [np.sqrt(np.mean(error_with[:idx + 1] ** 2)) for idx in range(len(years))]
    rmse_without_all = [np.sqrt(np.mean(error_without[:idx + 1] ** 2)) for idx in range(len(years))]
    rmse_organism_all = [np.sqrt(np.mean(error_organism[:idx + 1] ** 2)) for idx in range(len(years))]

    plt.figure(figsize=(10, 6))
    plt.plot(years, rmse_with_all, marker='o', linestyle='-', color='blue', label='Chronos with NFCI')
    plt.plot(years, rmse_without_all, marker='o', linestyle='--', color='cyan', label='Chronos without NFCI')
    plt.plot(years, rmse_organism_all, marker='x', linestyle='-', color='orange', label='Organism')
    plt.xlabel('Forecast Year')
    plt.ylabel('Cumulative RMSE')
    plt.title('Cumulative RMSE Over Years')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{target_subject}_RPCH_rmse_all_years_compare.png'))
    plt.close()

    print(f"Plots for {country} saved to {save_dir}")

if __name__ == "__main__":
    truth_series, forecast_with_series, forecast_without_series, organism_series = load_country_data(
        country='USA',
        target_subject='NGDP',
        start_year=2007,
        end_year=2024,
        truth_file='copieofWEO2024.xlsx'
    )
    plot_results(
        country='USA',
        target_subject='NGDP',
        start_year=2007,
        end_year=2024,
        truth_file='copieofWEO2024.xlsx'
    )

