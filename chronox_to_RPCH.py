import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set_theme(style='whitegrid')

def load_country_data(country, target_subject, start_year, end_year, truth_file):
    # 确保 figure 目录存在
    save_dir = os.path.join("figure", country)
    os.makedirs(save_dir, exist_ok=True)

    # === 读取 truth file ===
    truth_df = pd.read_excel(truth_file, skiprows=1)
    truth_df.columns = truth_df.columns.map(str)
    years = list(range(start_year-1, end_year+1))

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

    # === 读取 forecast_results.csv ===
    forecast_path = f'result_csv/{country}_{target_subject}_forecast_results.csv'
    all_forecasts = pd.read_csv(forecast_path, index_col=0)
    all_forecasts.index = all_forecasts.index.astype(int)
    all_forecasts = all_forecasts.loc[start_year:end_year+1]

    # 补全缺失年份（自动填 NaN）
    full_years = list(range(start_year, end_year + 1))
    all_forecasts = all_forecasts.reindex(full_years)

    target_col_with = f"{country}_{target_subject}_with_NFCI"
    target_col_without = f"{country}_{target_subject}_without_NFCI"
    forecast_with_vals = all_forecasts[target_col_with].values
    forecast_without_vals = all_forecasts[target_col_without].values

    # === 读取 organism_predictions.csv ===
    organism_path = f'result_csv/{country}_{target_subject}_WEO_predictions.csv'
    organism_df = pd.read_csv(organism_path, index_col=0)
    organism_df.columns = ['prediction']
    organism_df.index = organism_df.index.astype(int)
    organism_df = organism_df.loc[start_year:end_year]
    organism_vals = organism_df['prediction'].values

    # === 构造 Pandas Series
    truth_series = pd.Series(truth_vals, index= list(range(start_year-1, end_year+1)))
    forecast_with_series = pd.Series(forecast_with_vals, index=list(range(start_year, end_year+1)))
    forecast_without_series = pd.Series(forecast_without_vals, index=list(range(start_year, end_year+1)))
    organism_series = pd.Series(organism_vals, index=list(range(start_year, end_year+1)))

    return truth_series, forecast_with_series, forecast_without_series, organism_series



def plot_results(country, target_subject, start_year, end_year, truth_file):
    # 确保 figure/{country}/ 文件夹存在
    save_dir = os.path.join("figure", country)
    os.makedirs(save_dir, exist_ok=True)

    # === 读取 truth file ===
    truth_df = pd.read_excel(truth_file, skiprows=1)
    truth_df.columns = truth_df.columns.map(str)
    years = list(range(start_year, end_year + 1))
    base_years = [y - 1 for y in years]

    # === 读取 forecast_results.csv ===
    all_forecasts = pd.read_csv(f'result_csv/{country}_{target_subject}_forecast_results.csv', index_col=0)
    all_forecasts.index = all_forecasts.index.astype(str).str.extract(r'(\d+)')[0].astype(int)
    # 补全缺失年份（自动填 NaN）
    all_forecasts = all_forecasts.reindex(years)
    target_col_with = f"{country}_{target_subject}_with_NFCI"
    target_col_without = f"{country}_{target_subject}_without_NFCI"
    forecast_with_series = pd.Series(all_forecasts[target_col_with].values, index=years)
    forecast_without_series = pd.Series(all_forecasts[target_col_without].values, index=years)

    # === 读取 organism_predictions.csv ===
    organism_df = pd.read_csv(f'result_csv/{country}_{target_subject}_WEO_predictions.csv', index_col=0)
    organism_df.columns = ['prediction']
    organism_df.index = organism_df.index.map(str).str.extract(r'(\d+)')[0].astype(int)
    organism_df = organism_df.loc[years]
    weo_series = pd.Series(organism_df['prediction'].values, index=years)

    # === 提取 truth 数据 ===
    truth_target_series = pd.Series(
        [float(
            truth_df[(truth_df['ISO'] == country) & (truth_df['WEO Subject Code'] == target_subject)][str(y)].values[0])
         for y in years],
        index=years
    )
    truth_target_vals = truth_target_series.values

    # === 提取 base year truth 数据
    truth_series = pd.Series(
        [float(
            truth_df[(truth_df['ISO'] == country) & (truth_df['WEO Subject Code'] == target_subject)][str(y)].values[0])
         for y in base_years],
        index=base_years
    )
    truth_base_vals = truth_series.values

    # ✅ 冗余逻辑1：提取 RPCH 行（暂时没用，但保留）
    rpch_row = truth_df[(truth_df['ISO'] == country) & (truth_df['WEO Subject Code'] == 'NGDP_RPCH')]
    str_year = [str(col) for col in years]
    rpch_values_truth = rpch_row[str_year].values.flatten()
    rpch_values_truth = pd.to_numeric(rpch_values_truth, errors='coerce')
    RPCH_truth = pd.Series(rpch_values_truth, index=years)
    # 拿到 Estimates Start After 阈值
    try:
        estimates_start_after = rpch_row["Estimates Start After"].values[0]
        print(f"estimates_start_after, set {estimates_start_after}")
    except:
        estimates_start_after = 2024
        print("no estimates_start_after, set 2024")
    # 找出 > estimates_start_after 的年份，把它们设为 NaN
    for y in RPCH_truth.index:
        if y > estimates_start_after:
            RPCH_truth.loc[y] = np.nan
    # 筛选出符合条件的年份
    valid_indices = [i for i, y in enumerate(years) if y <= estimates_start_after]
    years_plot = [years[i] for i in valid_indices]

    # ✅ 冗余逻辑2：读取 weo-predictions.csv （暂时未用）
    file_path = 'weo-predictions.csv'
    df = pd.read_csv(file_path)
    iso_code = country
    filtered_df = df[
        (df['WEO Subject Code'] == 'NGDP_RPCH') &
        (df['ISO'] == iso_code) &
        (df['Year'].isin(years))
    ]

    # === 计算增长率
    forecast_with_vals = forecast_with_series.loc[years].values
    forecast_without_vals = forecast_without_series.loc[years].values

    forecast_with_growth = (forecast_with_vals - truth_base_vals) / truth_base_vals * 100
    forecast_without_growth = (forecast_without_vals - truth_base_vals) / truth_base_vals * 100
    organism_growth = (weo_series - truth_base_vals) / truth_base_vals * 100
    truth_growth = RPCH_truth

    # === 误差计算
    error_with = np.abs(truth_growth - forecast_with_growth)
    error_without = np.abs(truth_growth - forecast_without_growth)
    error_weo = np.abs(truth_growth - organism_growth)
    rmse_with = np.sqrt(np.nanmean(error_with ** 2))
    rmse_without = np.sqrt(np.nanmean(error_without ** 2))
    rmse_organism = np.sqrt(np.nanmean(error_weo ** 2))

    # 1️⃣ 误差柱状图
    error_with_plot = error_with[years_plot]
    error_without_plot = error_without[years_plot]
    error_weo_plot = error_weo[years_plot]

    plt.figure(figsize=(12, 6))
    x_ticks = np.arange(len(years_plot))
    width = 0.25
    plt.bar(x_ticks - width, error_with_plot, width, label='Chronos with NFCI', color='blue')
    plt.bar(x_ticks, error_without_plot, width, label='Chronos without NFCI', color='cyan')
    plt.bar(x_ticks + width, error_weo_plot, width, label='WEO', color='gray')
    plt.xlabel('Forecast Year')
    plt.ylabel('Absolute Error')
    plt.xticks(x_ticks, years_plot)
    plt.title(f'{country} Prediction Errors Comparison for NGDP_RPCH ({country})')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{country} NGDP_RPCH_error_compare.png'), dpi=300)
    plt.close()

    # 2️⃣ 总体 RMSE
    plt.figure(figsize=(8, 5))
    plt.bar(['Chronos with NFCI', 'Chronos without NFCI', 'WEO'],
            [rmse_with, rmse_without, rmse_organism],
            color=['blue', 'cyan', 'grey'])
    plt.ylabel('RMSE')
    plt.title(f'{country} Overall RMSE Comparison for NGDP_RPCH Predictions ({country})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{country} NGDP_RPCH_rmse_compare.png'), dpi=300)
    plt.close()

    # 3️⃣ 预测 vs 真实值
    plt.figure(figsize=(10, 6))
    plt.plot(years, forecast_with_growth, marker='o', markersize=10, linewidth=2, linestyle='-', color='blue',
             label='Chronos with NFCI')
    plt.plot(years, forecast_without_growth, marker='o', markersize=10, linewidth=2, linestyle='-', color='cyan',
             label='Chronos without NFCI')
    plt.plot(years, organism_growth, marker='x', markersize=10, linewidth=2, linestyle='-', color='grey',
             label='WEO')
    plt.plot(years, truth_growth, marker='s', markersize=10, linewidth=2, linestyle='-', color='green', label='Truth')

    plt.xlabel('Forecast Year', fontsize=14)
    plt.ylabel('Growth Rate (%)', fontsize=14)
    plt.title(f'NGDP_RPCH Prediction Comparison for {country}', fontsize=16)
    plt.legend(frameon=True, fontsize=12, loc='best')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{country} NGDP_RPCH_RESULT.png'), dpi=400)
    plt.close()

    # 保存 CSV
    data = {
        'Year': years,
        'Chronos_with_NFCI': forecast_with_growth,
        'Chronos_without_NFCI': forecast_without_growth,
        'WEO': organism_growth,
        'Truth': truth_growth
    }
    df_out = pd.DataFrame(data)
    output_path = f'figure/{country}/{country} NGDP_RPCH_pred_vs_true_compare.csv'
    df_out.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")

    # 4️⃣ 累积 RMSE
    rmse_with_all = [np.sqrt(np.mean(error_with[:idx + 1] ** 2)) for idx in range(len(years))]
    rmse_without_all = [np.sqrt(np.mean(error_without[:idx + 1] ** 2)) for idx in range(len(years))]
    rmse_organism_all = [np.sqrt(np.mean(error_weo[:idx + 1] ** 2)) for idx in range(len(years))]

    # 筛选后的数据
    rmse_with_plot = [rmse_with_all[i] for i in valid_indices]
    rmse_without_plot = [rmse_without_all[i] for i in valid_indices]
    rmse_organism_plot = [rmse_organism_all[i] for i in valid_indices]

    plt.figure(figsize=(10, 6))

    plt.plot(years_plot, rmse_with_plot,
             marker='o', markersize=10, linewidth=2, linestyle='-', color='blue', label='Chronos with NFCI')

    plt.plot(years_plot, rmse_without_plot,
             marker='o', markersize=10, linewidth=2, linestyle='-', color='cyan', label='Chronos without NFCI')

    plt.plot(years_plot, rmse_organism_plot,
             marker='x', markersize=10, linewidth=2, linestyle='-', color='grey', label='WEO')

    plt.xlabel('Forecast Year', fontsize=14)
    plt.ylabel('Cumulative RMSE', fontsize=14)
    plt.title(f'{country} Cumulative RMSE Over Years', fontsize=16, fontweight='bold')
    plt.legend(frameon=True, fontsize=12, loc='best')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{country} NGDP_RPCH_rmse_all_years_compare.png'), dpi=400)
    plt.close()

    print(f"Plots for {country} saved to {save_dir}")

if __name__ == "__main__":
    start_year = 2007
    end_year = 2025
    truth_file = 'copieofWEO2025.xlsx'

    for country in ['CHN', 'CAN', 'FRA', 'ITA', 'DEU', 'JPN', 'GBR', 'USA', 'SGP']:
    # for country in ['USA']:
        truth_series, forecast_with_series, forecast_without_series, organism_series = load_country_data(
            country=country,
            target_subject='NGDP_R',
            start_year=start_year,
            end_year=end_year,
            truth_file=truth_file
        )
        plot_results(
            country=country,
            target_subject='NGDP_R',
            start_year=start_year,
            end_year=end_year,
            truth_file=truth_file
        )

