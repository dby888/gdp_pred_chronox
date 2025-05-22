import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import prepare_data
from chronos import BaseChronosPipeline
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import visualization_part
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
        df_long_filtered = df_long[df_long['Year'] < file_year]
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

def run_rolling_forecast(country, target_subject, forecast_year, horizon, truth_file, cov_tensor_dict=None):
    # 初始化模型
    local_path = r"D:\huggingFaceModels\chronos-t5-large"
    chronos_pipeline = BaseChronosPipeline.from_pretrained(
        local_path,
        device_map="cuda",
        torch_dtype=torch.float16,
    )

    # 读取真实值（可选）
    truth_df = pd.read_excel(truth_file, skiprows=1)
    truth_df.columns = truth_df.columns.map(str)

    # 准备结果目录
    os.makedirs("result_rolling", exist_ok=True)
    forecast_result = pd.DataFrame()

    # 读取WEO数据
    file_name = f"WEOApr{forecast_year}all.xlsx"
    df = pd.read_excel(file_name)
    df.columns = df.columns.map(str)
    df_long = df.melt(id_vars=[df.columns[0], df.columns[1]], value_vars=df.columns[2:],
                      var_name='Year', value_name='Value')
    df_long.columns = ['ISO', 'WEO Subject Code', 'Year', 'Value']
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

    # 仅保留历史数据（< forecast_year）
    df_hist = df_long[(df_long['Year'] < forecast_year) &
                      (df_long['ISO'] == country) &
                      (df_long['WEO Subject Code'] == target_subject)].copy()
    df_hist['variable'] = df_hist['ISO'] + "_" + df_hist['WEO Subject Code'].astype(str)
    df_pivot = df_hist.pivot(index='Year', columns='variable', values='Value').dropna()
    df_pivot = df_pivot.astype(float)

    for col in df_pivot.columns:
        context = torch.tensor(df_pivot[col].tolist(), dtype=torch.float32)
        cov_tensor = cov_tensor_dict.get(forecast_year, None) if cov_tensor_dict else None

        preds_with = []
        preds_without = []

        for i in range(0, horizon+1):
            year = forecast_year + i

            # with NFCI
            if cov_tensor is not None:
                pred_with = chronos_pipeline.predict(
                    context, 1, num_samples=20,
                    top_p=1.0, top_k=50, temperature=1.0,
                    covariates=cov_tensor
                )
                mean_with = np.mean([pred_with[j][0][0].item() for j in range(pred_with.size(0))])
                preds_with.append(mean_with)
            else:
                preds_with.append(np.nan)

            # without NFCI
            pred_without = chronos_pipeline.predict(
                context, 1, num_samples=20,
                top_p=1.0, top_k=50, temperature=1.0
            )
            mean_without = np.mean([pred_without[j][0][0].item() for j in range(pred_without.size(0))])
            preds_without.append(mean_without)

            # 更新 context 用上一次预测的 without 值
            context = torch.cat([context, torch.tensor([mean_without], dtype=torch.float32)])

            # 协变量随之滚动（复制最后一个时间片）
            if cov_tensor is not None:
                cov_tensor = torch.cat([cov_tensor, cov_tensor[:, -1:].clone()], dim=1)

            # 保存结果
            forecast_result.loc[year, f'{col}_with_NFCI'] = preds_with[-1]
            forecast_result.loc[year, f'{col}_without_NFCI'] = preds_without[-1]


    # 保存
    output_path = f"result_rolling/{country}_{target_subject}_forecast_results.csv"
    forecast_result.to_csv(output_path)
    print(f"Rolling forecast saved: {output_path}")


def prepare_covariates(start_year, end_year, country):
    # test_df = prepare_data.process_nfci_features()
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
def plot_results(country, target_subject, start_year, end_year, horizon, truth_file):
    os.makedirs("figure", exist_ok=True)
    os.makedirs("figure_rolling", exist_ok=True)

    truth_df = pd.read_excel(truth_file, skiprows=1)
    truth_df.columns = truth_df.columns.map(str)
    years = list(range(start_year, end_year+1))
    years_rolling = list(range(start_year, start_year+horizon+1))

    target_col_with = f"{country}_{target_subject}_with_NFCI"
    target_col_without = f"{country}_{target_subject}_without_NFCI"

    all_forecasts = pd.read_csv(f'result_csv/{country}_{target_subject}_forecast_results.csv', index_col=0)
    all_forecasts.index = all_forecasts.index.astype(int)
    all_forecasts = all_forecasts.loc[start_year:end_year]

    all_forecasts_rolling = pd.read_csv(f'result_rolling/{country}_{target_subject}_forecast_results.csv', index_col=0)
    all_forecasts_rolling.index = all_forecasts_rolling.index.astype(int)
    all_forecasts_rolling = all_forecasts_rolling.loc[start_year:start_year+horizon]

    forecast_with_vals = None
    if target_col_with in all_forecasts.columns:
        forecast_with_vals = all_forecasts[target_col_with].values

    forecast_without_vals = all_forecasts[target_col_without].values if target_col_without in all_forecasts.columns else None

    forecast_with_vals_rolling = None
    if target_col_with in all_forecasts_rolling.columns:
        forecast_with_vals_rolling = all_forecasts_rolling[target_col_with].values

    forecast_without_vals_rolling = all_forecasts_rolling[
        target_col_without].values if target_col_without in all_forecasts_rolling.columns else None

    organism_df = pd.read_csv(f'result_csv/{country}_{target_subject}_WEO_predictions.csv', index_col=0)
    organism_df.columns = ['prediction']
    organism_df.index = organism_df.index.astype(int)
    organism_df = organism_df.loc[start_year:end_year]
    organism_df_rolling = organism_df.loc[start_year:start_year+horizon]
    organism_vals = organism_df['prediction'].values
    organism_vals_rolling = organism_df_rolling['prediction'].values

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

    truth_vals_rolling = []
    for y in years_rolling:
        if str(y) in truth_df.columns:
            row = truth_df[(truth_df['ISO'] == country) & (truth_df['WEO Subject Code'] == target_subject)]
            if y > row["Estimates Start After"].values[0]:
                truth_vals_rolling.append(np.nan)
                continue
            if not row.empty and not pd.isna(row[str(y)].values[0]):
                truth_vals_rolling.append(float(row[str(y)].values[0]))
                continue

        truth_vals_rolling.append(np.nan)

    save_dir = os.path.join("figure", country)
    save_dir_rolling = os.path.join("figure_rolling", country)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_rolling, exist_ok=True)

    # 1️⃣ 折线图
    save_path = os.path.join(save_dir, f'{country}_{target_subject}_pred_vs_true_compare.png')
    save_path_rolling = os.path.join(save_dir_rolling, f'{country}_{target_subject}_pred_vs_true_compare.png')
    visualization_part.draw_line_chart(forecast_with_vals, forecast_without_vals, organism_vals, truth_vals, years, save_path, country)
    print(len(forecast_with_vals_rolling),len(organism_vals_rolling),len(truth_vals_rolling))
    visualization_part.draw_line_chart(forecast_with_vals_rolling, forecast_without_vals_rolling, organism_vals_rolling, truth_vals_rolling, years_rolling, save_path_rolling, country)

    # ✅ 保存 CSV 数据
    data = {
        'Year': years,
        'Chronos_with_NFCI': forecast_with_vals if forecast_with_vals is not None else [None] * len(years),
        'Chronos_without_NFCI': forecast_without_vals if forecast_without_vals is not None else [None] * len(years),
        'WEO': organism_vals,
        'Truth': truth_vals
    }

    data_rolling = {
        'Year': years_rolling,
        'Chronos_with_NFCI': forecast_with_vals_rolling if forecast_with_vals is not None else [None] * len(years_rolling),
        'Chronos_without_NFCI': forecast_without_vals_rolling if forecast_without_vals is not None else [None] * len(years_rolling),
        'WEO': organism_vals_rolling,
        'Truth': truth_vals_rolling
    }

    df_out = pd.DataFrame(data)
    df_out_rolling = pd.DataFrame(data_rolling)
    csv_path = os.path.join(save_dir, f'{country}_{target_subject}_pred_vs_true_compare.csv')
    csv_path_rolling = os.path.join(save_dir_rolling, f'{country}_{target_subject}_pred_vs_true_compare.csv')
    df_out.to_csv(csv_path, index=False)
    df_out_rolling.to_csv(csv_path_rolling, index=False)
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
    # countries = ['USA']
    target_subject = 'NGDP_R'
    # target_subject = 'NGDP'
    start_year = 2010
    end_year = 2025
    truth_file = 'copieofWEO2025.xlsx'


    # for country in countries:
    #     cov_tensor_dict = prepare_covariates(start_year, end_year, country)
    #     run_forecast(country, target_subject, start_year, end_year, truth_file, cov_tensor_dict)

    # for country in countries:
    #     cov_tensor_dict = prepare_covariates(start_year, end_year, country)
    #     run_rolling_forecast(country, target_subject, start_year, 10, truth_file, cov_tensor_dict)

    # 绘图部分
    for country in countries:
        plot_results(country, target_subject, start_year, end_year, 10, truth_file)
