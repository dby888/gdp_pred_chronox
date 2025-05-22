import pandas as pd

def process_nfci_features(csv_path='NFCI.csv'):
    """
    从 NFCI.csv 中提取每年Q1（1-3月）的主值和特征，包括：
    - NFCI（1-3月均值）
    - 同比增长（与去年Q1均值比较）
    - 以当年3月为终点的 rolling 3/12月偏离值与 Z-score
    """
    nfci_df = pd.read_csv(csv_path, parse_dates=['observation_date'])
    nfci_df['Year'] = nfci_df['observation_date'].dt.year
    nfci_df['Month'] = nfci_df['observation_date'].dt.month

    # 只取1-3月用于计算 NFCI 主值
    nfci_q1 = nfci_df[(nfci_df['Month'] <= 3) & (nfci_df['Year'] >= 1980) & (nfci_df['Year'] <= 2024)]

    # Step 1：主值 = 每年 Q1 均值
    q1_means = nfci_q1.groupby('Year')['NFCI'].mean().rename('NFCI')

    # Step 2：同比增长
    yoy_growth = q1_means.pct_change().rename('NFCI_Q1_YoY')

    # Step 3：提取每年3月的 observation 日期（作为 rolling 截止点）
    march_rows = nfci_df[nfci_df['Month'] == 3].copy()
    march_rows = march_rows[march_rows['Year'].between(1981, 2024)]  # 从1981开始，因为同比需要前一年
    march_rows = march_rows.sort_values('observation_date')

    rolling_records = []

    for _, row in march_rows.iterrows():
        year = row['Year']
        date = row['observation_date']

        # rolling 区间含当年3月
        window_12m = nfci_df[
            (nfci_df['observation_date'] <= date) &
            (nfci_df['observation_date'] > date - pd.DateOffset(months=12))
            ]['NFCI']

        window_3m = nfci_df[
            (nfci_df['observation_date'] <= date) &
            (nfci_df['observation_date'] > date - pd.DateOffset(months=3))
            ]['NFCI']

        # 确保足够数据再计算
        if len(window_12m) >= 10 and len(window_3m) >= 2:
            mean_12 = window_12m.mean()
            std_12 = window_12m.std()
            mean_3 = window_3m.mean()
            current_val = row['NFCI']

            record = {
                'Year': year,
                'NFCI_Deviation_12M': current_val - mean_12,
                'NFCI_Z_12M': (current_val - mean_12) / std_12 if std_12 != 0 else None,
                'NFCI_Deviation_3M': current_val - mean_3
            }
            rolling_records.append(record)

    rolling_df = pd.DataFrame(rolling_records).set_index('Year')

    # 合并所有特征
    nfci_yearly = pd.concat([q1_means, yoy_growth, rolling_df], axis=1).dropna()

    return nfci_yearly

if __name__ == "__main__":
    process_nfci_features()