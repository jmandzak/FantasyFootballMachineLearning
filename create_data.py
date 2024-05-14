from __future__ import annotations

import pandas as pd

def create_dataframe(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)

def main() -> None:
    STATS_22_FILE = 'stats/2022.csv'
    STATS_23_FILE = 'stats/2023.csv'

    df_22 = create_dataframe(STATS_22_FILE)
    df_23 = create_dataframe(STATS_23_FILE)

    # get the columns PLAYER NAME and PPR_AVG_FAN PTS from the 2023 dataset
    df_23 = df_23[['PLAYER NAME', 'PPR_AVG_FAN PTS']]
    # Rename column to 2023_FINAL_PTS
    df_23 = df_23.rename(columns={'PPR_AVG_FAN PTS': 'ACTUAL_PPG'})

    # Add the PPR_AVG_FAN PTS values to the 2022 dataset with the column name 2023_FINAL_PTS using the PLAYER NAME as the key
    df_22 = df_22.merge(df_23, how='left', left_on='PLAYER NAME', right_on='PLAYER NAME')
    
    # Remove any rows with NaN values in the 2023_FINAL_PTS column
    df_22 = df_22.dropna(subset=['ACTUAL_PPG'])

    # Save the final dataset to a new csv file
    df_22.to_csv('stats/combined_data.csv', index=False)

if __name__ == "__main__":
    main()