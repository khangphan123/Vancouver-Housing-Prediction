import pandas as pd
import sys

def main(filename):
    df = pd.read_csv(filename, sep=',')
    df = df.iloc[:,0:2]
    df = df.rename(columns={'V41690973': 'TotalCPI'})
    df = df.loc[0:345]
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df[(df["year"] == 2019) | (df["year"] == 2020) | (df["year"] == 2021) | (df["year"] == 2022) | (df["year"] == 2023)]
    df['TotalCPI'] = pd.to_numeric(df['TotalCPI'])
    AverageCPIByYear = df.groupby('year')['TotalCPI'].mean()
    AverageCPIByYear_df = AverageCPIByYear.to_frame(name='TotalCPI')
    AverageCPIByYear_df.reset_index(inplace=True)
    AverageCPIByYear_df.rename(columns={'index': 'year'}, inplace=True)
    baseYear = AverageCPIByYear_df[AverageCPIByYear_df["year"] == 2019]["TotalCPI"][0]
    AverageCPIByYear_df['Inflation_Rate'] = ((AverageCPIByYear_df['TotalCPI'] - baseYear) / baseYear) * 100
    inflationRate = AverageCPIByYear_df.loc[1:4]
    inflationRate = inflationRate[['year', 'Inflation_Rate']]
    print(inflationRate)
    inflationRate.to_csv('inflation_rate.csv')

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/CPI_MONTHLY.csv"
    main(filename)
