# Analyzing and Predicting Vancouver House Prices
A CMPT 353 project

## Instructions
To reproduce the project findings, follow the instructions below.
1. Gather all data necessary. 
- Run `$ ./00-download-data.sh` to download the Property tax report from the [City of Vancouver Open Data Portal](https://opendata.vancouver.ca/explore/dataset/property-tax-report/information/)
- Other data files have been included in the repo
2. Ensure all dependencies are installed
- PySpark, numpy, pandas, scipy and sklearn for analysis
- matplotlib and seaborn for visualization
- geopy for processing geographic data
3. Run `$ python3 ./01-clean_cpi_data.py data/CPI_MONTHLY.csv`. This will create `inflation_rate.csv` in the project root directory.
4. Run `$ spark-submit ./02-clean_data.py data/property-tax-report.csv inflation_rate.csv` to clean and consolidate the data. This outputs `pid_addresses.csv`, `all_years.csv`, `adjusted_house_prices.csv` and `adjusted_house_prices_2023.csv` for the following pipeline steps. This also outputs `dataForMLToPredict2024.csv` and `dataForMLwithout2023HousePrice.csv` to be evaluated in the final step.
5. Run `$ python3 ./03-add_geo_data.py pid_addresses.csv data/property-addresses.csv` to incorporate public transit data. This outputs `pid_coords.csv`.
6. Run `$ python3 ./04-analyze_data.py adjusted_house_prices.csv adjusted_house_prices_2023.csv pid_coords.csv data/skytrains.csv`.
7. Run `$ python3 ./05-analyze_ml_result.py dataForMLwithout2023HousePrice.csv dataForMLToPredict2024.csv`. This step produces a graphic and analysis data on the ML predictions.

## Data sources and attribution
This project uses data from the City of Vancouver Open Data Portal. The data is licensed under the [Open Government Licence – Vancouver](https://opendata.vancouver.ca/pages/licence/).

CPI data is obtained from the [Bank of Canada website](https://www.bankofcanada.ca/). No changes were made to the data files, but information has been derived from them. The data usage is not endorsed by the Bank of Canada.
