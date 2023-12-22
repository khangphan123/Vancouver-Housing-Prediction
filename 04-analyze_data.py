import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def custom_formatter(x, pos):
    return "%1.1fM" % (x * 1e-6)

def closest_point(point, points):
    """ Based on https://stackoverflow.com/a/39318808 CC BY-SA 3.0 Find closest point from a list of points. """
    dist = cdist([point], points)
    return np.min(dist)

def main(adjusted_house_prices_path, adjusted_house_prices_2023_path, pid_coords_path, skytrains_path):
    adjustedHousePrice = pd.read_csv(adjusted_house_prices_path)
    adjusted_house_price_2023 = pd.read_csv(adjusted_house_prices_2023_path)
    pid_coords = pd.read_csv(pid_coords_path)
    skytrains = pd.read_csv(skytrains_path)

    fit = stats.linregress(adjustedHousePrice['Year'], adjustedHousePrice["Adjusted_Price"])
    plt.figure(figsize=(10, 6))
    plt.xticks([2019, 2020, 2021, 2022, 2023])
    prediction = adjustedHousePrice['Year']*fit.slope + fit.intercept
    adjustedHousePrice["prediction"] = prediction
    min_value = adjustedHousePrice["prediction"].min()
    max_value = adjustedHousePrice["prediction"].max()
    # plt.ylim(min_value, max_value)
    plt.ylim(1.5e6, 5.5e6) 
    ## Sample the data from 200k houses to 70000, to make the graph look readable
    sampled_data = adjustedHousePrice.sample(n=70000)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.plot(sampled_data['Year'], sampled_data["Adjusted_Price"], 'b.', alpha = 0.01 )
    plt.plot(sampled_data['Year'], sampled_data["prediction"], 'r-', linewidth = 2)
    plt.xlabel('Year')
    plt.ylabel('House Price')
    plt.title("House price over the years")
    plt.savefig("linregress.png")
    
   
    print("Slope: " + str(fit.slope) + " intercept: " + str(fit.intercept) + " p-value: "+ str(fit.pvalue))
    
    # Null hypothesis: Average House price does not increase over the year without inflation
    # Alternative Hypothesis: Average House price does increase over the year without inflation
    # p-value = 2.771680685860566e-20 < 0.05
    # At 5 percent confidence level, we reject the null hypothesis. Meaning that Average House price does increase over the year without inflation.
    # this implies that there are other factors that improve house prices



    # Based on https://stackoverflow.com/a/39318808
    adjusted_house_price_2023 = pid_coords.merge(adjusted_house_price_2023, on="PID")
    adjusted_house_price_2023["coords"] = [(lat, lon) for lat, lon in zip(adjusted_house_price_2023["lat"], adjusted_house_price_2023["lon"])]
    skytrains["coords"] = [(lat, lon) for lat, lon in zip(skytrains["lat"], skytrains["lon"])]
    adjusted_house_price_2023["closest_skytrain"] = [closest_point(x, list(skytrains["coords"])) for x in adjusted_house_price_2023["coords"]]
    print(stats.linregress(adjusted_house_price_2023["closest_skytrain"], adjusted_house_price_2023["HOUSE_PRICE_2023_ADJUSTED"]).rvalue)




if __name__ == "__main__":
    adjusted_house_prices = sys.argv[1]
    adjusted_house_prices_2023 = sys.argv[2]
    pid_coords = sys.argv[3]
    skytrains = sys.argv[4]
    main(adjusted_house_prices, adjusted_house_prices_2023, pid_coords, skytrains)
