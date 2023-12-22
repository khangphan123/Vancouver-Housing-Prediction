import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn
from matplotlib.ticker import FuncFormatter
import numpy as np
def custom_formatter(x, pos):
    return '%1.1fM' % (x * 1e-6)
    
def main(data2023, data2024):
    data2023 = pd.read_csv(data2023)
    data2024 = pd.read_csv(data2024)

    data2024["predicted_price_2023"] = data2023["2023Prediction"]
    HousePrice2023 = data2024["HOUSE_PRICE_2023"]
    predictedPrice2023 = data2024["predicted_price_2023"]

    # plt.figure(figsize=(10, 6))
    # seaborn.kdeplot(HousePrice2023, bw_adjust=0.5,label='Actual Prices 2023', shade=True)
    # seaborn.kdeplot(predictedPrice2023, bw_adjust=0.5,label='Predicted Prices 2023', shade=True)

    # # Set the x-axis limits to a reasonable range based on your data
    # plt.xlim([0, 1e7])
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    # plt.title('Density Plot of Actual vs Predicted Prices for 2023')
    # plt.xlabel('Price')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.savefig("compare.png")
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(HousePrice2023, predictedPrice2023, alpha=0.5)
    plt.title('Actual vs Predicted Prices for 2023')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.plot([min(HousePrice2023), max(HousePrice2023)], [min(predictedPrice2023), max(predictedPrice2023)], 'd--')  # Diagonal line for reference
    plt.grid(True)
    plt.ylim([0, 3e7])
    plt.xlim([0, 3e7])

    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.savefig("compare.png")
  

    print("Mean Actual House Price 2023: " + str(np.mean(HousePrice2023)))
    print("Median Actual House Price 2023: " + str(np.median(HousePrice2023)))
    print("Standard deviation Actual House Price 2023: " + str(np.std(HousePrice2023)))


    print("Mean Predicted House Price 2023: " + str(np.mean(predictedPrice2023)))
    print("Median Predicted House Price 2023: " + str(np.median(predictedPrice2023)))
    print("Standard deviation Predicted House Price 2023: " + str(np.std(predictedPrice2023)))


if __name__ == "__main__":
    data2023 = sys.argv[1]
    data2024 = sys.argv[2]
    main(data2023, data2024)
