import sys
import math
from pyspark.sql import SparkSession, functions, types
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pyspark.sql.functions import col
import pandas as pd
import seaborn
seaborn.set()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

spark = SparkSession.builder.appName('clean Data').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+                                                                

def custom_formatter(x, pos):
    return '%1.1fM' % (x * 1e-6)

schema = types.StructType([
    types.StructField('PID', types.StringType(), True),
    types.StructField('LEGAL_TYPE', types.StringType(), True),
    types.StructField('FOLIO', types.StringType(), True),
    types.StructField('LAND_COORDINATE', types.LongType(), True),
    types.StructField('ZONING_DISTRICT', types.StringType(), True),
    types.StructField('ZONING_CLASSIFICATION', types.StringType(), True),
    types.StructField('LOT', types.StringType(), True),
    types.StructField('PLAN', types.StringType(), True),
    types.StructField('BLOCK', types.StringType(), True),
    types.StructField('DISTRICT_LOT', types.StringType(), True),
    types.StructField('FROM_CIVIC_NUMBER', types.IntegerType(), True),
    types.StructField('TO_CIVIC_NUMBER', types.IntegerType(), True),
    types.StructField('STREET_NAME', types.StringType(), True),
    types.StructField('PROPERTY_POSTAL_CODE', types.StringType(), True),
    types.StructField('NARRATIVE_LEGAL_LINE1', types.StringType(), True),
    types.StructField('NARRATIVE_LEGAL_LINE2', types.StringType(), True),
    types.StructField('NARRATIVE_LEGAL_LINE3', types.StringType(), True),
    types.StructField('NARRATIVE_LEGAL_LINE4', types.StringType(), True),
    types.StructField('NARRATIVE_LEGAL_LINE5', types.StringType(), True),
    types.StructField('CURRENT_LAND_VALUE', types.LongType(), True),
    types.StructField('CURRENT_IMPROVEMENT_VALUE', types.LongType(), True),
    types.StructField('TAX_ASSESSMENT_YEAR', types.IntegerType(), True),
    types.StructField('PREVIOUS_LAND_VALUE', types.LongType(), True),
    types.StructField('PREVIOUS_IMPROVEMENT_VALUE', types.LongType(), True),
    types.StructField('YEAR_BUILT', types.IntegerType(), True),
    types.StructField('BIG_IMPROVEMENT_YEAR', types.IntegerType(), True),
    types.StructField('TAX_LEVY', types.LongType(), True),
    types.StructField('NEIGHBOURHOOD_CODE', types.StringType(), True),
    types.StructField('REPORT_YEAR', types.IntegerType(), True)
])

def main(property_tax_report_path, inflation_rate):
    data = spark.read.csv(property_tax_report_path, schema = schema,sep=";", header = True)
    inflationRate = spark.read.csv(inflation_rate, header = True)
    # inflationRate.show()    
    data = data.filter(data["ZONING_CLASSIFICATION"] != "Industrial")
    data = data.filter(data["ZONING_CLASSIFICATION"] != "Historial Area")
    data = data.filter(data["ZONING_CLASSIFICATION"] != "Other")
    data = data.filter(data["ZONING_CLASSIFICATION"] != "Limited Agriculture")
    data = data.drop("NARRATIVE_LEGAL_LINE1")
    data = data.drop("NARRATIVE_LEGAL_LINE2")
    data = data.drop("NARRATIVE_LEGAL_LINE3")
    data = data.drop("NARRATIVE_LEGAL_LINE4")
    data = data.drop("NARRATIVE_LEGAL_LINE5")
    data = data.drop("FROM_CIVIC_NUMBER")
    
    data2020 = data.filter(data["REPORT_YEAR"] == 2020)
    data2021 = data.filter(data["REPORT_YEAR"] == 2021)
    data2022 = data.filter(data["REPORT_YEAR"] == 2022)
    data2023 = data.filter(data["REPORT_YEAR"] == 2023)


    # Join with the inflation Rate
    data2020 = data2020.join(inflationRate, data2020.REPORT_YEAR == inflationRate.year)
    data2020 = data2020.withColumnRenamed('Inflation_Rate', 'inflationRate2020')

    data2021 = data2021.join(inflationRate, data2021.REPORT_YEAR == inflationRate.year)
    data2021 = data2021.withColumnRenamed('Inflation_Rate', 'inflationRate2021')

    data2022 = data2022.join(inflationRate, data2022.REPORT_YEAR == inflationRate.year)
    data2022 = data2022.withColumnRenamed('Inflation_Rate', 'inflationRate2022')

    data2023 = data2023.join(inflationRate, data2023.REPORT_YEAR == inflationRate.year)
    data2023 = data2023.withColumnRenamed('Inflation_Rate', 'inflationRate2023')


    # data2020.show(n=1)
    # House price = current land price + current improvement value
    data2020 = data2020.withColumn("HOUSE_PRICE_2020", data2020["CURRENT_LAND_VALUE"] + data2020["CURRENT_IMPROVEMENT_VALUE"])
    data2021 = data2021.withColumn("HOUSE_PRICE_2021", data2021["CURRENT_LAND_VALUE"] + data2021["CURRENT_IMPROVEMENT_VALUE"])
    data2022 = data2022.withColumn("HOUSE_PRICE_2022", data2022["CURRENT_LAND_VALUE"] + data2022["CURRENT_IMPROVEMENT_VALUE"])
    data2023 = data2023.withColumn("HOUSE_PRICE_2023", data2023["CURRENT_LAND_VALUE"] + data2023["CURRENT_IMPROVEMENT_VALUE"])
    # Extract out the House Price in 2019
    data2020 = data2020.withColumn("HOUSE_PRICE_2019", data2020["PREVIOUS_LAND_VALUE"] + data2020["PREVIOUS_IMPROVEMENT_VALUE"])



    
    # Merge into 1 data frame where we have land value of house from 2019-2023

    # Each house is identified by the PID (same for every year)
    AllYear = data2023
    AllYear = AllYear.join(data2022.select("PID", "HOUSE_PRICE_2022", "inflationRate2022"), on="PID", how="left")
    AllYear = AllYear.join(data2021.select("PID", "HOUSE_PRICE_2021", "inflationRate2021"), on="PID", how="left")
    AllYear = AllYear.join(data2020.select("PID", "HOUSE_PRICE_2020", "HOUSE_PRICE_2019", "inflationRate2020"), on="PID", how="left")



    # AllYear.show(n=1)


    # Now, we need to calculate house prices in 2023, 2022, 2021, 2020 in term of the 2019 values. Then we will do linear regression to check if there is an increase in house prices if not accounting for inflation rate
    # : housePrice2023 = price2019 * (1 + inflationRate2023 / 100)
    AllYear = AllYear.withColumn("HOUSE_PRICE_2023_ADJUSTED", AllYear["HOUSE_PRICE_2019"] * (1 + (AllYear["inflationRate2023"] / 100)))
    AllYear = AllYear.withColumn("HOUSE_PRICE_2022_ADJUSTED", AllYear["HOUSE_PRICE_2019"] * (1 + (AllYear["inflationRate2022"] / 100)))
    AllYear = AllYear.withColumn("HOUSE_PRICE_2021_ADJUSTED", AllYear["HOUSE_PRICE_2019"] * (1 + (AllYear["inflationRate2021"] / 100)))
    AllYear = AllYear.withColumn("HOUSE_PRICE_2020_ADJUSTED", AllYear["HOUSE_PRICE_2019"] * (1 + (AllYear["inflationRate2020"] / 100)))

    # AllYear.show(n=1)
    adjustedHousePrice = AllYear.select("HOUSE_PRICE_2019", "HOUSE_PRICE_2020_ADJUSTED",
    "HOUSE_PRICE_2021_ADJUSTED", "HOUSE_PRICE_2022_ADJUSTED", "HOUSE_PRICE_2023_ADJUSTED")
    ## Transform it so we can use the data to do linear regression
    adjustedHousePrice = pd.melt(adjustedHousePrice.toPandas(), var_name='Year', value_name='Adjusted_Price')
    adjustedHousePrice['Year'] = adjustedHousePrice['Year'].str.extract('(\d+)').astype(int)
    adjustedHousePrice = adjustedHousePrice.dropna(subset=['Adjusted_Price'])
    adjustedHousePrice.to_csv("adjusted_house_prices.csv")

    adjustedHousePrice2023 = AllYear.select("PID", "HOUSE_PRICE_2023_ADJUSTED")
    adjustedHousePrice2023.toPandas().to_csv("adjusted_house_prices_2023.csv", index=False)

    # AllYear.show()
    # Lets look at how house prices varies based on Postal Code, we can extract the forward sortation area(FSA)
    AllYear = AllYear.withColumn("FSA", AllYear["PROPERTY_POSTAL_CODE"].substr(1,3))
    # Theres one None in the FSA
    AllYear = AllYear.filter(AllYear["FSA"].isNotNull())

    AllYear.toPandas().to_csv("all_years.csv")
    pid_addresses: pd.DataFrame = AllYear.select("PID", "TO_CIVIC_NUMBER", "STREET_NAME", "PROPERTY_POSTAL_CODE").toPandas()
    pid_addresses["TO_CIVIC_NUMBER"] = pid_addresses["TO_CIVIC_NUMBER"].apply(lambda n: 0 if math.isnan(n) else int(n))
    pid_addresses.to_csv("pid_addresses.csv", index=False)

    # First, we want to see if the year built has any effect on average house prices. (Use 2020 house price)
    housePriceByYearBuilt = AllYear.groupBy("YEAR_BUILT").agg(functions.avg("HOUSE_PRICE_2020").alias('AVG_HOUSE_PRICE_2020'))
    housePriceByYearBuilt = housePriceByYearBuilt[housePriceByYearBuilt["YEAR_BUILT"] < 2020]

#   This gives median house prices

    # housePriceByYearBuilt = AllYear.groupBy("YEAR_BUILT").agg(expr('percentile_approx(HOUSE_PRICE_2020, 0.5)').alias('AVG_HOUSE_PRICE_2020'))

    orderedByYear = housePriceByYearBuilt.orderBy("YEAR_BUILT")
    orderedByYear_pd = orderedByYear.toPandas()
    plt.figure(figsize=(10, 6))
    plt.plot(orderedByYear_pd['YEAR_BUILT'], orderedByYear_pd['AVG_HOUSE_PRICE_2020'], marker='o')
    
    plt.title('Average House Price by Year Built')
    plt.xlabel('Year Built')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.ylabel('Average House Price in 2020')
    plt.grid(True)
    plt.savefig("housePriceByBuiltYear.png")

    # From that graph, it does seems that older house built are more expensive than newer built house

    # Group by VSA and get house prices

    HousePriceByFSA =  AllYear.groupBy("FSA").agg(functions.avg("HOUSE_PRICE_2023").alias('AVG_HOUSE_PRICE_2023'))
    FSA_count = AllYear.groupby("FSA").count()
    FSA_with_more_than_3_observations = FSA_count.filter(FSA_count["count"] > 3)
    HousePriceByFSA = HousePriceByFSA.join(FSA_with_more_than_3_observations, "FSA")
    HousePriceByFSA = HousePriceByFSA.orderBy(col("AVG_HOUSE_PRICE_2023").asc())
    HousePriceByFSA = HousePriceByFSA.toPandas()
    # print(HousePriceByFSA)
    plt.figure(figsize=(10, 6))
    plt.bar(HousePriceByFSA["FSA"], HousePriceByFSA["AVG_HOUSE_PRICE_2023"])
    plt.title('Average House Price by FSA in 2023')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.xlabel('FSA (First 3 letters of Postal Code)')
    plt.xticks(rotation=90)
    plt.ylabel('Average House Price')
    plt.savefig("barchartByFSA.png")
    # print(data2020.count())


    # We want to see if more expensive




    # Machine Learning part:
    dataForML = AllYear.select("HOUSE_PRICE_2019", "HOUSE_PRICE_2020", "HOUSE_PRICE_2021",
     "HOUSE_PRICE_2022", "HOUSE_PRICE_2023", "FSA", "YEAR_BUILT")
    
    dataForML = dataForML.toPandas()
    # Drop any row that has null value in it
    dataForML = dataForML.dropna()

    onehotEncoding = pd.get_dummies(dataForML["FSA"], prefix = "FSA")
    dataForML = pd.concat([dataForML, onehotEncoding], axis = 1)    
    dataForML = dataForML.drop("FSA", axis = 1)

    dataForMLwithout2023HousePrice = dataForML.drop("HOUSE_PRICE_2023", axis = 1)


    X = dataForML.drop(columns=["HOUSE_PRICE_2023"]).values
    y = dataForML['HOUSE_PRICE_2023'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    regressor = RandomForestRegressor(n_estimators=100, random_state=50)
    regressor.fit(X_train, y_train)
    print(regressor.score(X_valid, y_valid))



    X_no_2023 = dataForMLwithout2023HousePrice
    prediction = regressor.predict(X_no_2023)
    dataForMLwithout2023HousePrice["2023Prediction"] = prediction
    print(prediction)

    # Now that we have regressor for 2023, we want to use dataset for 2020,2021,2022,2023 to predict for 2024
    dataForMLToPredict2024 = dataForML.drop("HOUSE_PRICE_2019", axis = 1)
    X_Predict_2024 = dataForMLToPredict2024
    predictionFor2024 = regressor.predict(X_Predict_2024)
    dataForMLToPredict2024['predicted_price_2024'] = predictionFor2024

    df = pd.DataFrame({'truth':y_valid,'prediction': regressor.predict(X_valid) })
    print(df)
    print(dataForMLToPredict2024.iloc[0])
    dataForMLToPredict2024.to_csv('dataForMLToPredict2024.csv', index=False)

# Export dataForMLwithout2023HousePrice to a CSV file
    dataForMLwithout2023HousePrice.to_csv('dataForMLwithout2023HousePrice.csv', index=False)
if __name__ == "__main__":
    property_tax_report = sys.argv[1]
    inflation_rate = sys.argv[2]
    main(property_tax_report, inflation_rate)
