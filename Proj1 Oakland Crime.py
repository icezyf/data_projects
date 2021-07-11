# Databricks notebook source
# MAGIC %md # Oakland California Crime Data Analysis and Modeling

# COMMAND ----------

# MAGIC %md Programmer: Crystella Yufei Zheng (10/11/2020)

# COMMAND ----------

# MAGIC %md ####Oakland ranks top 2 in the state of California in the past years according to https://www.roadsnacks.net/most-dangerous-cities-in-california/. Here we are going to study and analyze data during Covid-19 2020.

# COMMAND ----------

# MAGIC %md
# MAGIC ##0. Import Packages and Load Data

# COMMAND ----------

# DBTITLE 1,Import packages
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

# DBTITLE 0,Download Crime Watch Data Past 90 Days (prior to 10/05/2020)
data_path = "dbfs:/FileStore/tables/CrimeWatch_Maps_Past_90_Days.csv"
#after upload to Databricks data; then copy the path of location from dbfs

# COMMAND ----------

import urllib.request
display(dbutils.fs.ls("dbfs:/FileStore/tables/CrimeWatch_Maps_Past_90_Days.csv"))

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt = spark.read.format("csv").option("header", "true").load(data_path)
df_opt = df_opt.dropna() #drop all nulls
df_opt.limit(5).display()

# COMMAND ----------

# check the data types of all values and whether there are missing values
df_pd = df_opt.toPandas()
df_pd.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ##1. Data Preprocessing##

# COMMAND ----------

# Convert data type 
from pyspark.sql.functions import to_date
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import hour
from pyspark.sql.functions import year
from pyspark.sql.functions import month
from pyspark.sql.functions import date_format
from pyspark.sql.functions import col
from pyspark.sql.functions import concat
from pyspark.sql.functions import lit

# COMMAND ----------

spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df = df_opt.withColumn("DATE", to_date("DATETIME", "MM/dd/yyyy")).withColumn("DATETIME", to_timestamp("DATETIME", "MM/dd/yyyy hh:mm:ss a")).withColumn("HOUR", hour("DATETIME")).withColumn("MINUTE", date_format("DATETIME", "m")).withColumn("YEAR", year("DATETIME")).withColumn("MONTH", month('DATETIME')).withColumn("DAY", date_format('DATETIME', 'd')).withColumn("DOW", date_format("DATETIME", "E"))
df.limit(5).display()
df.createOrReplaceTempView("crime_df")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct CRIMETYPE)
# MAGIC from crime_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##2. OLAP##

# COMMAND ----------

# MAGIC %md ####Q1 question (OLAP):
# MAGIC #####Write a Spark program that counts the number of crimes for different category

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q1
crimeCategory = df.groupBy('CRIMETYPE').count().orderBy('count', ascending=False)
crimeCategory_top20 = df.groupBy('CRIMETYPE').count().orderBy('count', ascending=False).limit(20)
crimeCategory.display()

# COMMAND ----------

# visualize result
crime_pd_top20 = crimeCategory_top20.toPandas()
plt.figure()
ax = crime_pd_top20.plot(kind='barh', x='CRIMETYPE', y='count', color='green', legend=False, align='center')
ax.set_ylabel('count', fontsize = 14)
ax.set_xlabel('CRIMETYPE', fontsize=14)
plt.xticks(fontsize=10, rotation=90)
plt.title('Top 20 Types of Crimes in Oakland')
display()

# COMMAND ----------

from pyspark.sql.functions import col

# convert “object" into "float"
crimeCategory = crimeCategory.withColumn('count', col('count').cast('float'))

# convert column into list
output = crimeCategory.select('count').collect()
cnts = [i[0] for i in output]

# to check which types criminals tend to do
high_rate = 0
mid_rate = 0
low_rate = 0
population_rate = 435224/100000

for cnt in cnts: 
  if cnt >= 1000:
    high_rate += (cnt/population_rate)
  elif cnt >= 100 and cnt < 999:
    mid_rate += (cnt/population_rate)
  elif cnt < 100:
    low_rate += (cnt/population_rate)
print(f'''High Rate:{round(high_rate, 4)}, 
Mid Rate: {round(mid_rate, 4)}, 
Low Rate: {round(low_rate, 4)}''')
print("Total Rate:" + str(round(high_rate+mid_rate+low_rate, 4)))

# COMMAND ----------

# MAGIC %md #####Q1 Insight:
# MAGIC The population of Oakland, California 2020 is 435,224. During past 90 days, namely Covid-19, per 100,000 people, the high rate of criminal events remains 1359.76; the medium rate is 1672.93. The low rate is 155.78. Total rate is 3188.47. The medium rate occupies 52.47% of overall, which reminds us of California's Prop 47 causing to rise in shoplifting, thefts, criminal activity --- a law that changed certain low-level crimes from potential felonies to misdemeanors. 

# COMMAND ----------

# MAGIC %md ####Q2 question (OLAP):
# MAGIC #####Count the number of crimes for different streets, and visualize your results

# COMMAND ----------

# choose the criminal events that are over or equilvant to 20 times in the streets.
import pyspark.sql.functions as f
crimeStreet = df.groupBy('ADDRESS').count().orderBy('count', ascending = False).filter(f.col('count') >= 20)
display(crimeStreet)

# COMMAND ----------

# visualize result
fig_dims = (15,4)
fig = plt.subplots(figsize = fig_dims)
oak_crime_add = crimeStreet.toPandas()
chart2 = sns.swarmplot(x = oak_crime_add['ADDRESS'], y = oak_crime_add['count'])
chart2.set_xticklabels(chart2.get_xticklabels(), rotation=45, horizontalalignment='right')
display()

# COMMAND ----------

# MAGIC %md #####Q2 Insight:
# MAGIC From the datasets, the top criminal site is 400 7th street in Oakland, followed by 2600 73rd av and 3400 Telegraph Av.

# COMMAND ----------

# MAGIC %md ####Q3 question (OLAP):
# MAGIC #####Count the number of each "Sunday" during past 90 days (7/5/2020 - 10/5/2020) at top 3 criminal sites in Oakland CA. 
# MAGIC *Becase Oakland government did not provide latitude and longitude coordinates in open data, I replaced it by string of addresses instead.*

# COMMAND ----------

df_90days_top3 = spark.sql('select ADDRESS, count(*) as crime_counts from crime_df where DOW = "Sun" and (DATE between "2020-07-05" and "2020-10-05") group by 1 order by 2 desc limit 3')
display(df_90days_top3)

# COMMAND ----------

# visualize result
pd_90days_top3 = df_90days_top3.toPandas()
plt.figure()
ax = pd_90days_top3.plot(kind='barh', x='ADDRESS', y='crime_counts', color='blue', legend=False, align='center')
ax.set_ylabel('Crime Counts', fontsize = 14)
ax.set_xlabel('Address', fontsize=14)
plt.xticks(fontsize=10, rotation=90)
plt.title('Top 3 Criminal Sites in Oakland during Sundays')
display()

# COMMAND ----------

# MAGIC %md ####Q4 question (OLAP):
# MAGIC #####Analyze 2 comparisons from the crime data
# MAGIC 1. Initial Breakout Stage (March, April, May) **vs.** Most Commonly Spreadying Season (July, August, September)
# MAGIC 2. Non-Covid time **vs.** Covid time *(2015; Oakland gov provides latest open data of Crime Statistic 2015 only)*

# COMMAND ----------

# how many cases occurred in every month in 2020
crime_months = df.groupBy('YEAR', 'MONTH').count().orderBy('count', ascending = False).filter(f.col('YEAR') == 2020)
display(crime_months)
crime_months.createOrReplaceTempView('crime_mos_view')

# COMMAND ----------

display(crime_months)

# COMMAND ----------

diff_mos = spark.sql('with begin_mos as (select sum(count) as begin_mos_sum from crime_mos_view where MONTH in (3,4,5)), curr_mos as (select sum(count) as curr_mos_sum from crime_mos_view where MONTH in (7,8,9)) select curr_mos_sum - begin_mos_sum from begin_mos join curr_mos')
diff_mos.display()

# COMMAND ----------

# MAGIC %sql
# MAGIC with begin_cat as (select CRIMETYPE, count(*) as begin_cnts from crime_df where MONTH IN (3,4,5) group by CRIMETYPE order by 2 desc),
# MAGIC      curr_cat as (select CRIMETYPE, count(*) as curr_cnts from crime_df where MONTH IN (7,8,9) group by CRIMETYPE order by 2 desc)
# MAGIC select a.CRIMETYPE, curr_cnts - begin_cnts
# MAGIC from begin_cat a join curr_cat b on a.CRIMETYPE=b.CRIMETYPE
# MAGIC order by 2 desc

# COMMAND ----------

# DBTITLE 0,Download Crime Data 2015
#download the crime 2015 report
data_path2 = "dbfs:/FileStore/tables/Crime_Data_2015.csv"

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis 2015") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt2 = spark.read.format("csv").option("header", "true").load(data_path2)
df_opt2 = df_opt2.dropna() #drop all nulls
df_opt2.limit(5).display()

# COMMAND ----------

spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df_15 = df_opt2.withColumn("DATE", to_date("DATETIME", "MM/dd/yyyy")).withColumn("DATETIME", to_timestamp("DATETIME", "MM/dd/yyyy hh:mm:ss a")).withColumn("HOUR", hour("DATETIME")).withColumn("DOW", date_format("DATETIME", "E")).withColumn("MONTH", month('DATETIME')).withColumn("YEAR", year("DATETIME"))
df_15.display()
df_15.createOrReplaceTempView("crime_df_15")

# COMMAND ----------

diff_year = spark.sql('with 2015_cnts_tb as (select count(distinct CASENUMBER) as 2015_cnts from crime_df_15 where MONTH between 1 and 9), 2020_cnts_tb as (select sum(count) as 2020_cnts from crime_mos_view where MONTH between 1 and 9) select 2015_cnts - 2020_cnts from 2015_cnts_tb join 2020_cnts_tb')
diff_year.display()

# COMMAND ----------

# MAGIC %md #####Q4 Insight:
# MAGIC 1. During the days when pandemic last, total number of criminal events in the begining (March, April, May) is 12,514 fewer than summer (July, August, September). 
# MAGIC     * It is not surprising to see many massive upsurge, such as misdemeanor assault, vandalism, felony assault, robbery, etc. 
# MAGIC     > From 22 May to 22 August, there were more than 10,600 BLM protest events in the United States according to Wikipedia (https://en.wikipedia.org/wiki/Black_Lives_Matter#2020).
# MAGIC     * Due to the governer's administrative order of lockdown, most skyrocketing cases are 'STOLEN VEHICLE', followed by 'MISDEMEANOR ASSAULT' and 'VANDALISM'. Besides that, 'Auto Burglary', 'DOMESTIC VIOLENCE' and 'PETTY THEFT' cannot be ignored.  
# MAGIC 
# MAGIC 2. From the datasets of Crime 2015, criminal occurrences are far more than the pandemic period. Total number of criminal cases in 2015 are 39,105 more than in 2020. The lockdown order may cause duduced decresement.

# COMMAND ----------

# MAGIC %md ####Q5 question (OLAP):
# MAGIC #####Analyze the number of crime w.r.t the hour from pandemic year and from non-Covid year. Then provide suggestion for residents or people who have to go there about the relatively safe time.

# COMMAND ----------

# merge 2 dataframes of hourly crime case counts
crime_hr_cnts_15 = spark.sql('select HOUR, count(*) as hourly_crime_cnts_15 from crime_df_15 group by HOUR order by HOUR')
crime_hr_cnts = spark.sql('select HOUR, count(*) as hourly_crime_cnts from crime_df group by HOUR order by HOUR')

output = crime_hr_cnts_15.join(crime_hr_cnts,['HOUR'],how='outer')
output.display()

# COMMAND ----------

output.display()

# COMMAND ----------

# MAGIC %md #####Q5 Insight:
# MAGIC * There are 5 peaks in the timeline graph - 12am, 12-2pm, 4-6pm, 8-9pm. 
# MAGIC * Residents and must-go visitors try to arrange your activities in the morning. 
# MAGIC * They are also requried to stay alert anytime even though it is at the relatively safe time.

# COMMAND ----------

# MAGIC %md ####Q6 question (OLAP):
# MAGIC * Step 1: Find out the top-3 danger district
# MAGIC * Step 2: Find out the crime event w.r.t category and time (hour) from the result of Step 1
# MAGIC * Step 3: Give your advice to distribute the police based on your analysis results

# COMMAND ----------

# DBTITLE 1,Step 1: Find out the top-3 dangerous districts
top3_zone = df.groupBy('ADDRESS').count().orderBy('count', ascending = False).head(3)
top3_zone_lst = [top3_zone[i][0] for i in range(0,3)]
top3_zone_lst

# COMMAND ----------

display(top3_zone)

# COMMAND ----------

# DBTITLE 1,Step 2: Find out the crime event w.r.t category and time (hour) from the result of Step 1
q6_s2_result = df.filter(df.ADDRESS.isin(top3_zone_lst)).groupby('CRIMETYPE','HOUR').count().orderBy('HOUR')
display(q6_s2_result)

# COMMAND ----------

# DBTITLE 1,Step 3: Give your advice to distribute the police based on your analysis results
# MAGIC %md #####Q6 Advice:
# MAGIC - The top 3 danger streets are 400 7TH ST, 2600 73RD AV and  3400 TELEGRAPH AV.
# MAGIC - From Step 2,  the most dangerous hour in the top 3 danger streets is 2pm, followed by midnight 0am and noon 12pm. Police officers are better stay watch for onsite standby instead of patrol.

# COMMAND ----------

# MAGIC %md ####Q7 question (OLAP):
# MAGIC #####For different category of crime, the percentage of weapon. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

q3_res_1 = spark.sql("select HOUR, count(distinct CASENUMBER) as case_cnt from crime_df where (DESCRIPTION like '%WEAPON%' or DESCRIPTION like '%SHOOT%') and YEAR == 2020 group by HOUR order by 2 desc limit 10")
q3_res_1.display()

# COMMAND ----------

q3_res_2 = spark.sql("select DOW, count(distinct CASENUMBER) as case_cnt from crime_df where (DESCRIPTION like '%WEAPON%' or DESCRIPTION like '%SHOOT%') and YEAR==2020 group by DOW order by 2 desc limit 10")
q3_res_2.display()

# COMMAND ----------

q3_res_3 = spark.sql("select count(distinct CASENUMBER) as case_cnt from crime_df where (DESCRIPTION like '%WEAPON%' or DESCRIPTION like '%SHOOT%') and  YEAR == 2020")
q3_res_3.display()

# COMMAND ----------

# MAGIC %sql
# MAGIC with weapon_cnts as (select count(distinct CASENUMBER) as shoot_cnt from crime_df where (DESCRIPTION like '%WEAPON%' or DESCRIPTION like '%SHOOT%') and  YEAR == 2020), total_cnts as (select count(distinct CASENUMBER) as case_cnt from crime_df where YEAR == 2020)
# MAGIC select shoot_cnt/case_cnt*100 as percentage
# MAGIC from weapon_cnts join total_cnts

# COMMAND ----------

# MAGIC %md #####Q7 Insight:
# MAGIC 1. Usually, 8pm-10pm and Saturdays are the peak times of shoot crime.
# MAGIC 2. It is unbelievable to see the percentage of weapon-related criminal events occupies only 5.6%, which seems that gun use is under control in such an unsafety city.
# MAGIC 3. Police should continue to keep watch for that even thought the percentage is not that high.

# COMMAND ----------

# MAGIC %md ####Q8 question (optional):
# MAGIC #####Analyze the enviornment, housing, races and living status in 2016

# COMMAND ----------

# MAGIC %md * How many parks and recreation facilities in Oakland?

# COMMAND ----------

data_path3 = 'dbfs:/FileStore/tables/Parks_and_Recreation_Facilities.csv'

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("parks and facilities") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_fa = spark.read.format("csv").option("header", "true").load(data_path3)
df_fa = df_fa.dropna()
df_fa.limit(10).display()

# COMMAND ----------

df_fac = df_fa["Parks and Recreation Facilities","Facility Type","Location 1"]
df_fac = df_fac.withColumnRenamed("Facility Type", "FacilityType").withColumnRenamed("Parks and Recreation Facilities", "ParkandRec").withColumnRenamed("Location 1", "Location")
df_fac.limit(10).display()
df_fac.createOrReplaceTempView("fac_view")

# COMMAND ----------

fac_count = df_fac.groupby("FacilityType").count().orderBy('count', ascending = False)
display(fac_count)

# COMMAND ----------

display(fac_count)

# COMMAND ----------

loc_count = df_fac.groupby("Location").count().orderBy('count', ascending = False)
display(loc_count)

# COMMAND ----------

df_fac.groupby("Location").count().orderBy('count', ascending = False).filter(f.col('count') > 1).display()

# COMMAND ----------

# MAGIC %md * What are the percentages of races who own house(s)?

# COMMAND ----------

data_path4 = "dbfs:/FileStore/tables/homeownership.csv"

spark = SparkSession \
    .builder \
    .appName("homeownership") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_hou = spark.read.format("csv").option("header", "true").load(data_path4)
df_hou = df_hou.dropna()
display(df_hou.take(5))

# COMMAND ----------

# MAGIC %md * What are the percentages of races who rent house(s)?

# COMMAND ----------

file_location = "/FileStore/tables/rent_burden.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_rent = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", "true") \
  .option("sep", delimiter) \
  .load(file_location)

display(df_rent.take(5))

# COMMAND ----------

# MAGIC %md * What are rates of races who are homeless?

# COMMAND ----------

#The rates of races who are homeless
file_location = "/FileStore/tables/homelessness.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_homeless = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", "true") \
  .option("sep", delimiter) \
  .load(file_location)

display(df_homeless)

# COMMAND ----------

# MAGIC %md #####Q8 Insight:
# MAGIC - There are still lots of parks and leisure facilities for residents to choose even though criminal events happen almost everyday. Based on the previous analysis of perilous streets and boom hours, residents are still able to chill out in those facilities. Not only that government strived as much as possible on urban planning.
# MAGIC - For races who rent or own houses, all races almost enjoy the relatively same shares except Asians. Asians who rent houses are more than themselves who own houses, of which this finding is bizarre. Asians usually prefer taking possession of houses than renting. Here's my guess that the rental fee is cheaper than other areas so Asians just temporarily rent houses until their saving is enough to afford a house in their dream places and then they will move out of Oakland.
# MAGIC - With respect to homelessness, African American is majority.

# COMMAND ----------

# MAGIC %md
# MAGIC #3. Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC #####The Steps of this Modeling:
# MAGIC 1. Apply StringIndexer and One Hot Encoding on Categorical columns, then assemble all those converted Categorical columns and numeric columns by VectorAssembler; in the end, pipeline all those methods together.
# MAGIC 2. Due to a great many features, we'd better theck whether they are highly correlated.
# MAGIC 3. Dimensionality reduction by PCA
# MAGIC 4. Apply K-Means clustering for unsupervised learning and evaluate by ClusteringEvaluator
# MAGIC 5. Choose the best score according to Silhouette method for the final K
# MAGIC 6. Obtain the prediction and the centers of all clusters throught the K-means clustering model

# COMMAND ----------

# convert Numberic Columns into Float
cols = ['HOUR', 'MINUTE', 'YEAR', 'MONTH', 'DAY']
for col_name in cols:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC ####3.1 Feature Engineering

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
categoricalColumns = ["CRIMETYPE", "ADDRESS", "DOW"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

# Transform all features into a vector using VectorAssembler
numericCols = ["HOUR", "MINUTE", "YEAR", "MONTH", "DAY"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(df)
assembled_df = pipelineModel.transform(df)
assembled_df.show(2)

# COMMAND ----------

# check whether features are highly correlated
from pyspark.ml.stat import Correlation
pearsonCorr = Correlation.corr(assembled_df, "features", "pearson").collect()[0][0]
print("Pearson correlation matrix:")
print(str(pearsonCorr).replace('nan', 'NaN'))

# COMMAND ----------

from pyspark.ml.feature import PCA
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(assembled_df)
pca_df = pca_model.transform(assembled_df)
pca_df.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2 Model Building and Evaluation

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='pcaFeatures', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(2,10):
    KMeans_algo=KMeans(featuresCol='pcaFeatures', k=i)
    KMeans_fit=KMeans_algo.fit(pca_df)
    output=KMeans_fit.transform(pca_df)
    
    score=evaluator.evaluate(output)
    silhouette_score.append(score)
    print("Silhouette with squared euclidean distance = " + str(score))

# COMMAND ----------

# visualize the silhouette scores in a plot
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,10),silhouette_score)
ax.set_xlabel('k')
ax.set_ylabel('cost')

# COMMAND ----------

# MAGIC %md
# MAGIC #####Graph and Score Insight:
# MAGIC I prefer K=5 where a local maxima of Silhouette Score is observed. Therefore, the final model is executed with k=5.

# COMMAND ----------

KMeans_algo=KMeans(featuresCol='pcaFeatures', k=5)
KMeans_fit=KMeans_algo.fit(pca_df)
output=KMeans_fit.transform(pca_df)
ctr = []
centers = KMeans_fit.clusterCenters()
print("Cluster Centers: ")
for center in centers:
      ctr.append(center)
      print(center)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.3 Visualization

# COMMAND ----------

# merge multiple dataframes into one and plot them in 3D
pca_pd = pca_df.toPandas()
output_pd = output.toPandas()
graph_pd = assembled_df.select("CRIMETYPEIndex", "DOWIndex", "HOUR", "features").toPandas()
new_graph_pd = graph_pd.merge(pca_pd, on='features').merge(output_pd,on='pcaFeatures')
new_graph_pd.head(3)

# COMMAND ----------

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection='3d')

ax.scatter3D(new_graph_pd.CRIMETYPEIndex, new_graph_pd.DOWIndex, new_graph_pd.HOUR, c=new_graph_pd.prediction, cmap='viridis', linewidth=1)
ax.scatter()

ax.set_xlabel('CrimeType')
ax.set_ylabel('Day')
ax.set_zlabel('Hour')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.4 Analysis

# COMMAND ----------

output.select('prediction').groupBy('prediction').count().orderBy('prediction').display()

# COMMAND ----------

watch_cluster = output.filter(f.col("prediction") == 1)
watch_cluster.display()
watch_cluster.createOrReplaceTempView('watch_view')

# COMMAND ----------

# MAGIC %sql
# MAGIC select CRIMETYPE, count(*) AS type_cnt
# MAGIC from watch_view
# MAGIC group by 1
# MAGIC order by 2 desc
# MAGIC limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC select HOUR, count(*) as hour_cnt
# MAGIC from watch_view
# MAGIC group by HOUR
# MAGIC order by 2 desc
# MAGIC limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC select DOW, count(*) as day_cnt
# MAGIC from watch_view
# MAGIC group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %sql
# MAGIC with cluster1_tb as (select count(distinct CASENUMBER) as shoot_cnt 
# MAGIC                      from watch_view 
# MAGIC                      where (DESCRIPTION like '%WEAPON%' or DESCRIPTION like '%SHOOT%') and  YEAR == 2020), 
# MAGIC      total_cnts as (select count(distinct CASENUMBER) as case_cnt 
# MAGIC                      from crime_df 
# MAGIC                      where (DESCRIPTION like '%WEAPON%' or DESCRIPTION like '%SHOOT%') and  YEAR == 2020)
# MAGIC select shoot_cnt/case_cnt*100 as percentage
# MAGIC from cluster1_tb join total_cnts

# COMMAND ----------

# MAGIC %md
# MAGIC #####3.4 Insight:
# MAGIC - As we can see, Cluster 1 has the most cases among all clusters. After zooming in all data that belong to the Cluster 1, we can see something in common that criminals will act on STOLEN CAR, VANDALISM, PETTY THEFT at midnight during weekends and Wednesdays. 
# MAGIC - Whenever falling on these factors during patrolling, police should stay sensitively alerted. Tourists and residents should take away valuable belongings whenever they leave the car, keep an eye on bags or wallets and make notice around when they walk outside.
# MAGIC - Gun shot in Cluster 1 occupies 32.72% among all clusters, which is higher than average. 
# MAGIC - All factors in Cluster 1 should be analyzed as detailed as possible and concentrated to monitor and solve; then take turns to the next most cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC #4. Conclustion

# COMMAND ----------

# MAGIC %md
# MAGIC #####Business Insight:
# MAGIC 1. Oakland CA acutally is a perfect place for living as it is close to San Francisco downtown. Especially, housing in San Francisco has been skyrocketing during recent years. Housing in Oakland is definitely lower than in San Francisco and even other bay areas. People who cannot afford housing may consider this substitute. Hence, the factors of crime, governmental-constructed leisure facilities, races of residents, etc need to be analyzed and elaborated for those real estate companies to figure out the potential customers, market value and safety issues.
# MAGIC 
# MAGIC 2. It's not only pandemic but also previous years that Oakland keeps a high trend of ciminal occurences. Perhaps, it causes low market values for lands. If there are other options to develop, Oakland should be the last option. If the group has to choose Oakland CA, location is the vitalest and keep them furthest from downtown.  
# MAGIC 
# MAGIC #####Method:
# MAGIC 1. Make use of spark SQL and pyspark for data cleaning, then data parsing and exploratory and comparison tables building by pyspark Dataframe, pandas and spark SQL.
# MAGIC 2. Take advantage of built-in graphing tools to visualize most of dataframes; others are applied with pythong Matplotlib package.
# MAGIC 3. For modeling and evaluation, pyspark MLlib packages have been applied, such as PCA, correlation, k-means clustering, clustering evaluator, string indexer, one hot encoder, etc.
