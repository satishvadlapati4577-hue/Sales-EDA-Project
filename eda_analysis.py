import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
print("Libraries imported successfully")
#Read csv file 
df = pd.read_csv("Dataset.csv")
print("First 5 rows of the dataset:")
print(df.head())

#Data Shape
print("Dataset shape:")
print(df.shape)

print ("\nDataset Info:")
print (df.info())

#Missing Value Analysis
print ("\nmissing value analysis:")
print (df.isnull().sum())

df["profit"] = df["profit"].fillna(df["profit"].mean())
print ("\nmissing value analysis after filling missing values:")
print (df.isnull().sum())

#correlation Heatmap
plt.figure(figsize=(10,5))
numeric_df = df.select_dtypes(include=["float64", "int64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Profit Discount Analysis
plt.figure(figsize=(10,5))
sns.scatterplot(x='discount', y='profit', data=df)
plt.title("Profit vs Discount")
plt.xlabel("discount")
plt.ylabel("profit")
plt.show()

#Yearly Sales Analysis
plt.figure(figsize=(10,5))
sns.lineplot(x="year", y="sales", data=df, estimator=np.mean, errorbar=None)
plt.title("Total Sales by Year")
plt.xlabel("year")
plt.ylabel("sales")
plt.show()