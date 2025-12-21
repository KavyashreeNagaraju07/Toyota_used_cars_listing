# Understand the data (Load Dataset & Basic Inspection)
import pandas as pd
import numpy as np
df = pd.read_csv("toyota.csv")
print("heads",df.head()) 
print("Information",df.info())
print(df.columns)

# Data Quality Analysis (Data cleaning - Missing values & duplicates)
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.isnull().mean() * 100) # Mean percentage of misisng  values

# Descriptive Analysis (shows what is happened)
print("Summary statistics",df.describe()) # Summary statistics
print("rows & coloumns", df.shape)
# Mean price by fuel type
print(df.groupby("fuelType")["price"].mean())
# Average price by car model
print(df.groupby("model")["price"].mean().sort_values(ascending=False))

#EDA (Exploratory Data Analysis- Visualization using matplotlib & seaborn)
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg"
import matplotlib.pyplot as plt

# Price distribution
plt.hist(df["price"], bins=20)
plt.xlabel("Price")
plt.ylabel("Count")
plt.title("Price Distribution")
plt.savefig("plot_name.png", dpi=150, bbox_inches="tight")
plt.show()
print("Second plot is executing")
# Price vs Age
import matplotlib.pyplot as plt
plt.scatter(df["year"], df["price"])
plt.xlabel("year")
plt.ylabel("price")
plt.title("Price vs year")
plt.show()

