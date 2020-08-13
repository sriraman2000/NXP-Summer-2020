import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 19)

pba = pd.read_csv("pba.csv")
gba = pd.read_csv("gba.csv")
print(pba)
print(gba)

combined = gba
combined.insert(11, "PBA Delay", pba["Delay"], True)
combined.insert(5, "PBA Path", pba["Path"], True)
combined.insert(1, "PBA ID", pba["ID"], True)
combined.insert(3, "PBA Cell", pba["Cell"], True)
print(combined)

combined["Delay"] = combined["Delay"].astype(float)
combined["PBA Delay"] = combined["PBA Delay"].astype(float)
combined["PBA Path"] = combined["PBA Path"].astype(int)

combined = combined[(combined["Path"] == combined["PBA Path"]) & (combined["ID"] == combined["PBA ID"]) & (combined["Cell"] == combined["PBA Cell"])]


# Multiple Linear Regression

model = LinearRegression()

features = ["Fanout", "Load", "Trans", "Delay", "VT-Class", "Drive Strength", "Effort"]

predictor = combined[features]
target = combined["PBA Delay"]

# Split into the training set and test set

predictorTrain, predictorTest, targetTrain, targetTest = train_test_split(predictor, target, test_size=0.2, random_state=1)

# Train the model
model = model.fit(predictorTrain, targetTrain)

# Predict the PBA values for the test set
predictions = model.predict(predictorTest)

test_set_rmse = (np.sqrt(mean_squared_error(targetTest, predictions)))

test_set_r2 = r2_score(targetTest, predictions)

print(test_set_rmse)
print(test_set_r2)


comparison = pd.DataFrame(columns=["Actual PBA Value", "Predicted PBA Value"])
comparison["Actual PBA Value"] = targetTest
comparison["Predicted PBA Value"] = predictions  # currently 2611 predictions
comparison["Actual PBA Value"].astype(float)
comparison["Predicted PBA Value"].astype(float)
comparison["Error"] = abs((comparison["Predicted PBA Value"] - comparison["Actual PBA Value"]))/comparison["Actual PBA Value"] * 100
#print(comparison.sort_values("Error"))
print(comparison)
comparison["ID"] = ""
comparison["Path"] = ""
comparison["Cell"] = ""
comparison["GBA Load"] = ""
comparison["GBA Delay"] = np.nan
comparison["GBA Trans"] = np.nan
comparison["Drive Strength"] = np.nan
comparison["Effort"] = np.nan

for row in comparison.index:
    comparison.at[row, "Cell"] = combined.at[row, "Cell"]
    comparison.at[row, "GBA Delay"] = combined.at[row, "Delay"]
    comparison.at[row, "GBA Trans"] = combined.at[row, "Trans"]
    comparison.at[row, "GBA Load"] = combined.at[row, "Load"]
    comparison.at[row, "Drive Strength"] = combined.at[row, "Drive Strength"]
    comparison.at[row, "Effort"] = combined.at[row, "Effort"]
    comparison.at[row, "ID"] = combined.at[row, "ID"]
    comparison.at[row, "Path"] = combined.at[row, "Path"]

comparison["Error(ps, abs)"] = 1000 * abs(comparison["Actual PBA Value"] - comparison["Predicted PBA Value"])
comparison["Error(ps)"] = (comparison["Predicted PBA Value"] - comparison["Actual PBA Value"]) * 1000

#Scatterplot for data
plt.scatter(comparison["Actual PBA Value"], comparison["Predicted PBA Value"] ,color='red', s=2)
plt.title('Predictions for PBA Timing using Multiple Linear Regression')
plt.xlabel('Predicted PBA Time (ns)')
plt.ylabel('Actual PBA Time (ns)')
plt.show()

#histogram for Error
a = np.array(comparison["Error"].tolist(), dtype=float)

print(a.mean())
print(a.std())
print(a.max())
print(a.min())

print(comparison)

# Probability Density Curve

sns.distplot(comparison["Error"], hist=True, kde=True, kde_kws={'linewidth': 2})
plt.legend(prop={'size': 16}, title='Cell')
plt.title('Density Plot for Error')
plt.xlabel('Error (%)')
plt.ylabel('Density')
plt.xlim(0, 30)


plt.show()

# Plot for Absolute Error in picoseconds

sns.kdeplot(comparison["Error(ps, abs)"], bw=0.5)

plt.title('Density Plot for Absolute Error')
plt.xlabel('Error (ps)')
plt.ylabel('Density')
plt.xlim(0, 10)


plt.show()

# Plot for Error in picoseconds

#sns.distplot(comparison["Error(ps)"], hist=True, kde=True, kde_kws={'linewidth': 2} )
sns.kdeplot(comparison["Error(ps)"], bw=0.5)
plt.title('Density Plot for Error')
plt.xlabel('Error (ps)')
plt.ylabel('Density')
plt.xlim(-5, 5)


plt.show()
