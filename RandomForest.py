import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import RandomizedSearchCV

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

pba = pd.read_csv("pba.csv")
gba = pd.read_csv("gba.csv")
#print(pba)
#print(gba)

combined = gba
combined.insert(11, "PBA Delay", pba["Delay"], True)
combined.insert(5, "PBA Path", pba["Path"], True)
combined.insert(1, "PBA ID", pba["ID"], True)
combined.insert(3, "PBA Cell", pba["Cell"], True)
print(combined)

combined["Delay"] = combined["Delay"].astype(float)
combined["PBA Delay"] = combined["PBA Delay"].astype(float)
combined["PBA Path"] = combined["PBA Path"].astype(int)

print(combined[combined["Path"] == combined["PBA Path"]])
print(combined[combined["Path"] != combined["PBA Path"]])
combined = combined[(combined["Path"] == combined["PBA Path"]) & (combined["ID"] == combined["PBA ID"]) & (combined["Cell"] == combined["PBA Cell"])]
print(combined)

# Random Forest

model = RandomForestRegressor(n_estimators=20, n_jobs = -1, random_state=1)

features = ["Fanout", "Load", "Trans", "Delay", "VT-Class", "Drive Strength", "Effort"]

predictor = combined[features]
target = combined["PBA Delay"]

# Split into the training set and test set

predictorTrain, predictorTest, targetTrain, targetTest = train_test_split(predictor, target, test_size=0.2, random_state=1)

# Train the model
model = model.fit(predictorTrain, targetTrain)

# Predict the PBA values for the test set
predictions = model.predict(predictorTest)



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
print(comparison)
print(comparison[comparison["Error"] > 10])

comparison["Error(ps, abs)"] = 1000 * abs(comparison["Actual PBA Value"] - comparison["Predicted PBA Value"])
comparison["Error(ps)"] = (comparison["Predicted PBA Value"] - comparison["Actual PBA Value"]) * 1000

#Scatterplot for data
plt.scatter(comparison["Actual PBA Value"], comparison["Predicted PBA Value"] ,color='red', s=2)
plt.title('Predictions for PBA Timing using Random Forest')
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

sns.kdeplot(comparison["Error"], bw=0.5)

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

'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(predictorTrain, targetTrain)

print(rf_random.best_params_)

'''