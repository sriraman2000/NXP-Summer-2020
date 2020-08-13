import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

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
#print(combined[combined["Delay"] == 0])
combined["PBA Path"] = combined["PBA Path"].astype(int)


combined = combined[(combined["Path"] == combined["PBA Path"]) & (combined["ID"] == combined["PBA ID"]) & (combined["Cell"] == combined["PBA Cell"])]



# Decision Tree Regressor

model = tree.DecisionTreeRegressor()

features = ["Fanout", "Load", "Trans", "Delay", "Effort", "VT-Class"]

predictor = combined[features]
target = combined["PBA Delay"]

# training set and test set

predictorTrain, predictorTest, targetTrain, targetTest = train_test_split(predictor, target, test_size=0.2,
                                                                          random_state=1)

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
# determine what amount of data falls into each margin of error

pbaListA = comparison["Actual PBA Value"].tolist()
pbaListP = comparison["Predicted PBA Value"].tolist()
errorDistribution = []
for i in range(1, 26):
    numPass = 0

    for index in range(len(comparison["Actual PBA Value"])):
        max = pbaListA[index] * (1 + 0.01 * i)
        min = pbaListA[index] * (1 - 0.01 * i)
        if min <= pbaListP[index] <= max:
            numPass += 1
        proportion = numPass/len(pbaListA)
    print("The amount within " + str(i) + "% is " + str(proportion))
    errorDistribution.append(proportion)
print(errorDistribution)


print(comparison)
print(comparison[comparison["Error"] > 60])

#Scatterplot for data
plt.scatter(comparison["Actual PBA Value"], comparison["Predicted PBA Value"] ,color='red', s=2)
plt.title('Predictions for PBA Timing using Decision Tree Regressor')
plt.xlabel('Predicted PBA Time (ns)')
plt.ylabel('Actual PBA Time (ns)')
plt.show()

#histogram for Error
a = np.array(comparison["Error"].tolist(), dtype=float)

print(a.mean())
print(a.std())
print(a.max())
print(a.min())


#comparison = comparison[comparison["Error"] < 0.2]
#plt.hist(comparison["Error"], bins=30)

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


#testing different max_depths
means = []
stds = []
maxs = []
max_depths = np.linspace(1, 32, 32, endpoint=True)

for maxDepth in max_depths:
    model = tree.DecisionTreeRegressor(max_depth=maxDepth)
    features = ["Load", "Trans", "Delay"]

    predictor = combined[features]
    target = combined["PBA Delay"]

    # training set and test set

    predictorTrain, predictorTest, targetTrain, targetTest = train_test_split(predictor, target, test_size=0.2,
                                                                              random_state=1)

    # Train the model
    model = model.fit(predictorTrain, targetTrain)

    # Predict the PBA values for the test set
    predictions = model.predict(predictorTest)

    comparison = pd.DataFrame(columns=["Actual PBA Value", "Predicted PBA Value"])
    comparison["Actual PBA Value"] = targetTest
    comparison["Predicted PBA Value"] = predictions  # currently 2611 predictions
    comparison["Actual PBA Value"].astype(float)
    comparison["Predicted PBA Value"].astype(float)
    comparison["Error"] = abs((comparison["Predicted PBA Value"] - comparison["Actual PBA Value"])) / comparison[
        "Actual PBA Value"] * 100

    a = np.array(comparison["Error"].tolist(), dtype=float)
    means.append(a.mean())
    stds.append(a.std())
    maxs.append(a.max())

    from matplotlib.legend_handler import HandlerLine2D


plt.title("Mean and Standard Deviation of Error vs. Max Depth")
plt.xlabel("Max Depth of Tree")
plt.ylabel("Mean, Standard Deviation, and Max Error")
line1, = plt.plot(max_depths, means, color="b", label="MEAN")
line2, = plt.plot(max_depths, stds, color="r", label="STD")
line3, = plt.plot(max_depths, maxs, color="g", label="MAX")

plt.legend(loc="upper right")


plt.show()



#testing different min_samples_leaf values
means = []
stds = []
maxs = []
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)

for min_sample in min_samples_leaf:
    model = tree.DecisionTreeRegressor(min_samples_leaf=min_sample)
    features = ["Cell Type", "Fanout", "Load", "Trans", "Delay", "VT-Class", "Drive Strength"]

    predictor = combined[features]
    target = combined["PBA Delay"]

    # training set and test set

    predictorTrain, predictorTest, targetTrain, targetTest = train_test_split(predictor, target, test_size=0.2,
                                                                              random_state=1)

    # Train the model
    model = model.fit(predictorTrain, targetTrain)

    # Predict the PBA values for the test set
    predictions = model.predict(predictorTest)

    comparison = pd.DataFrame(columns=["Actual PBA Value", "Predicted PBA Value"])
    comparison["Actual PBA Value"] = targetTest
    comparison["Predicted PBA Value"] = predictions  # currently 2611 predictions
    comparison["Actual PBA Value"].astype(float)
    comparison["Predicted PBA Value"].astype(float)
    comparison["Error"] = abs((comparison["Predicted PBA Value"] - comparison["Actual PBA Value"])) / comparison[
        "Actual PBA Value"] * 100

    a = np.array(comparison["Error"].tolist(), dtype=float)
    means.append(a.mean())
    stds.append(a.std())
    maxs.append(a.max())

    from matplotlib.legend_handler import HandlerLine2D


plt.title("Mean and Standard Deviation of Error vs. Min Sample Leaf")
plt.xlabel("Min Samples per Leaf of Tree")
plt.ylabel("Mean, Standard Deviation, and Max Error")
line1, = plt.plot(min_samples_leaf, means, color="b", label="MEAN")
line2, = plt.plot(min_samples_leaf, stds, color="r", label="STD")
line3, = plt.plot(min_samples_leaf, maxs, color="g", label="MAX")

plt.legend(loc="upper right")
plt.show()



#testing different min_samples_split values
means = []
stds = []
maxs = []
min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)

for min_sample_split in min_samples_split:
    model = tree.DecisionTreeRegressor(min_samples_split=min_sample_split)
    features = ["Cell Type", "Fanout", "Load", "Trans", "Delay", "VT-Class", "Drive Strength"]

    predictor = combined[features]
    target = combined["PBA Delay"]

    # training set and test set

    predictorTrain, predictorTest, targetTrain, targetTest = train_test_split(predictor, target, test_size=0.2,
                                                                              random_state=1)

    # Train the model
    model = model.fit(predictorTrain, targetTrain)

    # Predict the PBA values for the test set
    predictions = model.predict(predictorTest)

    comparison = pd.DataFrame(columns=["Actual PBA Value", "Predicted PBA Value"])
    comparison["Actual PBA Value"] = targetTest
    comparison["Predicted PBA Value"] = predictions  # currently 2611 predictions
    comparison["Actual PBA Value"].astype(float)
    comparison["Predicted PBA Value"].astype(float)
    comparison["Error"] = abs((comparison["Predicted PBA Value"] - comparison["Actual PBA Value"])) / comparison[
        "Actual PBA Value"] * 100

    a = np.array(comparison["Error"].tolist(), dtype=float)
    means.append(a.mean())
    stds.append(a.std())
    maxs.append(a.max())

    from matplotlib.legend_handler import HandlerLine2D


plt.title("Mean and Standard Deviation of Error vs. Min Sample Split")
plt.xlabel("Min Samples per Leaf of Tree")
plt.ylabel("Mean, Standard Deviation, and Max Error")
line1, = plt.plot(min_samples_split, means, color="b", label="MEAN")
line2, = plt.plot(min_samples_split, stds, color="r", label="STD")
line3, = plt.plot(min_samples_split, maxs, color="g", label="MAX")

plt.legend(loc="upper right")


plt.show()


