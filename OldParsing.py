import time

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
pd.set_option('display.max_columns', 14)

# initialize columns
columns = ["ID", "Cell", "Cell Type", "Path", "Group", "Arc", "Fanout", "Load", "Trans", "Delay", "Arrival",
           "VT-Class", "Effort"]
df = pd.DataFrame(columns=columns)
pba = pd.DataFrame(columns=columns)
gba = pd.DataFrame(columns=columns)
effortsDict = np.load("EffortDictionary.npy",allow_pickle='TRUE').item()

dictDF = {}
dictGBA = {}
dictPBA = {}
counterDF = 0
counterGBAPBA = 0

# Set columns to null

path = None
group = None
id = None
cell = None
arc = None
fanout = None
load = None
trans = None
delay = None
arrival = None
vtclass = None
cellType = None
effort = None

pbaTrans = None
pbaDelay = None
pbaArrival = None
pbaNewRow = None

# read file and fill dataframe

print("Generating new dataframes")

timeStart = time.time()

with open("setup.gba.rpt", "r") as a_file:
    for line in a_file:

        stripped_line = line.strip()
        #print(stripped_line)

        if (stripped_line.startswith("Path")):
            path = stripped_line[5:stripped_line.index(":")]
        elif (stripped_line.startswith("Group")):
            group = stripped_line[7:]
        elif (stripped_line.startswith("Timing Path")):
            pbaArrival = 0
            dataPath = False
            for index in range(6):
                next(a_file, None)  # skip the next 6 lines of parsing
            tempString = a_file.readline().strip()

            while not tempString.startswith("#-"):

                #print(tempString)
                start = time.time()

                id = tempString[:tempString.index(" ")]
                tempString = tempString[tempString.index(" "):].strip()

                cell = tempString[0:tempString.index(" ")]
                try:
                    effort = effortsDict[cell]
                except:
                    effort = 0.5

                # Determine Cell Type
                cellType = cell[0:3]
                try:
                    vtclass = cell[cell.index("NOD") + 3:]
                except:
                    vtclass = "LVT"

                # Have we found the starting flip flop?
                if not(cell.startswith("SDF") or cell.startswith("MB2")) and not dataPath:
                    tempString = a_file.readline().strip()
                    continue
                else:
                    dataPath = True

                tempString = tempString[tempString.index(" "):].strip()
                arc = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                fanout = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                load = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                trans = tempString[:tempString.index(" ")]

                # make random pba value
                randomNumber = random.uniform(0.91, 0.98)
                pbaTrans = round(float(trans) * randomNumber, 3)

                tempString = tempString[tempString.index(" "):].strip()
                delay = tempString[:tempString.index(" ")]

                pbaDelay = round(float(delay) * randomNumber, 3)
                pbaArrival = pbaArrival + pbaDelay

                tempString = tempString[tempString.index(" "):].strip()
                arrival = tempString[:tempString.index(" ")]

                newRow = {"ID": id, "Cell": cell, "Cell Type": cellType, "Path": path, "Group": group, "Arc": arc,
                          "Fanout": fanout, "Load": load, "Trans": trans, "Delay": delay, "Arrival": arrival,
                          "VT-Class": vtclass, "Effort": effort}
                #df = df.append(newRow, ignore_index=True)
                dictDF[counterDF] = newRow
                pbaNewRow = {"ID": id, "Cell": cell, "Cell Type": cellType, "Path": path, "Group": group,
                             "Arc": arc, "Fanout": fanout, "VT-Class": vtclass, "Load": load, "Trans": pbaTrans,
                             "Delay": pbaDelay, "Arrival": pbaArrival}

                if dataPath and not (arc == "-"):
                    dictGBA[counterGBAPBA] = newRow
                    dictPBA[counterGBAPBA] = pbaNewRow
                    counterGBAPBA = counterGBAPBA + 1
                    #pba = pba.append(pbaNewRow, ignore_index=True)
                    #gba = gba.append(newRow, ignore_index=True)

                tempString = a_file.readline().strip()
                counterDF = counterDF + 1

df = pd.DataFrame.from_dict(dictDF, "index")
gba = pd.DataFrame.from_dict(dictGBA, "index")
pba = pd.DataFrame.from_dict(dictPBA, "index")

#print(df)
#print(gba)
#print(pba)


print(time.time() - timeStart)

# add a drive strength column to the data

print("Just finished looping")


def getDriveStrength(cell):
    columns = ["Drive Strength"]
    df1 = pd.DataFrame(columns=columns)
    itemNumber = 0
    for eachCell in cell:
        driveStrength = ""
        if "BWP" in eachCell:
            firstHalf = eachCell[0:eachCell.index("BWP")]
            for index in range(len(firstHalf) - 1, 0, -1):
                if firstHalf[index].isdigit():
                    driveStrength = firstHalf[index] + driveStrength
                else:
                    break
        else:
            driveStrength = 6
        df1.at[itemNumber, "Drive Strength"] = driveStrength
        itemNumber += 1

    return df1


df["Drive Strength"] = getDriveStrength(df["Cell"])
pba["Drive Strength"] = getDriveStrength(pba["Cell"])
gba["Drive Strength"] = getDriveStrength(gba["Cell"])

# change the VT-class column in the pba/gba dataframes to a numeric value

vtClassDictionary = {
    "SVT": 1,
    "LVTLL": 2,
    "LVT": 3,
    "ULVTLL": 4,
    "ULVT": 5,
    "ELVT": 6
}
gba["VT-Class"] = [vtClassDictionary.get(k, 1) for k in gba["VT-Class"].tolist()]
pba["VT-Class"] = [vtClassDictionary.get(k, 1) for k in pba["VT-Class"].tolist()]

# Add column for the Cell Type (Inverter, Buffer, NOR, etc...)

cellTypeDictionary = {
    "AN": 1,
    "AO": 2,
    "AOI": 3,
    "BUFF": 4,
    "BUFT": 5,
    "IAO": 6,
    "IND": 7,
    "INR": 8,
    "IIND": 9,
    "IINR": 10,
    "INV": 11,
    "IOA": 12,
    "MAOI": 13,
    "MOAI": 14,
    "MUX": 15,
    "MUXxN": 16,
    "ND": 17,
    "NR": 18,
    "OA": 19,
    "OAI": 20,
    "OR": 21,
    "XNR": 22,
    "XOR": 23,
    "ANTENNA": 24,
    "BHD": 25,
    "CKBX": 26,
    "CKLNQ": 27,
    "CKLHQ": 28,
    "CKNX": 29,
    "CKND2": 30,
    "CKAN2": 31,
    "CKXOR2": 32,
    "CLMUX2": 33,
    "DCAP": 34,
    "DEL": 35,
    "FA1": 36,
    "HA1": 37,
    "FILL": 38,
    "TIEH": 39,
    "TIEL": 40,
    "TOH": 41,
    "TOL": 42,
    "TOHL": 43,
    "HDRSI": 44,
    "HDRDI": 45,
    "HDRDIAON": 46,
    "FTRSI": 47,
    "FTRDI": 48,
    "PTDFCN": 49,
    "FTRDIAON": 50,
    "PTINV": 51,
    "PTBUFF": 52,
    "RSDFC": 53,
    "RSDFCR": 54,
    "RSDFCS": 55,
    "RSDFCSR": 56,
    "SDF": 57,
    "SRAM": 58
}

cellTypes = gba["Cell Type"].unique()
print(cellTypes)
cellTypeDict = {cellTypes[i]: i + 1 for i in range(0, len(cellTypes))}

gba["Cell Type"] = [cellTypeDict.get(k, 1) for k in gba["Cell Type"].tolist()]
pba["Cell Type"] = [cellTypeDict.get(k, 1) for k in pba["Cell Type"].tolist()]

print(gba)

gba.to_pickle("gba.pkl")
gba.to_csv("gba.csv")
pba.to_pickle("pba.pkl")
pba.to_csv("pba.csv")
df.to_pickle("setup.gba_timing.pkl")
df.to_csv("setup.gba_timing.csv")
print(time.time() - timeStart)
