import time

import pandas as pd
import numpy as np
import random

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 14)

# initialize columns
columns = ["ID", "Cell", "Cell Type", "Path", "Group", "Arc", "Fanout", "Drive Strength","Load" "Trans", "Delay", "Arrival",
           "VT-Class", "Effort"]
pba = pd.DataFrame(columns=columns)
gba = pd.DataFrame(columns=columns)
effortsDict = np.load("EffortDictionary.npy",allow_pickle='TRUE').item()

dictGBA = {}
dictPBA = {}
counterGBA = 0
counterPBA = 0

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
driveStrength = None

dataPath = None


# read file and fill dataframe

print("Generating new dataframes")

timeStart = time.time()

with open("reg2reg_50k_gba_r2_fc.rpt", "r") as a_file:
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
                next(a_file, None)  # skip the next 3 lines of parsing
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
                driveStrength = ""
                if "BWP" in cell:
                    firstHalf = cell[0:cell.index("BWP")]
                    for index in range(len(firstHalf) - 1, 0, -1):
                        if firstHalf[index].isdigit():
                            driveStrength = firstHalf[index] + driveStrength
                        else:
                            break
                else:
                    driveStrength = 6

                # Determine Cell Type
                cellType = cell[0:3]
                try:
                    vtclass = cell[cell.index("NOD") + 3:]
                except:
                    vtclass = "LVT"

                tempString = tempString[tempString.index(" "):].strip()
                arc = tempString[:tempString.index(" ")]

                # Have we found the starting flip flop?
                if not(cell.startswith("SDF") or cell.startswith("MB")) and not dataPath:
                    tempString = a_file.readline().strip()
                    continue
                else:
                    dataPath = True





                tempString = tempString[tempString.index(" "):].strip()
                fanout = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                load = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                trans = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                delay = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                tempString = tempString[tempString.index(" "):].strip()


                tempString = tempString[tempString.index(" "):].strip()
                arrival = tempString[:tempString.index(" ")]


                newRow = {"ID": id, "Cell": cell, "Cell Type": cellType, "Path": path, "Group": group, "Arc": arc,
                          "Fanout": fanout, "Load": load, "Trans": trans, "Delay": delay, "Arrival": arrival,
                          "VT-Class": vtclass, "Effort": effort, "Drive Strength": driveStrength}

                if dataPath and not (arc == "-"):
                    #print(newRow)
                    dictGBA[counterGBA] = newRow
                    counterGBA = counterGBA + 1

                tempString = a_file.readline().strip()
print(counterGBA)
gba = pd.DataFrame.from_dict(dictGBA, "index")
with open("reg2reg_50k_pba_r2.rpt", "r") as a_file:
    for line in a_file:

        stripped_line = line.strip()
        #print(stripped_line)

        if (stripped_line.startswith("Path")):
            path = stripped_line[5:stripped_line.index(":")]
            #counter += 1
            #print(path)
        elif (stripped_line.startswith("Group")):
            group = stripped_line[7:]
        elif (stripped_line.startswith("# Pin")):
            pbaArrival = 0
            dataPath = False
            for index in range(4):
                next(a_file, None)  # skip the next 4 lines of parsing
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
                driveStrength = ""
                if "BWP" in cell:
                    firstHalf = cell[0:cell.index("BWP")]
                    for index in range(len(firstHalf) - 1, 0, -1):
                        if firstHalf[index].isdigit():
                            driveStrength = firstHalf[index] + driveStrength
                        else:
                            break
                else:
                    driveStrength = 6

                # Determine Cell Type
                cellType = cell[0:3]
                try:
                    vtclass = cell[cell.index("NOD") + 3:]
                except:
                    vtclass = "LVT"

                tempString = tempString[tempString.index(" "):].strip()
                arc = tempString[:tempString.index(" ")]






                # Have we found the starting flip flop?
                if not(cell.startswith("SDF") or cell.startswith("MB")) and not dataPath:
                    tempString = a_file.readline().strip()
                    continue
                else:
                    dataPath = True





                tempString = tempString[tempString.index(" "):].strip()
                fanout = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                load = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                trans = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                delay = tempString[:tempString.index(" ")]

                tempString = tempString[tempString.index(" "):].strip()
                tempString = tempString[tempString.index(" "):].strip()

                tempString = tempString[tempString.index(" "):].strip()
                arrival = tempString[:tempString.index(" ")]


                newRow = {"ID": id, "Cell": cell, "Cell Type": cellType, "Path": path, "Group": group, "Arc": arc,
                          "Fanout": fanout, "Trans": trans, "Delay": delay, "Arrival": arrival,
                          "VT-Class": vtclass, "Effort": effort, "Drive Strength": driveStrength}

                if dataPath and not (arc == "-"):
                    dictPBA[counterPBA] = newRow
                    counterPBA = counterPBA + 1

                tempString = a_file.readline().strip()
print(counterPBA)
pba = pd.DataFrame.from_dict(dictPBA, "index")


#print(gba)
#print(pba)

print(time.time() - timeStart)

print("Just finished looping")
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
#print(cellTypes)
cellTypeDict = {cellTypes[i]: i + 1 for i in range(0, len(cellTypes))}

gba["Cell Type"] = [cellTypeDict.get(k, 1) for k in gba["Cell Type"].tolist()]
pba["Cell Type"] = [cellTypeDict.get(k, 1) for k in pba["Cell Type"].tolist()]

print(gba)
print(pba)


#gba.to_pickle("gba.pkl")
gba.to_csv("gba.csv")
#pba.to_pickle("pba.pkl")
pba.to_csv("pba.csv")
print(time.time() - timeStart)
