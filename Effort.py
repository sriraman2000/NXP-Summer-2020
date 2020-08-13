import numpy as np


cell = None
delay = None
newRow = None

dictionary = {}

with open("LogicalEffort.srt", "r") as a_file:
    for line in a_file:
        stripped_line = line.strip()
        cell = stripped_line[0:stripped_line.index(" ")]
        delay = float(stripped_line[stripped_line.index(" ") + 1:])
        dictionary[cell] = 0.004/delay
print(dictionary)
np.save('EffortDictionary', dictionary)