"""Write a python program to peform the following steps:

1. Import the matplotlib library

2. Load a tab delimited file named "coors.txt" containing XYZ coordinates

3. Create a heat map of those coordinates using the matplotlib python library. Bins that have many points should be colored red, and bins with fewer should be blue.

4. Save the heat map to a PNG file."""

# 1. Import the matplotlib library
import matplotlib
import math

# 2. Load a tab delimited file named "coors.txt" containing XYZ coordinates
import csv

#with open('coors.txt', 'r') as f:
#    reader = csv.reader(f, delimiter='\t')
#    for row in reader:
#        print(row)

# 3. Create a heat map of those coordinates using the matplotlib python library. Bins that have many points should be colored red, and bins with fewer should be blue.
import matplotlib.pyplot as plt
import numpy as np

x = []
y = []

with open('coors.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

heatmap = np.log(heatmap)

plt.clf()
plt.imshow(heatmap, extent=extent)
plt.show()

# 4. Save the heat map to a PNG file.
plt.savefig('heatmap.png')
