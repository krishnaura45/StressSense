# -*- coding: utf-8 -*-
# ***Research on feature extraction using SPAR***

**Creating function for SPAR (Symmetric Projection Attractor Reconstruction)**
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.spatial.distance import pdist, squareform
def spar_features(ppg_data):
    # Extract PPG signal values
    ppg_signal = ppg_data.values

    # Define time delay and embedding dimension
    tau = 5
    m = 3

    # Create phase space reconstruction matrix
    N = len(ppg_signal)
    X = np.zeros((N - (m-1)*tau, m))
    for i in range(N - (m-1)*tau):
        for j in range(m):
            X[i,j] = ppg_signal[i + j*tau]

    # Compute Euclidean distances between each pair of points in the reconstruction matrix
    dist_matrix = squareform(pdist(X))

    # Compute mean distance to neighbors for each point
    mean_dist = np.mean(dist_matrix, axis=1)

    # Compute SPAR features
    spar1 = np.mean(mean_dist)   # corresponding to MEAN
    spar2 = np.std(mean_dist)    # corresponding to STANDARD DEVIATION
    spar3 = skew(mean_dist)      # corresponding to SKEWNESS

    # Return SPAR features
    return spar1, spar2, spar3

"""***Importing LOW MENTAL WORKLOAD Dataset of PPG signals for 1st person***"""

df=pd.read_csv('inf_ppg.csv')
col1=df['Trial 1:0back']

"""*Trial 1:0back*"""

from tables.atom import Float128Atom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
f1=[]
f2=[]
f3=[]
j=1
i=0
count=0
while j<101:
  while i<768*j:
    x,y,z=spar_features(col1.iloc[i:i+48])
    f1.append(x)
    f2.append(y)
    f3.append(z)
    i=i+48
  count=count+len(f1)
  print("\nExtracted",count,"number of spar features")
  # Collecting feature values in dataframe
  a = pd.DataFrame(f1)
  b = pd.DataFrame(f2)
  d = pd.DataFrame(f3)

  # Create 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter3D(a, b, d, s=5, cmap='viridis')

  # Set plot title and labels
  print("PLOT",j)
  ax.set_xlabel('Feature 1')
  ax.set_ylabel('Feature 2')
  ax.set_zlabel('Feature 3')

  # Show plot
  plt.show()
  j=j+1
  f1=[]
  f2=[]
  f3=[]

"""*Trial 6:0back*"""

from tables.atom import Float128Atom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
f1=[]
f2=[]
f3=[]
j=1
i=0
count=0
while j<101:
  while i<768*j:
    x,y,z=spar_features(col2.iloc[i:i+48])
    f1.append(x)
    f2.append(y)
    f3.append(z)
    i=i+48
  count=count+len(f1)
  print("\nExtracted",count,"number of spar features")
  # Collecting feature values in dataframe
  a = pd.DataFrame(f1)
  b = pd.DataFrame(f2)
  d = pd.DataFrame(f3)

  # Create 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter3D(a, b, d, s=5, cmap='viridis')

  # Set plot title and labels
  print("PLOT",j)
  ax.set_xlabel('Feature 1')
  ax.set_ylabel('Feature 2')
  ax.set_zlabel('Feature 3')

  # Show plot
  plt.show()
  j=j+1
  f1=[]
  f2=[]
  f3=[]

"""**Low Workload for person 1 (n=500)**

**High Workload for person 1 (n=500)**

New Code
"""

! pip install pyrem

import numpy as np
import pandas as pd
import pyrem as pr

# Load GSR data from CSV file
data = pd.read_csv('/content/inf_gsr.csv')

# Preprocess the data (filtering, normalization, resampling, etc.)
# ...

# Generate 3D SPAR features using pyrem
n_components = 3 # number of components to use for the projection
spar = pr.features.SPAR(n_components=n_components)
spar_features = spar.transform(data)

# Extract the 3D SPAR features
spar_3d = spar_features[:, :3]

print(spar_3d)

!pip install pyts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

# Load PPG data from CSV file
data = pd.read_csv('/content/p2_gsr_hwl.csv')
display(data)

# Extract PPG signal from data
df=data.loc[0:499]
ppg = df['Trial 3:3back'].values

# Set parameters for symmetric projection attractor reconstruction
embedding_dimension = 3
time_delay = 1
threshold = 0.1

# Perform symmetric projection attractor reconstruction
X = np.array([ppg[:-2*time_delay*embedding_dimension],
              ppg[time_delay*embedding_dimension:-time_delay*embedding_dimension],
              ppg[2*time_delay*embedding_dimension:]])
X = X.T
d = np.abs(X[:, None] - X)
D = np.sqrt(np.sum(np.square(np.minimum(d, threshold)), axis=2))
R = np.heaviside(threshold - D, 0)

# Create recurrence plot from symmetric projection attractor reconstruction
rp = RecurrencePlot(threshold='point', percentage=30)
rp.fit_transform(R)

# Plot recurrence plot as a 2D image
plt.imshow(rp.transform(R)[0], cmap='binary', origin='lower')
plt.xlabel('Time')
plt.ylabel('Time')
plt.title('Symmetric Projection Attractor Reconstruction Recurrence Plot')
