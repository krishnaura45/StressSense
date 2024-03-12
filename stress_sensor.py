# -*- coding: utf-8 -*-
***STRESS ESTIMATION USING PPG SIGNALS***

# Basics/Foundation

*Ppg Data*
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file
df1=pd.read_csv('/content/p2l.csv')
df1

# Read the CSV file
df2=pd.read_csv('/content/p2h.csv')
df2

"""**Plot of PPG signal**

Low Workload
"""

# Extract the required ppg values
ppg_1=df1['Trial 1:0back']
ppg_2=df2['Trial 3:3back']

# Set the sampling rate of ppg data 256 Hz as mentioned in MAUS documentation
sampling_rate=256

# Create a time array based on the sampling rate and length of the ppg values
time=np.arange(len(ppg_1))/sampling_rate

# Plot the ppg values against time
plt.plot(time,ppg_1)
plt.xlabel('Time')
plt.ylabel('PPG Amplitude')
plt.title('PPG Signal 1')

"""High Workload"""

# Plot the ppg values against time
plt.plot(time,ppg_2)
plt.xlabel('Time')
plt.ylabel('PPG Amplitude')
plt.title('PPG Signal 2')

"""# Part-1(Chirp Transform Trial)

> Indented block

**Chirp Z Transform Algorithm**

# Part-2(HRV Trial)
"""

!pip install heartpy

"""**Finding heart rate variability (HRV) features**"""

import heartpy as hp
import pandas as pd

# Load the PPG data from a CSV file
data = pd.read_csv('/content/inf_ppg.csv')
ppg1 = data['Trial 1:0back'].values
ppg2 = data['Trial 6:0back'].values

# Set the sampling rate of the PPG signal
sampling_rate = 256  # Hz

# Process the PPG signal to extract the HRV features
wd, m = hp.process(ppg1, sample_rate=sampling_rate)
wb, n = hp.process(ppg2, sample_rate=sampling_rate)

# Print the HRV features extracted from ppg data of person 002 for trial 1
print('HRV features:', m)

hrv1=pd.DataFrame(m,index=[0])
hrv1

"""Bar plot"""

import matplotlib.pyplot as plt

plt.bar(range(len(m)), m.values(), align='center')
plt.xticks(range(len(m)), list(m.keys()))
plt.show()

l=list(n.values())
hrv1.loc[len(hrv1.index)]=l
hrv1

import seaborn as sns

# Compute the correlation matrix of the HRV features
corr = hrv1.corr()

# Create a heatmap of the correlation matrix
sns.heatmap(corr, annot=True)
plt.show()

"""# Part-3(Emperical Mode Decomposition)"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install EMD-signal

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from scipy.signal import hilbert
from PyEMD import EMD
pd.options.display.precision = 10
from os import listdir

"""# **EMD on LOW sample**"""

print(df1.head(10))

# Selecting column 1 of dataframe along with 768 rows
col1=df1['Trial 1:0back'].iloc[0:768]
S1 = col1.values
t=np.arange(len(col1))/sampling_rate

print('S shape: ', S1.shape)
print('t shape: ', t.shape)

dt = t[0] - t[1]
print(dt)

# Compute IMFs with EMD
config = {'spline_kind':'cubic', 'MAX_ITERATION':100}
emd = EMD(**config)
imfs = emd(S1, max_imf=7)
print('imfs = ' + f'{imfs.shape[0]:4d}')

def instant_phase(imfs):
    """Extract analytical signal through Hilbert Transform."""
    analytic_signal = hilbert(imfs)  # Apply Hilbert transform to each row
    # Compute angle between img and real
    phase = np.unwrap(np.angle(analytic_signal))
    return phase

# Extract instantaneous phases and frequencies using Hilbert transform
instant_phases = instant_phase(imfs)
instant_freqs = np.diff(instant_phases)/(2*np.pi*dt)

# Create a figure consisting of 3 panels which from the top are the input signal, IMFs and instantaneous frequencies
fig, axes = plt.subplots(3, figsize=(12, 12))

# The top panel shows the input signal
ax = axes[0]
ax.plot(t, S1)
ax.set_ylabel("Amplitude [arbitrary units]")
ax.set_title("Input signal")

# The middle panel shows all IMFs
ax = axes[1]
for num, imf in enumerate(imfs):
    ax.plot(t, imf, label='IMF %s' %( num + 1 ))

# Label the figure
#ax.legend()
ax.set_ylabel("Amplitude [arb. u.]")
ax.set_title("IMFs")

# The bottom panel shows all instantaneous frequencies
ax = axes[2]
for num, instant_freq in enumerate(instant_freqs):
    ax.plot(t[:-1], instant_freq, label='IMF %s'%(num+1))

# Label the figure
#ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Inst. Freq. [Hz]")
ax.set_title("Huang-Hilbert Transform")

#plt.tight_layout()
#plt.savefig('S1', dpi=120)
plt.show()

# Plot results
nIMFs = imfs.shape[0]
plt.figure(figsize=(24,24))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(S1, 'r')
plt.ylabel('PPG Amplitude')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(imfs[n], 'g')
    plt.ylabel("IMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
#plt.tight_layout()
plt.savefig('Imfs_0-768_low', dpi=120)
plt.show()

"""# **EMD on HIGH sample**"""

print(df2.head())

# Selecting column 1 of dataframe along with 768 rows
col1=df2['Trial 3:3back'].iloc[0:768]
S2 = col1.values
t=np.arange(len(col1))/sampling_rate

print('S shape: ', S2.shape)
print('t shape: ', t.shape)

dt = t[0] - t[1]
print(dt)

# Compute IMFs with EMD
config = {'spline_kind':'cubic', 'MAX_ITERATION':100}
emd = EMD(**config)
imfs = emd(S2, max_imf=7)
print('imfs = ' + f'{imfs.shape[0]:4d}')

def instant_phase(imfs):
    """Extract analytical signal through Hilbert Transform."""
    analytic_signal = hilbert(imfs)  # Apply Hilbert transform to each row
    # Compute angle between img and real
    phase = np.unwrap(np.angle(analytic_signal))
    return phase

# Extract instantaneous phases and frequencies using Hilbert transform
instant_phases = instant_phase(imfs)
instant_freqs = np.diff(instant_phases)/(2*np.pi*dt)

# Create a figure consisting of 3 panels which from the top are the input
# signal, IMFs and instantaneous frequencies
fig, axes = plt.subplots(3, figsize=(12, 12))

# The top panel shows the input signal
ax = axes[0]
ax.plot(t, S2)
ax.set_ylabel("Amplitude [arbitrary units]")
ax.set_title("Input signal")

# The middle panel shows all IMFs
ax = axes[1]
for num, imf in enumerate(imfs):
    ax.plot(t, imf, label='IMF %s' %( num + 1 ))

# Label the figure
#ax.legend()
ax.set_ylabel("Amplitude [arb. u.]")
ax.set_title("IMFs")

# The bottom panel shows all instantaneous frequencies
ax = axes[2]
for num, instant_freq in enumerate(instant_freqs):
    ax.plot(t[:-1], instant_freq, label='IMF %s'%(num+1))

# Label the figure
#ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Inst. Freq. [Hz]")
ax.set_title("Huang-Hilbert Transform")

#plt.tight_layout()
#plt.savefig('Sig_002_high', dpi=120)
plt.show()

# Plot results
nIMFs = imfs.shape[0]
plt.figure(figsize=(24,24))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(S2, 'r')
plt.ylabel('PPG Amplitude')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(imfs[n], 'g')
    plt.ylabel("IMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
#plt.tight_layout()
plt.savefig('Imfs_0-768_high', dpi=120)
plt.show()

"""# Emd as a whole"""

# Low WL
col1=df1['Trial 1:0back']
S1 = col1.values

t=np.arange(len(col1))/sampling_rate

dt = t[0] - t[1]

# Compute IMFs with EMD
config = {'spline_kind':'linear', 'MAX_ITERATION':100}
emd = EMD(**config)
imfs = emd(S1, max_imf=7)
print('imfs = ' + f'{imfs.shape[0]:4d}')

# Extract instantaneous phases and frequencies using Hilbert transform
instant_phases = instant_phase(imfs)
instant_freqs = np.diff(instant_phases)/(2*np.pi*dt)

# Plot results
nIMFs = imfs.shape[0]
plt.figure(figsize=(24,24))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(S1, 'r')
plt.ylabel('PPG Amplitude')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(imfs[n], 'g')
    plt.ylabel("IMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
#plt.tight_layout()
#plt.savefig('Imfs_002_high', dpi=120)
plt.show()

# High WL
col3=df2['Trial 3:3back']
S2 = col3.values
# Compute IMFs with EMD
config = {'spline_kind':'linear', 'MAX_ITERATION':100}
emd = EMD(**config)
imfs = emd(S2, max_imf=7)
print('imfs = ' + f'{imfs.shape[0]:4d}')

# Extract instantaneous phases and frequencies using Hilbert transform
instant_phases = instant_phase(imfs)
instant_freqs = np.diff(instant_phases)/(2*np.pi*dt)

# Plot results
nIMFs = imfs.shape[0]
plt.figure(figsize=(24,24))
plt.subplot(nIMFs+1, 1, 1)
plt.plot(S2, 'r')
plt.ylabel('PPG Amplitude')

for n in range(nIMFs):
    plt.subplot(nIMFs+1, 1, n+2)
    plt.plot(imfs[n], 'g')
    plt.ylabel("IMF %i" %(n+1))
    plt.locator_params(axis='y', nbins=5)

plt.xlabel("Time [s]")
#plt.tight_layout()
#plt.savefig('Imfs_002_high', dpi=120)
plt.show()

"""# Operating EMD on all the 76800 samples of person 002 in trial 1 (by looping)"""

print(df1.head())

c1=df1['Trial 1:0back']
S1 = c1.values
t1=np.arange(len(c1))/sampling_rate

print('S shape: ', S1.shape)
print('t shape: ', t1.shape)

"""Visualising EMD in form of imfs for sets of 768 s each"""

# Instantaneous time interval
dt = t1[0] - t1[1]

j=1
i=0
count=0
while j<101:
  while i<768*j:
    ext_set=c1.iloc[i:i+768]
    i=i+768

  signal=ext_set.values
  time=np.arange(len(ext_set))/sampling_rate

  # Plotting Counter
  count=count+1
  print("\nPlot",count)

  # Compute IMFs with EMD
  config = {'spline_kind':'linear', 'MAX_ITERATION':100}
  emd = EMD(**config)
  imfs = emd(signal, max_imf=10)
  print('imfs = ' + f'{imfs.shape[0]:4d}')

  # Extract instantaneous phases and frequencies using Hilbert transform
  instant_phases = instant_phase(imfs)
  instant_freqs = np.diff(instant_phases)/(2*np.pi*dt)

  # Plot results
  nIMFs = imfs.shape[0]
  plt.figure(figsize=(24,24))
  plt.subplot(nIMFs+1, 1, 1)
  plt.plot(signal, 'r')
  plt.ylabel('PPG Amplitude')

  for n in range(nIMFs):
      plt.subplot(nIMFs+1, 1, n+2)
      plt.plot(imfs[n], 'g')
      plt.ylabel("IMF %i" %(n+1))
      plt.locator_params(axis='y', nbins=5)

  plt.xlabel("Time [s]")
  #plt.tight_layout()
  #plt.savefig('Imfs_002_high', dpi=120)
  plt.show()
  j=j+1

"""# Operating EMD on all the 76800 samples of person 002 in trial 3 (by windowing or looping)"""

print(df2.head())

c2=df2['Trial 3:3back']
S2 = c2.values
t2=np.arange(len(c2))/sampling_rate

print('S shape: ', S2.shape)
print('t shape: ', t2.shape)

# Instantaneous time interval
dt = t2[0] - t2[1]

j=1
i=0
count=0
while j<101:
  while i<768*j:
    ext_set=c2.iloc[i:i+768]
    i=i+768

  signal=ext_set.values
  time=np.arange(len(ext_set))/sampling_rate

  # Plotting Counter
  count=count+1
  print("\nPlot",count)

  # Compute IMFs with EMD
  config = {'spline_kind':'cubic', 'MAX_ITERATION':100}
  emd = EMD(**config)
  imfs = emd(signal, max_imf=10)
  print('imfs = ' + f'{imfs.shape[0]:4d}')

  # Extract instantaneous phases and frequencies using Hilbert transform
  instant_phases = instant_phase(imfs)
  instant_freqs = np.diff(instant_phases)/(2*np.pi*dt)

  # Plot results
  nIMFs = imfs.shape[0]
  plt.figure(figsize=(24,24))
  plt.subplot(nIMFs+1, 1, 1)
  plt.plot(signal, 'r')
  plt.ylabel('PPG Amplitude')

  for n in range(nIMFs):
      plt.subplot(nIMFs+1, 1, n+2)
      plt.plot(imfs[n], 'g')
      plt.ylabel("IMF %i" %(n+1))
      plt.locator_params(axis='y', nbins=5)

  plt.xlabel("Time [s]")
  #plt.tight_layout()
  #plt.savefig('Imfs_002_high', dpi=120)
  plt.show()
  j=j+1

"""# *Finding Peaks of Imf Signals in trial 1(low)*"""

from scipy.signal import find_peaks
import numpy as np

# Instantaneous time interval
dt = t1[0] - t1[1]

j=1
i=0
count=0
while j<101:
  while i<768*j:
    ext_set=c1.iloc[i:i+768]
    i=i+768

  signal=ext_set.values
  time=np.arange(len(ext_set))/sampling_rate

  # Plotting Counter
  count=count+1
  print("\nSet",count)

  # Compute Intrinsic Mode Functions with EMD
  config = {'spline_kind':'linear', 'MAX_ITERATION':1000}
  emd = EMD(**config)
  imfs = emd(signal, max_imf=10)
  print('imfs =' + f'{imfs.shape[0]:4d}')

  p, _= find_peaks(imfs[1])
  print('Number of peaks in 2nd IMF:',len(p))
  print(p)
  q, _= find_peaks(imfs[2])
  print('Number of peaks in 3rd IMF:',len(q))
  print(q)

  j=j+1

"""# *Finding Peaks of Imf Signals in trial 3(high)*"""

# Instantaneous time interval
dt = t2[0] - t2[1]

j=1
i=0
count=0
while j<101:
  while i<768*j:
    ext_set=c2.iloc[i:i+768]
    i=i+768

  signal=ext_set.values
  time=np.arange(len(ext_set))/sampling_rate

  # Plotting Counter
  count=count+1
  print("\nSet",count)

  # Compute IMFs with EMD
  config = {'spline_kind':'linear', 'MAX_ITERATION':1000}
  emd = EMD(**config)
  imfs = emd(signal, max_imf=10)
  print('imfs = ' + f'{imfs.shape[0]:4d}')

  p, _= find_peaks(imfs[1])
  print('Number of peaks in 2nd IMF:',len(p))
  print(p)
  q, _= find_peaks(imfs[2])
  print('Number of peaks in 3rd IMF:',len(q))
  print(q)
  j=j+1

"""# ***Final shot***"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install EMD-signal

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from scipy.signal import hilbert
from PyEMD import EMD
pd.options.display.precision = 10
from os import listdir

# merging twenty two csv files of Low stress // 0 back
ldata = pd.concat(
    map(pd.read_csv, ['/content/p2l.csv','/content/p3l.csv','/content/p4l.csv','/content/p5l.csv','/content/p6l.csv','/content/p8l.csv','/content/p10l.csv','/content/p11l.csv','/content/p12l.csv','/content/p13l.csv','/content/p14l.csv','/content/p15l.csv','/content/p16l.csv','/content/p17l.csv','/content/p18l.csv','/content/p19l.csv','/content/p20l.csv','/content/p21l.csv','/content/p22l.csv','/content/p23l.csv','/content/p24l.csv','/content/p25l.csv']), ignore_index=True)

# merging twenty two csv files of High workload  // 3 back
hdata = pd.concat(
    map(pd.read_csv, ['/content/p2h.csv','/content/p3h.csv','/content/p4h.csv','/content/p5h.csv','/content/p6h.csv','/content/p8h.csv','/content/p10h.csv','/content/p11h.csv','/content/p12h.csv','/content/p13h.csv','/content/p14h.csv','/content/p15h.csv','/content/p16h.csv','/content/p17h.csv','/content/p18h.csv','/content/p19h.csv','/content/p20h.csv','/content/p21h.csv','/content/p22h.csv','/content/p23h.csv','/content/p24h.csv','/content/p25h.csv']), ignore_index=True)

hdata.to_csv('PPG_high.csv', index = True)
ldata.to_csv('PPG_low.csv', index = True)

"""LOW Section"""

# Pre-processing the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(ldata)

low=pd.DataFrame(X)
low['Label']=0
display(low.head())

low=low.iloc[:,[0,1,5]]
low

"""**Strategy 1**"""

from scipy.stats import skew
import numpy as np
# df.drop('1',axis=1,inplace=True)

#%%capture
setnos,f1,f2,f3,f4,labels=[],[],[],[],[],[]
sampling_rate=256    # Hz
i,j,k=0,1,1
count,c=1,0
print("For Low Mental Stress")
while k<3:
  print("\n\n\nColumn",k)
  while j<2201:
    while i<768*j:
      set=low.iloc[i:i+768,k]
      i=i+768

    signal=set.values
    time=np.arange(len(set))/sampling_rate

    # Plotting Counter
    # c=c+1
    # print("\nPlot",c)

    # Compute IMFs with EMD
    config = {'spline_kind':'cubic', 'MAX_ITERATION':100}
    emd = EMD(**config)
    imfs = emd(signal, max_imf=10)
    print('imfs = ' + f'{imfs.shape[0]:4d}')

    # Grouping Counter
    print("\nSet",count,"captured")
    setnos.append(count)
    count=count+1

    labels.append(0)
    f1.append(np.mean(imfs[0]))
    f2.append(np.min(imfs[0]))
    f3.append(np.max(imfs[0]))
    f4.append(skew(imfs[0]))

    j=j+1
  k=k+1
  j=1

df_imf_low=pd.DataFrame(zip(setnos,f1,f2,f3,f4,labels),columns=['Set_no','Imf_1_MEAN','Imf_1_MIN','Imf_1_MAX','Imf_1_SKEWNESS','Label'])
df_imf_low

"""# *Strategy 2*"""

'''Commented out IPython magic to ensure Python compatibility.
# %%capture
# setnos,feature_1,feature_2,labels=[],[],[],[]
# sampling_rate=256    # Hz
# i,j,k=0,1,1
# count,c=1,0
# print("For Low Mental Stress")
# while k<3:
#   print("\n\n\nColumn",k)
#   while j<2201:
#     while i<768*j:
#       set=low.iloc[i:i+768,k]
#       i=i+768
# 
#     signal=set.values
#     time=np.arange(len(set))/sampling_rate
# 
#     # Plotting Counter
#     # c=c+1
#     # print("\nPlot",c)
# 
#     # Compute IMFs with EMD
#     config = {'spline_kind':'cubic', 'MAX_ITERATION':100}
#     emd = EMD(**config)
#     imfs = emd(signal, max_imf=10)
#     print('imfs = ' + f'{imfs.shape[0]:4d}')
# 
#     # Grouping Counter
#     print("\nSet",count,"captured")
#     setnos.append(count)
#     count=count+1
# 
#     labels.append(0)
#     feature_1.append(imfs[0])
#     feature_2.append(imfs[1])
# 
#     j=j+1
#   k=k+1
#   j=1
'''

df_imf_low=pd.DataFrame(zip(setnos,feature_1,feature_2,labels),columns=['Set_no','Imf_1','Imf_2','Label'])
df_imf_low

"""# Strategy 1 cont....

HIGH Section
"""

# Pre-processing the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(hdata)

high=pd.DataFrame(X)
high['Label']=1
display(high.head())

"""HIGH section"""

#%%capture
setnos2,f1,f2,f3,f4,labels2=[],[],[],[],[],[]
sampling_rate=256    # Hz
i,j,k=0,1,1
count,c=4401,0
print("For High Mental Stress")
while k<3:
  print("\n\n\nColumn",k)
  while j<2201:
    while i<768*j:
      set=high.iloc[i:i+768,k]
      i=i+768

    signal=set.values
    time=np.arange(len(set))/sampling_rate

    # Plotting Counter
    # c=c+1
    # print("\nPlot",c)

    # Compute IMFs with EMD
    config = {'spline_kind':'cubic', 'MAX_ITERATION':100}
    emd = EMD(**config)
    imfs = emd(signal, max_imf=10)
    print('imfs = ' + f'{imfs.shape[0]:4d}')

    # Grouping Counter
    print("\nSet",count,"captured")
    setnos2.append(count)
    count=count+1

    labels2.append(1)
    f1.append(np.mean(imfs[0]))
    f2.append(np.min(imfs[0]))
    f3.append(np.max(imfs[0]))
    f4.append(skew(imfs[0]))

    j=j+1
  k=k+1
  j=1

df_imf_high=pd.DataFrame(zip(setnos2,f1,f2,f3,f4,labels2),columns=['Set_no','Imf_1_MEAN','Imf_1_MIN','Imf_1_MAX','Imf_1_SKEWNESS','Label'])
df_imf_high

df_imf_low.to_csv('Imf_low.csv', index = True)
df_imf_high.to_csv('Imf_high.csv', index = True)

# merging two csv files of whole featured data
data = pd.concat(
    map(pd.read_csv, ['/content/Imf_low.csv','/content/Imf_high.csv']), ignore_index=True)

final_data=data.drop(['Unnamed: 0'], axis=1)
display(final_data)

len(imfs[0])

"""# ***Pre-CLASSIFICATION***"""

from scipy.stats import skew
import numpy as np
# df.drop('1',axis=1,inplace=True)

df=final_data

df0=df[df.Label==0]
df0.head()

df1=df[df.Label==1]
df1.head()

plt.xlabel('MEAN of Imf')
plt.ylabel('SKEWNESS of Imf')
plt.scatter(df0['Imf_1_MEAN'], df0['Imf_1_SKEWNESS'],color="green",marker='+')
plt.scatter(df1['Imf_1_MEAN'], df1['Imf_1_SKEWNESS'],color="blue",marker='.')

x=final_data.iloc[:,[1,2,3,4]]
display(x)

y=final_data.iloc[:,5]
y

"""*Train Test Split*"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

"""# **Random Forest Classifier**"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000)
classifier.fit(X_train, y_train)

import matplotlib.pyplot as plt
plt.plot(x.iloc[0])
plt.plot(x.iloc[4400])

y_pred = classifier.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix : \n", conf)

rec = conf[0][0]/(conf[0][0]+conf[1][0])
print("Recall : \n",rec)

prec = conf[0][0]/(conf[0][0]+conf[0][1])
print("Precision:\n",prec)

"""# **SVM Classifier**"""

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

model.score(X_test, y_test)

"""Kernalization"""

model_kernal = SVC(kernel='sigmoid')
model_kernal.fit(X_train, y_train)

model_kernal.score(X_test, y_test)

"""Regularization(C)"""

model_C = SVC(C=200)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)

y_prd = model_C.predict(X_test)
y_prd

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_prd)
print ("Confusion Matrix : \n", cm)

prec = cm[0][0]/(cm[0][0]+cm[0][1])
print("Precision:\n",prec)

rec = cm[0][0]/(cm[0][0]+cm[1][0])
print("Recall : \n",rec)

"""# Next"""

