# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install EMD-signal

from scipy.stats import skew
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

# Pre-processing the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(ldata)

low=pd.DataFrame(X)
low=low.iloc[:,[0,1]]
display(low.head())

#%%capture
setnos,f1,f2,f3,f4,f5,f6,f7,f8,labels=[],[],[],[],[],[],[],[],[],[]
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
    f5.append(np.mean(imfs[1]))
    f6.append(np.min(imfs[1]))
    f7.append(np.max(imfs[1]))
    f8.append(skew(imfs[1]))

    j=j+1
  k=k+1
  j=1

df_imf_low=pd.DataFrame(zip(setnos,f1,f2,f3,f4,f5,f6,f7,f8,labels),columns=['Set_no','Imf_1_MEAN','Imf_1_MIN','Imf_1_MAX','Imf_1_SKEWNESS','Imf_2_MEAN','Imf_2_MIN','Imf_2_MAX','Imf_2_SKEWNESS','Label'])
df_imf_low

# Pre-processing the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(hdata)

high=pd.DataFrame(X)
display(high.head())

#%%capture
setnos2,f1,f2,f3,f4,f5,f6,f7,f8,labels2=[],[],[],[],[],[],[],[],[],[]
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
    f5.append(np.mean(imfs[1]))
    f6.append(np.min(imfs[1]))
    f7.append(np.max(imfs[1]))
    f8.append(skew(imfs[1]))

    j=j+1
  k=k+1
  j=1

df_imf_high=pd.DataFrame(zip(setnos2,f1,f2,f3,f4,f5,f6,f7,f8,labels2),columns=['Set_no','Imf_1_MEAN','Imf_1_MIN','Imf_1_MAX','Imf_1_SKEWNESS','Imf_2_MEAN','Imf_2_MIN','Imf_2_MAX','Imf_2_SKEWNESS','Label'])
df_imf_high

df_imf_low.to_csv('Imf_low.csv', index = True)
df_imf_high.to_csv('Imf_high.csv', index = True)

"""*Final Data*"""

# merging two csv files of whole featured data
data = pd.concat(
    map(pd.read_csv, ['/content/Imf_low.csv','/content/Imf_high.csv']), ignore_index=True)

df=data.drop(['Unnamed: 0'], axis=1)
df.to_csv('Final_data.csv',index=True)
display(df)

"""Low Stress DATA"""

df0=df[df.Label==0]
df0.head()

"""High Stress DATA"""

df1=df[df.Label==1]
df1.head()

"""Scatter Plot of Skewness for IMFs"""

plt.xlabel('MEAN of Imf')
plt.ylabel('SKEWNESS of Imf')
plt.scatter(df0['Imf_1_SKEWNESS'], df0['Imf_2_SKEWNESS'],color="green",marker='+')
plt.scatter(df1['Imf_1_SKEWNESS'], df1['Imf_2_SKEWNESS'],color="red",marker='*')

"""Separation of features and labels"""

x=df.iloc[:,[1,2,3,4,5,6,7,8]]
y=df.iloc[:,9]

#plotting the heatmap for correlation between features
ax = sns.heatmap(x.corr(), annot=True)

"""Train Test Split"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

from sklearn import svm
model = svm.SVC()
model.fit(X_train, y_train)

# Acc
model.score(X_test, y_test)

"""Kernelization"""

model_k = svm.SVC(kernel='linear')
model_k.fit(X_train, y_train)

# Accuracy on testing
model_k.score(X_test, y_test)

"""Regularization(C)"""

model_C = svm.SVC(C=600)
model_C.fit(X_train, y_train)

# Accuracy on testing
model_C.score(X_test, y_test)

y_prd = model_C.predict(X_test)
y_prd

"""Saving the model"""

import pickle

with open('SVM_model','wb') as f:
  pickle.dump(model,f)

with open('SVM_Reg','wb') as f:
  pickle.dump(model_C,f)

"""# **Confusion matrix**"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_prd)
print ("Confusion Matrix : \n", cm)

import seaborn as sns
import matplotlib.pyplot as plt

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('PREDICTED');ax.set_ylabel('ACTUAL');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Low Stress', 'High Stress']); ax.yaxis.set_ticklabels(['Low Stress', 'High Stress']);

"""0 Class-wise Metrics"""

prec = cm[0][0]/(cm[0][0]+cm[0][1])
print("Precision:\n",prec)

rec = cm[0][0]/(cm[0][0]+cm[1][0])
print("Recall : \n",rec)

f1_score=(2*prec*rec)/(prec+rec)
f1_score

"""1 Class-wise Metrics"""

p = cm[1][1]/(cm[0][1]+cm[1][1])
print("Precision:\n",p)

r = cm[1][1]/(cm[1][1]+cm[1][0])
print("Recall : \n",r)

f1_score=(2*p*r)/(p+r)
f1_score

"""# Other Metrics and Visualizations"""
