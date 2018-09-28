import numpy as np
import matplotlib.pyplot as plt

from mne.io import concatenate_raws, read_raw_edf, find_edf_events, read_raw_fif
from mne import read_events
import pyedflib
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#file2 = read_raw_edf('S001R03.edf', preload=True, stim_channel='auto')
#raw = read_raw_fif('S001R03.edf')
# event = find_edf_events('S001R03.edf.event')

file = pyedflib.EdfReader("S001R03.edf")

def transpose(list):
    return [[list[j][i] for j in range(len(list))] for i in range(len(list[0]))]
''' Annotation (3 X 30):
    first array is the time stamp of each code
    second array is the interval between each code .... there's one extra don't know why
    third array is the code'''
annotation = file.readAnnotations()
marker = []
for i in annotation[1]:
    marker.append(i*160)

''' Building target
    160 EEG are recorded every second (total 19920 without counting 80 zeros in the end)
    annotations gives the interval between each marker(code)
    so filled target y with the target T0 = 0, T1 = 1, T2 = 2 
    for 19920 data points '''
y = []
for counter, dataPoints in enumerate(marker):
    for i in range(int(dataPoints)):
        code = annotation[2][counter]
        if code == 'T0':
            y.append(0)
        elif code == 'T1':
            y.append(1)
        elif code == 'T2':
            y.append(2)
        else:
            #TODO
            print("catch error here")

''' NOTE !length of y is 19920, but data set has 20000
        this is because the last 80 data sets are all zeros'''


'''number of channels: 64'''
totalSignals = file.signals_in_file #totalSignals = 64

signal_labels = file.getSignalLabels() #label names of electrode in 10-10 system

''' -80 removes the ending 80 zero data points'''
data = np.zeros((totalSignals, file.getNSamples()[0]-80))
for i in np.arange(totalSignals):
    data[i, :] = file.readSignal(i)[:-80]

print(data)

plt.figure(1, figsize=(30, 5))
plt.clf()

# Plot the training points
''' xaxix (1,19920) for filling x-axis'''
xaxis = []
for i in range(19920):
    xaxis.append(i)
plt.scatter(xaxis, data[23], c=y, cmap=plt.cm.Set1, marker=".", label=['T0','T1','T2'])
plt.xlabel('1/160 of a second')
plt.ylabel('EEG µV(?)')
plt.title('EEG Motor Movement/Imagery Dataset')

y = np.asarray(y)
lda = LinearDiscriminantAnalysis(n_components=2)
data = np.transpose(data)
y = np.transpose(y)
#X_fit = lda.fit(data.reshape(19920,64), y.reshape(19920,1)).transform(data)
X_fit = lda.fit(data, y).transform(data)
print(X_fit.shape)



plt.figure(2, figsize=(8, 6))
#plt.scatter(X_fit[:, 0], X_fit[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k',label=target_names)
plt.scatter(X_fit[:, 0], X_fit[:, 1], c=y, cmap=plt.cm.Set1,label=['T0','T1','T2'])

plt.title('LDA of EEG Motor Movement/Imagery Dataset')


plt.show()
'''
counter = 0
for i in sigbufs:
    print(len(i))
    counter += 1
print(counter)
print(len(signal_labels))
for i in annotation:
    print(len(i))
for i in annotation[1]
'''