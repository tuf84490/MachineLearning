import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import pprint #used to make output easier to read for demo purposes
f1 = open('outlet_power1.txt', 'r') # This file is the combination of the three below
fc = open('outlet_power_coffee.txt', 'r')
fs = open('outlet_power_stove.txt', 'r')
fk = open('outlet_power_kettle.txt', 'r')
lines = f1.readlines()
lines_c = fc.readlines()
lines_s = fs.readlines()
lines_k = fk.readlines()


def populate_time_ticks(powers, times, ticks): # convert points(time, power) to frame-wise data. For example, if two nearest points are (10.0, 1300) and (10.5, 0),
                                                # we suppose the power value keeps 1300 between 10.0 and 10.5
                                                # Besides, we convert the time domain from the format hour.(minute/60) to hour*60+minute. Therefore, the time dimension
                                                # becomes the number of minutes away from 00:00
    last_big_power = 0
    last_big_time = 0
    for power, time in zip(powers, times):
        hour = int(time)
        minute = int((time - hour)*60)
        if last_big_power == 0:
            if power > 10:
                last_big_power = power
                last_big_time = hour * 60 + minute
            else:
                last_big_power = 0
                last_big_time = 0
        elif last_big_power != 0:
            if power > 10:
                start = last_big_time
                end = hour * 60 + minute
                ticks[start:end] = map(lambda x: x+last_big_power, ticks[start:end])
                last_big_power = power
                last_big_time = hour * 60 + minute
            else:
                start = last_big_time
                end = hour * 60 + minute
                ticks[start:(end+1)] = map(lambda x: x+last_big_power, ticks[start:(end+1)])
                last_big_power = 0
                last_big_time = 0
    return ticks


def slice_data(powers, times):
    threshold = 500
    instances = []
    last_big_power = 0
    last_big_time = 0
    for power, time in zip(powers, times):
        hour = int(time)
        minute = int((time - hour)*60)
        if last_big_power == 0:
            if power > threshold:
                last_big_power = power
                last_big_time = hour * 60 + minute
            else:
                last_big_power = 0
                last_big_time = 0
        elif last_big_power != 0:
            if power > threshold:
                start = last_big_time
                end = hour * 60 + minute
                instance = {}
                instance['power'] = last_big_power
                instance['duration'] = end - start
                instance['start'] = start
                instance['end'] = end
                instances.append(instance)
                last_big_power = power
                last_big_time = hour * 60 + minute
            else:
                start = last_big_time
                end = hour * 60 + minute
                instance = {}
                instance['power'] = last_big_power
                instance['duration'] = end - start
                instance['start'] = start
                instance['end'] = end
                instances.append(instance)
                last_big_power = 0
                last_big_time = 0
    return instances

def findMaxPower(data, cluster):
    max = 0
    for dic in data:
        if (dic['id'] == cluster):
            if (dic['power'] > max):
                max = dic['power']
    for dic2 in data:
        if (dic2['id'] == cluster):
            dic2['max_power'] = max

def findMinPower(data, cluster):
    min = 99999
    for dic in data:
        if (dic['id'] == cluster):
            if (dic['power'] < min):
                min = dic['power']
    for dic2 in data:
        if (dic2['id'] == cluster):
            dic2['min_power'] = min

def findMinVal(data, cluster):
    min = 99999
    for dic in data:
        if (dic['id'] == cluster):
            if (dic['duration'] < min):
                min = dic['duration']
    for dic2 in data:
        if (dic2['id'] == cluster):
            dic2['min_duration'] = min

def findMaxVal(data, cluster):
    max = 0
    for dic in data:
        if (dic['id'] == cluster):
            if (dic['duration'] > max):
                max = dic['duration']
    for dic2 in data:
        if (dic2['id'] == cluster):
            dic2['max_duration'] = max

# read the merged data
fri = []
fri_time = []
sat = []
sat_time = []
sun = []
sun_time = []
for line in lines:
    date, time, value, = line.split()
    hour, minute, second = time.split(":")
    day = date[-2:]
    if day == "25":   # fri
        fri.append(float(value))
        fri_time.append(float(hour)+float(minute)/60)
    elif day == "26":   # sat
        sat.append(float(value))
        sat_time.append(float(hour)+float(minute)/60)
    elif day == "27":   # sun
        sun.append(float(value))
        sun_time.append(float(hour)+float(minute)/60)

# read the coffee machine data
fri_coffee = []
fri_coffee_time = []
sat_coffee = []
sat_coffee_time = []
sun_coffee = []
sun_coffee_time = []
for line in lines_c:
    date, time, value, = line.split()
    hour, minute, second = time.split(":")
    day = date[-2:]
    time = float(hour)+float(minute)/60
    if day == "25":   # fri
        fri_coffee.append(float(value))
        fri_coffee_time.append(time)
    elif day == "26":   # sat
        sat_coffee.append(float(value))
        sat_coffee_time.append(time)
    elif day == "27":   # sun
        sun_coffee.append(float(value))
        sun_coffee_time.append(time)

# read the stove data
fri_stove = []
fri_stove_time = []
sat_stove = []
sat_stove_time = []
sun_stove = []
sun_stove_time = []
for line in lines_s:
    date, time, value, = line.split()
    hour, minute, second = time.split(":")
    day = date[-2:]
    time = float(hour)+float(minute)/60
    if day == "25":   # fri
        fri_stove.append(float(value))
        fri_stove_time.append(time)
    elif day == "26":   # sat
        sat_stove.append(float(value))
        sat_stove_time.append(time)
    elif day == "27":   # sun
        sun_stove.append(float(value))
        sun_stove_time.append(time)


# read the kettle data
fri_kettle = []
fri_kettle_time = []
sat_kettle = []
sat_kettle_time = []
sun_kettle = []
sun_kettle_time = []
for line in lines_k:
    date, time, value, = line.split()
    hour, minute, second = time.split(":")
    day = date[-2:]
    time = float(hour)+float(minute)/60
    if day == "25":   # fri
        fri_kettle.append(float(value))
        fri_kettle_time.append(time)
    elif day == "26":   # sat
        sat_kettle.append(float(value))
        sat_kettle_time.append(time)
    elif day == "27":   # sun
        sun_kettle.append(float(value))
        sun_kettle_time.append(time)



instances_fri = slice_data(fri, fri_time)
# print('Fri:')
# for each in instances_fri:
#     print(each)
instances_sat = slice_data(sat, sat_time)
# print('Sat:')
# for each in instances_sat:
#     print(each)
instances_sun = slice_data(sun, sun_time)
# print('Sun:')
# for each in instances_sun:
#     print(each)

# plot the instances in fri, sat, and sun
x = range(24*60)
fig = plt.figure()
ax1 = plt.subplot(313)
p1 = plt.plot(x, populate_time_ticks(fri, fri_time, [0]*24*60), label="power")
ax1.set_ylim(500, 1500)
ax1.set_xlim(0*60, 24*60)
plt.xticks(range(0*60, 24*60+60, 120), range(0, 24+1, 2))
plt.yticks(range(500, 1501, 250))
ax1.set_xlabel('Time of Day (hour)', fontsize=15)

ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
p2 = plt.plot(x, populate_time_ticks(sat, sat_time, [0]*24*60), label="power")
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = plt.subplot(311, sharex=ax1, sharey=ax1)
p3 = plt.plot(x, populate_time_ticks(sun, sun_time, [0]*24*60), label="power")
plt.setp(ax3.get_xticklabels(), visible=False)
fig.text(0.02, 0.78, 'Outlet Power (watt) of 3 Days', rotation='vertical', fontsize=15)
ax1.annotate('local max', xy=(2, 1), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
size = 30
alpha = 0.5
markers = ['^', '*', '8', 'p', 'D', 's']
size = 10
plt.show()


######################################################################
###   instances_merged format: [{value: , duration:, start:, end:}, {}, ]
###   power: the power
###   duration: the number of minutes a power value lasts for
###   start, end: the start and end time of a power value
###   \TODO: use the (value, duration) of each instance as features to perform a K-mean clustering to partition the instances into clusters, make sure the value of K can be adjusted by code
###   K = 3 ~ 8 may be the good values for debugging
###   After the clustering, add 5 fields to each element in instances_merged: 
###   id: the id of the cluster
###   max_cluster_duration: the maximum duration value of instances in the current instance's cluster
###   min_cluster_duration: 
###   max_cluster_power: the maximum power value of instances in the current instance's cluster
###   min_cluster_power: 
instances_merged = instances_fri + instances_sat + instances_sun

#put the data into a data frame object for easy use by python KMeans class
x_clusters = [i['duration'] for i in instances_merged]
y_clusters = [i['power'] for i in instances_merged]
data_frame = pd.DataFrame({
    'x': x_clusters,
    'y': y_clusters
})

#create the KMeans object with K value = 8 and the number of clusters set to be 3
kmeans = KMeans(n_clusters=3, max_iter=8, init='random') #max_iter is the K value. Just change that to any value to set the K for K clustering
kmeans.fit(data_frame)  #fit the data to the model
labels = kmeans.predict(data_frame) #run the model
centroids = kmeans.cluster_centers_

#add an ID to each datapoint based on their cluster
index = 0
for i in labels:
    instances_merged[index]["id"] = i
    index = index + 1

#add the max and min of power and duration on each datapoint in cluster
for i in range(0,3):
    findMaxPower(instances_merged, i)
    findMaxVal(instances_merged, i)
    findMinPower(instances_merged, i)
    findMinVal(instances_merged, i)

#print each datapoint cleanly
for dictionary in instances_merged:
    pprint.pprint(dictionary)

#graph the clusters
fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b'}
col = map(lambda x: colmap[x+1], labels) #this works on python 3. to make this work on python 2, change col to colors
colors = list(col)                       #and then delete this line
plt.scatter(data_frame['x'], data_frame['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 90)
plt.xlabel("duration", fontsize=15)
plt.ylim(850, 1500)
plt.ylabel("power", fontsize=12)
plt.show()

### The clusters will be matched to the coffee machine, stove or kettle based on some common sense knowledge



# plot the frame-wise data from three ground truth files
x = range(24*60)
fig = plt.figure()
ax1 = plt.subplot(313)
p1 = plt.plot(x, populate_time_ticks(fri_coffee, fri_coffee_time, [0]*24*60), 'b:', label="coffee machine")
p2 = plt.plot(x, populate_time_ticks(fri_stove, fri_stove_time, [0]*24*60), 'r--', label="stove")
p3 = plt.plot(x, populate_time_ticks(fri_kettle, fri_kettle_time, [0]*24*60), 'g-', label="kettle")
ax1.set_ylim(500, 1500)
ax1.set_xlim(0*60, 24*60)
plt.xticks(range(0*60, 24*60+60, 120), range(0, 24+1, 2))
plt.yticks(range(500, 1501, 250))
ax1.set_xlabel('Time of Day (hour)', fontsize=15)
ax1.legend(loc='upper left')

ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
p1 = plt.plot(x, populate_time_ticks(sat_coffee, sat_coffee_time, [0]*24*60), 'b:', label="coffee machine")
p2 = plt.plot(x, populate_time_ticks(sat_stove, sat_stove_time, [0]*24*60), 'r--', label="stove")
p3 = plt.plot(x, populate_time_ticks(sat_kettle, sat_kettle_time, [0]*24*60), 'g-', label="kettle")
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = plt.subplot(311, sharex=ax1, sharey=ax1)
p1 = plt.plot(x, populate_time_ticks(sun_coffee, sun_coffee_time, [0]*24*60), 'b:', label="coffee machine")
p2 = plt.plot(x, populate_time_ticks(sun_stove, sun_stove_time, [0]*24*60), 'r--', label="stove")
p3 = plt.plot(x, populate_time_ticks(sun_kettle, sun_kettle_time, [0]*24*60), 'g-', label="kettle")
plt.setp(ax3.get_xticklabels(), visible=False)
fig.text(0.02, 0.78, 'Outlet Power (watt) of 3 Days', rotation='vertical', fontsize=15)
#ax1.annotate('local max', xy=(2, 1), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))

size = 30
alpha = 0.5
markers = ['^', '*', '8', 'p', 'D', 's']
size = 10
plt.show()

