import numpy as np
import matplotlib.pyplot as plt

fr = open('three-day-temperature1.txt', 'r')
lines = fr.readlines()
data = []
dates = []
times = []
for line in lines:
    value, date, time = line.split()
    data.append(float(value))
    dates.append(int(date[-2:])-14)
    times.append(float(time[0:2])+float(time[3:5])/60)

x_time = range(len(data))

colors1 = ['red', 'cyan', 'black']
markers = ['^', 'D', 's']

fig = plt.figure(1)
ax = fig.gca()
#ax.set_title('temperature')
ax.set_ylim(65, 72)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
#plt.xticks(range(len(x_time)), times, rotation=90)
#plt.yticks([each+1 for each in range(10)], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed'], rotation=0)
ax.set_xlabel('Time of Day')
ax.set_ylabel('Living Room Temperature')

day1 = []
time1 = []
day2 = []
time2 = []
day3 = []
time3 = []

for i in range(len(data)):
    if dates[i] == 0:
        day1.append(data[i])
        time1.append(times[i])
    if dates[i] == 1:
        day2.append(data[i])
        time2.append(times[i])
    if dates[i] == 2:
        day3.append(data[i])
        time3.append(times[i])

xavg = []
yavg = []
largest = []
L1 = len(time1)
L2 = len(time2)
L3 = len(time3)
#determine which is largest
if (L1 > L2) and (L1 > L3):
   largest = L1
elif (L2 > L1) and (L2 > L3):
   largest = L2
else:
   largest = L3

#get the average at that point, if theres not an average accross all three, get the average across what is available
for i in range(largest):
    if(i>=L1 and i<L2 and i<L3):   #(time1 is none)change all these to use the lengths of each array L1 L2 L3 instead of index checks
        xavg.append( (time2[i] + time3[i]) / 2 )
        yavg.append( (day2[i] + day3[i]) / 2 )
    elif(i>=L2 and i<L1 and i<L3): #time2 is none
        xavg.append( (time1[i] + time3[i]) / 2 )
        yavg.append( (day1[i] + day3[i]) / 2 )
    elif(i>=L3 and i<L1 and i<L2): #time3 is none
        xavg.append( (time1[i] + time2[i]) / 2 )
        yavg.append( (day1[i] + day2[i]) / 2 )
    elif(i>=L1 and i>=L2 and i<L3): #time1[i] is None and time2[i] is None
        xavg.append(time3[i])
        yavg.append(day3[i])
    elif(i>=L2 and i>=L3 and i<L1): #(time2[i] is None and time3[i] is None)
        xavg.append(time1[i])
        yavg.append(day1[i])
    elif(i>=L1 and i>=L3 and i<L2): #time1[i] is None and time3[i] is None
        xavg.append(time2[i])
        yavg.append(day2[i])
    else:
        xavg.append( (time1[i] + time2[i] + time3[i]) / 3 )
        yavg.append( (day1[i] + day2[i] + day3[i]) / 3 )

fit = np.polyfit(xavg,yavg,5)
fit_fn = np.poly1d(fit)

plt.plot(xavg,yavg, 'gx', xavg, fit_fn(xavg), '--g')

p1 = plt.scatter(time1, day1, marker=markers[0], color=colors1[0], label='1', s = 30)
p2 = plt.scatter(time2, day2, marker=markers[1], color=colors1[1], label='1', s = 30)
p3 = plt.scatter(time3, day3, marker=markers[2], color=colors1[2], label='1', s = 30)
plt.legend((p1, p2, p3), ('day 1', 'day 2', 'day 3'), loc='upper left')
plt.show()


