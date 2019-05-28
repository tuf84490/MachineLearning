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
    
x = time1
y = day1
fit1 = np.polyfit(x,y,5)
fit_fn1 = np.poly1d(fit1)

x2 = time2
y2 = day2
fit2 = np.polyfit(x2,y2,5)
fit_fn2 = np.poly1d(fit2)

x3 = time3
y3 = day3
fit3 = np.polyfit(x3,y3,5)
fit_fn3 = np.poly1d(fit3)

plt.plot(x,y, 'r^', x, fit_fn1(x), '--r')
plt.plot(x2,y2, 'cD', x2, fit_fn2(x2), '--c')
plt.plot(x3,y3, 'ks', x3, fit_fn3(x3), '--k')
p1 = plt.scatter(time1, day1, marker=markers[0], color=colors1[0], label='1', s = 30)
p2 = plt.scatter(time2, day2, marker=markers[1], color=colors1[1], label='1', s = 30)
p3 = plt.scatter(time3, day3, marker=markers[2], color=colors1[2], label='1', s = 30)
plt.legend((p1, p2, p3), ('day 1', 'day 2', 'day 3'), loc='upper left')
plt.show()


