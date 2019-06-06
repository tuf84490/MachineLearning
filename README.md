# MachineLearning
summer research work on IoT security using machine learning

# Audio Processing
## musicAudio and MIVIA folders
Currently working on taking audio files and processing them to be more usable by the LSTM RNN.

# Regression
## temperatureRegression folder
taking temperature data from three days and using polynomial regression of degree 5 (adjustable) to train a best fit line for the mean of all three days temperature readings

# K-Means Clustering
## powerDetection folder
simulating an attack where the attacker has compromised a smartplug and is trying to identify what devices are plugged into it by reading the power usage and timestamp data. This data can be plotted and then kmeans clustering can be performed to group (k value is 8 for this, but can be easily adjusted) the plots into 3 groups. We can then look at these groups and easily determine which devices are using the smartplug at a specific time.  