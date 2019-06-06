MIVIA audio events data set for surveillance applications
version: MIVIA_DB4
date: 11/03/2013

CREATED BY:
Pasquale Foggia, Alessia Saggese, Nicola Strisciuglio, Mario Vento
University of Salerno (ITALY), Dept. of Information Eng., Electrical Eng. and Applied Math., MIVIA Lab

Contact person: 
Nicola Strisciuglio
email: nstrisciuglio@unisa.it

The data set
The MIVIA audio events data set is composed of a total of 6000 events for surveillance applications, namely glass breaking, gun shots and screams. The 6000 events are divided into a training set (composed of 4200 events) and a test set (composed of 1800 events).

In audio surveillance applications, the events of interest (for instance a scream) can occur at different distances from the microphone that correspond to different levels of the signal-to-noise ratio. Moreover, in these applications the events are generally mixed with a complex background, usually composed of several types of different sounds depending on the specific environments both indoor and outdoor (household appliances, cheering of crowds, talking people, traffic jam, passing cars or motorbikes etc.).

The data set is designed to provide each audio event at 6 different values of signal-to-noise ratio (namely 5dB, 10dB, 15dB, 20dB, 25dB and 30dB) and overimposed to different combinations of environmental sounds in order to simulate their occurrence in different ambiences.

Description
The sounds have been registered with an Axis P8221Audio Module and an Axis T83 omnidirectional microphone for audio surveillance applicationsare, sampled at 32000 Hz and quantized at 16 bits per PCM sample. The audio clips are distributed as WAV files. Every audio clip comes with a XML file that contains the metadata of the events and the description of the sounds.
The training set has a duration of about 20 hours while the test set of about 9 hours.
More details are reported on the web site http://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/.

MATLAB Script
In order to parse the XML files in MATLAB, the xml_toolbox is available in the folder MATLAB. Please use the function xml_load in order to read a XML file and obtain as result a MATLAB struct with the related information.
