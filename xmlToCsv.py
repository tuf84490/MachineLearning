from xml.dom import minidom
#initialize the csv file
csv = open('audioDataMap.csv', 'w')
firstRow = "pathname,class_id,start,end\n"
csv.write(firstRow)
#get every xml metadata file in the training set
for i in range(1,67):
    if(i < 10):
        xmlFile = 'MIVIA_DB4_dist/training/0000' + str(i) + '.xml'
    else:
        xmlFile = 'MIVIA_DB4_dist/training/000' + str(i) + '.xml'
    mydoc = minidom.parse(xmlFile)
    events = mydoc.getElementsByTagName('events') #get the list of events
    items = events[0].getElementsByTagName('item') #get each event
    #for each event, print its file, the class, and the start and end of the event
    for elem in items:
        #print('id: ' + elem.attributes['idx'].value)
        print('location: ' + elem.getElementsByTagName('PATHNAME')[0].firstChild.data)
        print('class: ' + elem.getElementsByTagName('CLASS_ID')[0].firstChild.data)
        print('start: ' + elem.getElementsByTagName('STARTSECOND')[0].firstChild.data)
        print('end: ' + elem.getElementsByTagName('ENDSECOND')[0].firstChild.data)
    #put this data into a csv for preprocessing
    for elem in items:
        #do some string parsing to get the actual file name
        pathname = elem.getElementsByTagName('PATHNAME')[0].firstChild.data
        slashIndex = pathname.find('/')
        fileNumber = '0' + pathname[slashIndex+1:]
        fullFileName = fileNumber.replace('.wav','_00.wav')
        row = fullFileName + "," + elem.getElementsByTagName('CLASS_ID')[0].firstChild.data + "," + elem.getElementsByTagName('STARTSECOND')[0].firstChild.data + "," + elem.getElementsByTagName('ENDSECOND')[0].firstChild.data + "\n"
        csv.write(row)