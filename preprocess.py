from xml.dom import minidom
import csv

mydoc = minidom.parse('MIVIA_DB4_dist/training/00001.xml')
items = mydoc.getElementsByTagName('item')

for elem in items:  
    print(elem.attributes['idx'].value)