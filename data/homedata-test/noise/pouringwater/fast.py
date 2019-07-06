import os, sys


FILE_NAME = 'water-pouring'
FILE_EXT = 'm4a'
numbering_from = 0


fp=r'./'
os.chdir(fp)


def rename(folder):
    files=[]
    filedir = os.listdir(folder)
    index = numbering_from
    for fl in filedir:
        if os.path.isfile(fl):
            if os.path.splitext(fl)[-1] != ".py":
                print(fl)
                os.rename(fl, FILE_NAME + '-' + str(index) + '.' + FILE_EXT)
                index = index + 1



#rename files in fp
rename(fp)

#convert .m4a to wav
import argparse
from pydub import AudioSegment
AudioSegment.converter = r"C:\\ffmpeg\\bin\\ffmpeg.exe"


for filename in os.listdir(fp):
       infilename = os.path.join(fp,filename)
       if not os.path.isfile(infilename): continue
       oldbase = os.path.splitext(filename)
       newname = infilename.replace('.tmp', '.m4a')
       output = os.rename(infilename, newname)
	   
formats_to_convert = ['.m4a']

for (dirpath, dirnames, filenames) in os.walk(fp):
    for filename in filenames:
        if filename.endswith(tuple(formats_to_convert)):

            filepath = dirpath + '/' + filename
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                track = AudioSegment.from_file(filepath,
                        file_extension_final)
                wav_filename = filename.replace(file_extension_final, 'wav')
                wav_path = dirpath + '/' + wav_filename
                print('CONVERTING: ' + str(filepath))
                file_handle = track.export(wav_path, format='wav')
                os.remove(filepath)
            except:
                print("ERROR CONVERTING " + str(filepath))
