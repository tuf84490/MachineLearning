import os


FILE_NAME = 'driving'
FILE_EXT = 'wav'
fp=r'./'
os.chdir(fp)
index_from = 0


def eachfile(filepath):
    files=[]
    #pathdir = os.listdir(filepath)

    # for s in pathdir:
    #     newdir = os.path.join(filepath,s) # 
    #     if os.path.isfile(newdir):  
    #         if os.path.splitext(newdir)[1]==".au": # ".au"
    #             files.append(newdir)

    filedir = os.listdir(filepath)
    index = index_from

    for fl in filedir:
        #print(os.path.splitext(fl)[0])
        if os.path.isfile(fl):
            if os.path.splitext(fl)[-1] != ".py":
                #os.rename()
                print(fl)
                os.rename(fl, FILE_NAME + '-' + str(index) + '.' + FILE_EXT)
                index = index + 1

# f=eachfile(fp)


# def rename_txtfilename(fp):

#     f=eachfile(fp)

#     for i in range(len(f)):
#         nowdir = os.path.split(f[i])[0]
#         filename = os.path.split(f[i])[1]
#         # print(nowdir+"\\\\"+filename)
#         # print("file path:%s, filename:%s" %(nowdir,filename))
#         os.rename(nowdir+"\\\\"+filename,nowdir+"\\\\"+str(i)+'.au')

# rename_txtfilename(fp)

# f = os.listdir(fp)

# def music_convert(fp):

#     f = os.listdir(fp)
#     g=[] 

#     for i in range(len(f)):
#         g.append(f[i].split('.')[0]) 
#         # g.append(str(i))              
#     for i in range(len(f)):
#         os.system('sox %s %s' %(f[i], g[i]+'.wav'))

#music_convert(fp)

eachfile(fp)
