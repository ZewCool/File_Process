# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:21:30 2019

@author: guz4
"""

import os

### Part One: count the number of samples by 'sample_num()'. 
###           Rename every sample folders with an end of '00x' or '0xx'. 
###           Modify the random number in every 'RandomFibre' file by looking for the '00x' in the name of folder. 

def sample_num(path):                       
    ''' count the number of sample folders under 'path'. '''   
    names = os.listdir(path)                               # get the names of all files and folders to a list
    return FileProp(names, len(names))                                   # return the number of samples
    
def markFolders(numSample, markOrder=3, startNum=0):
    ''' the mark value is the order of folder with certain format'''
    mark = str(startNum + numSample).zfill(markOrder)
    return mark
    
def rename_folders(oldName, mark, newName):
    ''' rename the names of folders and files under 'path' by 'newName'. '''
    os.rename(os.path.join(path,oldName),
              os.path.join(path,newName + mark))  
    return 
                  

class FileProp:
    ''' the file properties of the folders. '''
    def __init__(self, names, sampleNum):
        self.FoNames = names
        self.SaNum = sampleNum
        
class NewName:
    ''' the new name of the file. '''
    def __init__(self, newName):
        self.NewName = newName
        
class ModifyFiles(object):
    ''' modify the line 'modeLine' in the file 'filePath' by 'newContent'
            the modLine should be an int. '''
    def __init__(self, modLine, newContent):
        self.ModLine = modLine
        self.NewCont = newContent
    
    def __call__(self, filePath):
        self.modify_file_line(filePath)
        return
        
    def modify_file_line(self, filePath):
        with open(filePath) as f:
            lines = []
            for line in f:
                lines.append(line)       
            lines[self.ModLine] = self.NewCont
            f.close() 
        # replace the old content  by newLine in 'lines'
        with open(filePath, 'w') as fw:   
            for newLine in lines:
                fw.write(newLine)                             # write the new content into 'RandomFibre.fis'
            fw.close()
        return
        
def mainFunc_part01():
    ''' rename the folders and modify the content of the files. '''
    fileProp = sample_num(path)                                 # assign the values in the Class 'FileProp' to 'FP'
    mfp = ModifyFiles(1, 'SET random 10%s\n')
    for numSample in range(fileProp.SaNum):
        mark = markFolders(numSample)
        rename_folders(fileProp.FoNames[numSample], mark, newName)       
        filePath = path + '\\%s%s\\%s' % (newName, mark, fileName)
        mfp(filePath)

if __name__ == '__main__':
    path = 'C:\\PhD_ValveSpringDynamics\\Others\\DingML\\SBoard_OneFibSamples'
    newName = 'Sample'
    fileName = 'RandomFibre.fis'
    mainFunc_part01()
