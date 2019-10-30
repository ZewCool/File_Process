# -*- coding: utf-8 -*-
"""
@author: guz4

Count the number of samples by 'sample_num()'.
Rename every sample folders by 'rename_folders()'.
Modify the content by 'modify_file_line()'.

"""

import os

def sample_num(path):                       
    ''' count the number of sample folders under 'path'. '''   
    names = os.listdir(path)                               # get the names of all files and folders to a list
    return FileProp(path, names, len(names))                                   # return the number of samples
    
def markFolders(startNum, numSample, markOrder=3):
    ''' the mark value is the order of folder with certain format'''
    mark = str(startNum + numSample).zfill(markOrder)
    return mark
    
def rename_folders(path, oldName, mark, newName):
    ''' rename the names of folders and files under 'path' by 'newName'. '''
    os.rename(os.path.join(path,oldName),
              os.path.join(path,newName + mark))          # rename files and folders under the 'path' folder  
    return NewName(newName)
                  
def modify_file_line(filePath, modLine, newContent):
    ''' modify the line 'modeLine' in the file 'filePath' by 'newContent'
        the modLine should be an int. '''
    with open(filePath) as f:
        lines = []
        for line in f:
            lines.append(line)       
        lines[modLine] = newContent
        f.close() 
    # replace the old content  by newLine in 'lines'
    with open(filePath, 'w') as fw:   
        for newLine in lines:
            fw.write(newLine)                             # write the new content into 'RandomFibre.fis'
        fw.close()
    return

class FileProp:
    def __init__(self, path, names, sampleNum):
        self.Path = path
        self.FoNames = names
        self.SaNum = sampleNum
        
class NewName:
    def __init__(self, newName):
        self.NewName = newName
        
class ModFileProp:
    ''' chose the line 'modLine' and the content 'newContent' needs be modified'''
    def __init__(self, modLine, newContent):
        self.ModLine = modLine
        self.NewCont = newContent
        
def mainFunc_batch():
    startNum = 0
    FP = sample_num(path)                                 # assign the values in the Class 'FileProp' to 'FP'
    MFP = ModFileProp(1, 'SET random 10%s\n')
    for numSample in range(FP.SaNum):
        mark = markFolders(startNum, numSample)
        rename_folders(FP.Path, FP.FoNames[numSample], mark, newName)   
        
        filePath = path + '\\%s%s\\%s' % (newName, mark, fileNam)
        modLine = MFP.ModLine
        newContent = MFP.NewCont % mark
        modify_file_line(filePath, modLine, newContent)

if __name__ == '__main__':
    path = 'Fill file directory here!'
    newName = 'Type new folders names here'
    fileNam = 'Type the name of the file needs to be modified'
    mainFunc_batch()