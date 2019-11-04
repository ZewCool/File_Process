# -*- coding: utf-8 -*-
"""
@author: guz4
"""

import os

def sample_num(path):                       
    ''' count the number of sample folders under 'path'. '''   
    names = os.listdir(path)                 # get the names of all objects under the path and save to a list
    return FileProp(names, len(names))       # return the number of subfolders

def rename_folders(path, oldName, newName, mark):
    ''' rename the names of folders and files under 'path' by 'newName'. '''
    os.rename(os.path.join(path,oldName),
              os.path.join(path,newName + str(mark)))  
    return 

def map_str_to_float(oldListStr):                                   ## function to alter a string list to a float list 
    newListFloat = []
    for eachEle in oldListStr:                                       # currently support only 2 layers
        eachEleFloat = list(map(float, eachEle))
        newListFloat.append(eachEleFloat)
    return newListFloat

### classes below###
    
class FileName(object):
    def __init__(self, names):
        self.FoNames = names
        
class FileProp(FileName):
    ''' the properties of the folders under 'path'. '''
    def __init__(self, names, sampleNum):
        FileName.__init__(self, names)
        self.SaNum = sampleNum 
        
    def __call__(self):
        return self.SaNum       
  
class Files(object):
    ''' 
    modify the line 'modeLine' in the file 'filePath' by 'newContent'
    the modLine should be an int. 
    '''
    def __init__(self, filePath):
        self.filePath = filePath
        
    def modify_file_line(self, modLine, newContent):
        with open(self.filePath) as f:
            lines = []
            for line in f:
                lines.append(line)       
            lines[modLine] = newContent 
            f.close() 
        # replace the old content  by newLine in 'lines'
        with open(self.filePath, 'w') as fw:   
            for newLine in lines:
                fw.write(newLine)                             # write the new content into 'RandomFibre.fis'
            fw.close()          
        return

    def rewrite_file(self, newCont):         ## function to write the content of 'varNameR' into 'fileNameW'
        file = open(self.filePath, 'a+')
        file.read()
        file.seek(0)                                        # reset the pointer to the beginning
        file.truncate()                                     # clear all the data from the position of 'f.seek(0)'
        for line in newCont:
            file.write(line +'\n')                         # write every x in every row
        file.close()  
        return 
        
    def read_file(self, readStart, readEnd, 
                  readAll=False, readCommand=False, ceaseCommand=False):   
        '''
        By default, read file from 'readStart' to 'readEnd'
        If readAll=True, read all the lines in the file
        Read from beginning when readCommand=True; Read until end when ceaseCommand=True
        '''
        with open(self.filePath) as oriFile:
            content = []
            for line in oriFile:
                if readAll:
                    content.append(line)
                else:
                    if line == readStart:
                        readCommand = True
                        continue
                    if line == readEnd:
                        ceaseCommand = True
                    if readCommand and not ceaseCommand:
                        content.append(line)
            oriFile.close()     
        return content  
