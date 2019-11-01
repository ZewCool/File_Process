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
  
class ModifyFiles(object):
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
    
def main_func(startNum=0, digitNum=3):  
    '''
    example main function which rename multiple folders and 
    modify the file 'randomFibre' under each folder
    '''
    for oldName in sample_num(path).FoNames:
        startNum += 1
        mark = str(startNum).zfill(digitNum)
        rename_folders(path, oldName, newName, mark)
        
        filePath = path + '\\%s%s\\RandomFibre.fis' % (newName, mark)
        modify_files = ModifyFiles(filePath)  
        modify_files.modify_file_line(1, 'SET random 10%s\n' % mark)
        
if __name__ == '__main__':     
    path = 'C:\\PhD_ValveSpringDynamics\\Others\\DingML\\SBoard_OneFibSamples'
    newName = 'Sample'
    main_func()
