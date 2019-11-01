# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:53:16 2019

@author: guz4
"""

import FileProcessFunctions as FPF

def fibre_xyr(filePath, readStart, readEnd):    ## output fiber centre coordinates of file number under 'folderNum'
    fibInfo = []
    fibInformation = FPF.Files(filePath).read_file(readStart, readEnd)
    for fibProp in fibInformation:
        fibInfo.append(fibProp.split())
    fibx, fiby, fibr = [], [], []
    for fib in fibInfo:
        fibx.append(fib[1]), fiby.append(fib[2]), fibr.append(fib[3])
    return fibx, fiby, fibr

def file_path(num, mark):
    fileName = fileNames[num]
    filePath = path + '\\%s%s\\%s' % (newName, mark, fileName)
    return filePath

def main_func(stepNum=3, startNum=0, digitNum=3):  
    '''
    example main function which rename multiple folders and 
    modify the file 'randomFibre' under each folder
    '''
    if stepNum == 1:
        for oldName in FPF.sample_num(path).FoNames:
            startNum += 1
            mark = str(startNum).zfill(digitNum)
            
            FPF.rename_folders(path, oldName, newName, mark)
            
            filePathRanfib = file_path(0, mark)
            fileRanfib = FPF.Files(filePathRanfib)  
            fileRanfib.modify_file_line(1, 'SET random 10%s\n' % mark)
            
    if stepNum == 2:
        for oldName in FPF.sample_num(path).FoNames:
            startNum += 1
            mark = str(startNum).zfill(digitNum)
            
            filePathFib = file_path(1, mark)
            fibX, fibY, fibR = fibre_xyr(filePathFib, 'FibreCentreStart\n', 'FibreCentreEnd\n')
            
            filePathHex = file_path(2, mark) 
            fileHex = FPF.Files(filePathHex)  
            linNums = [115, 117, 124, 131, 145]
            newContent = ['  array xxf(%s) yyf(%s) raddf(%s)\n' % (len(fibX), len(fibY), len(fibR)),
                          '  status = read(xxf, %s)\n' % len(fibX), 
                          '  status = read(yyf, %s)\n' % len(fibY),
                          '  status = read(raddf, %s)\n' % len(fibR),
                          '  loop _kk (1, %s)\n' % len(fibR)]
            for line, cont in zip(linNums, newContent):
                fileHex.modify_file_line(line, cont)
        
if __name__ == '__main__':     
    path = 'C:\\PhD_ValveSpringDynamics\\Others\\DingML\\SBoardSamples'
    newName = 'Sample'
    fileNames = ['RandomFibre.fis','FibreCentre.log', 'Hex1.fis']
    main_func(2)

               
               


    