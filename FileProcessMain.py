# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:53:16 2019

@author: guz4
"""

import FileProcessFunctions as FPF
import numpy as np

def fibre_xyr(filePath, readStart, readEnd):    ## output fiber centre coordinates of file number under 'folderNum'
    fibInfo = []
    fibInformation = FPF.Files(filePath).read_file(readStart, readEnd)
    for fibProp in fibInformation:
        fibInfo.append(fibProp.split())
    fibx, fiby, fibr = [], [], []
    for fib in fibInfo:
        fibx.append(fib[1]), fiby.append(fib[2]), fibr.append(fib[3])
    return fibx, fiby, fibr

def matrix_result(filePath, readStart, readEnd, blackBall=False):
    matInfo = FPF.Files(filePath).read_file(readStart, readEnd) 
    matInfoStr = []
    for eachBall in matInfo:
        if blackBall == False:
            if eachBall[-2] == '0':
                matInfoStr.append(eachBall.split())
        if blackBall == True:
            matInfoStr.append(eachBall.split())
    return matInfoStr

def crackId_result(filePath, readStart, readEnd, matInfoStr):
    crackInfo = FPF.Files(filePath).read_file(readStart, readEnd)
    matCid = []
    for eachBall in matInfoStr:
        if eachBall[0] + '\n' in set(crackInfo):
            matCid.append('1')
        else:
            matCid.append('0')  
    return matCid

def file_path(num, mark):
    fileName = fileNames[num]
    filePath = path + '\\%s%s\\%s' % (newName, mark, fileName)
    return filePath

def unit_meterTomm(mar):
    mar_new = []
    for mar_board in mar:
        mar_board = mar_board*1e6
        mar_new.append(mar_board)
    return mar_new

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
    
    if stepNum == 3:
        marx, mary, marBW, crackId, fibCx, fibCy = [], [], [], [], [], []
        
        for oldName in FPF.sample_num(path).FoNames:
            startNum += 1
            mark = str(startNum).zfill(digitNum)
            
            filePathOut = file_path(3, mark)
            matInfoStr = matrix_result(filePathOut, 'matrix_and_fibre start\n', 'matrix_and_fibre end\n', blackBall=True)
            matCidStr = crackId_result(filePathOut, 'cracksid start\n', 'cracksid end\n', matInfoStr)
            filePathFib = file_path(1, mark)
            fibXStr, fibYStr, fibRStr = fibre_xyr(filePathFib, 'FibreCentreStart\n', 'FibreCentreEnd\n')
            
            # change the type from 'string' to 'float'
            matInfoFloat = FPF.map_str_to_float(matInfoStr)
            matCidFloat = FPF.map_str_to_float(matCidStr)
            fibXFloat, fibYFloat = list(map(float, fibXStr)), list(map(float, fibYStr))
            
            # make the data in each board as an array
            matInfoArr, matCidArr = np.array(matInfoFloat), np.array(matCidFloat)   
            fibXArr, fibYArr = np.array(fibXFloat), np.array(fibYFloat)
            
            # achieve datas on each board
            marXEachB, marYEachB, marBWEachB, crackIdEachB = matInfoArr[:, 1], matInfoArr[:, 2], matInfoArr[:, 3], matCidArr[:]
            
            fibCxEachB, fibCyEachB = fibXArr[:], fibYArr[:]
            
            # list the datas of all boards
            marx.append(marXEachB)
            mary.append(marYEachB)
            crackId.append(crackIdEachB)
            fibCx.append(fibCxEachB)
            fibCy.append(fibCyEachB)
            
            marBW.append(marBWEachB)
            
        return marx, mary, marBW, crackId, fibCx, fibCy
                   
        
if __name__ == '__main__':   
    # parameters defined outside functions
    path = 'D:\\1010NewSample1'
    newName = 'New1010Sample'
    fileNames = ['RandomFibre.fis',
                 'FibreCentre.log', 
                 'Hex1.fis', 
                 'output1.log']
    
    # main function 
    # stepNum == 1 when we want to change the names of sub-folders 
    # stepNum == 2 when we want to change the content in several files
    # stepNum == 3 can achieve the matrix and fibre data
    sampleNum = FPF.sample_num(path).SaNum
    marx, mary, marBW, crackId, fibCx, fibCy= main_func(stepNum=3, startNum=200)
    
    def crackOne(crackId):
        Class = []
        for i in crackId:
            c = np.array([])
            for j in i:
                c = np.append(c, j)
            Class.append(c)
        return Class
    
    Class = crackOne(crackId)
    marx = unit_meterTomm(marx)
    mary = unit_meterTomm(mary)
