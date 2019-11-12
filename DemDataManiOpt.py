# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:29:39 2019

@author: guz4
"""
import math 
import numpy as np

def dis_mar_to_fibC(marx, mary, fibCx, fibCy):
    distAngle = []
    for numB in range(len(marx)): 
        marXY, fibCXY = np.vstack((marx[numB], mary[numB])) .T, np.vstack((fibCx[numB], fibCy[numB])) .T
        distB = []
        for marxy in marXY:
            distFunc = np.sqrt(np.sum(np.square(marxy - fibCXY), axis=1))
            angleFunc = angle_to_fib(marxy, distFunc, fibCXY)
            disAndAng = np.vstack((distFunc, angleFunc)) .T
            distB.append(disAndAng)
        distAngle.append(distB)
    return distAngle

def angle_to_fib(marxy, distFunc, fibCXY):
    angleFunc = []
    for numxy in range(len(fibCXY)):
        angle = math.degrees(math.asin((marxy[1] - fibCXY[numxy][1]) / distFunc[numxy]))
        if angle > 0:
            angle = angle if marxy[0] > fibCXY[numxy][0] else 180 - angle
        elif angle < 0: 
            angle = 360 + angle if marxy[0] > fibCXY[numxy][0] else 180 - angle
        elif angle == 0:
            angle = angle if marxy[0] > fibCXY[numxy][0] else 180
        else:
            print('Error in Angle!')
        angleFunc.append(angle)
    return np.array(angleFunc)

def comb_distToArr(marx, mary, fibCx, fibCy):
    distArr = dis_mar_to_fibC(marx, mary, fibCx, fibCy)
    for numB in range(len(distArr)):
        distArr[numB] = np.r_[distArr[numB]] .T
    return distArr
 
def sort_distAngleByDist(marx, mary, fibCx, fibCy):   
    testda = dis_mar_to_fibC(marx, mary, fibCx, fibCy)       
    sortList = []
    for da in testda:
        danlist=[]
        for dan in da:
            danlist.append(sorted(dan, key = lambda x:x[0]))
        sortList.append(danlist)
    return sortList

def numst_min_dist(numst, marx, mary, fibCx, fibCy):
    sortList = sort_distAngleByDist(marx, mary, fibCx, fibCy)
    numstDist = []
    numstAngle = []
    for matDAa in sortList:
        numstDist1 = []
        numstAngle1 = []
        for matDA in matDAa:
            try:
                numstDist1.append(matDA[numst][0])
                numstAngle1.append(matDA[numst][1])
            except IndexError:
                numstDist1.append(float(substitute))
                numstAngle1.append(float(substitute))
        numstDist.append(numstDist1)
        numstAngle.append(numstAngle1)
    return numstDist, numstAngle

def min_da_all(numAll, marx, mary, fibCx, fibCy):
    minDa = []
    minAng = []
    for num in range(numAll):
        dist, ang = numst_min_dist(num, marx, mary, fibCx, fibCy)
        minDa.append(dist)
        minAng.append(ang)
    return minDa, minAng
        
if __name__ == '__main__':
    substitute = 10000
    minDist, minAng = min_da_all(32, marx, mary, fibCx, fibCy)
