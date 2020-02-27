# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:36:41 2014

@author: djd
"""
import numpy as np

def warpFileToMatrix(warp,plot=False):
    for i in range(np.shape(warp)[0]):
        if warp[i][1] < warp[i+1][1]:
            numCols = i+1
            break
    numRows = (np.shape(warp)[0])/numCols
    X=np.zeros((numRows,numCols))
    Y=np.zeros((numRows,numCols))
    U=np.zeros((numRows,numCols))
    V=np.zeros((numRows,numCols))
    Intensity=np.zeros((numRows,numCols))
    for i in range(np.shape(warp)[0]):
        row = numRows - 1 - np.floor(i/numCols)
        col = i % numCols
        X[row][col]=warp[i][0]
        Y[row][col]=warp[i][1]
        U[row][col]=warp[i][2]
        V[row][col]=warp[i][3]
        Intensity[row][col]=warp[i][4]
    if plot:
        plt.subplot(151);plt.imshow(X)
        plt.subplot(152);plt.imshow(Y)
        plt.subplot(153);plt.imshow(U)
        plt.subplot(154);plt.imshow(V)
        plt.subplot(155);plt.imshow(Intensity)
    return X,Y,U,V,Intensity

def changeWarpAspect(warp,x=1):
    warpout = np.zeros((0,5))#make an empty first row of correct shape
    for i in range(np.shape(warp)[0]):
        if warp[i][0] < x and warp[i][0] > -1*x:
            warpout = np.vstack((warpout,warp[i,:]))
    np.delete(warpout,0,0)#get rid of the empty first row
    return warpout

def matrixToWarpFile(X,Y,U,V,I):
    numCols = np.shape(X)[1];print numCols
    numRows = np.shape(X)[0];print numRows
    warp = np.zeros((numCols*numRows,5))
    for i in range(np.shape(warp)[0]):
        row = int(np.floor(i/numCols))
        col = i % numCols
        print (row,col)
        warp[i][0]=X[row][col]
        warp[i][1]=Y[row][col]
        warp[i][2]=U[row][col]
        warp[i][3]=V[row][col]
        warp[i][4]=I[row][col]
    return warp

##strecth the Y warp only in specified xRange. default is the whole xRange
#def warpStretchY(warp,amount,area=(0,1)):
# warpout = warp
# xAspect = abs(warp[0][0])
# for i in range(np.shape(warp)[0]):
# if abs(warp[i][0]) <= area[1]*xAspect and abs(warp[i][0]) >= area[0]*xAspect :
# warpout[i][3]=warp[i][3]*amount #stretch warp
# return warpout
#
##strecth the X warp only in specified yRange. default is the whole yRange
#def warpStretchX(warp,amount,area=(0,1)):
# warpout = warp
# yAspect = abs(warp[0][1])
# for i in range(np.shape(warp)[0]):
# if abs(warp[i][1]) < area[1]*yAspect and abs(warp[i][1]) > area[0]*yAspect :
# warpout[i][2]=warp[i][2]*amount #stretch warp
# return warpout

#strecth the Y warp only in specified xRange. default is the whole xRange
def warpStretchY(warp,amount,area=(0,1)):
    warpout = warp
    xAspect = abs(warp[0][0])
    for i in range(np.shape(warp)[0]):
        if abs(warp[i][0]) <= area[1]*xAspect and abs(warp[i][0]) >= area[0]*xAspect :
            warpout[i][3]=warp[i][3]*amount #stretch warp
    return warpout
#strecth the X warp only in specified yRange. default is the whole yRange
def warpStretchX(warp,amount,area=(0,1)):
    warpout = warp
    yAspect = abs(warp[0][1])
    for i in range(np.shape(warp)[0]):
            if abs(warp[i][1]) < area[1]*yAspect and abs(warp[i][1]) > area[0]*yAspect :
                warpout[i][2]=warp[i][2]*amount #stretch warp
                return warpout

#strecth the warp only in specified yRange. default is the whole yRange
def warpStretch(warp,amount=0.1,limits=(0,0)):
# warpout = warp
# for i in range(np.shape(warp)[0]):
# if abs(warp[i][0]) > limits[0] and warp[i][1] > limits[1] :
# scale = ((abs(warp[i][0])*(1/abs(warp[0][0])))*amount)
# warpout[i][1]=(1+scale)*warp[i][1] #stretch warp
# return warpout
    X,Y,U,V,I = warpFileToMatrix(warp,plot=False)
    #newY = (Y+0.5) * np.ceil(I);newY[0,0]=Y[0,0];newY[Y.shape[0]-1,0]=Y[Y.shape[0]-1,0];
    newY = (Y) * np.ceil(I)   
    
    middleCol = int(np.shape(Y)[1]/2)
    middleRow = np.nonzero(newY)[0][0]+(np.nonzero(newY)[0][len(np.nonzero(newY)[0])-1] - np.nonzero(newY)[0][0])/2
    for i in range(middleCol):
        columnL = middleCol-i-1;
        columnR = middleCol+i
        bottom = middleRow+5
        if sum(np.abs(newY[:,columnL]))>0 and columnL>0:
            top = np.nonzero(newY[:,columnL])[0][0];#print (top,bottom)
            newTop = top + (i/2)
            newY[top:bottom,columnL]= np.linspace(Y[newTop,columnL],Y[bottom,columnL],(bottom-top))
            newY[top:bottom,columnR]= np.linspace(Y[newTop,columnR],Y[bottom,columnR],(bottom-top))
    for i in range(np.shape(newY)[0]):
        for j in range(np.shape(newY)[1]):
            if I[i,j]==0:
                newY[i,j]=Y[i,j]
                
#    for i in range(np.shape(newY)[1]):
#        if sum(np.abs(newY[:,i]))>0 and i>0:
#             if (np.abs(50-i)) > 12:
#                 for j in range(np.shape(newY)[0]-1):
#                     if newY[j,i] == 0 and newY[j+1,i] != 0:
#                        top = j+1
#                     if newY[j,i] != 0 and newY[j+1,i] == 0 or j == len(newY[:,i])-1:
#                        bottom = j+1
#                 stretch = round(((bottom-top)-13)/2)
#                 print (54-top)
#                 newY[top:bottom,i]=np.linspace(newY[47+((57-top)/2),i],newY[bottom,i],num=bottom-top)
##                    print 59+(np.abs(24-np.abs(50-i)))
#            #newY[:,i] = newY[:,i]*(abs(50-i)*amount+1)
#            #newY[:,i] =  newY[:,i]*(np.abs(50-i)/10+1)
    return newY