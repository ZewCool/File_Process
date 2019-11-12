# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras import regularizers 
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE

def namestr(obj, namespace):
    ''' retreive the name of a variable '''
    return [name for name in namespace if namespace[name] is obj]

def pd_x(marVar):                                        
    ''' transfer marx, mary and cId from array to Series '''
    mar = []
    for m in marVar:
        marxTrain = pd.Series(m, name = namestr(marVar, globals())[0])
        mar.append(marxTrain)
    return mar

def pd_DA(minDA):                                        # transfer dist and ang from array to Series
    daPD = []
    for num in range(len(minDA)):
        da = []
        for bDist in minDA[num]:
            nm = 'dist0%d' % (num+1)
            da.append(pd.Series(bDist, name = nm))
        daPD.append(da)
    return daPD

def numC_to_numB(distDF):                                
    distDaF = []
    for num in range(len(distDF[1])):
        dist = []
        for nstDist in distDF:
            dist.append(nstDist[num])
        dist = pd.concat(dist, axis=1)
        distDaF.append(dist)
    return distDaF

def concat_pds(marx, mary, marBW, crackId, minDist, minAng):
    mxPDTrain, myPDTrain, mBWPDTrain, crackPDTrain = pd_x(marx), pd_x(mary), pd_x(marBW), pd_x(crackId)
    distDaF, angDaF = numC_to_numB(pd_DA(minDist)), numC_to_numB(pd_DA(minAng))
    train_data = []
    for bNum in range(len(mxPDTrain)): 
        tr_data = pd.concat(
                [mxPDTrain[bNum], myPDTrain[bNum], mBWPDTrain[bNum],
                 distDaF[bNum], angDaF[bNum], crackPDTrain[bNum]], axis=1)
        train_data.append(tr_data)
    return train_data

def under_sample(train_data):
    
    crack_indices = np.array(train_data[train_data.crackIdone == 1].index)
    norm_indices = np.array(train_data[train_data.crackIdone == 0].index)
    number_record_crack = len(train_data[train_data.crackIdone==1])
    leftNum = 100 - number_record_crack 
    randNormIndices = np.array(np.random.choice(norm_indices, leftNum, replace=False))
    
    under_sample_indices = np.concatenate([crack_indices,randNormIndices])
    under_sample_data = train_data.iloc[under_sample_indices,:]
    
    return under_sample_data

def upSample(trainData):    
    ''' upsampling input data 'trainData',  '''
    X_smo, Y_smo = [], []
    for tr in trainData:
        count_classes=pd.value_counts(tr['Class'],sort=True).sort_index()
        numCra = 2200 - 1476 + np.array(count_classes)[1]
        trainFea, trainRes = np.array(tr)[:, 0:67], np.array(tr)[:, 67:68]
        smo = SMOTE(ratio={1:numCra},random_state=2)
        x_smo, y_smo = smo.fit_sample(trainFea, trainRes)
        X_smo.append(x_smo)
        Y_smo.append(y_smo)   
    trainFea = np.array([item for sublist in X_smo for item in sublist])
    trainRes = np.array([item for sublist in Y_smo for item in sublist])
    return trainFea, trainRes

def concat_everyUnderBoard(trainData):
    trDaUnd = []
    for trData in trainData:
        trDaUnd.append(trData)  
    trDaUnd = pd.concat(trDaUnd, axis=0)
    return trDaUnd

if __name__ == '__main__':
    allData = concat_pds(marx, mary, marBW, Class, minDist, minAng)
    trainData = allData[0:490]

    # underSample
    # trDaUnd = concat_everyUnderBoard(trainData)
    # trainFea = np.array(trDaUnd.iloc[:,trDaUnd.columns != 'Class'])
    # trainRes = np.array(trDaUnd.iloc[:,trDaUnd.columns == 'Class'])  
    
    # upSample
    trainFea, trainRes = upSample(trainData)
   
    preData = allData[490:500]
    preDaUnd = concat_everyUnderBoard(preData)  
    preFea = np.array(preDaUnd.iloc[:,preDaUnd.columns != 'Class'])
    preCra = np.array(preDaUnd.iloc[:,preDaUnd.columns == 'Class'])    


    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = {'batch':[], 'epoch':[]}
            self.accuracy = {'batch':[], 'epoch':[]}
            self.val_loss = {'batch':[], 'epoch':[]}
            self.val_acc = {'batch':[], 'epoch':[]}
    
        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))
    
        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))
    
        def loss_plot(self, loss_type):
            iters = range(len(self.losses[loss_type]))
            plt.figure()
            # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # val_acc
                plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
                # val_loss
                plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.show()
        
        
        
    # create model
    model = Sequential()
    #model.add(Dropout(0.2))
    #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0.0, amsgrad=False)
    model.add(Dense(units = 201, input_shape=(67,), 
                    kernel_initializer ='glorot_uniform', activation='tanh', name='Dense_0'))
    #model.add(Dropout(0.05))
    #use_bias=True, bias_initializer='ones',
    #keras.layers.noise.GaussianNoise(0.01)
    model.add(Dense(67, kernel_initializer ='glorot_uniform', activation='tanh', name='Dense_1'))
    #kernel_regularizer=regularizers.l2(0.001), 
    #activity_regularizer=regularizers.l1(0.001),kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.01),
    #model.add(Dropout(0.05))
    ## model.add(Dense(70, kernel_initializer ='glorot_uniform', activation='tanh', name='Dense_2'))
    #model.add(GlobalAveragePooling1D())#model.add(Dropout(0.1))
    #model.add(Dropout(0.05))
    #model.add(Dense(52, kernel_initializer ='glorot_uniform', activation='tanh'))
    #model.add(Dropout(0.1))
    #model.add(Dense(61, use_bias=True, bias_initializer='random_uniform',kernel_initializer ='glorot_uniform', activation='relu'))
    
    #model.add(Dense(30, kernel_initializer ='glorot_uniform', activation='tanh'))
    #model.add(Dropout(0.05))
    model.add(Dense(1, kernel_initializer ='glorot_uniform', activation='sigmoid'))
    # Compile model
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #创建一个实例history
    history = LossHistory()
    # Fit the model‼
    model.fit(trainFea, trainRes, verbose=1, epochs = 100, batch_size = 2200, 
              validation_data = (preFea, preCra),
              callbacks = [history])

    for preB in preData:
        preFeaB = np.array(preB.iloc[:,preB.columns != 'Class'])
        oriClaB = np.array(preB.iloc[:,preB.columns == 'Class'])
    
        # Prediction
        predictions = model.predict(preFeaB)
        preResB = [round(x[0]) for x in predictions]
        
        # plot prediction
        rounda = pd.Series(preResB, name='preCraB')
        dataOriPre = pd.concat([preB, rounda], axis=1)
        
        oriCra = dataOriPre[dataOriPre.Class.isin(['1'])]
        oriMar = dataOriPre[dataOriPre.Class.isin(['0'])]
        preCra = dataOriPre[dataOriPre.preCraB.isin(['1'])]
        
        plt.figure(figsize = (10, 10))
        plt.scatter(oriCra['marx'], oriCra['mary'], s=150)
        plt.scatter(oriMar['marx'], oriMar['mary'], s=150, c = 'grey')
        plt.scatter(preCra['marx'], preCra['mary'], s=20, c = 'red')
       
        plt.show()