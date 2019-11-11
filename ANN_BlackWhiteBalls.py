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

def concat_pds(marx, mary, marBW, crackId):
    mxPDTrain, myPDTrain, mBWPDTrain, crackPDTrain = pd_x(marx), pd_x(mary), pd_x(marBW), pd_x(crackId)
    train_data = []
    for bNum in range(len(mxPDTrain)): 
        tr_data = pd.concat(
                [mxPDTrain[bNum], myPDTrain[bNum], 
                 mBWPDTrain[bNum], crackPDTrain[bNum]], axis=1)
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

def concat_everyUnderBoard(trainData):
    trDaUnd = []
    for trData in trainData:
        trDaUnd.append(trData)  
    trDaUnd = pd.concat(trDaUnd, axis=0)
    return trDaUnd

if __name__ == '__main__':
    allData = concat_pds(marx, mary, marBW, Class)
    trainData = allData[0:491]
    preData = allData[491:500]
    
    trDaUnd = concat_everyUnderBoard(trainData)
    preDaUnd = concat_everyUnderBoard(preData)
    
    trainFea = np.array(trDaUnd.iloc[:,trDaUnd.columns != 'Class'])
    trainRes = np.array(trDaUnd.iloc[:,trDaUnd.columns == 'Class'])
    count_classes=pd.value_counts(trDaUnd['Class'],sort=True).sort_index()
    count_classes.value_counts()
    count_classes.plot(kind='bar')
    plt.show()
    
    smo = SMOTE(ratio={1:400000},random_state=42)
    #smo = SMOTE(ratio={1:30500},kind='borderline1',random_state=42)
    #smo = SMOTE(ratio={1:40000},kind='borderline2',random_state=42)
    X_smo, Y_smo = smo.fit_sample(trainFea, trainRes)
    print(Counter(Y_smo))
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
    model.add(Dense(units = 9, input_shape=(3,), 
                    kernel_initializer ='glorot_uniform', activation='tanh', name='Dense_0'))
    #model.add(Dropout(0.05))
    #use_bias=True, bias_initializer='ones',
    #keras.layers.noise.GaussianNoise(0.01)
    model.add(Dense(3, kernel_initializer ='glorot_uniform', activation='tanh', name='Dense_1'))
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
    model.fit(X_smo, Y_smo, verbose=1, epochs = 200, batch_size = 760, 
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
        plt.scatter(preCra['marx'], preCra['mary'], s=20, c = 'blue')
       
        plt.show()
