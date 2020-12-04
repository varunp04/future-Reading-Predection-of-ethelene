import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers 


def update_dictionary_items(dict1, dict2):
    
    if dict2 is None:
        return dict1
    for k in dict1:
        if k in dict2:
            dict1[k] = dict2[k]

    return dict1


class lstmnet():
    def __init__(self, parameters={}):
        self.params = update_dictionary_items({'epochs':10,'act':'sigmoid','filter1':32,'filter2':32,'filter3':32,'filter4':32,'layers':1}, parameters)
    def learn(self, X, y,s,v_s):
        
        #construction of model
        self.net = models.Sequential()
        if(self.params['layers'] > 1):
            self.net.add(layers.LSTM(units=self.params['filter1'],dropout=0.2,recurrent_dropout=0.5,return_sequences=True,input_shape=(None,s)))
        elif(self.params['layers'] == 1):
            self.net.add(layers.LSTM(units=self.params['filter1'],dropout=0.2,recurrent_dropout=0.5,input_shape=(None,s)))
        if(self.params['layers'] > 2):
            self.net.add(layers.LSTM(units=self.params['filter2'],activation=self.params['act'],recurrent_dropout=0.5,return_sequences=True))
        elif(self.params['layers'] == 2):
            self.net.add(layers.LSTM(units=self.params['filter2'],activation=self.params['act'],recurrent_dropout=0.5))
        if(self.params['layers'] > 3):
            self.net.add(layers.LSTM(units=self.params['filter3'],activation=self.params['act'],recurrent_dropout=0.5,return_sequences=True))
        elif(self.params['layers'] == 3):
            self.net.add(layers.LSTM(units=self.params['filter3'],activation=self.params['act'],recurrent_dropout=0.5))
        if(self.params['layers'] > 4):
            self.net.add(layers.LSTM(units=self.params['filter4'],activation=self.params['act'],recurrent_dropout=0.5,return_sequences=True))
        elif(self.params['layers'] == 4):
            self.net.add(layers.LSTM(units=self.params['filter4'],activation=self.params['act'],recurrent_dropout=0.5))
        self.net.add(layers.Dense(1))
        self.net.compile(optimizer = 'rmsprop', loss = 'mae')
        #training
        his = self.net.fit_generator(X,steps_per_epoch=500,epochs=self.params['epochs'],validation_data=y,validation_steps=v_s)


        return np.mean(his.history['val_loss'])

    def predict(self, Xtest,t_s):
        test_loss = self.net.evaluate(Xtest,steps = t_s)
        return test_loss
      
class conv1d():
    def __init__(self, parameters={}):
        self.params = update_dictionary_items({'epochs':5,'filter2':64,'act':'relu','filter1':32,'filter3':64}, parameters)
    def learn(self, X, y,s,v_s):
        #construction of model
        self.net = models.Sequential()
        self.net.add(layers.Conv1D(self.params['filter1'],3,activation = self.params['act'],input_shape = (None,s)))
        self.net.add(layers.MaxPooling1D(1))
        self.net.add(layers.Conv1D(self.params['filter1'],3,activation  = self.params['act']))
        self.net.add(layers.MaxPooling1D(1))
        self.net.add(layers.Conv1D(self.params['filter1'],3,activation = self.params['act']))
        self.net.add(layers.GlobalMaxPooling1D())
        self.net.add(layers.Dense(1))
        #compile
        self.net.compile(optimizer='rmsprop',loss='mae')

        #training
        his = self.net.fit_generator(X,steps_per_epoch=500,epochs=self.params['epochs'],validation_data=y,validation_steps=v_s)

        return np.mean(his.history['val_loss'])
        

    def predict(self, Xtest,t_s):
        test_loss = self.net.evaluate(Xtest,steps = t_s)
        return test_loss