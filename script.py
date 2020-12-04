import numpy as np
import tensorflow as tf
import regressor as algs
import sklearn.preprocessing as pr 
import generator as gt




    #data
file_name = "ethylene_CO.txt" # "ethylene_methane.txt"

data_arr = np.loadtxt("ethylene_CO.txt")
data_arr[:,3:] = pr.normalize(data_arr[:,3:],norm='l1',axis= 0 )
cut = int((data_arr.shape[0]*2)/3)
val_cut = cut - int(cut/10)
if(file_name == "ethylene_CO.txt"):
    m1 = 14
    train_step = 13132
    train_lookback = train_step * m1
    m2 = 14
    test_step = 13424
    test_lookback = test_step * m2
elif(file_name == "ethylene_methane.txt"):
    m1 = 13
    train_step = 15725
    train_lookback = train_step * m1
    m2 = 12
    test_step = 15034
    test_lookback = test_step * m2



tran_gen = gt.gene(data_arr[:,3:],train_lookback,1, 0, val_cut,False,128,train_step)

val_gen = gt.gene(data_arr[:,3:],train_lookback,1,val_cut,cut,False,128,train_step)

test_gen = gt.gene(data_arr[:,3:],test_lookback,1, cut, data_arr.shape[0],False,128, test_step)
val_s = (cut - val_cut - train_lookback) // 128
test_s = (data_arr.shape[0] - cut - test_lookback) //128  
    
    
classalgs = {
    'ConvNNet': algs.conv1d,
    'LStM' : algs.lstmnet,
    }

## hyper-parameters
parameters = {
    'ConvNNet' : [
            {'epochs':10,'filter2':64,'act':'relu','filter1':32,'filter3':64},
            {'epochs':10,'filter2':32,'act':'relu','filter1':32,'filter3':32},
            {'epochs':10,'filter2':64,'act':'relu','filter1':64,'filter3':64},
    ],
    'LStM' : [
            {'epochs':10,'filter1':32,'act':'relu','layers':1},
            {'epochs':10,'filter1':32,'act':'relu','filter2':64, 'layers': 2},
            {'epochs':10,'filter1':32,'act':'sigmoid','filter2':64,'layers':2},
            {'epochs':10,'filter1':64,'act':'relu','filter2':64,'filter3':64,'layers':3},
            {'epochs':10,'filter1':64,'act':'relu','filter2':64,'filter3':64,'filter4':128,'layers':4}
    ],
    
    }



errors = np.zeros((2,5))
i=0;
best_parameters = {}
for learnername, Learner in classalgs.items():
    params = parameters.get(learnername, [ None ])
    j=0;
    for p in range(len(params)):

        learner = Learner(params[p])
        print ('Running learner = ' + learnername )
        # Train model
        val_error = learner.learn(tran_gen,val_gen,data_arr[:,3:].shape[-1],val_s)
        
        errors[i][j] = val_error
        j += 1
    i += 1
print(errors)


######### only for best  comment the othersin parameter dictionary

b_erro = np.zeros((2,2))
i=0;
for learnername, Learner in classalgs.items():
    params = parameters.get(learnername, [ None ])
    j=0;
    for p in range(len(params)):

        learner = Learner(params[p])
        print ('Running learner = ' + learnername )
        # Train model
        val_error = learner.learn(tran_gen,val_gen,data_arr[:,3:].shape[-1],val_s)
        predictions = learner.predict(test_gen,test_s)
        print ('lowest Error for ' + learnername + ': ' + str(predictions))
        b_erro[i][j] = predictions
        j += 1
    i += 1
b_erro