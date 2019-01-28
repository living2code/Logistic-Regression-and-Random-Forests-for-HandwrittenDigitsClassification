
# coding: utf-8

# In[72]:


import numpy as np
import matplotlib as plt
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
import pickle
import time


# In[91]:


# Preprocess the training data and testing data
def preProcess(x_training,x_testing):
    x_training=np.reshape(x_training,(x_training.shape[0],784))
    x_training = x_training/255.0
    x_testing=np.reshape(x_testing, (x_testing.shape[0],784))
    x_testing =x_testing /255.0
    
    return x_training,x_testing


# In[125]:


#Modeling Logistic Regression as a function 
def mlModelLR(xDataTrain,yDataTrain,xDataTest,yDataTest):
    lr=LogisticRegression(solver='lbfgs',max_iter=150)
    learnedLRModel=lr.fit(xDataTrain,yDataTrain)
    predictionsLR=lr.predict(xDataTest)
    accuracyLR=lr.score(xDataTest,yDataTest)
    print("Accuracy of the Trained Logistic Regression Model is %2.2f percent" % (accuracyLR*100))
    return learnedLRModel,predictionsLR


# In[126]:


#Modeling RandomForest classifier as a function
def mlModelRF(xDataTrain,yDataTrain,xDataTest,yDataTest):
    rf=RandomForestClassifier(n_estimators=150)
    learnedRFModel=rf.fit(xDataTrain,yDataTrain)
    predictionsRF=rf.predict(xDataTest)
    accuracyRF=rf.score(xDataTest,yDataTest)
    print("Accuracy of the Trained Random Forest Model is %2.2f percent" % (accuracyRF*100))
    return learnedRFModel,predictionsRF


# In[96]:


def saveModel(modelName,learnedModel):
    filename = modelName+'.sav'
    pickle.dump(learnedModel, open(filename, 'wb'))


# In[97]:


def oneHotEncoding(final_predictions):
    oneHotEnc_final_pred=np.zeros((final_predictions.shape[0],10),dtype=int)
    i=0
    for pred in final_predictions:
        oneHotEnc_final_pred[i][pred]=1
        i+=1
    return oneHotEnc_final_pred


# In[98]:


def exportToCSV(oneHCPred,modelName):
    np.savetxt(modelName+".csv",oneHCPred,delimiter=",",fmt='%d')


# In[106]:


def main():
    print("This Script has the MNIST Dataset preloaded. To perform ML on this dataset")
    option=input('Enter the option for model you want to choose \n 1.Logistic Regression 2.Random Forest \n')
    save_model=input('Enter 1 if you want to save model: ')
    if save_model!='1':
        print("Not Saving the ML model")
    #preprocess data
    ((x_training,y_training),(x_testing,y_testing))=mnist.load_data()
    
    startTime=time.time()
    x_training,x_testing=preProcess(x_training,x_testing) 
    #select the ML model
    #make a prediction
    #print the score
    #display misclassified images
    if(option=='1'):
        print('Performing Logistic Regression')
        lrModel,lrPredictions=mlModelLR(x_training,y_training,x_testing,y_testing)
        oneHCPred=oneHotEncoding(lrPredictions)
        exportToCSV(oneHCPred,'lr')
        if save_model=='1':
            saveModel('LogisticRegressionModelPkl',lrModel)
        endTime=time.time()-startTime
        print("time taken for Logistic Regression model=",endTime)
    if(option =='2'):
        print('Performing Random Forest')
        rfModel,rfPredictions=mlModelRF(x_training,y_training,x_testing,y_testing)
        oneHCPred=oneHotEncoding(rfPredictions)
        exportToCSV(oneHCPred,'rf')
        if save_model=='1':
            saveModel('RandomForestModelPkl',rfModel)
        endTime=time.time()-startTime
        print("time taken for Random Forest model=",endTime)
    elif(option!='1' or option!='2'):
        print("Invalid Choice")
    
        
        
                 


# In[123]:


if __name__=="__main__":
    main()
    


# In[102]:


'''
Future Work: Want to Make it visually appeaking by plotting some training examples
and also some testing examples along with their labels.
also a separate section for the testing images where the model mispredicts the Labels.
Random Training Images
Random Testing Images
Ramdom Mispredicted Images to give a holistic comparision of the chosen ML Models
Later extending this to a Conv Neural Net
'''

