"""
Created on Sun Feb 27 21:00 2019

spam-classifier.py: a script to classify spam-email using Naive Bayes.

@author: Lei Zhao
"""
import numpy as np
import sys,getopt,glob,os
import re,time
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class CommandLine:

    def __init__(self):
        opts,args=getopt.getopt(sys.argv[1:],'')
        self.train_folder=args[0]
        self.test_folder=args[1]
        
    def getFolder(self):
        return self.train_folder,self.test_folder

class Tokenize:
    def __init__(self,folderName,testFolder):
        self.folderName=folderName
        self.testFolder=testFolder
    
    def readData(self):
        
        dataset=[]
        fileDic={}
        dicIndex={}
        testFile=[]
        
        trueLabeltest=[]
        
        for filename in glob.glob(os.path.join(self.folderName,'*.txt')):
           with open(filename ,'r') as f:
               content=re.findall('[a-zA-Z]+',f.read())
               if 'spm' in filename:
                   fileDic[tuple(content)]=0    
                   
               else:
                   fileDic[tuple(content)]=1
               dataset.extend(content)
               
        #we only consider the most common 3000 word to speed up
        frequentWord=Counter(dataset).most_common(3000)
        frequentWord=[k[0] for k in frequentWord]
        
        for i,w in enumerate(frequentWord):
            dicIndex[w]=i
        
        for filename in glob.glob(os.path.join(self.testFolder,'*.txt')):
            with open(filename,'r') as f:
                content=re.findall('[a-zA-Z]+',f.read())
                testFile.append(content)
                #get spam email
                if 'spm' in filename:
                    trueLabeltest+=[0] #get the true lable for test email set
                else:
                    trueLabeltest+=[1]
        return fileDic,dicIndex,testFile,trueLabeltest
    
    
class buildModel:
    def __init__(self):
        print('a')
        
    def train(self,fileDict,dicIndex,testFile):
        
        feature_matrix=np.zeros((len(fileDict),3000))
        trainLabels=np.zeros(len(fileDict))
        
        docId=0
        
        for k,v in fileDict.items():
            for w in k:
                if w in dicIndex:
                        feature_matrix[docId][dicIndex[w]]=k.count(w)

            trainLabels[docId]=v
            
            docId+=1
        
    
        #=============predict========================================
        test_fea_matrix=np.zeros((len(testFile),3000)) #set metrix for test data
        testDoc=0
        
        for sentence in testFile:
            for word in sentence:
                if word in dicIndex:
                    test_fea_matrix[testDoc][dicIndex[word]]=sentence.count(word)
                    
            testDoc+=1
            
        #using naive bayes
        NB_model=GaussianNB()
        NB_model.fit(feature_matrix,trainLabels)
        result=NB_model.predict(test_fea_matrix)
        
        return result


    def getAccuracy(self,trueLable,predictLable):
        
        accuracy=accuracy_score(trueLable,predictLable)
        print(accuracy)
                        
            
            

if __name__=='__main__':
    
    startTime=time.time()
    
    command=CommandLine()
    train_folder,test_folder=command.getFolder()
    
    token=Tokenize(train_folder,test_folder)
    fileDic,dicIndex,testFile,testLable=token.readData()
    
    model=buildModel() #build model
    predictLable=model.train(fileDic,dicIndex,testFile)
    model.getAccuracy(testLable,predictLable)
    
    endTime=time.time()
    
    print(endTime-startTime)
    
    
    