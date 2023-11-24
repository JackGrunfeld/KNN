from ast import List
import csv
import math 
import operator
import sys
import pandas as pd

training_data = list()


class Number:
    def __init__(self, num, classification):
        self.num = num
        self.classification = classification

# make print function    
def printList(list):
    for i in range(len(list)):
        print(list[i].num, list[i].classification)


def loadTrainingData(filename, filename2):
   
    dh = pd.read_csv(filename, nrows=0, delimiter=' ').columns.tolist()
    df = pd.read_csv(filename, delimiter=' ')
    df2 = pd.read_csv(filename2, delimiter=' ')
    objectList = list()
    objectList2 = list()
    
    for i in range(len(df.columns)-1):
        min = df.iloc[:, i].min()
        max = df.iloc[:, i].max()
        df.iloc[:, i] = (df.iloc[:, i] - min) / (max - min)
        df2.iloc[:, i] = (df2.iloc[:, i] - min) / (max - min)

    for i in range(len(df)):
        tempList = list()
        classification = df.iloc[i, len(dh)-1] 
        for j in range(len(df.columns)-1):
            tempList.append(df.iloc[i, j])
            
        n = Number(tempList, classification)
        objectList.append(n) 
        

    for i2 in range(len(df2)):
        tempList2 = list()
        classification2 = df2.iloc[i2, len(dh)-1] 
        for j2 in range(len(df2.columns)-1):
            tempList2.append(df2.iloc[i2, j2])  
        n2 = Number(tempList2, classification2)
        objectList2.append(n2)    
        
    return objectList, objectList2

  
def getDistances(wine):
    DistancesList = list()
    for w in training_data:
        dist = euclideanDistance(wine, w)
        DistancesList.append((w, dist))
     
    return DistancesList


def euclideanDistance(wine1, wine2):
    distance = 0
    for i in range(len(wine1.num)):
        distance += pow((wine1.num[i] - wine2.num[i]), 2)
    sDist = math.sqrt(distance)
   
    return sDist

def getKNeighbors(wine, K):
    distances = getDistances(wine)
    distances = sorted(distances, key=lambda w: w[1])
    
    neighbors = list() 
    for i in range(K):
      neighbors.append(distances[i][0])
      
    return neighbors


def getPredict(wine, K):
    neighbors = getKNeighbors(wine, K)
    mode = 0
    modeCount = 0
    
    for i in range(len(neighbors)):
        count = 0
        for j in range(len(neighbors)):
            if neighbors[i].classification == neighbors[j].classification:
                count += 1
        if count > modeCount:
            modeCount = count
            mode = neighbors[i].classification
            
            
    return mode , mode == wine.classification


if __name__ == '__main__': 
    training_data, test_data = loadTrainingData(sys.argv[2], sys.argv[1])
    K = int(input("Type the value of K:"))
    strr = None
    predictions = list()
    correct = 0
    preList = list()
    for wine in test_data:
        predicted_class, right = getPredict(wine, K)
        preList.append(predicted_class)
        if(right):
            correct+=1
    print(preList)
    print("The accuracy is:", (correct/float(len(test_data))) * 100.0, "%")

