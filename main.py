import math
import os
from statistics import stdev
import seaborn as sns
import numpy
import numpy as np    # do odczytu z pliku
import pandas as pd   # do analiz
import random as rn   # do randomowych liczb
from numpy import mean
import matplotlib.pyplot as plt


class DataProcessing:

    #implementujemy algorytm Knuta, tasujemy wiersze zeby nie byly po kolei
    @staticmethod
    def shuffleData(x):
        for i in range(len(x)-1, 0, -1):
            j = rn.randint(0, i-1)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]

    #metoda do przeskalowania wszytkich dancyh w kolumnie na mnijsze ( mamy na to wzor xprim = x-min/max-min
    @staticmethod
    def normalizeData(x):
        # metoda pozwala nam wykluczyc jakas kolumne
        # object to obiekt czyli string bo string w pythonie to object
        values = x.select_dtypes(
            exclude="object")  # w skrocie to wyrzuca z bazy nienumeryczne dane np string, bo by sie nie dalo zrobic obliczen
        values = values
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = x.loc[:,
                   column]  # to prawdopodobnie bierze wszystkie wartosci z danej kolumny i pozniej bierze min i max wartosc z niej
            # srednia danego atrybutu
            min1 = min(data)
            max1 = max(data)

            for row in range(len(x)):
                # wzór na normalizacje
                xprim = (x.at[row, column] - min1) / (max1 - min1)

                x.at[row, column] = xprim  # tu podmiana wartsci oryginalnej na przeskalowaną



    # metoda do dzielenia calego zbioru na dwa podzbiory
    @staticmethod
    def splitData(data, percentage):
        # x treningowy, y walidacyjny
        num_rows = len(data)
        split_index = int(num_rows * percentage)
        training = data.iloc[:split_index]
        testing = data.iloc[split_index:]
        return training, testing






class NaiveBayes:
    @staticmethod
    def classify(x, sample):
        probability = []
        classNames = x['Outcome'].unique().tolist()
        columnNames = x.columns.tolist()[:8]

        for className in classNames:
            prob = 1
            tmp = x[x["Outcome"] == className]

            for columnName in columnNames:
                data = tmp.loc[:, columnName]
                mu = mean(data)
                sigma = stdev(data)

                if columnName == "Glucose":
                    prob *= 2 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                if columnName == "BMI":
                    prob *= 1.5 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                if columnName == "DiabetesPedigreeFunction":
                    prob *= 1.5 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                if columnName == "Pregnancies":
                    prob *= 0.8 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                else:
                    prob *= 1 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)

            prob *= len(tmp) / len(x)
            probability.append([className, prob])

        maxprobNameAndValue = max(probability, key=lambda x: x[1])
        return maxprobNameAndValue






diabetesData = pd.read_csv(r'C:\Users\domin\Desktop\diabetes.csv')






# sns.pairplot(diabetesData, hue="Outcome",height=8, aspect=1.2)
# plt.subplots_adjust(bottom=0.10, left=0.08)
# sns.set(font_scale=0.2)
# plt.show()


# shuffle
print("before shuffle :")
print(diabetesData.head(5))

DataProcessing.shuffleData(diabetesData)
print("after shuffle :")
print(diabetesData.head(5))


# # split
percentage = 0.6
train, test = DataProcessing.splitData(diabetesData, percentage)
print("length of data :")
print(len(diabetesData))
print("split percentage :"+str(percentage*100)+"%")
print("length of training data :")
print(len(train))
print("length of testing data :")
print(len(test))



# znormalizowanie i podzielenie na treningowa testowa
DataProcessing.normalizeData(diabetesData)
print("length of normalized data :" + str(len(diabetesData)))
normTrain, normTest = DataProcessing.splitData(diabetesData, 0.6)

print("normalized training data :")
print(normTrain.head(5))

print("normalized testing data :")
print(normTest.head(5))


# Analiza
def analize(tr, tst):
    counter = 0
    for i in range(len(tst)):
        tmp = NaiveBayes.classify(tr, tst.iloc[i])[0]
        if tmp == tst.iloc[i]['Outcome']:
            counter += 1


    dokladnosc = float(counter) / len(tst) * 100
    print(dokladnosc, "%")

print("nieznormalizowana :")
analize(train, test)

print("znormalizowana :")
analize(normTrain, normTest)