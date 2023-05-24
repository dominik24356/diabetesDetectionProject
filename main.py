import math
from statistics import stdev
import seaborn as sns
import numpy as np
import pandas as pd
import random as rn
from numpy import mean
import matplotlib.pyplot as plt


class DataProcessing:

    #implementujemy algorytm Knuta, tasujemy wiersze zeby nie byly po kolei
    @staticmethod
    def shuffleData(x):
        for i in range(len(x)-1, 0, -1):
            j = rn.randint(0, i-1)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]


    @staticmethod
    def normalizeData(x):
        values = x.select_dtypes(exclude="object")
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = x[column]
            min1 = min(data)
            max1 = max(data)
            if max1 - min1 == 0:
                continue  # Skip normalization if the range is zero
            x[column] = (data - min1) / (max1 - min1)



    # metoda do dzielenia calego zbioru na dwa podzbiory
    @staticmethod
    def splitData(data, percentage):
        # x treningowy, y walidacyjny
        num_rows = len(data)
        split_index = int(num_rows * percentage)
        training = data.iloc[:split_index]
        testing = data.iloc[split_index:]
        return training, testing

    def cleanColumn(data, columns, thr=2):
        column_desc = data[columns].describe()

        q3 = column_desc[6]
        q1 = column_desc[4]
        IQR = q3 - q1

        top_limit_clm = q3 + thr * IQR
        bottom_limit_clm = q1 - thr * IQR

        filter_clm_bottom = bottom_limit_clm < data[columns]
        filter_clm_top = data[columns] < top_limit_clm

        filters = filter_clm_bottom & filter_clm_top

        data = data[filters]

        print("{} rows left after cleaning column : {}".format(len(data), columns))

        return data






class NaiveBayes:
    @staticmethod
    def classify(x, sample):
        probability = []
        classNames = x['Outcome'].unique().tolist()
        columnNames = x.columns.tolist()[:7]

        for className in classNames:
            prob = 1
            tmp = x[x["Outcome"] == className]

            for columnName in columnNames:
                data = tmp.loc[:, columnName]
                mu = mean(data)
                sigma = stdev(data)

                if columnName == "Glucose":
                    prob *= 0.47 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "BMI":
                    prob *= 0.29 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "Age":
                    prob *= 0.24 / (sigma * math.sqrt(np.pi * 2)) * np.exp(-0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "Pregnancies":
                    prob *= 0.22 / (sigma * math.sqrt(np.pi * 2)) * np.exp(
                        -0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "DiabetesPedigreeFunction":
                    prob *= 0.17 / (sigma * math.sqrt(np.pi * 2)) * np.exp(
                        -0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "Insulin":
                    prob *= 0.13 / (sigma * math.sqrt(np.pi * 2)) * np.exp(
                        -0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "SkinThickness":
                    prob *= 0.075 / (sigma * math.sqrt(np.pi * 2)) * np.exp(
                        -0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                elif columnName == "BloodPressure":
                    prob *= 0.065 / (sigma * math.sqrt(np.pi * 2)) * np.exp(
                        -0.5 * ((sample[columnName] - mu) / sigma) ** 2)
                else:
                    raise Exception("Wrong column name " + columnName)

            prob *= len(tmp) / len(x)
            probability.append([className, prob])

        maxprobNameAndValue = max(probability, key=lambda x: x[1])
        return maxprobNameAndValue






diabetesData = pd.read_csv(r'C:\Users\domin\Desktop\diabetes.csv')



# sns.pairplot(diabetesData, hue="Outcome",height=8, aspect=1.2)
# plt.subplots_adjust(bottom=0.10, left=0.08)
# sns.set(font_scale=0.2)
# plt.show()

# czysczenie
print("number of rows after clearing each column :")
for column in diabetesData.columns:
    diabetesData = DataProcessing.cleanColumn(diabetesData, column)


# shuffle
print("before shuffle :")
print(diabetesData.head(5))

DataProcessing.shuffleData(diabetesData)
print("after shuffle :")
print(diabetesData.head(5))


# # split
percentage = 0.7
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
normTrain, normTest = DataProcessing.splitData(diabetesData, 0.7)

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
    return dokladnosc

# print("nieznormalizowana :")
# analize(train, test)


print("outcome of normalized data :" + str(analize(normTrain, normTest)) + "%")



