from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
import Regression
import datetime
import csv
import re

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

actorsLen = 10
crewLen = 7
genresLen = 19
companiesLen = 48

class MovieData:
    pass

class ActorData:
    pass

def average(listt):
    rats = list(map(lambda x: x.Rating, listt))
    sumr = sum(rats)
    avr = sumr/len(rats)
    return avr

def sumUpCrewData(inPath, files, releaseDate):
    results = []
    for file in files:
        path = join(inPath, file + '.csv')
        try:
            data = pd.read_csv(path)
        except FileNotFoundError:
            file = re.sub(r'[^A-Za-z0-9_. ]+', '', file)
            path = join(inPath, file.replace(" ", "") + '.csv')
            data = pd.read_csv(path)
        dataM = []
        for index, row in data.iterrows():
            ob = ActorData()
            ob.Rating = row.Rating
            ob.Date = datetime.datetime.strptime(row.Date, '%d/%m/%Y')
            dataM.append(ob)
        dataM = sorted(dataM, key=lambda x: x.Date)
        recentRanks = list(filter(lambda x: x.Date <= releaseDate, dataM))
        for i in range (0, 8):
            recentRanks.insert(0, recentRanks[recentRanks.__len__() - 1])
        for i in range(0, 3):
            recentRanks.insert(0, recentRanks[recentRanks.__len__() - 2])
        for i in range(0, 2):
            recentRanks.insert(0, recentRanks[recentRanks.__len__() - 2])

        avr = average(recentRanks)
        results.append(avr)
    return  results


def processMovieCsv(names):
    movies = []
    for name in names:
        movData = pd.read_csv(name, header=None)
        actors = movData[3][0].split(';')
        actorsPop = movData[4][0].split(';')
        crew = movData[5][0].split(';')
        crewPop = movData[6][0].split(';')
        budget = movData[8][0]
        genreFlags = movData[10][0].split(';')
        companiesFlags = movData[11][0].split(';')
        target = movData[13][0]

        releaseDate = datetime.datetime.strptime(movData[12][0], '%Y-%m-%d')
        actorsAv = sumUpCrewData('../ActorsCsvs',actors, releaseDate)

        movieData = MovieData();
        setattr(movieData, 'target', target)
        setattr(movieData, 'budget', budget)
        for i in range (0, actorsLen):
            setattr(movieData, 'actorAv'+ str(i), actorsAv[i])
        for i in range(0, actorsLen):
            setattr(movieData, 'actorPo'+ str(i), float(actorsPop[i]))
        crewAv = sumUpCrewData('../CrewCsvs', crew, releaseDate)
        for i in range (0, crewLen):
            setattr(movieData, 'crewAv'+ str(i), crewAv[i])
        for i in range(0, crewLen):
            setattr(movieData, 'crewPo' + str(i), float(crewPop[i]))
        for i in range(0, genresLen):
            setattr(movieData, 'genre' + str(i), float(genreFlags[i]))
        for i in range(0, companiesLen):
            setattr(movieData, 'comp' + str(i), float(companiesFlags[i]))

        movies.append(movieData)
    return movies

def createTrainData():
    files = [f for f in listdir('../MoviesCsvs') if isfile(join('../MoviesCsvs', f))]
    files = list(map(lambda x: '../MoviesCsvs/' + x, files))
    moviesModels = processMovieCsv(files)

    with open("combined_csv.csv", 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(moviesModels[0].__dict__.keys())
        for mv in moviesModels:
            writer.writerow(mv.__dict__.values())

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


#createTrainData();
dataframe = pd.read_csv("combined_csv.csv")
dataframe.head()
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

feature_columns = []
headers = list(dataframe.columns)
headers.remove('target')
for header in headers:
    print(header)
    feature_columns.append(feature_column.numeric_column(header))


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu', dtype='float64'),
  layers.Dense(128, activation='relu', dtype='float64'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=500)