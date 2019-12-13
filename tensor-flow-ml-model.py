from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

URL = '<your-path>/new-memes-dataset.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('videojuegos')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('Cada caracteristica: ', list(feature_batch.keys()))
    print('Grupo de edad: ', feature_batch['edad'])
    print('Grupo de objetivos: ', label_batch)

# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]


# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

feature_columns = []

# Creamos diferentes columnas para ver resultados de cada caracteristica
age = feature_column.numeric_column("edad")
demo(age)

age_buckets = feature_column.bucketized_column(age, boundaries=[16, 25, 30, 35, 40, 45, 50])
demo(age_buckets)
feature_columns.append(age_buckets)

genero = feature_column.categorical_column_with_vocabulary_list(
    'genero', ['Hombre', 'Mujer'])

genero_one_hot = feature_column.indicator_column(genero)
demo(genero_one_hot)
feature_columns.append(genero_one_hot)

ubicacion = feature_column.categorical_column_with_vocabulary_list(
    'ubicacion', ['Pueblo', 'Ciudad'])

ubicacion_one_hot = feature_column.indicator_column(ubicacion)
demo(ubicacion_one_hot)
feature_columns.append(ubicacion_one_hot)

estado_civil = feature_column.categorical_column_with_vocabulary_list(
    'estado_civil', ['Casado', 'Soltero', 'Union Libre'])

estado_civil_one_hot = feature_column.indicator_column(estado_civil)
demo(estado_civil_one_hot)
feature_columns.append(estado_civil_one_hot)

orientacion_politica = feature_column.categorical_column_with_vocabulary_list(
    'orientacion_politica', ['Anarquista', 'Centro', 'Centro-izquierda', 'Derecha', 'Izquierda', 'Ninguna'])

orientacion_politica_one_hot = feature_column.indicator_column(orientacion_politica)
demo(orientacion_politica_one_hot)
feature_columns.append(orientacion_politica_one_hot)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)


feature_columns = []
# Definimos y probamos combinaciones de columnas
crossed_caracteristicas = feature_column.crossed_column([age_buckets, genero, ubicacion, estado_civil, orientacion_politica], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_caracteristicas))
feature_columns.append(crossed_caracteristicas)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)



