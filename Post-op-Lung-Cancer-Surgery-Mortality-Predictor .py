import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
import matplotlib.style as style


columns = ['DGN', 'PRE4', 'PRE5', 'PRE6', 'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE14', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'AGE', 'Risk1Yr']

ds = pd.read_csv('ThoraricSurgery.csv', names=columns, delim_whitespace=True)
ds = ds.reindex(np.random.permutation(ds.index))
train_features = ds.drop('Risk1Yr',axis=1)
train_label = ds['Risk1Yr']

for x in train_features.columns:
    if x != ['PRE4','PRE5','AGE']:
        train_features[x] = train_features[x].astype('category').cat.codes


s = StandardScaler()
for x in train_features.columns:
        train_features[x] = s.fit_transform(train_features[x].values.reshape(-1, 1)).astype('float64')


label_encoder = LabelEncoder()
train_label = label_encoder.fit_transform(train_label)


input_dim = train_features.shape[1]
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_dim = input_dim, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(1, activation='sigmoid',kernel_initializer='he_uniform'))


model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics = 'binary_accuracy')


history = model.fit(train_features, train_label, epochs=50, validation_split=0.5)

metrics = np.mean(history.history['val_binary_accuracy'])
results = model.evaluate(train_features, train_label)
print('\nLoss, Binary_accuracy: \n',(results))


style.use('dark_background')
pd.DataFrame(history.history).plot(figsize=(11, 7),linewidth=4)
plt.title('Binary Cross-entropy',fontsize=14, fontweight='bold')
plt.xlabel('Epochs',fontsize=13)
plt.ylabel('Metrics',fontsize=13)
plt.show() 