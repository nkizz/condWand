from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

data = {}
NUM_CLASSES = 13
x_train, y_train, x_test, y_test,x_val,y_val = [], [], [], [],[],[]
NUM_SEC = 180
split = NUM_SEC * 0.8
split2 = split + NUM_SEC * 0.1
accel = None
gyro = None

for i in range(60,125,5):
    data[i] = np.loadtxt("training_data/"str(i)+"bpm3.csv",skiprows=1,delimiter=',')[0:119*60*3,1:]
    if accel is None:
        accel = data[i][:, :3]
        gyro = data[i][:, 3:]
    else:
        accel = np.append(accel, data[i][:, :3], axis=0)
        gyro = np.append(gyro, data[i][:, 3:], axis=0)
    print(accel.shape)
    print(gyro.shape)
scaler = StandardScaler()
scaler2 = StandardScaler()
accelScaler = scaler.fit(accel)
gyroScaler = scaler2.fit(gyro)
joblib.dump(accelScaler, "accel.scaler")
joblib.dump(gyroScaler, "gyro.scaler")
for i in range(60, 125, 5):
    data[i][:, :3] = accelScaler.transform(data[i][:, :3])
    data[i][:, 3:] = gyroScaler.transform(data[i][:, 3:])

for i, array in data.items():
    for j in range(0, NUM_SEC-1):
        if j < split:
            x_train.append(array[j*119:(j+2)*119])
            y_train.append(i)
        elif j >= split and j < split2:
            x_test.append(array[j*119:(j+2)*119])
            y_test.append(i)
        else:
            x_val.append(array[j*119:(j+2)*119])
            y_val.append(i)
#print(x_test)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_val = np.array(x_val)
y_val = np.array(y_val)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train.reshape(-1, 1))

y_train = enc.transform(y_train.reshape(-1, 1))
y_test = enc.transform(y_test.reshape(-1, 1))
y_val = enc.transform(y_val.reshape(-1, 1))
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='lstm6.h5',
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.005, patience=12, verbose=1, mode='auto')
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,
            input_shape=[x_train.shape[1],
                         x_train.shape[2]], 
                         return_sequences=True
        )
    )
)
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128
        )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
#model.add(keras.layers.Dense(1))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
#model = load_model('lstm.h5')
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(x_val, y_val),
    shuffle=True, callbacks=[model_checkpoint_callback, early_stop]
)
model = load_model('lstm6.h5')
model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

def plot_cm(y_true, y_pred, class_names):
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = plt.subplots(figsize=(18, 16))
  ax = sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap=sns.diverging_palette(220, 20, n=7),
      ax=ax
  )

  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  ax.set_xticklabels(class_names)
  ax.set_yticklabels(class_names)
  b, t = plt.ylim()  # discover the values for bottom and top
  b += 0.5  # Add 0.5 to the bottom
  t -= 0.5  # Subtract 0.5 from the top
  plt.ylim(b, t)  # update the ylim(bottom, top) values
  plt.show(block=True)  # ta-da!
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Training Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show(block=True)
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show(block=True)


plot_cm(
    enc.inverse_transform(y_test),
    enc.inverse_transform(y_pred),
    enc.categories_[0]
)
