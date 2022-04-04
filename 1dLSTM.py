import datetime
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# %% _____________________________Data________________________________________
PATH = os.getcwd()
model_data = pd.read_csv(PATH+'/model_data.csv', index_col=0)


# %% ----------------------------------- SETTINGS---------------------------
LR = 1e-3
N_EPOCHS = 1000
BATCH_SIZE = 1
DROPOUT = 0
SEQ_LEN = 7  # Number of previous time steps to use as inputs in order to predict the output at the next time step
HIDDEN_SIZE = 50

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x = model_data[['power_demand']]
x.index = pd.to_datetime(x.index)
x = x.resample('D').sum()
x = np.array(x['power_demand'])
x = x[:-1]

x_train, x_test = x[:int(0.8*len(x))], x[int(0.8*len(x)):]
# format size input data (batch_size, timesteps, input_dimension) (20,200, 1)
x_train_prep = np.empty((len(x_train)-SEQ_LEN, SEQ_LEN, 1))
y_train_prep = np.empty((len(x_train)-SEQ_LEN, SEQ_LEN, 1))
for idx in range(len(x_train)-SEQ_LEN):
    x_train_prep[idx, :, :] = x_train[idx:SEQ_LEN+idx].reshape(-1,1)
    y_train_prep[idx,:] = x_train[idx+1:SEQ_LEN+idx+1].reshape(-1,1)

x_test_prep = np.empty((len(x_test)-SEQ_LEN, SEQ_LEN, 1))
y_test_prep = np.empty((len(x_test)-SEQ_LEN, SEQ_LEN, 1))
for idx in range(len(x_test)-SEQ_LEN):
    x_test_prep[idx,:,:] = x_test[idx:SEQ_LEN+idx].reshape(-1,1)
    y_test_prep[idx, :] = x_test[idx+1:SEQ_LEN+idx+1].reshape(-1,1)

x_train, y_train, x_test, y_test = x_train_prep, y_train_prep, x_test_prep, y_test_prep
del x_train_prep, y_train_prep, x_test_prep, y_test_prep

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([LSTM(units = HIDDEN_SIZE, dropout=DROPOUT, stateful=True,
                        batch_input_shape=(BATCH_SIZE, SEQ_LEN,1), return_sequences = True),
                    Dense(64, activation='relu'),
                    Dense(1, activation='relu')])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)
check_point = tf.keras.callbacks.ModelCheckpoint(PATH+'/model_{}.h5'.format('BASE_MODEL'),
                                                 monitor='val_loss', save_best_only=True)

model.compile(optimizer=Adam(lr=LR), loss='mse')

print(model.summary())

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
          validation_data=(x_test, y_test),
          callbacks=[early_stop, check_point])
mse = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE,verbose=0)

# %% -------------------------------------- Predictions & Plots ----------------------------------------------------------
base_model = tf.keras.models.load_model(PATH+'/model_{}.h5'.format('BASE_MODEL'))
pred_test = base_model.predict(x_test, batch_size=BATCH_SIZE)
pred_train = base_model.predict(x_train, batch_size=BATCH_SIZE)
predictions_train, predictions_test = [], []
for i in range(len(pred_train)):
    predictions_train.append(pred_train[i,-1].reshape(-1))

for i in range(len(pred_test)):
    predictions_test.append(pred_test[i,-1].reshape(-1))

pred_train, pred_test = np.array(predictions_train), np.array(predictions_test)
time = [i for i in range(len(x))]
plt.style.use('dark_background')
plt.plot(time[150:], x[150:], label='Total data', linewidth=0.5)
plt.plot(time[len(pred_train)+2*SEQ_LEN:], pred_test,
            label='Test Prediction - MSE = {}'.format(round(mse,2)), color='r', linestyle='dashed')
plt.xlabel('Time'); plt.ylabel('Power demand')
plt.legend()
plt.show()