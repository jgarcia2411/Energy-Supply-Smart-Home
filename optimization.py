import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from random import sample
import matplotlib.pyplot as plt
import os

# %% _____________________________Data________________________________________
PATH = os.getcwd()
model_data = pd.read_csv(PATH+'/model_data.csv', index_col=0)

# %% ----------------------------------- SETTINGS---------------------------
LR = 1e-3
N_EPOCHS = 20
BATCH_SIZE = 1

SEQ_LEN = 7  # Number of previous time steps to use as inputs in order to predict the output at the next time step
MODEL_NAME = 'OPTIMIZED'
#HIDDEN_SIZE = 32


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

class genetic_algorithm():
    def __init__(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            batch_size,
            seq_len,
            epochs,
            learning_rate,
            population_size=10,
            num_iter=50,
            keep_top_n =0.5,
            mutation_rate=0.5):

        self.population_size = population_size
        self.num_iter = num_iter
        self.keep_top_n = keep_top_n
        self.mutation_rate = mutation_rate
        self._iteration_progress = []
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.BATCH_SIZE = batch_size,
        self.SEQ_LEN = seq_len,
        self.N_EPOCHS = epochs
        self.LR=learning_rate
        self.starting_states = self._generate_starting_state()

    def _generate_starting_state(self):
        final_population = []
        for _ in range(self.population_size):
            hidden_sate_1 = random.randrange(16,500,2)
            dropout_1 = random.uniform(0,0.6)
            hidden_sate_2 = random.randrange(16, 500, 2)
            dropout_2 = random.uniform(0, 0.6)
            final_population.append([hidden_sate_1, dropout_1, hidden_sate_2, dropout_2 ])
        return final_population
    def _evaluate(self, individual):
        model = Sequential([LSTM(units=int(individual[0]), dropout=individual[1], stateful=True,
                                 batch_input_shape=(self.BATCH_SIZE[0], self.SEQ_LEN[0], 1), return_sequences=True),
                            Dense(int(individual[2]), activation='relu'),
                            Dropout(individual[-1],seed=42),
                            Dense(1, activation='relu')])
        model.compile(optimizer=Adam(lr=self.LR), loss='mse', metrics =['mse'])
        model.fit(self.x_train, self.y_train, batch_size=self.BATCH_SIZE[0], epochs=self.N_EPOCHS,
                  validation_data=(self.x_test, self.y_test), verbose=2)
        results = model.evaluate(
            x=self.x_test,
            y=self.y_test,
            batch_size=self.BATCH_SIZE[0],
            verbose=0,
            return_dict=True
        )
        return results

    def evaluate(self, final_population):
        fitness_results =[]
        for individual in final_population:
            metrics = self._evaluate(individual)
            fitness_results.append(metrics['mse'])
        return fitness_results

    def reduce_population(self, population, fitness_results):
        sorted_indices = np.argsort(fitness_results)
        if isinstance(self.keep_top_n, float):
            top_n = max(int(self.keep_top_n*len(fitness_results)),1)
        else:
            top_n = self.keep_top_n
        final_indices = sorted_indices[:top_n]

        reduced_population = np.array(population)[final_indices].tolist()
        return reduced_population

    def crossover(self, reduced_population, r_cross=0.6):
        #parents = sample(reduced_population,2)
        parents = reduced_population[:2]
        c1 = parents[0]
        c2 = parents[-1]

        if np.random.rand() < r_cross:
            #pt = np.random.randint(1,len(parents[0])-1)
            c1[0]= c1[0]
            c1[-2] = c2[-2]
            c1[-1] = c2[-1]
            c2[0] = c2[0]
            c2[-2]= c1[-2]
            c2[-1] = c1[-1]
        offspring_list = [c1, c2]
        return offspring_list
    def _mutate(self, offspring_list):
        for individual in offspring_list:
            for i in range(len(individual)):
                if np.random.rand() < self.mutation_rate:
                    if i < 3:
                        individual[i] = int(random.choice([0.5, 1, 1.5]) * individual[i])
                    else:
                        individual[i] = random.choice([0.5, 1, 1.5]) * individual[i]
        return offspring_list
    def procreate(self, reduced_population):
        reduced_population_size = len(reduced_population)
        new_population = []

        while len(new_population) < self.population_size - reduced_population_size:

            new_population.extend(self.crossover(reduced_population))
        new_population = self._mutate(new_population)
        return new_population

    def optimize(self, verbose=False):
        best_mse = 1e10
        best_sequence = []
        for i in range(self.num_iter):
            if verbose:
                print(f'OPTIMIZING ROUND {i + 1}/{self.num_iter}'.center(100, '='))
            if verbose:
                print('Procreating...')
            starting_idx = -self.population_size
            population = (
                self.starting_states if i == 0
                else self.procreate(reduced_population)
            )
            if verbose:
                print('Evaluating...')
            fitness_results = self.evaluate(population)
            best_idx = np.argsort(fitness_results)[0]

            iter_best = fitness_results[best_idx]
            is_better = iter_best < best_mse  # compare with the last best result and keep it if the mse is lower than the previous.
            best_mse = iter_best if is_better else best_mse  # take the new best mse value
            best_sequence = self.starting_states[best_idx] if is_better else best_sequence  # select the best individual from starting_states

            if verbose:
                print('Selecting best...')
            reduced_population = self.reduce_population(population, fitness_results)

            self._iteration_progress.append(best_mse)

            if verbose:
                print(f'Best mse achieved: {best_mse} - {best_sequence}')
                print('=' * 100)
        print(f'Best Overall:{best_mse} - {best_sequence}')
        units_1 = best_sequence[0]
        dropout_1 = best_sequence[1]
        units_2 = best_sequence[-2]
        dropout_2 = best_sequence[-1]
        return units_1, dropout_1, units_2, dropout_2

GA = genetic_algorithm(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    epochs= N_EPOCHS,
    learning_rate=LR,
    population_size= 32,
    num_iter=4,
    keep_top_n=0.5,
    mutation_rate=0.5

)

units_1, dropout_1, units_2, dropout_2 = GA.optimize(verbose=False)


model = Sequential([LSTM(units = units_1, dropout=dropout_1, stateful=True,
                        batch_input_shape=(BATCH_SIZE, SEQ_LEN,1), return_sequences = True),
                    Dense(units_2, activation='relu'),
                    Dropout(dropout_2, seed=42),
                    Dense(1, activation='relu')])

print(model.summary())
with open(PATH+'/summary_{}.txt'.format('OPTIMIZED MODEL'), 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)
check_point = tf.keras.callbacks.ModelCheckpoint(PATH+'/model_{}.h5'.format(MODEL_NAME),
                                                 monitor='val_loss', save_best_only=True)
model.compile(optimizer=Adam(lr=LR), loss='mse')


# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1000,
          callbacks = [early_stop, check_point],
          validation_data=(x_test, y_test))

# %% -------------------------------------- Predictions & Plots ----------------------------------------------------------
# Optimized model predicitons:
final_model = tf.keras.models.load_model(PATH+'/model_{}.h5'.format(MODEL_NAME))
mse_opt = final_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
pred_test = final_model.predict(x_test, batch_size=BATCH_SIZE)
pred_train = final_model.predict(x_train, batch_size=BATCH_SIZE)
predictions_train, predictions_test = [], []
for i in range(len(pred_train)):
    predictions_train.append(pred_train[i,-1].reshape(-1))

for i in range(len(pred_test)):
    predictions_test.append(pred_test[i,-1].reshape(-1))

pred_train, pred_test = np.array(predictions_train), np.array(predictions_test)

#Base model predictions:
base_model = tf.keras.models.load_model(PATH+'/model_{}.h5'.format('BASE_MODEL'))
mse_b = base_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
pred_test_b = base_model.predict(x_test, batch_size=BATCH_SIZE)
predictions_test_b = []

for i in range(len(pred_test_b)):
    predictions_test_b.append(pred_test_b[i,-1].reshape(-1))

pred_test_b = np.array(predictions_test_b)

time = [i for i in range(len(x))]
plt.style.use('dark_background')
plt.plot(time[150:], x[150:], label='Total data', linewidth=0.5)
plt.plot(time[len(pred_train)+2*SEQ_LEN:], pred_test,
            label='Forecast after optimization, MSE = {}'.format(round(mse_opt),3),
         color='r', linestyle='dashed')

plt.plot(time[len(pred_train)+2*SEQ_LEN:], pred_test_b,
         label='Forecast base model, MSE = {}'.format(round(mse_b),3),
         color='w', linestyle='dashed')

plt.xlabel('Time steps'); plt.ylabel('Power demand [kW]')
plt.title('Forecast Power Demand')
plt.legend()
plt.show()







