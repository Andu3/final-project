import numpy as np
from statsmodels.tsa import stattools
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from keras.optimizers import SGD
from sklearn.model_selection import TimeSeriesSplit
from keras.constraints import maxnorm
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten
import tensorflow as tf
from Models.Helper import *
from sklearn.model_selection import RandomizedSearchCV
import keras.backend as K





tf.random.set_seed(123)




# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_regressor_attributes(df, attribute, list_of_prev_t_instants) :
    
    """
    Function to convert a time series into a supervised learning problem.
    ...

    Attributes
    ----------
    df : Dataframe
        Dataframe of stock data.
    attribute : --------------------------------------------------------------------------------------
        
    list_of_prev_t_instants : list
    
    df : list
    
    """
        
    list_of_prev_t_instants.sort()
    start = list_of_prev_t_instants[-1] 
    end = len(df)
    df['datetime'] = df.index
    df.reset_index(drop=True)

    df_copy = df[start:end]
    df_copy.reset_index(inplace=True, drop=True)

    for attribute in attribute :
            foobar = pd.DataFrame()

            for prev_t in list_of_prev_t_instants :
                new_col = pd.DataFrame(df[attribute].iloc[(start - prev_t) : (end - prev_t)])
                new_col.reset_index(drop=True, inplace=True)
                new_col.rename(columns={attribute : '{}_(t-{})'.format(attribute, prev_t)}, inplace=True)
                foobar = pd.concat([foobar, new_col], sort=False, axis=1)

            df_copy = pd.concat([df_copy, foobar], sort=False, axis=1)
            
    df_copy.set_index(['datetime'], drop=True, inplace=True)
    return df_copy



def find_input_dim(data):
    
    acf_djia, confint_djia, qstat_djia, pvalues_djia = stattools.acf(data,
                                                                 adjusted=True,
                                                                 nlags=500,
                                                                 qstat=True,
                                                                 fft=True,
                                                                 alpha = 0.05)
    
    plt.figure(figsize=(7, 5))
    plt.plot(pd.Series(acf_djia), color='r', linewidth=2)
    plt.title('Autocorrelation of Closing Price', weight='bold', fontsize=16)
    plt.xlabel('Lag', weight='bold', fontsize=14)
    plt.ylabel('Value', weight='bold', fontsize=14)
    plt.xticks(weight='bold', fontsize=12, rotation=45)
    plt.yticks(weight='bold', fontsize=12)
    plt.grid(True)
    
    input_dim = sum([1 for x in acf_djia if x>0.9])
    print("Number of values above 0.9 autocorrelation is: ", input_dim)
    return input_dim



def build_baseline_model(num_of_input):
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=num_of_input, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def build_big_model (input_dim):
    
    # define model
    model = Sequential()
    model.add(Dense(input_dim, activation='relu', input_dim=input_dim))
    model.add(Dense(input_dim, activation='relu'))
    model.add(Dense(input_dim, activation='relu'))
    model.add(Dense(input_dim, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    
    model.summary()
    return model

def build_MLP_model (input_dim):
    
    # define model
    model = Sequential()
    model.add(Dense(input_dim, activation='relu', input_dim=input_dim))
    model.add(Dense(input_dim, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    
    model.summary()
    return model




#########################
#new try


def get_mlp_model(input_dim, hidden_layer_one=50, hidden_layer_two=25,
    dropout=0.2, learn_rate=0.01):
    # initialize a sequential model and add layer to flatten the
    # input data
    model = Sequential()
    #model.add(Flatten())
    
    model.add(Dense(hidden_layer_one, activation="relu",
        input_dim=input_dim))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_two, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    # compile the model
    model.compile(
        optimizer=Adam(learning_rate=learn_rate),
        loss="mean_squared_error",
        metrics=["mse", "mae"])
    # return compiled model
    return model
    



    
def baseline_test(n_iter, input_dim, X_train, y_train, X_valid, y_valid, X_test, y_test, scaler):
    
        
    baseline_MSE=[]
    baseline_MAE=[]
    baseline_r2=[]
        
    
    for i in range(n_iter):
        model = build_baseline_model(input_dim)
        # train the network (i.e., no hyperparameter tuning)
        print("[INFO] training model...")
        H = model.fit(x=X_train, y=y_train,
            validation_data=(X_valid, y_valid),
            batch_size=128,
            epochs=100, verbose=0, shuffle=False)
            # mke predictions on the test set and evaluate it
        
        
        baseline_pred = model.predict(X_test)
        baseline_pred_rescaled = scaler.inverse_transform(baseline_pred)
        
        
        plot_comparison_graph(y_test, baseline_pred_rescaled)
        plot_loss(H)
        
        measures = calculate_error_measures(y_test, baseline_pred_rescaled)
        baseline_MSE.append(measures[0])
        baseline_MAE.append(measures[1])
        baseline_r2.append(measures[2])
        
        K.clear_session()
    

    return sum(baseline_MSE)/n_iter, sum(baseline_MAE)/n_iter, sum(baseline_r2)/n_iter




def optimize_parameters(model, grid, X_train, y_train):
    
    
    
    tss = TimeSeriesSplit(n_splits=10)
    
    
    print("[INFO] performing random search...")
    searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, n_iter=10, cv=tss,
        param_distributions=grid, scoring=("r2", 'neg_mean_squared_error','neg_mean_absolute_error'), refit='neg_mean_squared_error')
    searchResults = searcher.fit(X_train, y_train)
    # summarize grid search information
    bestScore = searchResults.best_score_
    bestParams = searchResults.best_params_
    print("[INFO] best score is {:.2f} using {}".format(bestScore,
    bestParams))
    
    return bestParams


#"r2", 'neg_mean_squared_error',

#########################





