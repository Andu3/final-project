import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time



def calculate_error_measures(X, Y):
    
    """
    Function to calculate the mean squared error, mean absolute error and the coefficient of determination (r2) between two lists.
    ...

    Attributes
    ----------
    X : list
    
    Y : list
    
    """
    
        
    MSE = mean_squared_error(X, Y)
    print('Mean Squared Error: ', MSE)
    
    MAE = mean_absolute_error(X, Y)
    print('Mean Absolute Error: ', MAE)
        
    error_measures = [MSE, MAE]
    return error_measures



def create_train_valid_test(data, test_set_size, valid_set_size):
    
    """
    Function to split a dataset into training, testing and validation sets.
    ...

    Attributes
    ----------
    data : list
        
    test_set_size : int
        Int <= 1 for size of test set.
        
    valid_set_size : int
        Int <= 1 for size of validation set.   
    """
    
    
    df_copy = data.reset_index(drop=True)
    
    df_test = df_copy.iloc[ int((len(df_copy)*(1-test_set_size))) : ]
    df_train_plus_valid = df_copy.iloc[ : int((len(df_copy)*(1-test_set_size))) ]
    
    df_train = df_train_plus_valid.iloc[ : int((len(df_train_plus_valid)*(1-valid_set_size))) ]
    df_valid = df_train_plus_valid.iloc[ int((len(df_train_plus_valid)*(1-valid_set_size))) : ]
    
    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
    X_valid, y_valid = df_valid.iloc[:, 1:], df_valid.iloc[:, 0]
    X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]
    
    
    
    print('Shape of training inputs, training target:', X_train.shape, y_train.shape)
    print('Shape of validation inputs, validation target:', X_valid.shape, y_valid.shape)
    print('Shape of test inputs, test target:', X_test.shape, y_test.shape)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def scale_data(X_train, y_train, X_valid, y_valid, X_test, y_test, scaler):
    
    """
    Function to transform features by scaling them into a range a given range. A range of 0-1 is used for this project.
    ...

    Attributes
    ----------
    X_train : list
    
    y_train : list
    
    X_valid : list
    
    y_valid : list
        
    X_test : list
    
    y_test : list
    
    scaler : MinMaxScaler
    """
    
    
    X_train_scaled = scaler.fit_transform(np.array(X_train))
    X_valid_scaled = scaler.fit_transform(np.array(X_valid))
    X_test_scaled = scaler.fit_transform(np.array(X_test))
    
    y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1,1))
    y_valid_scaled = scaler.fit_transform(np.array(y_valid).reshape(-1,1))
    y_test_scaled = scaler.fit_transform(np.array(y_test).reshape(-1,1))
    
    return X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, X_test_scaled, y_test_scaled


def plot_comparison_graph(actual, prediction):
    
    """
    Function to plot a graph comparing the actual values to predicted values.
    ...

    Attributes
    ----------
    actual : list
    
    prediction : list   
    """
    timestamp = time.time()
    
    y_actual = pd.DataFrame(actual, columns=['Actual Close Price'])

    y_hat = pd.DataFrame(prediction, columns=['Predicted Close Price'])
    
    
    plt.figure(figsize=(11, 6))
    plt.plot(y_actual, linestyle='solid', color='r')
    plt.plot(y_hat, linestyle='dashed', color='b')
    
    plt.legend(['Actual','Predicted'], loc='best', prop={'size': 14})
    plt.title('Closing Prices', weight='bold', fontsize=16)
    plt.ylabel('USD ($)', weight='bold', fontsize=14)
    plt.xlabel('Test Set Day no.', weight='bold', fontsize=14)
    plt.xticks(weight='bold', fontsize=12, rotation=45)
    plt.yticks(weight='bold', fontsize=12)
    plt.grid(True)
    
    plt.savefig('Reports/{}.png'.format(timestamp))
    
    plt.show()
    
    
def plot_loss(history):
    
    """
    Function to plot the training and validation loss of the MLP model.
    ...

    Attributes
    ----------
    history :  History object  
    """
    
    plt.figure(figsize=(11, 6))
    history_dict=history.history
    loss_values = history_dict['loss']
    val_loss_values=history_dict['val_loss']
    plt.plot(loss_values,'bo',label='Training loss')
    plt.plot(val_loss_values,'r',label='Validation loss')
    plt.legend()
    
    
    
def load_data(ETF):
    
    """
    Function to load the ETF data from a file, remove NaN values and set the Date column as index.
    ...

    Attributes
    ----------
    ETF : filepath
    """
    
    
    data = pd.read_csv(ETF, usecols=[0,4], parse_dates=[0], header=0)

    data.dropna(subset = ['Close', 'Date'], inplace=True)
    
    data_close = pd.DataFrame(data['Close'])
    data_close.index = pd.to_datetime(data['Date'])
    
    return data_close



def equal_size(list1, list2):
    
    """
    Function to equalise the size of two lists.
    ...

    Attributes
    ----------
    list1: list
    list2: list
    
    """
    
    if len(list1) > len(list2):
        s = len(list1)-len(list2)
        list1 = list1[:-s]
        return list1, list2
    if len(list2) > len(list1):
        s = len(list2)-len(list1)
        list2 = list2[:-s]
        return list1, list2
    else:
        return list1, list2
        