import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



def calculate_error_measures(X, Y):
        
    MSE = mean_squared_error(X, Y)
    print('Mean Squared Error: {}'.format(MSE))
    
    MAE = mean_absolute_error(X, Y)
    print('Mean Absolute Error: {}'.format(MAE))
    
    r2 = r2_score(X, Y)
    print('R-squared score:', round(r2,4))

    
    error_measures = [MSE, MAE, r2]
    return error_measures



#Split dataset

def split_data(data, training_size):
    if training_size < 0 or training_size > 1:
        raise ValueError("Please use a value between 0 and 1 to split the dataset. Recommended values for the size of the training dataset are: 0.50, 0.67, 0.80.")
    train_data, test_data = data[0:int(len(data)*training_size)], data[int(len(data)*training_size):]
    
    train_data = train_data['Close'].values
    test_data = test_data['Close'].values
    
    print("Number of training samples:", len(train_data))
    print("Number of testing samples:", len(test_data))
    return training_size, train_data, test_data


def create_train_valid_test(data, test_set_size, valid_set_size):
    
    df_copy = data.reset_index(drop=True)
    
    df_test = df_copy.iloc[ int(np.floor(len(df_copy)*(1-test_set_size))) : ]
    df_train_plus_valid = df_copy.iloc[ : int(np.floor(len(df_copy)*(1-test_set_size))) ]
    
    df_train = df_train_plus_valid.iloc[ : int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) ]
    df_valid = df_train_plus_valid.iloc[ int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) : ]
    
    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
    X_valid, y_valid = df_valid.iloc[:, 1:], df_valid.iloc[:, 0]
    X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]
    
    
    
    print('Shape of training inputs, training target:', X_train.shape, y_train.shape)
    print('Shape of validation inputs, validation target:', X_valid.shape, y_valid.shape)
    print('Shape of test inputs, test target:', X_test.shape, y_test.shape)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def scale_data(X_train, y_train, X_valid, y_valid, X_test, y_test, scaler):
    
    X_train_scaled = scaler.fit_transform(np.array(X_train))
    X_valid_scaled = scaler.fit_transform(np.array(X_valid))
    X_test_scaled = scaler.fit_transform(np.array(X_test))
    
    y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1,1))
    y_valid_scaled = scaler.fit_transform(np.array(y_valid).reshape(-1,1))
    y_test_scaled = scaler.fit_transform(np.array(y_test).reshape(-1,1))
    
    return X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, X_test_scaled, y_test_scaled


def plot_comparison_graph(test, prediction):
    y_actual = pd.DataFrame(test, columns=['Actual Close Price'])

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
    plt.grid(color = 'y', linewidth='0.5')
    plt.show()
    
    
def plot_loss(history):
    history_dict=history.history
    loss_values = history_dict['loss']
    val_loss_values=history_dict['val_loss']
    plt.plot(loss_values,'bo',label='Training loss')
    plt.plot(val_loss_values,'r',label='Validation loss')
    plt.legend()
        