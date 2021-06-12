from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_error_measures(X, Y):
        
    MSE = mean_squared_error(X, Y)
    print('Mean Squared Error: {}'.format(MSE))
    
    MAE = mean_absolute_error(X, Y)
    print('Mean Absolute Error: {}'.format(MAE))
    
    error_measures = [MSE, MAE]
    print(error_measures)
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