from Models.Misc import *
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam



def forecast_HMLP_residuals(input_dim_hybrid, X_train, y_train, X_valid, y_valid, X_test):
    
    """
    Function to build the HMLP and forecast the residuals.
    ...

    Attributes
    ----------
    input_dim_hybrid: int
        Dimension of the input layer for the HMLP.
    X_train: list
    y_train: list
    X_valid: list
    y_valid: list
    X_test: list
    
    """
    #
    ## define model
    model = Sequential()
    model.add(Dense(input_dim_hybrid, activation='tanh', input_dim=input_dim_hybrid))
    model.add(Dense(input_dim_hybrid/2, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss="mean_squared_error",
            metrics=["mse", "mae"])
    
    model.summary()
    
    history_hybrid = model.fit(X_train, y_train,
                               validation_data=(X_valid, y_valid),
                               batch_size=32,
                               epochs=100, verbose=False)
    
    pred_HMLP = model.predict(X_test)
    
    return pred_HMLP, history_hybrid


def zero_centre(prediction):
    
    """
    Function to zero center the hybrid prediction.
    ...

    Attributes
    ----------
    prediction : list
    
    """
    
    if prediction[0] < 0:
        prediction = [abs(prediction[0])+i for i in prediction]
    if prediction[0] > 0:
        prediction = [i-prediction[0] for i in prediction]
    else:
        return prediction
    return prediction

