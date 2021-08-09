from pmdarima.arima import ndiffs
import statsmodels.tsa.arima.model as sm
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import time
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)



def find_differencing (train_data):
    
    """
    Function to find the maximum differencing from two unit tests - ADF and KPSS.
    ...

    Attributes
    ----------
    train_data : list
    
    """
    
    
    kpss_diffs = ndiffs(train_data, alpha=0.05, test='kpss', max_d=5)
    adf_diffs = ndiffs(train_data, alpha=0.05, test='adf', max_d=5)
    n_diffs = max(adf_diffs, kpss_diffs)
    
    print(f"Estimated differencing term: {n_diffs}")
    return n_diffs


def stepwise_ARIMA(train_data, test_data, optimal_order):
    
    """
    Function to perform stepwise ARIMA forecasting.
    ...

    Attributes
    ----------
    train_data : list
    
    test_data : list
    
    optimal_order : tuple
        Optimal order of the ARIMA model, determined by the auto_arima function in the pmdarima library.
   
    """
    
    history = [x for x in train_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=optimal_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    return model_predictions
    
    
def ARIMA_pred(train_data, test_data, optimal_order):
    
    """
    Function to perform an ARIMA forecast and plot the results.
    ...

    Attributes
    ----------
    train_data : list
    
    test_data : list
    
    optimal_order : tuple
        Optimal order of the ARIMA model, determined by the auto_arima function in the pmdarima library.
   
    """
    timestamp = time.time()

    
    # Build Model
    model = ARIMA(train_data, order=optimal_order)  
    fitted = model.fit()  
    print(fitted.summary())
        
    # Forecast
    fc, se, conf = fitted.forecast(len(test_data), alpha=0.10)
    
    # Make as pandas series
    fc_series = pd.Series(fc, index=test_data.index)
    lower_series = pd.Series(conf[:, 0], index=test_data.index)
    upper_series = pd.Series(conf[:, 1], index=test_data.index)
    
    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.grid(True)
    plt.plot(train_data, label='training')
    plt.plot(test_data, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=0.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    
    plt.savefig('Reports/{}.png'.format(timestamp))

    
    plt.show()
    
    return fc