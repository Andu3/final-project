
from sklearn.metrics import mean_absolute_error
from Models.Helper import *



def hybrid_average(ARIMA_forecast, MLP_forecast):

    average = [sum(x)/2 for x in zip(ARIMA_forecast, MLP_forecast)]

    return average


def weighted_average(ARIMA_forecast, MLP_forecast, y_test):
      
    arima_mae = mean_absolute_error(y_test, ARIMA_forecast)
    mlp_mae = mean_absolute_error(y_test, MLP_forecast)
    
    
    total_mae = arima_mae + mlp_mae
        
    mlp_weight = arima_mae/total_mae
    arima_weight = mlp_mae/total_mae   
    
    weighted_arima_pred = [x*arima_weight for x in ARIMA_forecast]
    weighted_mlp_pred = [x*mlp_weight for x in MLP_forecast]
    
    
    weighted_average = [sum(x) for x in zip(weighted_arima_pred, weighted_mlp_pred)]
        
    print("Weight for ARIMA: ", arima_weight)
    print("Weight for MLP: ", mlp_weight)
    
    return weighted_average