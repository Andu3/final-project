from pmdarima.arima import ndiffs
import statsmodels.tsa.arima.model as sm



def find_differencing (train_data):
    kpss_diffs = ndiffs(train_data, alpha=0.05, test='kpss', max_d=5)
    adf_diffs = ndiffs(train_data, alpha=0.05, test='adf', max_d=5)
    n_diffs = max(adf_diffs, kpss_diffs)
    
    print(f"Estimated differencing term: {n_diffs}")
    return n_diffs


def stepwise_ARIMA(train_data, test_data, optimal_order):
    history = [x for x in train_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = sm.ARIMA(history, order=optimal_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    return model_predictions
    