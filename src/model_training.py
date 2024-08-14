import pandas as pd
import xgboost
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score

def cross_validation(X_train, y_train, hyper_params):
    
    cv_data = xgboost.DMatrix(X_train, y_train)

    cv_df = xgboost.cv(
            params=hyper_params,
            dtrain=cv_data,
            num_boost_round=1000, 
            nfold = 4,
            early_stopping_rounds=25, 
            verbose_eval=False,
            shuffle=False,
            metrics='rmse',
        ) 
    
    return cv_df


# DEFINE OPTUNA OBJECTIVE FOR HYPER-PARAMETER OPTIMIZATION
def optuna_objective_xgboost(trial, optuna_bounds, X_train, y_train):

    # xgb parameters
    xgb_params = {}
    xgb_params['learning_rate'] = trial.suggest_float(
        'learning_rate', optuna_bounds['learning_rate'][0], optuna_bounds['learning_rate'][1]
        )
    xgb_params['max_depth'] = trial.suggest_int(
        'max_depth', optuna_bounds['max_depth'][0], optuna_bounds['max_depth'][1]
        )
    xgb_params['min_child_weight'] = trial.suggest_float(
        'min_child_weight', optuna_bounds['min_child_weight'][0], optuna_bounds['min_child_weight'][1]
        )
    xgb_params['gamma'] = trial.suggest_float(
        'gamma', optuna_bounds['gamma'][0], optuna_bounds['gamma'][1]
        )
    xgb_params['subsample'] = trial.suggest_float(
        'subsample', optuna_bounds['subsample'][0], optuna_bounds['subsample'][1]
        )
    xgb_params['colsample_bytree'] = trial.suggest_float(
        'colsample_bytree', optuna_bounds['colsample_bytree'][0], optuna_bounds['colsample_bytree'][1]
        )

    xgb_params['objective'] = 'reg:squarederror'
    
    cv_df = cross_validation(X_train, y_train, xgb_params)

    best_result = cv_df.sort_values(f'test-rmse-mean')[f'test-rmse-mean'].iloc[0]

    return best_result


def plot_optuna_study(study):
    '''PLOT INFORMATION FROM THE HYPER-PARAMETER TRAINING RUN'''

    print('OPTIMIZATION TRIALS DATA')
    test_scores = [study.get_trials()[i].values[0] for i in range(0,len(study.get_trials()))]
    params = [pd.DataFrame(study.get_trials()[i].params, index=[i]) for i in range(0,len(study.get_trials()))]
    trial_data = pd.concat(params)
    trial_data[f'test rmse'] = test_scores

    # plot trial data
    trial_data.plot(subplots=True)
    plt.show()

def train_xgboost(xgb_params, X_train, y_train, X_test=None, y_test=None):
    '''Train xgboost model with given hyper-parameters.'''

    # initialize model
    model = xgboost.XGBRegressor(
        objective='reg:squarederror', 
        eval_metric='rmse',  
        **xgb_params)
    
    # fit model to the whole training data
    model.fit(X_train, y_train)

    # predictions
    y_predicted = model.predict(X_train)

    # measure performance
    rmse_train = root_mean_squared_error(y_train, y_predicted)
    print(f'RMSE (train): {rmse_train}')
    r2_train = r2_score(y_train, y_predicted)
    print(f'R^2 (train): {r2_train}')

    if X_test is not None and y_test is not None:

        # predictions
        y_predicted = model.predict(X_test)        

        # measure performance
        rmse_test = root_mean_squared_error(y_test, y_predicted)
        print(f'RMSE (test): {rmse_test}')
        r2_test = r2_score(y_test, y_predicted)
        print(f'R^2 (test): {r2_test}')

        return model, {'train_rmse': rmse_train, 'train_r2': r2_train, 'test_rmse':rmse_test, 'test_r2':r2_test}
    
    else:
        return model, {'train_rmse': rmse_train, 'train_r2': r2_train}