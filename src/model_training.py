import xgboost
import optuna

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

    cv_data = xgboost.DMatrix(X_train, y_train)

    cv_df = xgboost.cv(
            params=xgb_params,
            dtrain=cv_data,
            num_boost_round=1000, 
            nfold = 4,
            early_stopping_rounds=25, 
            verbose_eval=False,
            shuffle=False,
            metrics='rmse',
        ) 
    
    best_result = cv_df.sort_values(f'test-rmse-mean')[f'test-rmse-mean'].iloc[0]

    return best_result