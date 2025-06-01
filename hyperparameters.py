# Гиперпараметры для оптимизации моделей с помощью hyperopt

ridge_params = {
    'alpha': hp.uniform('alpha', 0.01, 2)
}

lasso_params = {
    'alpha': hp.uniform('alpha', 0.01, 2)
}


dtr_params = {
    'max_depth': hp.quniform('max_depth', 5, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 6, 1)
}


br_params = {
    'n_estimators': hp.quniform('n_estimators', 10, 50, 5)
}


rf_params = {
    'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
    'max_depth': hp.quniform('max_depth', 5, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 6, 1)
}


gb_params = {
    "learning_rate": hp.uniform('learning_rate', 0.01,0.5),
    'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 6, 1)   
}


lgbm_params = {
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01,0.5),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), 
    'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
    'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
    'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
    'verbose': -1,
    'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
}


xgb_params = {
    'eta': hp.uniform('eta', 0.1, 0.5),
    'gamma': hp.uniform ('gamma', 0, 9),
    'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
    'max_delta_step': hp.quniform("max_delta_step", 0, 10, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    'alpha' : hp.uniform('alpha', 0, 1),
    'lambda' : hp.uniform('lambda', 0.5, 2)
    }

cb_params = {
    'loss_function': hp.choice('loss_function', ['RMSE']),
    "learning_rate": hp.uniform('learning_rate', 0.1,0.5),
    'depth': hp.quniform("depth", 4, 10, 1),
    "l2_leaf_reg": hp.quniform("l2_leaf_reg", 2, 5, 1),
    'bootstrap_type': hp.choice('bootstrap_type', ["Bayesian", "Bernoulli", "MVS"])
}


# Рассматриваемые модели 

model_params_grid = {
    'Ridge': (Ridge(random_state=42), ridge_params),
    'Lasso': (Lasso(random_state=42), lasso_params),
    "DecisionTreeRegressor": (DecisionTreeRegressor(random_state=42), dtr_params),
    "BaggingRegressor": (BaggingRegressor(random_state=42), br_params),
    'RandomForestRegressor': (RandomForestRegressor(random_state=42), rf_params),
    "GradientBoostingRegressor": (GradientBoostingRegressor(random_state=42), gb_params),
    "LGBMRegressor": (LGBMRegressor(random_state=42), lgbm_params),
    'XGBRegressor': (XGBRegressor(random_state=42), xgb_params),
    'CatBoostRegressor': (CatBoostRegressor(random_state=42), cb_params)
}
