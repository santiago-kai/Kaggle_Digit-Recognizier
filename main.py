# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd

train = pd.read_csv('./input/train.csv')
test  = pd.read_csv('./input/test.csv')

# -

print(train.shape)
train.head()

print(test.shape)
test.head()

train_temp = train.copy()
Xtrain = train_temp.drop(columns=['label'], axis=1)
ytrain = train.label

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute   import SimpleImputer
# my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())

from sklearn.model_selection import cross_val_score
# scores = cross_val_score(my_pipeline, Xtrain, ytrain, cv=5, 
#                          scoring='neg_mean_absolute_error')
# print(scores)

# +
# print('Mean Absolute Error %2f' %(-1 * scores.mean()))

# +
#Evaluation Metrics
from sklearn.metrics import mean_squared_error, make_scorer

def rmse(predict, actual):
    score = mean_squared_error(ytrain, y_pred) ** 0.5
    return score
rmse_score = make_scorer(rmse)

def score(model):
    score = cross_val_score(model, Xtrain, ytrain, cv=5, 
                            scoring=rmse_score).mean()
    return score

scores = {}
# -

from sklearn.metrics import mean_absolute_error, r2_score

# #### Simple Linear Regression

# +
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(Xtrain, ytrain)

#accuracies = cross_val_score(estimator=lr_model,
                         #   X=Xtrain,
                         #   y=Ytrain,
                          #  cv=5,
                          #  verbose=1)
                
y_pred = lr_model.predict(Xtrain)

print('')
print('####### Linear Regression #######')
meanCV = score(lr_model)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(ytrain,y_pred)
mae = mean_absolute_error(ytrain, y_pred)
rmse = mean_squared_error(ytrain, y_pred)**0.5
r2 = r2_score(ytrain, y_pred)
scores.update({'OLS':[meanCV, mse, mae, rmse, r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### Lasso Regression

# +
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso

model_lasso = Lasso(random_state=42, alpha=0.00035)
lr_lasso = make_pipeline(RobustScaler(), model_lasso)
lr_lasso.fit(Xtrain, ytrain)

y_pred = lr_lasso.predict(Xtrain)

print('')
print('####### Lasso Regression #######')
meanCV = score(lr_lasso)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(ytrain,y_pred)
mae = mean_absolute_error(ytrain, y_pred)
rmse = mean_squared_error(ytrain, y_pred)**0.5
r2 = r2_score(ytrain, y_pred)
scores.update({'Lasso':[meanCV, mse, mae, rmse, r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### Ridge Regression

# +
from sklearn.linear_model import Ridge

lr_ridge = make_pipeline(RobustScaler(), 
                         Ridge(random_state=42,alpha=0.002))
lr_ridge.fit(Xtrain,ytrain)

y_pred = lr_ridge.predict(Xtrain)

print('')
print('####### Ridge Regression #######')
meanCV = score(lr_ridge)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(ytrain,y_pred)
mae = mean_absolute_error(ytrain, y_pred)
rmse = mean_squared_error(ytrain, y_pred)**0.5
r2 = r2_score(ytrain, y_pred)
scores.update({'Ridge':[meanCV, mse, mae, rmse, r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### ElasticNet Regression

# +
from sklearn.linear_model import ElasticNet

lr_elasticnet = make_pipeline(RobustScaler(),
                              ElasticNet(alpha=0.02, l1_ratio=0.7,random_state=42))
lr_elasticnet.fit(Xtrain,ytrain)

y_pred = lr_elasticnet.predict(Xtrain)

print('')
print('####### ElasticNet Regression #######')
meanCV = score(lr_elasticnet)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(ytrain,y_pred)
mae = mean_absolute_error(ytrain, y_pred)
rmse = mean_squared_error(ytrain, y_pred)**0.5
r2 = r2_score(ytrain, y_pred)
scores.update({'ElasticNet':[meanCV, mse, mae, rmse, r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### KNN Regression

# +
from sklearn.neighbors import KNeighborsRegressor

knn = make_pipeline(RobustScaler(),KNeighborsRegressor())
knn.fit(Xtrain,ytrain)

y_pred = knn.predict(Xtrain)

print('')
print('####### KNN Regression #######')
meanCV = score(knn)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(ytrain,y_pred)
mae = mean_absolute_error(ytrain, y_pred)
rmse = mean_squared_error(ytrain, y_pred)**0.5
r2 = r2_score(ytrain, y_pred)
scores.update({'KNN':[meanCV, mse, mae, rmse, r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### GradientBoosting Regression

# +
from sklearn.ensemble import GradientBoostingRegressor

model_GBoost = GradientBoostingRegressor(n_estimators=3000, 
                                         learning_rate=0.05,
                                         max_depth=4, 
                                         max_features='sqrt',
                                         min_samples_leaf=15, 
                                         min_samples_split=10,
                                         loss='huber', 
                                         random_state =42)
model_GBoost.fit(Xtrain,ytrain)

y_pred = model_GBoost.predict(Xtrain)

print('')
print('####### GradientBoosting Regression #######')
meanCV = score(model_GBoost)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(ytrain,y_pred)
mae = mean_absolute_error(ytrain, y_pred)
rmse = mean_squared_error(ytrain, y_pred)**0.5
r2 = r2_score(ytrain, y_pred)
scores.update({'GradientBoosting':[meanCV, mse, mae, rmse, r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### RandomForest Regressor

# +
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(Xtrain, ytrain)

y_pred = forest_reg.predict(Xtrain)

print('')
print('####### RandomForest Regression #######')
meanCV = score(forest_reg)
print('Mean CV Score : %.4f' % meanCV)


mse = mean_squared_error(ytrain,y_pred)
mae = mean_absolute_error(ytrain, y_pred)
rmse = mean_squared_error(ytrain, y_pred)**0.5
r2 = r2_score(ytrain, y_pred)
scores.update({'RandomForest':[meanCV, mse, mae, rmse, r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

# #### Grid Search for finding best params for RandomForest

# +
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [70,100], 'max_features': [150]},
    {'bootstrap': [True], 'n_estimators': [70,100], 
     'max_features': [150]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)
grid_search.fit(Xtrain, ytrain)

y_pred = grid_search.predict(Xtrain) #???? missing

print('')
print('####### GridSearch RF Regression #######')
meanCV = score(grid_search)
print('Mean CV Score : %.4f' % meanCV)

mse = mean_squared_error(ytrain,y_pred)
mae = mean_absolute_error(ytrain, y_pred)
rmse = mean_squared_error(ytrain, y_pred)**0.5
r2 = r2_score(ytrain, y_pred)
scores.update({'GridSearchRF':[meanCV, mse, mae, rmse, r2]})

print('')
print('MSE(RSS)    : %0.4f ' % mse)
print('MAE         : %0.4f ' % mae)
print('RMSE        : %0.4f ' % rmse)
print('R2          : %0.4f ' % r2)
# -

grid_search.best_estimator_

# +
scores_list =[]
for k,v in scores.items():
    temp_lst =[]
    temp_lst.append(k)
    temp_lst.extend(v)
    scores_list.append(temp_lst)
    
scores_df = pd.DataFrame(scores_list, 
                         columns=['Model','CV_Mean_Score',
                                  'MSE(RSS)','MAE','RMSE',
                                  'R2Squared'])

scores_df.sort_values(['CV_Mean_Score'])
# -

_ = sns.scatterplot(x='Model',y='CV_Mean_Score',
                   data=scores_df,style='Model')

# +
Lasso_Predictions = lr_lasso.predict(test)

GBoost_Predictions = model_GBoost.predict(test)

KNN_Predictions = knn.predict(test)

GridSearch_Predictions = grid_search.best_estimator_.predict(test)

# +
submission=pd.read_csv('../input/sample_submission.csv')

submission['Label'] = Lasso_Predictions
submission.to_csv('../input/Lasso.csv',index=False)

submission['Label'] = GBoost_Predictions
submission.to_csv('../input/GBoost.csv',index=False)

submission['Label'] = KNN_Predictions
submission.to_csv('../input/KNN.csv',index=False)

submission['Label'] = GridSearch_Predictions
submission.to_csv('../input/GidSearch.csv',index=False)
