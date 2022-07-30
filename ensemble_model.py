import seaborn as sns
iris = sns.load_dataset('iris')
X = iris.drop('species', axis = 1)
y = iris['species']

from sklearn.preprocessing import LabelEncoder
classle = LabelEncoder()
y = classle.fit_transform(iris['species'].values)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_cl = LogisticRegression()
rf_cl = RandomForestClassifier()
svm_cl = SVC()
voting_cl = VotingClassifier(estimators = [('lr', log_cl), ('rf', rf_cl), ('svc', svm_cl)], voting= 'hard')
voting_cl.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for cl in (log_cl, rf_cl, svm_cl, voting_cl) :
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    print(cl.__class__.__name__, accuracy_score(y_test, y_pred))
    
    
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_cl = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, max_samples=100, bootstrap = True)


bag_cl.fit(X_train, y_train)
y_pred = bag_cl.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))



bag_cl2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, bootstrap = True, oob_score = True)
bag_cl2.fit(X_train, y_train)
print(bag_cl2.oob_score_)


from sklearn.ensemble import AdaBoostClassifier
ada_t = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=500, random_state=1)
ada_t.fit(X_train, y_train)
y_train_pred=ada_t.predict(X_train)
y_test_pred=ada_t.predict(X_test)

from sklearn.metrics import accuracy_score
ada_train=accuracy_score(y_train, y_train_pred)
ada_test=accuracy_score(y_test, y_test_pred)

print("Adaboost train/test accuracy %0.3f/%0.3f" %(ada_train, ada_test))



from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
gbcl=GradientBoostingClassifier(n_estimators=100, max_depth=2) # M=100, 나무 깊이=2
gbcl.fit(X_train, y_train)
accuracies=[accuracy_score(y_test,y_pred) for y_pred in gbcl.staged_predict(X_test)]
best_n_estimator=np.argmax(accuracies)
gbcl_best=GradientBoostingClassifier(max_depth=2, n_estimators=best_n_estimator)
gbcl_best.fit(X_train, y_train)
y_train_pred=gbcl_best.predict(X_train)
y_test_pred=gbcl_best.predict(X_test)
print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test, y_test_pred))
print(best_n_estimator)


import pandas as pd
house = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',header=None,sep='\s+')
house.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
house.head()


print(house.info())
house.describe()


from sklearn.tree import DecisionTreeRegressor
X=house.iloc[:, :-1].values
y=house['MEDV'].values

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

from sklearn.ensemble import GradientBoostingRegressor
gbrg=GradientBoostingRegressor(n_estimators=120, max_depth=3)
gbrg.fit(X_train, y_train)
errors=[mean_squared_error(y_test, y_pred) for y_pred in gbrg.staged_predict(X_test)]
bst_n_estimators=np.argmin(errors) # 가장 낮은 MSE를 보여주는 M을 결정함.
gbrg_best=GradientBoostingRegressor(max_depth=3, n_estimators=bst_n_estimators)
gbrg_best.fit(X_train, y_train)
y_train_pred=gbrg_best.predict(X_train)
y_test_pred=gbrg_best.predict(X_test)
from sklearn.metrics import mean_squared_error

print('RMSE train : %0.3f, test: %0.3f' 
      %(np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred))))
print(gbrg_best.feature_importances_)


from sklearn.inspection import plot_partial_dependence
fig1=plot_partial_dependence(gbrg_best, X_train, features=[5])
fig2=plot_partial_dependence(gbrg_best, X_train, features=[12])
fig3=plot_partial_dependence(gbrg_best, X_train, features=[(5,12)])


import pandas as pd
house = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',header=None,sep='\s+')
house.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

from sklearn.tree import DecisionTreeRegressor
X = house[['LSTAT']].values
y = house['MEDV'].values
tree = DecisionTreeRegressor(max_depth=4)
tree.fit(X, y)
sort_idx = X.flatten().argsort()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None

import matplotlib.pyplot as plt
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()



X=house.iloc[:, :-1].values
y=house['MEDV'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=500, criterion='mse', random_state=1)
forest.fit(X_train, y_train)
y_train_pred=forest.predict(X_train)
y_test_pred=forest.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('RMSE train : %0.3f, test: %0.3f' 
      %(np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred))))
print('R**2 train : %0.3f, test: %0.3f' %(r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))



import plotly.graph_objects as go
fig=go.Figure()
fig.add_trace(go.Scatter(x=y_train_pred,y=y_train_pred-y_train,mode='markers',name='Training data'))
fig.add_trace(go.Scatter(x=y_test_pred,y=y_test_pred-y_test,mode='markers',name='Test data'))
fig.update_layout(width=600,height=400, title_text='Residual Plots versus predicted values',title_x=0.5)
fig.update_xaxes(title_text='residuals')
fig.update_yaxes(title_text='predicted')
fig.show()



from sklearn.datasets import load_digits
digits=load_digits()
print(digits.keys())


import matplotlib.pyplot as plt
fig=plt.figure(figsize=(6,6)) # figure size in inches
for i in range(64):
    ax=fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))
    
    
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(digits.data, digits.target, random_state=0)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train, y_train)
y_test_pred=rfc.predict(X_test)

from sklearn import metrics
print(metrics.classification_report(y_test_pred, y_test))



import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

X=house.iloc[:, :-1].values
y=house['MEDV'].values
data_dim=xgb.DMatrix(data=X, label=y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

xg_reg=xgb.XGBRegressor(objective='reg:squarederror',booster='gbtree',colsample_bytree=0.75, 
                        learning_rate=0.1,max_depth=5, alpha=10, n_estimators=30)
xg_reg.fit(X_train, y_train)
pred_train=xg_reg.predict(X_train)
pred_test=xg_reg.predict(X_test)
rmse_train=np.sqrt(mean_squared_error(y_train,pred_train))
rmse_test=np.sqrt(mean_squared_error(y_test,pred_test))
print('RMSE train : %0.3f, test: %0.3f' %(rmse_train, rmse_test))



paras={'objective':'reg:squarederror','colsample_bytree':0.6,'max_depth':5, 'alpha':10}
data_dim1=xgb.DMatrix(data=X_train,label=y_train)
cv_result=xgb.cv(dtrain=data_dim1, params=paras, nfold=5,num_boost_round=60, 
                 early_stopping_rounds=20,metrics='rmse',as_pandas=True, seed=1)
cv_result.head()


print(cv_result['test-rmse-mean'].tail(1))

xg_reg1=xgb.train(params=paras, dtrain=data_dim1, num_boost_round=60)
                
import matplotlib.pyplot as plt
xgb.plot_tree(xg_reg1,num_trees=0)
plt.rcParams['figure.figsize']=[80,20]
plt.show()


xgb.plot_importance(xg_reg1)
plt.rcParams['figure.figsize']=[15,15]
plt.show()


from sklearn.model_selection import GridSearchCV
param={'max_depth':range(3,10,2),'colsample_bytree':[i/100.0 for i in range(75,90,5)]}
xgsearch=GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', max_depth=5,alpha=10),param_grid=param,
                      scoring='neg_mean_squared_error', cv=5)
xgsearch.fit(X_train,y_train)
xgsearch.best_params_, xgsearch.best_score_




from lightgbm import LGBMRegressor
lgbm_reg=LGBMRegressor(booster='gbtree',colsample_bytree=0.75, learning_rate=0.1,max_depth=5, 
                       alpha=10, n_estimators=100)
lgbm_reg.fit(X_train, y_train)
pred_train=lgbm_reg.predict(X_train)
pred_test=lgbm_reg.predict(X_test)
rmse_train=np.sqrt(mean_squared_error(y_train,pred_train))
rmse_test=np.sqrt(mean_squared_error(y_test,pred_test))
print('RMSE train : %0.3f, test: %0.3f' %(rmse_train, rmse_test))