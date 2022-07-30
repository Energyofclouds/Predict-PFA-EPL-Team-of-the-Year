from catboost import CatBoostRegressor
train_data=[[1,4,5,6],[4,5,6,7],[30,40,50,60]]
eval_data=[[2,4,6,8],[1,4,50,60]]
train_labels=[10,20,30]
model=CatBoostRegressor(iterations=2,learning_rate=1, depth=2)
model.fit(train_data,train_labels)

preds=model.predict(eval_data)
print(preds)

from catboost import CatBoostClassifier
cat_features=[0,1]
train_data=[['a','b',1,4,5,6],['a','b',4,5,6,7],['c','d',30,40,50,60],
           ['f','g',1,2,3,4]]
train_labels=[1,1,-1,-1]
eval_data=[['a','b',2,4,6,8],['a','d',1,4,50,60]]
model=CatBoostClassifier(iterations=3,learning_rate=1,depth=2)
model.fit(train_data,train_labels,cat_features)


predics_class=model.predict(eval_data)
predics_proba=model.predict_proba(eval_data)
preds_raw=model.predict(eval_data,prediction_type='RawFormulaVal')
print(predics_class)
print(predics_proba)
print(preds_raw)



from catboost import CatBoostClassifier,Pool
train_data=Pool(data=[[1,4,5,6],[4,5,6,7],[30,40,50,60]],
               label=[1,1,-1], weight=[0.1,0.2, 0.3])
model=CatBoostClassifier(iterations=10)
model.fit(train_data)

preds_class=model.predict(train_data)
print(preds_class)

train_data=[['summer',1924,44],['summer',1932,37],['winter',1980,37],['summer',2012,204]]
eval_data=[['winter',1996,197],['winter',1968,37],['summer',2002,77],['summer',1948,59]]
cat_features=[0]
train_labels=['FRA','USA','USA','UK']
eval_labels=['USA','FRA','USA','UK']

train_dataset=Pool(data=train_data,label=train_labels,cat_features=cat_features)
eval_dataset=Pool(data=eval_data,label=eval_labels,cat_features=cat_features)

model=CatBoostClassifier(iterations=10,learning_rate=1, depth=2,loss_function="MultiClass")
model.fit(train_dataset)


pred_class=model.predict(eval_dataset)
pred_proba=model.predict_proba(eval_dataset)
print(pred_class)
print(pred_proba)



train1_data=[[0,3],[4,1],[8,1],[9,1]]
train1_labels=[0,0,1,1]
eval1_data=[[2,1],[3,1],[9,0],[5,3]]
eval1_labels=[0,1,1,0]
eval1_dataset=Pool(eval1_data,eval1_labels)

model=CatBoostClassifier(learning_rate=0.03,custom_metric=['Logloss','AUC:hints=skip_train~false'])#to calculate AUC
model.fit(train1_data,train1_labels,eval_set=eval1_dataset,verbose=False)


print(model.get_best_score())




from catboost import Pool,cv
cv_data=[['fran',1924,44],['usa',1932,37],['korea',1928,25],['noeway',1952,30],['japan',1972,35],['maxico',1968,112]]
labels=[1,1,0,0,0,1]
cat_features=[0]
cv_dataset=Pool(data=cv_data,label=labels,cat_features=cat_features)
params={'iterations':100,'depth':2, 'loss_function':'Logloss','verbose':False}
scores=cv(cv_dataset,params,fold_count=2,plot='True')
print(scores)


from catboost import Pool,CatBoostClassifier

train_data=[['fra',1924,44],['usa',1932,37],['usa',1980,37]]
eval_data=[['usa',1996,197],['fra',1968,37],['usa',2002,77]]
cat_features=[0]
train_label=[1,1,0]
eval_label=[0,0,1]

train_dataset=Pool(data=train_data,label=train_label,cat_features=cat_features)
eval_dataset=Pool(data=eval_data,label=eval_label,cat_features=cat_features)

model=CatBoostClassifier(iterations=100,l2_leaf_reg=0.01, one_hot_max_size=3)
model.fit(train_dataset,use_best_model=True,eval_set=eval_dataset)


print('count of trees in model={}'.format(model.tree_count_))


model.plot_tree(tree_idx=53)

model.feature_importances_

model.feature_names_


import pandas as pd
house = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',header=None,sep='\s+')
house.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
house.head()


X=house.iloc[:, :-1]
y=house['MEDV']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


from catboost import CatBoostRegressor

model=CatBoostRegressor()
model.fit(X_train, y_train)


from catboost import Pool,CatBoostRegressor

train_dataset=Pool(data=X_train,label=y_train)
eval_dataset=Pool(data=X_test,label=y_test)

model=CatBoostRegressor(l2_leaf_reg=0.01)
model.fit(train_dataset,use_best_model=True,eval_set=eval_dataset, verbose=False)


model.get_best_score()


model.get_best_iteration()

model.feature_importances_

model.feature_names_



from catboost import cv
params={'iterations':300,'depth':6, 'loss_function':'RMSE','verbose':False, 'early_stopping_rounds':3}
scores=cv(train_dataset,params,fold_count=5,plot='True')


print(scores)


import seaborn as sns
iris = sns.load_dataset('iris')
X = iris.drop('species', axis=1)
y=iris['species'] 

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=1, stratify=y)

from catboost import Pool,CatBoostClassifier

train_dataset=Pool(data=X_train,label=y_train)
eval_dataset=Pool(data=X_test,label=y_test)

model=CatBoostClassifier(l2_leaf_reg=0.01,iterations=10,depth=2,eval_metric='Accuracy')
model.fit(train_dataset,use_best_model=True,eval_set=eval_dataset)



model.get_best_score()

model.get_best_iteration()

model.feature_names_

model.plot_tree(tree_idx=0, pool=train_dataset)

from catboost import cv
params={'iterations':10,'depth':2, 'loss_function':'MultiClass','verbose':False}
scores=cv(train_dataset,params,fold_count=2,plot='True')
print(scores)


import numpy as np
import plotly.express as px
df_tips=px.data.tips()
size_n=np.unique(df_tips['size'])
print(size_n)
#df_tips['size']=df_tips['size'].astype(np.float32)
df_tips.head()


X=df_tips.drop('total_bill', axis=1)
Y=df_tips['total_bill']
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=1)


from catboost import Pool,CatBoostRegressor
train_data=X_train
eval_data=X_test
train_label=y_train
eval_label=y_test
cat_features=[1,2,3,4]
train_dataset=Pool(data=train_data,label=train_label,cat_features=cat_features)
eval_dataset=Pool(data=eval_data,label=eval_label,cat_features=cat_features)

model=CatBoostRegressor(l2_leaf_reg=0.01,iterations=10,depth=3,eval_metric='RMSE',one_hot_max_size=3)
model.fit(train_dataset,use_best_model=True,eval_set=eval_dataset)

model.get_best_score()


import catboost
model.plot_tree(tree_idx=8,pool=train_dataset)

model.feature_importances_

model.feature_names_