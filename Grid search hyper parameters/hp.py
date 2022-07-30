import pandas as pd
import numpy as np
from collections import Counter
import random
from sklearn.metrics import roc_auc_score

def confu (conf) :
    accuracy = (conf[0][0] + conf[1][1]) / (conf[0][0] + conf[1][0] + conf[0][1] + conf[1][1])
    precision = (conf[1][1] ) / (conf[1][1] + conf[0][1] )
    recall = (conf[1][1]) / (conf[1][0]  + conf[1][1])
    f1 = 2*(1/(1/recall + 1/precision))
    print(conf)
    print('accuracy = {:.3f}'.format( accuracy))
    print('precision = {:.3f}'.format(precision))
    print('recall = {:.3f}'.format(recall))
    print('f1-score = {:.3f}'.format(f1))
    return 


df_15 = pd.read_csv("C:/Users/빈운기/Downloads/archive/pl_15-16.csv", encoding='cp949')
df_16 = pd.read_csv("C:/Users/빈운기/Downloads/archive/pl_16-17.csv", encoding='cp949')
df_17 = pd.read_csv("C:/Users/빈운기/Downloads/archive/pl_17-18.csv", encoding='cp949')
df_18 = pd.read_csv("C:/Users/빈운기/Downloads/archive/pl_18-19.csv", encoding='cp949')
df_19 = pd.read_csv("C:/Users/빈운기/Downloads/archive/pl_19-20.csv", encoding='cp949')
df_20 = pd.read_csv("C:/Users/빈운기/Downloads/archive/pl_20-21.csv", encoding='cp949')




df_15 = df_15.iloc[:,2:(len(df_15.columns))]
df_16 = df_16.iloc[:,2:(len(df_16.columns))]
df_17 = df_17.iloc[:,2:(len(df_17.columns))]
df_18 = df_18.iloc[:,2:(len(df_18.columns))]
df_19 = df_19.iloc[:,2:(len(df_19.columns))]
df_20 = df_20.iloc[:,2:(len(df_20.columns))]


df_15.fillna(0, inplace = True)
df_16.fillna(0, inplace = True)
df_17.fillna(0, inplace = True)
df_18.fillna(0, inplace = True)
df_19.fillna(0, inplace = True)
df_20.fillna(0, inplace = True)

X_15 = df_15.drop(['MVP'], axis=1)
X_16 = df_16.drop(['MVP'], axis=1)
X_17 = df_17.drop(['MVP'], axis=1)
X_18 = df_18.drop(['MVP'], axis=1)
X_19 = df_19.drop(['MVP'], axis=1)
X_20 = df_20.drop(['MVP'], axis=1)



X_15 = pd.get_dummies(X_15)
X_16 = pd.get_dummies(X_16)
X_17 = pd.get_dummies(X_17)
X_18 = pd.get_dummies(X_18)
X_19 = pd.get_dummies(X_19)
X_20 = pd.get_dummies(X_20)




y_15 = df_15['MVP']
y_16 = df_16['MVP']
y_17 = df_17['MVP']
y_18 = df_18['MVP']
y_19 = df_19['MVP']
y_20 = df_20['MVP']


y = pd.concat([y_15, y_16, y_17, y_18, y_19, y_20 ], ignore_index=True)

from sklearn.preprocessing import MinMaxScaler
MMscaler = MinMaxScaler()
X_15M = MMscaler.fit_transform(X_15)
X_16M = MMscaler.fit_transform(X_16)
X_17M = MMscaler.fit_transform(X_17)
X_18M = MMscaler.fit_transform(X_18)
X_19M = MMscaler.fit_transform(X_19)
X_20M = MMscaler.fit_transform(X_20)

X_15M = pd.DataFrame(X_15M)
X_16M = pd.DataFrame(X_16M)
X_17M = pd.DataFrame(X_17M)
X_18M = pd.DataFrame(X_18M)
X_19M = pd.DataFrame(X_19M)
X_20M = pd.DataFrame(X_20M)




X = pd.concat([X_15M, X_16M, X_17M, X_18M, X_19M, X_20M], ignore_index=True)





#li = list(range(2,(len(df.columns)-1)))

#df[['Goals']].hist()
#df[['Goals conceded']].hist()
#df[['Assists']].hist()
#df[['Passes']].hist()



X.columns = X_15.columns




from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.3, random_state=1, stratify = y)

print(Counter(y))



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.svm import SVC
from sklearn import tree





Log_reg = LogisticRegression(penalty= 'none', random_state=1)
Log_reg.fit(X_train, y_train)
conf_train_Log = confusion_matrix(y_train,Log_reg.predict(X_train))
conf_test_Log = confusion_matrix(y_test,Log_reg.predict(X_test))

accuracy_train = (conf_train_Log[0][0] + conf_train_Log[1][1]) / (conf_train_Log[0][0] + conf_train_Log[1][0] + conf_train_Log[0][1] + conf_train_Log[1][1])
precision_train = (conf_train_Log[1][1] ) / (conf_train_Log[1][1] + conf_train_Log[0][1] )
recall_train = (conf_train_Log[1][1]) / (conf_train_Log[1][0]  + conf_train_Log[1][1])



accuracy_test = (conf_test_Log[0][0] + conf_test_Log[1][1]) / (conf_test_Log[0][0] + conf_test_Log[1][0] + conf_test_Log[0][1] + conf_test_Log[1][1])
precision_test = (conf_test_Log[1][1] ) / (conf_test_Log[1][1] + conf_test_Log[0][1] )
recall_test = (conf_test_Log[1][1]) / (conf_test_Log[1][0]  + conf_test_Log[1][1])




print('')
print('confusion_matrix_Log_train')
print('penalty = none' )

confu(conf_train_Log)
print('ROC AUC = ',round(roc_auc_score(y_train,Log_reg.predict(X_train)),2))
print('')

print('')
print('confusion_matrix_Log_test')
print('penalty = none' )
confu(conf_test_Log)
print('ROC AUC = ',round(roc_auc_score(y_test,Log_reg.predict(X_test)),2))
print('')







knn = KNeighborsClassifier(n_neighbors = 3, p = 2)   # n_neighbors = 3 , 5 , 7  p = 5 ~ 10
knn.fit(X_train, y_train)
conf_train_knn = confusion_matrix(y_train, knn.predict(X_train))
conf_test_knn = confusion_matrix(y_test, knn.predict(X_test))

accuracy_train = (conf_train_knn[0][0] + conf_train_knn[1][1]) / (conf_train_knn[0][0] + conf_train_knn[1][0] + conf_train_knn[0][1] + conf_train_knn[1][1])
precision_train = (conf_train_knn[1][1] ) / (conf_train_knn[1][1] + conf_train_knn[0][1] )
recall_train = (conf_train_knn[1][1]) / (conf_train_knn[1][0]  + conf_train_knn[1][1])


accuracy_test = (conf_test_knn[0][0] + conf_test_knn[1][1]) / (conf_test_knn[0][0] + conf_test_knn[1][0] + conf_test_knn[0][1] + conf_test_knn[1][1])
precision_test = (conf_test_knn[1][1] ) / (conf_test_knn[1][1] + conf_test_knn[0][1] )
recall_test = (conf_test_knn[1][1]) / (conf_test_knn[1][0]  + conf_test_knn[1][1])



print('')
print('confusion_matrix_KNN_train')
print('n_neighbors = 3',  )
print('p = 2',  )
confu(conf_train_knn)
print('ROC AUC = ', round(roc_auc_score(y_train, knn.predict(X_train)),2))
print('')

print('')
print('confusion_matrix_KNN_test')
print('n_neighbors = 3')
print('p = 2')
confu(conf_test_knn)
print('ROC AUC = ', round(roc_auc_score(y_test, knn.predict(X_test)),2))
print('')




svm = SVC(kernel = 'linear', C = 1, random_state=1)  # kernel : linear, rbf, poly, sigmoid  ,  C  = 0.1 ~ 10 
svm.fit(X_train, y_train)
conf_train_svm = confusion_matrix(y_train, svm.predict(X_train))
conf_test_svm = confusion_matrix(y_test, svm.predict(X_test))

accuracy_train = (conf_train_svm[0][0] + conf_train_svm[1][1]) / (conf_train_svm[0][0] + conf_train_svm[1][0] + conf_train_svm[0][1] + conf_train_svm[1][1])
precision_train = (conf_train_svm[1][1] ) / (conf_train_svm[1][1] + conf_train_svm[0][1] )
recall_train = (conf_train_svm[1][1]) / (conf_train_svm[1][0]  + conf_train_svm[1][1])



accuracy_test = (conf_test_svm[0][0] + conf_test_svm[1][1]) / (conf_test_svm[0][0] + conf_test_svm[1][0] + conf_test_svm[0][1] + conf_test_svm[1][1])
precision_test = (conf_test_svm[1][1] ) / (conf_test_svm[1][1] + conf_test_svm[0][1] )
recall_test = (conf_test_svm[1][1]) / (conf_test_svm[1][0]  + conf_test_svm[1][1])




print('')
print('confusion_matrix_SVM_train')
print('kernel = linear' )
print('C = 1' )
confu(conf_train_svm)
print('ROC AUC = ', round(roc_auc_score(y_train, svm.predict(X_train)),2))
print('')

print('')
print('confusion_matrix_SVM_test')
print('kernel = linear' )
print('C = 1')
confu(conf_test_svm)
print('ROC AUC = ', round(roc_auc_score(y_test, svm.predict(X_test)),2))
print('')


dtc = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 1)   # criterion = gini, entropy , max_depth = 3 ~ 10 정도?
dtc.fit(X_train, y_train)
conf_train_dtc = confusion_matrix(y_train, dtc.predict(X_train))
conf_test_dtc = confusion_matrix(y_test, dtc.predict(X_test))


accuracy_train = (conf_train_dtc[0][0] + conf_train_dtc[1][1]) / (conf_train_dtc[0][0] + conf_train_dtc[1][0] + conf_train_dtc[0][1] + conf_train_dtc[1][1])
precision_train = (conf_train_dtc[1][1] ) / (conf_train_dtc[1][1] + conf_train_dtc[0][1] )
recall_train = (conf_train_dtc[1][1]) / (conf_train_dtc[1][0]  + conf_train_dtc[1][1])



accuracy_test = (conf_test_dtc[0][0] + conf_test_dtc[1][1]) / (conf_test_dtc[0][0] + conf_test_dtc[1][0] + conf_test_dtc[0][1] + conf_test_dtc[1][1])
precision_test = (conf_test_dtc[1][1] ) / (conf_test_dtc[1][1] + conf_test_dtc[0][1] )
recall_test = (conf_test_dtc[1][1]) / (conf_test_dtc[1][0]  + conf_test_dtc[1][1])


print('')
print('confusion_matrix_DT_train')
print('criterion = entropy' )
print('max_depth = 3')
confu(conf_train_dtc)
print('ROC AUC = ',round(roc_auc_score(y_train, dtc.predict(X_train)),2))
print('')

print('')
print('confusion_matrix_DT_test')
print('criterion = entropy')
print('max_depth = 3')
confu(conf_test_dtc)
print('ROC AUC = ',round(roc_auc_score(y_test, dtc.predict(X_test)),2))
print('')

