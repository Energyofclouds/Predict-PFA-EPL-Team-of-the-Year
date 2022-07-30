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


df_15.fillna(0, inplace = True)
df_16.fillna(0, inplace = True)
df_17.fillna(0, inplace = True)
df_18.fillna(0, inplace = True)
df_19.fillna(0, inplace = True)
df_20.fillna(0, inplace = True)




from sklearn.preprocessing import MinMaxScaler
MMscaler = MinMaxScaler()
X_15 = MMscaler.fit_transform(df_15.iloc[:,3:(len(df_15.columns)-1)])
X_16 = MMscaler.fit_transform(df_16.iloc[:,3:(len(df_15.columns)-1)])
X_17 = MMscaler.fit_transform(df_17.iloc[:,3:(len(df_15.columns)-1)])
X_18 = MMscaler.fit_transform(df_18.iloc[:,3:(len(df_15.columns)-1)])
X_19 = MMscaler.fit_transform(df_19.iloc[:,3:(len(df_15.columns)-1)])
X_20 = MMscaler.fit_transform(df_20.iloc[:,3:(len(df_15.columns)-1)])




X_15 = pd.DataFrame(X_15)
X_16 = pd.DataFrame(X_16)
X_17 = pd.DataFrame(X_17)
X_18 = pd.DataFrame(X_18)
X_19 = pd.DataFrame(X_19)
X_20 = pd.DataFrame(X_20)


X_15.columns = df_15.iloc[:,3:(len(df_15.columns)-1)].columns
X_16.columns = df_16.iloc[:,3:(len(df_16.columns)-1)].columns
X_17.columns = df_17.iloc[:,3:(len(df_17.columns)-1)].columns
X_18.columns = df_18.iloc[:,3:(len(df_18.columns)-1)].columns
X_19.columns = df_19.iloc[:,3:(len(df_19.columns)-1)].columns
X_20.columns = df_20.iloc[:,3:(len(df_20.columns)-1)].columns


df_15_MM = pd.concat([df_15.iloc[:,:3], X_15, df_15.iloc[:,54]], axis = 1)
df_16_MM = pd.concat([df_16.iloc[:,:3], X_16, df_16.iloc[:,54]], axis = 1)
df_17_MM = pd.concat([df_17.iloc[:,:3], X_17, df_17.iloc[:,54]], axis = 1)
df_18_MM = pd.concat([df_18.iloc[:,:3], X_18, df_18.iloc[:,54]], axis = 1)
df_19_MM = pd.concat([df_19.iloc[:,:3], X_19, df_19.iloc[:,54]], axis = 1)
df_20_MM = pd.concat([df_20.iloc[:,:3], X_20, df_20.iloc[:,54]], axis = 1)


df = pd.concat([df_15_MM, df_16_MM, df_17_MM, df_18_MM, df_19_MM, df_20_MM], axis = 0 )

df_FW = df[df['Position'] == 'Forward']
df_MF = df[df['Position'] == 'Midfielder']
df_DF = df[df['Position'] == 'Defender']
df_GK = df[df['Position'] == 'Goalkeeper']


df_FW=df_FW.iloc[:,[1,3,6,9,10,11,12,22,23,24,25,26,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,54]]
df_MF=df_MF.iloc[:,[1,3,6,7,9,10,11,12,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,54]]
df_DF=df_DF.iloc[:,[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,54]]
df_GK=df_GK.iloc[:,[1,3,4,5,20,21,22,23,24,29,30,31,32,34,46,47,48,49,50,51,52,53,54]]



df_FW.reset_index(inplace = True, drop = True)
df_MF.reset_index(inplace = True, drop = True)
df_DF.reset_index(inplace = True, drop = True)
df_GK.reset_index(inplace = True, drop = True)

X = df_FW.iloc[:,1:(len(df_FW.columns)-1)]
y= df_FW.iloc[:,(len(df_FW.columns)-1)]






from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.3, random_state=1, stratify = y)

print(Counter(y))



from sklearn.metrics import f1_score


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.svm import SVC
from sklearn import tree

f1_list_train = []
f1_list_test = []

for i in [100, 10, 1, 0.1, 0.01, 0.001] :

    kernel = 'poly'
    


    svm = SVC(kernel = kernel, C = i, random_state=1)  # kernel : linear, rbf, poly, sigmoid  ,  C  = 0.1 ~ 10 
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
    print('kernel = ', kernel )
    print('C = ', i )
    confu(conf_train_svm)
    f1_list_train.append(round(f1_score(y_train, svm.predict(X_train)),2))
    print('ROC AUC = ', round(roc_auc_score(y_train, svm.predict(X_train)),2))
    print('')
    
    print('')
    print('confusion_matrix_SVM_test')
    print('kernel = ', kernel )
    print('C = ', i)
    confu(conf_test_svm)
    f1_list_test.append(round(f1_score(y_test, svm.predict(X_test)),2))
    print('ROC AUC = ', round(roc_auc_score(y_test, svm.predict(X_test)),2))
    print('')


import matplotlib.pyplot as plt

plt.plot(range(0,len(f1_list_train)), f1_list_train, 'bo-', label = 'f1_score_train')
plt.plot(range(0,len(f1_list_test)), f1_list_test, 'ro-', label = 'f1_score_test')
plt.legend()
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score



log_cl = LogisticRegression(penalty='l2', C = 0.1)
knn_cl = KNeighborsClassifier(n_neighbors = 3, p = 2)
svm_cl = SVC(kernel = 'poly', C = 0.1)
voting_cl = VotingClassifier(estimators = [('lr', log_cl), ('knn', knn_cl), ('svc', svm_cl)], voting = 'hard')
voting_cl.fit(X_train, y_train)


for cl in ([log_cl, knn_cl, svm_cl, voting_cl]):
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    print(cl.__class__.__name__, 'acc : ',round(accuracy_score(y_test, y_pred),3))
    print(cl.__class__.__name__, 'pre : ',round(precision_score(y_test, y_pred, pos_label=1 ),3))
    print(cl.__class__.__name__, 'rec : ',round(recall_score(y_test, y_pred, pos_label=1),3))
    print(cl.__class__.__name__, 'f1  : ',round(f1_score(y_test, y_pred, pos_label=1),3))
    print(cl.__class__.__name__, 'AUC : ',round(roc_auc_score(y_test, y_pred),3))
    print('\n')




log_cl = LogisticRegression(penalty='l2', C = 0.1)
knn_cl = KNeighborsClassifier(n_neighbors = 3, p = 2)
svm_cl = SVC(kernel = 'poly', C = 0.1, probability = True)
voting_cl = VotingClassifier(estimators = [('lr', log_cl), ('knn', knn_cl), ('svc', svm_cl)], voting = 'soft')
voting_cl.fit(X_train, y_train)



for cl in ([log_cl, knn_cl, svm_cl, voting_cl]):
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    print(cl.__class__.__name__, 'acc : ',round(accuracy_score(y_test, y_pred),3))
    print(cl.__class__.__name__, 'pre : ',round(precision_score(y_test, y_pred, pos_label=1 ),3))
    print(cl.__class__.__name__, 'rec : ',round(recall_score(y_test, y_pred, pos_label=1),3))
    print(cl.__class__.__name__, 'f1  : ',round(f1_score(y_test, y_pred, pos_label=1),3))
    print(cl.__class__.__name__, 'AUC : ',round(roc_auc_score(y_test, y_pred),3))
    print('\n')






rf = RandomForestClassifier(max_depth = 3, n_estimators = 30, random_state = 6)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('acc : ',round(accuracy_score(y_test, y_pred),3))
print('pre : ',round(precision_score(y_test, y_pred, pos_label=1 ),3))
print('rec : ',round(recall_score(y_test, y_pred, pos_label=1),3))
print('f1  : ',round(f1_score(y_test, y_pred, pos_label=1),3))
print('AUC : ',round(roc_auc_score(y_test, y_pred),3))
print('\n')

