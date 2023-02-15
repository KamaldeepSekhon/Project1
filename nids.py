import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from pandas import factorize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier


column_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto','state',	'dur','sbytes','dbytes','sttl','dttl',	'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
'trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
'ct_srv_src','ct_srv_dst','ct_dst_ltm',	'ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label']
pdata = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv",header=None, names=column_names, skiprows=1)

df = pd.DataFrame(pdata, columns=column_names)

# delete the columns with null values
del df['ct_flw_http_mthd']
del df['is_ftp_login']
del df['ct_ftp_cmd']

# conversions
df['proto']=pd.factorize(df['proto'])[0]
df['state']=pd.factorize(df['state'])[0]
df['dsport']=pd.factorize(df['dsport'])[0]
df['srcip']=pd.factorize(df['srcip'])[0]
df['sport']=pd.factorize(df['sport'])[0]
df['dstip']=pd.factorize(df['dstip'])[0]
df['dur']=pd.factorize(df['dur'])[0]
df['service']=pd.factorize(df['service'])[0]

df["service"].replace('-','None')
df["attack_cat"].fillna('None', inplace = True)
# df["attack_cat"].value_counts()
 
def label_class():
    y=df["Label"] #Target
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)
    X_train.apply(LabelEncoder().fit_transform)
    X_test.apply(LabelEncoder().fit_transform)
    preprocessing.normalize(X_train)
#   Classification using KNN
    knnClassifier=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    metric_params=None, n_jobs=1, n_neighbors=5, p=2,
    weights='uniform')
    knnClassifier.fit(X_train, y_train)
    y_pred = knnClassifier.predict(X_validation)
    confusion_matrix(y_validation, y_pred)
    accuracy_score(y_validation, y_pred)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
    print(metrics.classification_report(y_validation, y_pred))
    micro_f1 = f1_score(y_validation, y_pred, average='micro')
    print("Micro F1 Score:", micro_f1) 

def attack_class():
    y=df["attack_cat"]
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)
    X_train.apply(LabelEncoder().fit_transform)
    X_test.apply(LabelEncoder().fit_transform)
    preprocessing.normalize(X_train)
#   classification using DecisionTree
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_validation)
    confusion_matrix(y_validation, y_pred)
    accuracy_score(y_validation, y_pred)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
    print(metrics.classification_report(y_validation, y_pred))
    micro_f1 = f1_score(y_validation, y_pred, average='micro')
    print("Micro F1 Score:", micro_f1) 

print("Feature analysis and classification")
X=df.drop(["attack_cat"],axis=1)
y=df["Label"]
corr_data = np.abs(X.corrwith(y))
# Selecting features with the high correlation coefficients
selected = corr_data.nlargest(n=26).index
X= X[selected]
X=X.drop(["Label"],axis=1)
print("\nLabel classification")
label_class()
print("\nattack_cat classification")
attack_class()
    
