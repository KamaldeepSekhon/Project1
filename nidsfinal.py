import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import difflib as dlib
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
from sklearn.metrics import get_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sys



column_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto','state',	'dur','sbytes','dbytes','sttl','dttl',	'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
'trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
'ct_srv_src','ct_srv_dst','ct_dst_ltm',	'ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label']
feature_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto','state',	'dur','sbytes','dbytes','sttl','dttl',	'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
'trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_srv_src','ct_srv_dst','ct_dst_ltm',	'ct_src_ ltm','ct_src_dport_ltm',
'ct_dst_sport_ltm','ct_dst_src_ltm']
label_cols = ['attack_cat', 'Label']

#process command line options with format <test file>, <selected classification method>, <task - label or attack_cat>, <optional load model name>
test_file_name = sys.argv[1]
class_selection = sys.argv[2]
task_selection = sys.argv[3]
load_pickles = False
if len(sys.argv) == 5:
    if sys.argv[4] == 'False':
        load_pickles = False
    elif sys.argv[4] == 'True':
        load_pickles = True
    else:
        print("invalid pickle option, not loading pickle")
#test_file_name = "UNSW-NB15-BALANCED-TRAIN.csv"
#class_selection = "CNB"
#task_selection = "attack_cat"

pdata = pd.read_csv(test_file_name, header=None, names=column_names, skiprows=1)

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

#preprocessing attack_cat for clones and similar words
df["attack_cat"] = df["attack_cat"].str.strip()
attack_cat_uniques = df["attack_cat"].unique()
#string comparison loop for similar words
cat_list = []
cat_list.append(attack_cat_uniques[0])
for index in attack_cat_uniques:
    matches = dlib.get_close_matches(index, cat_list, n=1, cutoff=0.8)
    if not matches:
        cat_list.append(index)
    else:
        df["attack_cat"] = df["attack_cat"].replace(index, matches[0])
    
X_train, X_validation, y_train, y_validation = train_test_split( df[feature_cols], df[label_cols], test_size = 0.20, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)
preprocessing.normalize(X_train)

def DT_classification(X_train,X_validation,task_selection):
    yT_train = y_train[task_selection]
    yT_validation = y_validation[task_selection]
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, yT_train)
    y_pred = dtree.predict(X_validation)
    confusion_matrix(yT_validation, y_pred)
    accuracy_score(yT_validation, y_pred)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(yT_validation, y_pred)*100))
    print(metrics.classification_report(yT_validation, y_pred))
    micro_f1 = f1_score(yT_validation, y_pred, average='micro')
    print("Micro F1 Score:", micro_f1)

def tree_features_selection(X_train,X_validation,y_train,y_validation,task_selection):
    print("\nFeature selection: correlation analysis and Classification: Decision Tree")
    X_train=X_train
    X_validation=X_validation

    if task_selection=="Label":
        correlation = df.corr()["Label"].abs()
        threshold = 0.60
        selected_corr_features = correlation[correlation > threshold].index
        if 'Label' in selected_corr_features:
         selected_corr_features=selected_corr_features.drop('Label')
        if 'attack_cat' in selected_corr_features:
         selected_corr_features=selected_corr_features.drop('attack_cat')
    else:
        df.drop('Label', axis=1, inplace=True)
        df['attack_cat'], index = pd.factorize(df['attack_cat'])
        pd.get_dummies(df['attack_cat'])
        correlation = df.corr()["attack_cat"].abs()
        threshold = 0.15
        selected_corr_features = correlation[correlation > threshold].index
        if 'Label' in selected_corr_features:
         selected_corr_features=selected_corr_features.drop('Label')
        if 'attack_cat' in selected_corr_features:
         selected_corr_features=selected_corr_features.drop('attack_cat')
        df['attack_cat'] = pd.Categorical.from_codes(df['attack_cat'], index)
        # print(X_train.shape)

    X_train=X_train[selected_corr_features]
    if 'Label' in X_train.columns:
        X_train.drop('Label', axis=1, inplace=True)
    if 'attack_cat' in X_train.columns:
        X_train.drop('attack_cat', axis=1, inplace=True)

    X_validation=X_validation[selected_corr_features]
    if 'Label' in X_validation.columns:
        X_validation.drop('Label', axis=1, inplace=True)
    if 'attack_cat' in X_validation.columns:
        X_validation.drop('attack_cat', axis=1, inplace=True)
    print("Number of features selected: "+ str(X_train.shape[1]))
    # call to decision tree classification
    DT_classification(X_train,X_validation,task_selection)

def rfe_featureselection(y_train, task_selection):
    y_train = y_train[task_selection]
    preprocessing.normalize(X_train)
    clf = DecisionTreeClassifier()
    cv = StratifiedKFold(5)

    min_features_to_select = 1
    rfe = RFECV(estimator=clf, step=1, cv=cv, scoring="f1_micro", min_features_to_select=1, n_jobs=12)
    rfe = rfe.fit(X_train, y_train)
    print("Num features: ", rfe.n_features_)
    print("Selected features")
    print(X_train.columns[rfe.support_])
    n_scores = len(rfe.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test f1_micro")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfe.cv_results_["mean_test_score"],
        yerr=rfe.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination")
    plt.show()
    return rfe, X_train.columns[rfe.support_]

def naive_bayes_classifier(x_train, y_train, x_val, y_val, task_selection, selected_features = [], load_pickle = False):
    if load_pickle:
        if task_selection == "Label":
            selected_features = ['srcip', 'sport', 'sbytes', 'dbytes', 'sttl', 'smeansz', 'Stime',
                            'Sintpkt', 'synack', 'ct_srv_dst']
        elif task_selection == "attack_cat":
            selected_features = ['dsport', 'sbytes', 'sttl', 'service']
    elif selected_features.empty:
        selected_features = feature_cols
    else:
        selected_features = selected_features

    X_train = x_train[selected_features]
    X_validation = x_val[selected_features]
    y_train = y_train[task_selection]
    y_validation = y_val[task_selection]
    if task_selection == "Label":
        gnb = GaussianNB()
        
    elif task_selection == "attack_cat":
        gnb = CategoricalNB(force_alpha=True)
        
    gnb = gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_validation)
    confusion_matrix(y_validation, y_pred)
    accuracy_score(y_validation, y_pred)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
    print(metrics.classification_report(y_validation, y_pred))
    micro_f1 = f1_score(y_validation, y_pred, average='micro')
    print("Micro F1 Score:", micro_f1) 
    
print("Feature analysis and classification")

print("RFE Feature Selection")

#Reverse Feature Elimination and Naive Bayes Classification:
if class_selection == "CNB":
    rfe_model, selected_colums = rfe_featureselection(y_train, task_selection)
    naive_bayes_classifier(X_train, y_train, X_validation, y_validation, task_selection, selected_features=selected_colums, load_pickle=load_pickles)


if class_selection == "DT":
    tree_features_selection(X_train,X_validation,y_train,y_validation,task_selection)
