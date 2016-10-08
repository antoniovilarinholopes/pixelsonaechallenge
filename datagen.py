from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import csv
#ID,SeasonCode,ProductCode,CycleCode,CountryCode,Biz_unit_Code,CatCode,SubCatCode,UnitBaseCode,GenderCode,GroupCode,ColourCode,First_Sale_Date,PVP,Dist_phase,Units_Sold,Stock,Flop


def ffs(x):
    if (int(x) == 0):
        return "0"
    else:
        return "1"


vals = []
with open('1_pixelcamp_train_test.csv', 'rb') as csvfile:
    ctx = csv.reader(csvfile)
    for i in ctx:
        vals += [i]
vals = vals[1:]
vals = [x for x in vals if str(int(x[1])%2) == '1']
X = [[str(int(x[1])%2), x[2], x[8], x[9], ffs(x[12]), float(x[13]), int(x[15]), int(x[16])] for x in vals]
y = [x[-1] for x in vals]

vals=[]
with open('2_pixelcamp_predict.csv', 'rb') as csvfile:
    ctx = csv.reader(csvfile)
    for i in ctx:
        vals += [i]
output = vals
vals = vals[1:]
Xf = [[str(int(x[1])%2), x[2], x[8], x[9], ffs(x[12]), float(x[13]), int(x[15]), int(x[16])] for x in vals]

clf = RandomForestClassifier(n_estimators=1000, criterion="entropy", max_depth=None, min_samples_split=5, random_state=0)
pred = clf.fit(X, y)
y_res = pred.predict(X)
y_res = map(int, y_res)
cf = confusion_matrix(map(int, y), y_res)
auc = roc_auc_score(map(int, y), y_res)
y_final = pred.predict(Xf)
print cf
print auc
print y_final
print list(y_final).count("0")
print list(y_final).count("1")

output[0] = output[0] +  ["Flop"]
for i in range(1,len(output)):
    output[i] = output[i] + [y_final[i-1]]

#TODO: train again for the type B models:




with open('4_pixelcamp_results.csv', 'w') as csvfile:
    ctx = csv.writer(csvfile)
    ctx.writerows(output)
