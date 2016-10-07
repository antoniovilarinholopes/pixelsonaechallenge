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

X = [[str(int(x[1])%2), x[2], x[8], x[9], ffs(x[12]), float(x[13]), int(x[15]), int(x[16])] for x in vals]
y = [x[-1] for x in vals]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

vals=[]
with open('train_test_balanced.csv', 'rb') as csvfile:
    ctx = csv.reader(csvfile)
    for i in ctx:
        vals += [i]
vals = vals[1:]
X = [[str(int(x[1])%2), x[2], x[8], x[9], ffs(x[12]), float(x[13]), int(x[15]), int(x[16])] for x in vals]
y = [x[-1] for x in vals]

vals=[]
with open('2_pixelcamp_predict.csv', 'rb') as csvfile:
    ctx = csv.reader(csvfile)
    for i in ctx:
        vals += [i]
vals = vals[1:]
Xf = [[str(int(x[1])%2), x[2], x[8], x[9], ffs(x[12]), float(x[13]), int(x[15]), int(x[16])] for x in vals]





def real_analise(clf, X_train, X_test, y_train, y_test, warm):
    for i in range(10):
        pred = clf.fit(X_train, y_train)
        y_res = pred.predict(X_test)
        clf.warm_start = True
    y_res = map(int, y_res)
    cf = confusion_matrix(map(int, y_test), y_res)
    print "Confusion matrix"
    print cf
    print "Accs: (a,b): ", (float(cf[0][0])/(cf[0][0]+cf[0][1]), float(cf[1][1])/(cf[1][0]+cf[1][1]))
    auc = roc_auc_score(map(int, y_test), y_res)
    print "AUC: ", auc
    print "Score: ", clf.score(X_test, y_test)
    y_final = pred.predict(Xf)
    print "0 freq: ", list(y_final).count('0')/float(list(y_final).count('0') + list(y_final).count('1'))

def analise(clf, warm=True):
    print "Raw:"
    real_analise(clf, X_train, X_test, y_train, y_test, warm)
    clf.warm_start = warm
    print "Normalized:"
    real_analise(clf, X, X_test, y, y_test, warm)


print ""
print "DecisionTreeClassifier"
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=10,
      random_state=0)
analise(clf)


print ""
print "RandomForestClassifier"
clf = RandomForestClassifier(n_estimators=1000, criterion="entropy", max_depth=None,
      min_samples_split=5, random_state=0)
analise(clf, False)

print ""
print "ExtraTreesClassifier"
clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,
      min_samples_split=2, random_state=0)
analise(clf, False)

print ""
print "SVC"
clf = SVC(decision_function_shape="ovo")
analise(clf)

print ""
print "AdaBoostClassifier"
clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
analise(clf)

print ""
print "BaggingClassifier"
clf = BaggingClassifier(n_estimators=1000, random_state=0)
analise(clf, False)