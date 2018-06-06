import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score

from ds.DSClassifierGD import DSClassifier

data = pd.read_csv("data/stroke_data.csv")

data = data.drop("pid", axis=1)
data = data.drop("index", axis=1)

data = data.dropna(thresh=10)
data["class"] = data["class"].map({0.: 0, 1.: 1})

data = data.apply(pd.to_numeric, args=("coerce",))
data = data.sample(frac=1).reset_index(drop=True)

cut = int(0.7*len(data))

X_train = data.iloc[:cut, :-1].values
y_train = data.iloc[:cut, -1].values
X_test = data.iloc[cut:, :-1].values
y_test = data.iloc[cut:, -1].values


DSC = DSClassifier(max_iter=200, debug_mode=True, balance_class_data=True, num_workers=4)
losses, epoch, dt = DSC.fit(X_train, y_train, add_single_rules=True, single_rules_breaks=5, add_mult_rules=False,
                            column_names=data.columns[:-1], print_every_epochs=1)
y_pred = DSC.predict(X_test)
y_score = DSC.predict_proba(X_test)

print "\nTraining Time: %.1f" % dt
print "Epochs: %d" % epoch
print "Min Loss: %.3f" % losses[-1]
print "Accuracy: %.1f%%" % (accuracy_score(y_test, y_pred) * 100.)
print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)
print "AUC score: %.3f" % (roc_auc_score(y_test, y_score))

print DSC.model.find_most_important_rules(threshold=0.2, class_names=["No Stroke", "Stroke"])