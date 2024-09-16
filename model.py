import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

df = pd.read_csv("data/diabetes.csv")
# df = data.sample(frac=1).reset_index(drop=True)
# plt.figure(figsize=(8, 8))
# sns.histplot(data["label"])
# plt.title("Distribution")
# plt.show()
# sns.heatmap(data.corr(), annot=True)
# plt.savefig("diabetes.png")

target = "label"
x = df.drop([target], axis=1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(scaler.mean_)
print(scaler.var_)

clf = SVC()

params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf', 'poly']
}
grid = GridSearchCV(SVC(), param_grid=params, scoring="recall", cv=5, verbose=1)
grid.fit(x_train, y_train)

print(grid.best_score_)
print(grid.best_params_)

y_predict = grid.predict(x_test)
print(classification_report(y_test, y_predict))
for i, j in zip(y_test, y_predict):
    print("Actual {} Predicted {}".format(i, j))
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)
