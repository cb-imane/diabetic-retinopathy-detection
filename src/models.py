import pickle
import uuid
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier


df = pd.read_pickle("../data/data_retino_preprocessed.pkl")
print(df.columns)

X = df.drop('Class',axis=1).values
y = df['Class'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)
model = DecisionTreeClassifier() # Per default, criterion="gini"; you could specify criterion="entropy"
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc_score_dt = accuracy_score(y_test,y_pred)
recall_score_dt = recall_score(y_test,y_pred)
f1_score_dt = f1_score(y_test,y_pred)

print(f"The accuracy score of the knn Model is {round(acc_score_dt,2)}")
print(f"The recall score of the Decision Tree Model is {round(recall_score_dt,2)}")
print(f"The F1 score of the Decision Tree Model is {round(f1_score_dt,2)}")
features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)
for i in indices:
    print ("{0} - {1:.3f}".format(features[i], importances[i]))


plt.figure()
plt.title("Feature importance of the tree")
plt.bar(range(num_features), importances[indices], color="g", align="center")
plt.xticks(range(num_features), [features[i] for i in indices], rotation='vertical')
plt.xlim([-1, num_features])
plt.savefig("figures/feature_importance_dt.png")
plt.show()

unique_id = uuid.uuid4().hex
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")

with open(f"trained_models/{unique_id}.pkl",'wb') as f:
    pickle.dump(model,f)



