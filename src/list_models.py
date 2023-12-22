import pandas as pd
import os
import pickle
import uuid
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier



from sklearn.preprocessing import MinMaxScaler,StandardScaler

#load data
df = pd.read_pickle("../data/data_retino_preprocessed.pkl")
print(df.columns)
X = df.drop('Class',axis=1).values
y = df['Class'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)

#scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


sgd = SGDClassifier()
sgd.fit(X_train_scaled,y_train)
y_pred_sgd = sgd.predict(X_test_scaled)
acc_sgd  = accuracy_score(y_test,y_pred_sgd)
recall_score_sgd = recall_score(y_test,y_pred_sgd)
f1_score_sgd = f1_score(y_test,y_pred_sgd)
print(f"The recall score of the SGD Model is {round(recall_score_sgd,2)}")
print(f"The F1 score of the SGD Model is {round(f1_score_sgd,2)}")
print(classification_report(y_test,y_pred_sgd))

#Fine tuning

param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1],
              'l1_ratio': [0, 0.1, 0.5, 0.7, 0.9],
              'penalty': ['l2','l1'],
              'max_iter': [10000,15000,20000],
              'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],

              }



# Create the GridSearchCV object
grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='recall')

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)


# Get the best parameters and the corresponding recall score
best_params = grid_search.best_params_
best_recall = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Recall Score: {best_recall}")


# Assuming you have the best-tuned model from the grid search
best_model = grid_search.best_estimator_
#best_model = SGDClassifier(loss='squared_hinge',alpha=0.1,max_iter=5000,penalty='l1')
best_model.fit(X_train_scaled,y_train)
# Predict on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
classification_rep = classification_report(y_test, y_pred)
print("Classification Report on Test Set:\n", classification_rep)

recall_score_sgd_tuned = recall_score(y_test,y_pred)
f1_score_sgd_tuned = f1_score(y_test,y_pred)

print(f"The recall score of the SGD Tuned Model is {round(recall_score_sgd_tuned,2)}")
print(f"The F1 score of the SGD Tuned Model is {round(f1_score_sgd_tuned,2)}")



unique_id = uuid.uuid4().hex
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")

with open(f"trained_models/sgd-{unique_id}.pkl",'wb') as f:
    pickle.dump(best_model,f)

