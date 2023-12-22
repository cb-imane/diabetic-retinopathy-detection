import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier



from sklearn.preprocessing import MinMaxScaler

#load data
df = pd.read_pickle("../data/data_retino_preprocessed.pkl")
print(df.columns)
X = df.drop('Class',axis=1).values
y = df['Class'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)

#scale data
scaler = MinMaxScaler()
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
              'max_iter': [1000, 2000, 3000],
              'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],

              }



# Create the GridSearchCV object
grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='recall')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)
# Print the best score
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate the best model on the test set
test_score = grid_search.score(X_test, y_test)
print("Test set score with best parameters: {:.2f}".format(test_score))





