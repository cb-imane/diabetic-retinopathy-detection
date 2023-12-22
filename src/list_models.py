import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report,confusion_matrix

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







