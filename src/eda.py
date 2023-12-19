import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import remove_collinear_features
import pickle



df = pd.read_csv("../data/data_retino.csv")
print(f"We have {df.shape[0]} sample and {df.shape[1]-1} feature")
features = df.columns
print("The features are {features}")
print(df.head())
print(df.info())

print(f"the number of rows for each class {df['Class'].value_counts()}")
plt.figure()
sns.countplot(x='Class',data = df)

plt.title("Distribution of positive & negative diabetic retinopathy")
plt.show()

print(f"The number of null values is {df.isna().sum().sum()}")
print(f"The number of duplicated rows is is {df.duplicated().sum().sum()}")
print(df.describe())
#sns.pairplot(df,hue='Class')


#preprocessing
df = df.drop(['quality','pre_screening','Unnamed: 0'],axis=1)
print(df.shape)
df_cleaned = remove_collinear_features(df,0.9)

print(f"After removing correlated features {df_cleaned.shape}")

df_cleaned.to_pickle("../data/data_retino_preprocessed.pkl")