import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,precision_score,f1_score,classification_report



df = pd.read_pickle("../data/data_retino_preprocessed.pkl")
print(df.columns)

X = df.drop('Class',axis=1).values
y = df['Class'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
def model(input_dim):
    model = models.Sequential()
    model.add(tf.keras.layers.Dense(512, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))  # Dropout layer for regularization
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification, so use 'sigmoid' activation
    return model
# Compile the model
retino_model = model(X_train_scaled.shape[1])
retino_model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary classification loss
              metrics=['accuracy'])

# Display the model summary
print(retino_model.summary())
callbacks = [EarlyStopping(monitor='val_loss', patience=100),
             TensorBoard(log_dir='./Graph', write_graph=True, write_images=True)]

retino_model.fit(x=X_train_scaled, y=y_train, validation_data=(X_test_scaled, y_test)
                 , epochs=500, batch_size=64)
predictions_cnn = retino_model.predict(X_test_scaled)
binary_predictions = (predictions_cnn > 0.5).astype(int)
recall = recall_score(y_test, binary_predictions)
f1 = f1_score(y_test, binary_predictions)

report = classification_report(y_test, binary_predictions)
print(f"The recall score of the CNN Model is {round(recall,2)}")
print("The classification report", classification_report)


accuracy = accuracy_score(y_test, binary_predictions)
precision = precision_score(y_test, binary_predictions)
print(f"The accuracy score of the cnn Model is {round(accuracy,2)}")
print(f"The F1 score of the CNN is {round(f1,2)}")
