import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from matplotlib import pyplot

# Importing data using pandas

credit_card_data = pd.read_csv('../dataset/creditcard.csv')

# Shuffle data and storing original data in _raw dataframe

credit_card_data = credit_card_data.sample(frac=1)
credit_card_data_raw = credit_card_data 

# Transforming output into one-hot-encoding format

credit_card_data = pd.get_dummies(credit_card_data, columns=['Class'])

# Transforming independent variables in numpy array

x=credit_card_data.iloc[:,:-2].values

# Getting y data from pandas dataframe and transforming x and y into numpy arrays with formats of float32

y = credit_card_data.iloc[:,-1].values

# Train-test split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Standardizing independent variables

scaler=StandardScaler()
x_train[:,:]=scaler.fit_transform(x_train[:,:])
x_test[:,:]=scaler.fit_transform(x_test[:,:])

# Since fraudulent data are sparse, we weight fraudulent outcomes so that the model will give preference to these data

count_legit_raw, count_fraud_raw = np.unique(credit_card_data_raw['Class'], return_counts=True)[1]
print("Legitimate and Fraudulent detections in raw dataset, respectively:")
print(count_legit_raw, count_fraud_raw)
fraud_ratio = float(count_fraud_raw / (count_legit_raw + count_fraud_raw))
weighting = 1 / fraud_ratio

# Did not show much improvement in the model performance... maybe no need for this in this case

y_train[:] = y_train[:] * weighting

classifier=MLPClassifier(random_state=0, max_iter=300)

"""

https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

hidden_layer_sizes: tuple, length = n_layers - 2, default=(100,)

activation {identity,  logistic,  tanh,  relu}, default=relu

solver {lbfgs, sgd, adam}, default=adam

alpha float, default=0.0001

learning_rate{constant, invscaling, adaptive}, default=constant

learning_rate_init float, default=0.001

power_t float, default=0.5

max_iter int, default=200

"""

# Training the model with the training dataset

classifier.fit(x_train,y_train)

# Using trained model to test the predictions on the test dataset

y_pred=classifier.predict(x_test)

model_accuracy=accuracy_score(y_test,y_pred)
classifier_probs=classifier.fit(x_train,y_train).predict_proba(x_test)
classifier_roc_auc=roc_auc_score(y_test,classifier_probs[:,1]) # We use column one to choose only the probability values of the positive outcome
model_roc_curve_fpr , model_roc_curve_tpr , _ = roc_curve(y_test,classifier_probs[:,1])

count_legit, count_fraud = np.unique(y_pred, return_counts=True)[1]

print("Accuracy:")
print(model_accuracy)
print("ROC AUC:")
print(classifier_roc_auc)
print("Legitimate and Fraudulent detections predicted, respectively:")
print(count_legit, count_fraud)

pyplot.plot(model_roc_curve_fpr, model_roc_curve_tpr, '--o')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.show()
