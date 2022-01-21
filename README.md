# fraud_detection_model
Classification problem to detect fraud using credit card data.

We used the dataset extracted from https://www.kaggle.com/mlg-ulb/creditcardfraud

Observation: To use the codes presented in this repository, the user must also download the dataset directly from the website above. Then, create a directory called dataset and put the downloaded file in it, with the name creditcard.csv

The dataset contains only numerical input variables which are the result of a Principal Component Analysis (PCA) transformation. Features V1 to V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

To train the model, we consider two approaches: A neural network implemented using TensorFlow, and a Multi-layer Perceptron classifier implemented using SKLearn. More details regarding the discussions, results and comparison of the performance of the two models will be given in a separate pdf file.
