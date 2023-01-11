# Import the necessary libraries
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Used to convert the strings to numeric value
train_data = pd.read_csv('mushrooms.csv')

# The head (first 5 elements) is printed
print(train_data.head())

# Read data from the dataset
pr = preprocessing.LabelEncoder()

# All the dataset is transformed into numeric values

# This data is the target variable
train_data['class'] = pr.fit_transform(train_data['class'])

# The rest of the data is the feature variable
train_data['cap-shape'] = pr.fit_transform(train_data['cap-shape'])
train_data['cap-surface'] = pr.fit_transform(train_data['cap-surface'])
train_data['cap-color'] = pr.fit_transform(train_data['cap-color'])
train_data['bruises'] = pr.fit_transform(train_data['bruises'])
train_data['odor'] = pr.fit_transform(train_data['odor'])
train_data['gill-attachment'] = pr.fit_transform(train_data['gill-attachment'])
train_data['gill-spacing'] = pr.fit_transform(train_data['gill-spacing'])
train_data['gill-size'] = pr.fit_transform(train_data['gill-size'])
train_data['gill-color'] = pr.fit_transform(train_data['gill-color'])
train_data['stalk-shape'] = pr.fit_transform(train_data['stalk-shape'])
train_data['stalk-root'] = pr.fit_transform(train_data['stalk-root'])
train_data['stalk-surface-above-ring'] = pr.fit_transform(train_data['stalk-surface-above-ring'])
train_data['stalk-surface-below-ring'] = pr.fit_transform(train_data['stalk-surface-below-ring'])
train_data['stalk-color-above-ring'] = pr.fit_transform(train_data['stalk-color-above-ring'])
train_data['stalk-color-below-ring'] = pr.fit_transform(train_data['stalk-color-below-ring'])
train_data['veil-type'] = pr.fit_transform(train_data['veil-type'])
train_data['veil-color'] = pr.fit_transform(train_data['veil-color'])
train_data['ring-number'] = pr.fit_transform(train_data['ring-number'])
train_data['ring-type'] = pr.fit_transform(train_data['ring-type'])
train_data['spore-print-color'] = pr.fit_transform(train_data['spore-print-color'])
train_data['population'] = pr.fit_transform(train_data['population'])
train_data['habitat'] = pr.fit_transform(train_data['habitat'])

# The X_train is for training with the features
X_train = train_data[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
                      'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                      'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                      'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']]

# The Y_train is for training with the target variable
Y_train = train_data[['class']]

# The train_test_split method is used to split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)

# The MLPClassifier function is used to classify the dataset
classifier = MLPClassifier(hidden_layer_sizes=(6, 5), random_state=0, verbose=True, learning_rate_init=0.01)

# The model is built
classifier.fit(X_train, Y_train)

# The prediction happens here
Y_pred = classifier.predict(X_test)

# The performance metrics are printed here
print('\n Accuracy: {}%'.format(round(metrics.accuracy_score(Y_test, Y_pred), 2)*100))
print('\n Precision: {}%'.format(round(metrics.precision_score(Y_test, Y_pred,), 2)*100))
print('\n Recall: {}%'.format(round(metrics.recall_score(Y_test, Y_pred), 2)*100))
