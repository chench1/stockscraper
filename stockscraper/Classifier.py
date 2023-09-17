import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split


plt.style.use('ggplot')


df = pd.read_csv('all-data.csv', encoding="ISO-8859-1")
df.head()

train = df
train.shape

#Removing punctuations 
data=train.iloc[:,1]
data.replace("[^a-zA-Z]", " ",regex=True, inplace=True)

#implement TF-IDF
tfvector = TfidfVectorizer(ngram_range=(2, 3))

# Fit and transform the 'text' column
tfidf_matrix = tfvector.fit_transform(df['Top1'])

# Use the 'Label' column as target labels
labels = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(X_train, y_train)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
# scikit-learn
# Confusion matrix
# Example of confusion matrix usage to evaluate the quality of the output of a classifier on the iris data set. The diagonal elements represent the number of points for which the predicted label is e...
# Image
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict for the Test Dataset
predictions = randomclassifier.predict(X_test)

# Evaluate the model using y_test (true labels) and predictions
matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(matrix)

score = accuracy_score(y_test, predictions)
print("Accuracy Score:", score)

report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)
