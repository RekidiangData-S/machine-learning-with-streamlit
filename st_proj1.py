import streamlit as st
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

############ set title ############
st.title("Rekidiang DataS")
st.write('=/=/*'*15)
image = Image.open('logot.jpg')
st.image(image, use_column_width=True)
############# set subtitle ############
st.write(""" 
# A Simple Data App with streamlit
""")
st.write(""" 
## Explore different classifiers and dataset
""")
############ set sidebar ############
dataset_name = st.sidebar.selectbox(
    'Select Dataset', ('Breast Cancer', 'Iris', 'Wine'))

classifier_name = st.sidebar.selectbox('Select Classsifier', ('KNN', 'SVM'))

############# Load dataset ############


def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y


x, y = get_dataset(dataset_name)
st.dataframe(x)
st.write('Shape of your dataset is : ', x.shape)
st.write('Unique Target Variable : ', len(np.unique(y)))

############# Plotting ############
fig = plt.figure()
sns.boxplot(data=x, orient='h')
st.pyplot()

plt.hist(x)
st.pyplot()

############# Building Algorithm ############


def add_parameter(name_of_clf):
    params = dict()
    if name_of_clf == 'SVM':
        C = st.sidebar.slider('C', 00.1, 15.0)
        params['C'] = C
    else:
        name_of_clf = 'KNN'
        k = st.sidebar.slider('K', 1, 15)
        params['k'] = k
    return params


params = add_parameter(classifier_name)

############# Accessing Classifiers ############


def get_classifier(name_of_clf, params):
    clf = None
    if name_of_clf == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    return clf


############ Model Training ############
clf = get_classifier(classifier_name, params)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10)

clf.fit(x_train, y_train)  # 80% for training
y_pred = clf.predict(x_test)  # 20% for testing
st.write(y_pred)
############ Model Evaluation ############
accuracy = accuracy_score(y_test, y_pred)
st.write('Classifier Name : ', classifier_name)
st.write('Model Accuracy  : ', accuracy)
