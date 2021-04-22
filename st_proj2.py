############ Import Libraries ############
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# from sklearn import datasets


############ set title ############
st.title("Rekidiang DataS")
st.write('=/=/*'*15)
image = Image.open('logot.jpg')
st.image(image, use_column_width=True)
############# set subtitle ############
st.write("""
# Machine Learning
""")

############ Project Structure ############


def main():
    activities = ['Menu', 'EDA', 'Visualization', 'Model', 'About Us']
    option = st.sidebar.selectbox('Selection Option', activities)
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')
        data = st.file_uploader("Upload Dataset : ", type=[
                                'csv', 'xlsx', 'json'])

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            st.success("Data sucessfully loaded")
            if st.checkbox('Dataset Shape :'):
                st.write(df.shape)
            if st.checkbox('Dataset Columns :'):
                st.write(df.columns)
            # if st.checkbox('Summary Statistics :'):
               # st.write(df.describe().T)
            if st.checkbox('Select Columns :'):
                columns = st.multiselect(
                    'Select your prefered column', df.columns)
                df1 = df[columns]
                st.dataframe(df1)
            if st.checkbox('Summary Statistics :'):
                # st.write(df1.describe().T)
                if len(columns) > 0:
                    st.write(df1.describe().T)
                elif len(columns) <= 0 or columns is not None:
                    st.write(
                        "No Column available, please Select one or more columns")
                if st.checkbox('Check Null Values :'):
                    st.write(pd.DataFrame(df.isnull().sum()).T)
                if st.checkbox('Check Data Type :'):
                    st.write(pd.DataFrame(df.dtypes).T)
                if st.checkbox('Check Correlation of Data Points :'):
                    st.write(pd.DataFrame(df.corr()).T)

    elif option == 'Visualization':
        st.subheader('Data Visualization')
        data = st.file_uploader("Upload Dataset : ", type=[
                                'csv', 'xlsx', 'json'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            st.success("Data sucessfully loaded")

            if st.checkbox('Select Multiple columns to plot :'):
                columns = st.multiselect(
                    'Select your prefered column', df.columns)
                df1 = df[columns]
                st.dataframe(df1)
            if st.checkbox('Display Heatmap Global'):
                st.write(sns.heatmap(df.corr(), vmax=1,
                                     square=True, annot=True, cmap='viridis'))
                st.pyplot()

                if st.checkbox('Display Heatmap Specific'):
                    st.write(sns.heatmap(df1.corr(), vmax=1,
                                         square=True, annot=True, cmap='viridis'))
                    st.pyplot()

            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1, diag_kind='kde'))
                st.pyplot()

                st.pyplot()

            if st.checkbox("Display Pie Chart"):
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox(
                    "Select columns to display : ", all_columns)
                piechart = df[pie_columns].value_counts().plot.pie(
                    autopct="%1.1f%%")
                st.write(piechart)
                st.pyplot()

    elif option == 'Model':
        st.subheader('Model Building')
        data = st.file_uploader("Upload Dataset : ", type=[
                                'csv', 'xlsx', 'json'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            st.success("Data sucessfully loaded")
            if st.checkbox('Select Multiple Columns'):
                new_data = st.multiselect(
                    "Select your preferred columns", df.columns)
                df1 = df[new_data]
                st.dataframe(df1)

                # Divideing my data into X and y variables
                X = df1.iloc[:, 0:-1]
                y = df1.iloc[:, -1]

            seed = st.sidebar.slider("Seed", 1, 200)
            classifier_name = st.sidebar.selectbox(
                "Select your preferred classifier:", ("KNN", "SVM", "LR", "Naive Bay_Bayes", "decision_tree"))

            def add_parameter(name_of_clf):
                params = dict()
                if name_of_clf == 'SVM':
                    C = st.sidebar.slider('C', 0.01, 15.0)
                    params['C'] = C
                if name_of_clf == 'KNN':
                    K = st.sidebar.slider('K', 1, 15)
                    params['K'] = K
                    return params
            params = add_parameter(classifier_name)

            # Defining a function for our classifier
            def get_classifier(name_of_clf, params):
                clf = None
                if name_of_clf == 'SVM':
                    clf = SVC(C=params['C'])
                elif name_of_clf == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=params['K'])
                elif name_of_clf == 'LR':
                    clf = LogisticRegression()
                elif name_of_clf == 'Naive Bay_Bayes':
                    clf = GaussianNB()
                elif name_of_clf == 'decision_tree':
                    clf = DecisionTreeClassifier()
                else:
                    st.warning('Select your choice of algorithm')
                return clf

            clf = get_classifier(classifier_name, params)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            st.write("Prediction : ", y_pred)

            accuracy = accuracy_score(y_test, y_pred)

            st.write('Name of Classifier : ', classifier_name)
            st.write('Accuracy : ', accuracy)

    elif option == 'About Us':
        st.markdown(
            """
            This is our interative web page for ML Projet feel free to use it.
            This Dataset is from UCI Machime Learning repository.
            The analysis in here is to demonstrate how we can present our work 
            to our stakeholders in an interactive by building a web app for 
            our machine learning algorithm using different dataset """)
        st.balloons()


if __name__ == "__main__":
    main()

############# Load dataset ############
