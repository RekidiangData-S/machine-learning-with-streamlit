############ Import Libraries ############
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import streamlit as st
from PIL import Image

# IMPORT THE NECESSARY LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
import pickle

import matplotlib.ticker as mtick
plt.style.use('fivethirtyeight')

warnings.filterwarnings('ignore')


############ set title ############
st.title("Rekidiang DataS")
st.write("***************")
image = Image.open('logot.jpg')
st.image(image, use_column_width=True)
st.header("  =====Zomato Restaurant Rating Prediction===")
st.text('======'*15)
############# set subtitle ############


def main():
    activities = ['Menu', 'EDA', 'Visualization', 'Prediction', 'About Us']
    option = st.sidebar.selectbox('Selection Option', activities)
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')
        if st.checkbox('Ten First Row of Dataset :'):
            df = pd.read_csv("zomato.csv")
            st.dataframe(df.head(10))
        if st.checkbox('Feature Description :'):
            image = Image.open('features_desc.jpg')
            st.image(image, use_column_width=True)
        # Deleting Unnnecessary Columns
        df = df.drop(['url', 'phone'], axis=1)
        if st.checkbox('Check Null Values :'):
            st.write(pd.DataFrame(df.isnull().sum()).T)
            miss_val_points = (df.shape[0] - df.dropna().shape[0])/df.shape[0]
            st.write("Missing Value points : ",
                     round(miss_val_points*100, 2), "%")
        if st.checkbox(' Most famous restaurants chains in Bangaluru :'):
            plt.figure(figsize=(17, 10))
            chains = df['name'].value_counts()[:20]
            sns.barplot(x=chains, y=chains.index, palette='deep')
            plt.xlabel("Number of outlets")
            st.pyplot()

    if option == 'Prediction':
        st.subheader('Prediction')
        model = pickle.load(open('models/etr93.9.sav', "rb"))
        data = []
        online_order = st.selectbox("online_order ?", (1, 0))

        book_table = st.selectbox("book_table ?", (1, 0))

        votes = st.number_input('Votes :')
        location = st.number_input('Location :')
        rest_type = st.number_input('Restaurant Type :')
        cuisines = st.number_input('Cuisines :')
        cost = st.number_input('Cost :')
        menu_item = st.number_input('Menu Item :')

        data = [online_order, book_table, votes, location,
                rest_type, cuisines, cost, menu_item]
        arr = pd.DataFrame([data])
        if st.checkbox('Prediction :'):
            pred_result = model.predict(arr.values)
            result = round(pred_result[0], 1)
            st.write(result)
        st.write("THANK YOU")


if __name__ == "__main__":
    main()

