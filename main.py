import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from plotly import express

def accept_user_data():
    duration = st.text_input("Enter the duration: ")
    start_station = st.text_input("Enter the start area: ")
    end_station = st.text_input("Enter the end station: ")
    return np.array([duration, start_station, end_station]).reshape(1,-1)


# =================== ML Models Below =================

@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, Y_train, Y_test):
    # training the model
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    tree.fit(X_train, Y_train)
    Y_pred = tree.predict(X_test)
    score = accuracy_score(Y_test, Y_pred) * 100
    report = classification_report(Y_test, Y_pred)

    return score, report, tree

@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, Y_train, Y_test):
    # scaling data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # start classifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    score = accuracy_score(Y_test, Y_pred) * 100
    report = classification_report(Y_test, Y_pred)

    return score, report, clf

@st.cache
def Knn_Classifier(X_train, X_test, Y_train, Y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    score = accuracy_score(Y_test, Y_pred) * 100
    report = classification_report(Y_test, Y_pred)

    return score, report, clf    

# =================== ML Models End   =================

@st.cache
def showMap():
    plotData = pd.read_csv('dataset-2010-latlong.csv')
    data = pd.DataFrame()
    data['lat'] = plotData['lat']
    data['lon'] = plotData['lon']
    return data
 
# enable st caching
@st.cache
def loadData():
    return pd.read_csv('dataset-2010.csv')

# basic preprocessing
def preprocessing(data):
    # Assign X (independent features) and y (dependent feature i.e. df['Member type'] column in dataset)
    X = data.iloc[:, [0,3,5]].values
    Y = data.iloc[:, -1].values

    # X and Y are in categories and hence need Encoding
    le = LabelEncoder()
    Y = le.fit_transform(Y.flatten())

    # splitting data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
    return X_train, X_test, Y_train, Y_test, le

def main():
    st.title('Prediction of Trip History Data using various ML Classification Algorithms')
    # load dataset
    data = loadData()
    # preprocessing dataset
    X_train, X_test, Y_train, Y_test, le = preprocessing(data)

    # first streamlit element, checkbox
    if st.checkbox('Show Raw Data'):
        st.subheader('Showing raw data -- >> ')
        st.write(data)  # displays our dataset

    model_choice = st.sidebar.selectbox("Choose the Model", 
        ["NONE", "Decision Tree", "Neural Network", "K-Nearest Neighbours"])

    if model_choice != "NONE":
        if model_choice == "Decision Tree":
            score, report, tree = decisionTree(X_train, X_test, Y_train, Y_test)
        elif model_choice == "Neural Network":
            score, report, clf = neuralNet(X_train, X_test, Y_train, Y_test)
        elif model_choice == "K-Nearest Neighbours":
            score, report, clf = Knn_Classifier(X_train, X_test, Y_train, Y_test)

        st.text("Accuracy of DT Model: ")
        st.write(score, "%")
        st.text("Report of DT Model: ")
        st.write(report)

        try:
            if (st.checkbox("Want to predict your own input?")):
                user_prediction_data = accept_user_data()
                if model_choice == "Decision Tree":
                    pred = tree.predict(user_prediction_data)
                elif model_choice == "Neural Network":
                    scaler = StandardScaler()
                    scaler.fit(X_train)
                    user_prediction_data = scaler.transform(user_prediction_data)
                    pred = clf.predict(user_prediction_data)
                elif model_choice == "K-Nearest Neighbours":
                    pred = clf.predict(user_prediction_data)
                st.write("The Predicted Class is: ", le.inverse_transform(pred)) # inverse tranformation is required to get original dependent value
        except:
            st.write("Failed to predict!")

    plotData = showMap()
    st.subheader("Bike Travel History Data Plot")
    st.map(plotData, zoom=14)

    # showing data histogram
    vis_choice = st.sidebar.selectbox("Choose the visualisation", 
        ['NONE',"Total number of vehicles from various Starting Points", "Total number of vehicles from various End Points",
        "Count of each Member Type"])

    if vis_choice != 'NONE':    
        if vis_choice == "Total number of vehicles from various Starting Points":
            key = 'Start station'
        elif vis_choice == "Total number of vehicles from various End Points":
            key = 'End station'
        elif vis_choice == "Count of each Member Type":
            key = 'Member type'
        fig = express.histogram(data[key], x = key)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()