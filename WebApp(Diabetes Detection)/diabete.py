import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# create a title and a sub-title
st.write("""
# Diabetes Detection
For diabetic person output is 1 else 0
""")

image=Image.open('/home/uddeshya/jupyter_files/ML_Project/Web App/diabetes.jpg')
st.image(image,caption='Machine Learning Project',use_column_width=True)

df=pd.read_csv('/home/uddeshya/jupyter_files/ML_Project/Web App/diabetes.csv')

st.subheader('Dataset Used: ')

# show data as table
st.dataframe(df)
print('')
st.write(df.describe())

# visualize data
st.subheader('Display Graph')
radio1=st.radio(" ",('Bar Chart','Line Chart'))
if radio1=='Bar Chart':
    #check1=st.checkbox("Show Bar Chart")
    #if check1:
        st.bar_chart(df)
else:
    #check2=st.checkbox("Show Line Chart")
    #if check2:
    st.line_chart(df)

check3=st.checkbox("Show Area Chart")
if check3:
    st.area_chart(df,height=0,width=0,use_container_width=True)

# split data
X=df.iloc[:,0:8].values
Y=df.iloc[:,-1].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

# get faeture input from the user
def get_user_input():
    pregnancies=st.sidebar.slider('pregnancies',0,15,2) #range0-15 and default is 2
    glucose=st.sidebar.slider('glucose',0,200,110)
    blood_pressure=st.sidebar.slider('blood_pressure',0,100,70)
    skin_thickness=st.sidebar.slider('skin_thickness',0,80,30)
    insulin=st.sidebar.slider('insulin',0.0,700.0,220.0)
    BMT=st.sidebar.slider('BMT',0.0,60.0,30.0)
    DFF=st.sidebar.slider('DFF',0.0,2.0,0.08)
    age=st.sidebar.slider('age',10,90,35)

    # store a dictionary into a variable
    user_data={'pregnancies':pregnancies,'glucose':glucose,'blood_pressure':blood_pressure,'skin_thickness':skin_thickness,'insulin':insulin,'BMT':BMT,'DFF':DFF,'age':age}

    # transform data into data frame

    features=pd.DataFrame(user_data,index=[0])
    return features

# store user input into  a variable
user_input=get_user_input()

# set subheader and display user input
st.subheader('User Input: ')

st.write(user_input)

# create and train model
RandomForestClassifier=RandomForestClassifier()

RandomForestClassifier.fit(X_train,Y_train)

# show the model metrics
st.subheader('Model Test Accuracy score: ')

st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

# store the model predictions in a variable
prediction=RandomForestClassifier.predict(user_input)

# set a  subheader and display the classifications
st.subheader('Classification: ')

st.write(prediction) 

if prediction==0:
    st.success('Healthy :)')
if prediction==1:
    st.warning('Diabetic :(')

button1=st.button('View source code')
if button1:
    code='''
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from PIL import Image
    import streamlit as st
    import matplotlib.pyplot as plt
    import time
    import altair as alt

    # create a title and a sub-title
    st.write("""
    # Diabetes Detection
    For diabetic person output is 1 else 0
    """)


    image=Image.open('/home/uddeshya/jupyter_files/ML_Project/Web App/background.jpg')
    st.image(image,caption='Machine Learning Project',use_column_width=True)

    df=pd.read_csv('/home/uddeshya/jupyter_files/ML_Project/Web App/diabetes.csv')

    st.subheader('Data information: ')

    # show data as table
    st.dataframe(df)
    print('')
    st.write(df.describe())

    # visualize data
    st.subheader('Display')
    radio1=st.radio('',('Bar Chart','Line Chart'))
    if radio1=='Bar Chart':
        #check1=st.checkbox("Show Bar Chart")
        #if check1:
            st.bar_chart(df)
    else:
        #check2=st.checkbox("Show Line Chart")
        #if check2:
        st.line_chart(df)

    check3=st.checkbox("Show Area Chart")
    if check3:
        st.area_chart(df,height=300,width=1100,use_container_width=True)

    #c = alt.Chart(df).mark_area().encode(
    #   x='glucose', y=['blood_pressure', 'insulin'])
    #st.altair_chart(c, use_container_width=True)
    #st.write(c)

    # split data
    X=df.iloc[:,0:8].values
    Y=df.iloc[:,-1].values

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

    # get faeture input from the user
    def get_user_input():
        pregnancies=st.sidebar.slider('pregnancies',0,15,2) #range0-15 and default is 2
        glucose=st.sidebar.slider('glucose',0,200,110)
        blood_pressure=st.sidebar.slider('blood_pressure',0,100,70)
        skin_thickness=st.sidebar.slider('skin_thickness',0,80,30)
        insulin=st.sidebar.slider('insulin',0.0,700.0,220.0)
        BMT=st.sidebar.slider('BMT',0.0,60.0,30.0)
        DFF=st.sidebar.slider('DFF',0.0,2.0,0.08)
        age=st.sidebar.slider('age',10,90,35)

        # store a dictionary into a variable
        user_data={'pregnancies':pregnancies,'glucose':glucose,'blood_pressure':blood_pressure,'skin_thickness':skin_thickness,'insulin':insulin,'BMT':BMT,'DFF':DFF,'age':age}

        # transform data into data frame

        features=pd.DataFrame(user_data,index=[0])
        return features

    # store user input into  a variable
    user_input=get_user_input()

    # set subheader and display user input
    st.subheader('User Input: ')

    st.write(user_input)

    # create and train model
    RandomForestClassifier=RandomForestClassifier()

    RandomForestClassifier.fit(X_train,Y_train)

    # show the model metrics
    st.subheader('Model Test Accuracy score: ')

    st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

    # store the model predictions in a variable
    prediction=RandomForestClassifier.predict(user_input)

    # set a  subheader and display the classifications
    st.subheader('Classification: ')

    st.write(prediction) 

    if prediction==0:
        st.success('Healthy :)')
    if prediction==1:
        st.warning('Diabetic :(')
    '''
    st.code(code, language='python')   
