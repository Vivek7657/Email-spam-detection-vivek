#1 copy pickle files spam,vectorizer,requirement file
#2 create virtual environment --- python -m venv myenv
#3 install requirements ---- pip install -r requirements.txt
#4 run app ---streamlit run app.py

import streamlit as st 
import pickle

model=pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))

st.title("Email Spam detection Application")
st.write("This is an Ai/ML Application to detect Spam Email")

user_input=st.text_area("enter an email to classify",height=150)
if st.button("classify"):
    if user_input:
        data=[user_input]
        vect=cv.transform(data).toarray()
        pred=model.predict(vect)
        if pred[0]==0:
            st.success("This is not a spam email")
        else:
            st.error("This is spam email")
    else:
        st.error("Please type Email")

#x = st.slider('Select a value')
#st.write(x, 'squared is', x * x)