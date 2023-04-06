

import streamlit as st

text = st.text_area(':blue[Enter text :]',height=100)

def Upper_Text(text):
    upper_text = text.upper()
    return upper_text

if st.button("Upper Text"):
    text = st.text_area(":blue[Enter text :]",Upper_Text(text),height=2000)