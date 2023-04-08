

import streamlit as st

default_text = "Hello, World!"

if st.button("Populate Text Field"):
    st.text_input("Enter Text", default_text)