import numpy as np
import pandas as pd
import streamlit as st

st.title("ABC")
st.text("ABC")
st.write("ABC")
st.info("ABC")
st.write("# ABC")

a = np.array([[1,2],[3,4],[5,6]])
st.write(a)
b = pd.DataFrame(a)
st.write(b)
st.table(a)

re = st.checkbox("RED",key="a")
if re:
    st.info("RED")
else:
    st.info("NOTHING")

re2 = st.checkbox("PINK")
if re2:
    st.info("PINK")
else:
    st.info("NOTHING")
