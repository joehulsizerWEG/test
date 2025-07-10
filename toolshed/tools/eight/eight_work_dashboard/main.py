# main.py
import streamlit as st
from toolshed.tools.eight.eight_utils import sidebar_filters

st.set_page_config(
    page_title="8x8 Call Performance",
    layout="wide",
    menu_items={"About": "Built with ❤️ & Streamlit"},
)

sidebar_filters()
st.write(
    f"### 8x8 Call Performance "
    f"({st.session_state['start_date']} → {st.session_state['end_date']})"
)
st.write("Use the navigation menu on the left to explore the data.")