import streamlit as st

from services.ui_pages import render_chat_page


st.set_page_config(page_title="OfficeMate", layout="wide")
render_chat_page()
