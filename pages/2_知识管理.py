import streamlit as st

from services.ui_pages import render_management_page


st.set_page_config(page_title="OfficeMate - 知识管理", layout="wide")
render_management_page()
