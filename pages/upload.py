import streamlit as st

from services.ui_pages import render_upload_page


st.set_page_config(page_title="OfficeMate - 知识上传", layout="wide")
render_upload_page()
