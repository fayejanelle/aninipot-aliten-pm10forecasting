import streamlit as st

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown('⛔ [ATTENTION] Please login through the main app to access this page.')
else:
    st.title("✨ About Us")
    st.markdown('HELLO WORLD ^_^')