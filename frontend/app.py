import streamlit as st
import requests

def run():
    st.title("Portfolio Optimization")
    depo = st.number_input("Deposit")
    
    data = {
        'deposit': depo
        }
    
    if st.button("Optimize"):
        response = requests.post("http://127.0.0.1:8000/optimize", json=data)
        portfolio = response.text
        st.success(f"Portfolio: {portfolio}")
    
if __name__ == '__main__':
    run()