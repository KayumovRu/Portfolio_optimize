import streamlit
import requests

def run():
    streamlit.title("Portfolio Optimization")
    depo = streamlit.number_input("Deposit")
    
    data = {
        'deposit': depo
        }
    
    if streamlit.button("Optimize"):
        response = requests.post("http://127.0.0.1:8000/optimize", json=data)
        portfolio = response.text
        streamlit.success(f"Portfolio: {portfolio}")
    
if __name__ == '__main__':
    run()