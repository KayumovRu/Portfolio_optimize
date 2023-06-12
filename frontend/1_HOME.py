import streamlit as st
import requests
import pandas as pd

hide_streamlit_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

PATH_DATA = "../datasets/"  # путь к данным

# Загрузка данных из файлов - происходит после авторизации любого уровня по кнопке Login
def getData():

    timeframe = st.session_state["timeframe"]

    # получение общего списка активов
    df_assets = pd.read_csv(PATH_DATA + "assets.csv")

    st.session_state["df_assets"] = df_assets
    tickers_list = df_assets["code"] # на самом деле используем код - измененный обработанный тикер в нижнем регистре

    # получение df по всем активам с сохранением в session_state и с обновлением файлов датасета
    # что позволяет формировать имена df динамически (альтернатива globals()["df_" + ticker])
    def getAssetsData(tickers_list):
        load_bar = st.sidebar.progress(0, text="Data Loading...")
        i = 0
        for ticker in tickers_list:
            df_tmp = pd.read_csv(PATH_DATA + timeframe + "_" + ticker + ".csv")
            st.session_state["df_" + ticker] = df_tmp
            i += 1
            load_bar.progress((i * 1)/len(tickers_list), text="Data Loading... " + ticker.upper())
        load_bar.empty()

    getAssetsData(tickers_list)


# СИСТЕМА АВТОРИЗАЦИИ ===============================================================
# инициализация при первой загрузке
if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["current_username"] = "None"
        st.session_state["timeframe"] = "w1"            # временная инициализация таймфрейма здесь

# если не введен корректный пароль, отобразить форму авторизации
if not st.session_state["password_correct"]:
    st.sidebar.warning("Please log in. You can use the password 'demo'")
    input_username = st.sidebar.text_input("Username", value="demo", key="username")
    input_password = st.sidebar.text_input("Password", value="demo", type="password", key="password")
    button_login = st.sidebar.button("Login")

    if button_login:
        if (
            input_username in st.secrets["passwords"]
            and input_password == st.secrets["passwords"][input_username]
        ):
            st.session_state["password_correct"] = True
            # del ...  # тут надо удалить сам пароль, чтобы не лежал в переменных
            st.session_state["current_username"] = input_username
            getData() # загружаются данные из файлов (один раз при входе за счет cache_data)
            st.experimental_rerun()
        else:
            st.session_state["password_correct"] = False            
            st.sidebar.error("User not known or password incorrect")
else:
    st.sidebar.success("User: " + st.session_state["current_username"])
    button_exit = st.sidebar.button("Exit", use_container_width=True)

    if button_exit:
        st.session_state["password_correct"] = False
        st.cache_data.clear() # не срабатывает, нужно заменить на другую очистку
        st.experimental_rerun()


# ABOUT
st.subheader("About")
st.markdown("Deep Reinforcement Learning for Investment Portfolio Rebalancing")
st.markdown("Master's degree project at HSE  \n[Ruslan Kayumov](https://kayumov.ru/)")

st.subheader("Help")
with st.expander("Tips & Caveats"):
    st.markdown("\
                - The weekly timeframe is used \n\
                - Training takes place with zero commission, commissions are applied already on top of \n\
                - Commissions are not taken into account in the maximum drawdown \n\
                - Sharpe 's calculation formula: (sqrt(freq) * mean(returns - rfr + EPS)) / std(returns - rfr + EPS) \n\
                - The formula for calculating Reward for DDGP based on sharpe + average logarithmic accumulated return\n\
                ")

st.subheader("Techs")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- Python  \n- FastAPI  \n- Streamlit  \n- PyTorch")
with col2:
    st.markdown("- PostgresSQL  \n- bit.io  \n- Plotly  \n- Requests, Threading")

# ================================================================
# при успешной авторизации любого уровня загрузить данные с файлов
# админ на странице DATA сможет обновить данные из облака DB
if st.session_state["password_correct"]:

    # показывать DEBUG и работу с источниками только админу
    if st.session_state["current_username"] == "admin":

        # инфа DEBUG
        st.markdown("""---""")
        st.subheader("Debug info")

        st.write("FastAPI:")
        try:
            response = requests.get("http://127.0.0.1:8000/home")
            st.success(response.text)
        except:
            st.warning("FastAPI error")

        st.write("Assets:")
        try:
            st.dataframe(st.session_state["df_assets"], use_container_width=True)
        except:
            st.warning("Datasets is not available")

        st.write("Session state:")
        st.write(st.session_state)