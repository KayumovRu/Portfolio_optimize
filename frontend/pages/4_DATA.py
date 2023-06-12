import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go

st.set_page_config(page_title="DRL Portfolio Simulator • DATA", layout="wide")

hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

PATH_DATA = "../datasets/"  # путь к данным

# ОБНОВЛЕНИЕ И ЗАГРУЗКА ДАННЫХ ================================================
# Загрузка данных с БД - доступно только админу по кнопке Update
def updateFromDB():

    timeframe = st.session_state["timeframe"]

    # ======== DB
    # устанавливаем соединение - убрал кеширование, т.к. по кнопке + есть проблема с сохранением соединения
    def initConnection():
        return psycopg2.connect(**st.secrets["postgres"])

    conn = initConnection()

    # выполнение запроса
    def runQuery(query):
        with conn.cursor() as cur:
            cur.execute(query)
            colnames = [desc[0] for desc in cur.description]
            return colnames, cur.fetchall()

    # получение общего списка активов
    def getAssetsList():
        names, rows = runQuery("SELECT * from assets;")
        df_assets = pd.DataFrame(rows, columns=names)
        df_assets["menu_item"] = df_assets["ticker"] + " (" + df_assets["trivial"] + ")"
        return df_assets
    
    df_assets = getAssetsList()
    df_assets.to_csv(PATH_DATA + "assets.csv", index=False) # сохраняем общий список ассетов с тикерами и кодами
    tickers_list = df_assets["code"] # на самом деле используем код - измененный обработанный тикер в нижнем регистре

    # получение df по всем активам с сохранением в session_state и с обновлением файлов датасета
    # что позволяет формировать имена df динамически (альтернатива globals()["df_" + ticker])
    def updateAssetsData(tickers_list):
        load_bar = st.sidebar.progress(0, text="Data Loading...")
        i = 0
        for ticker in tickers_list:
            str_query = "SELECT * from asset_" + timeframe + "_" + ticker + ";" # получаем запрос вида "SELECT * from asset_w1_aapl;"
            names, rows = runQuery(str_query)
            df_tmp = pd.DataFrame(rows, columns=names)

            df_tmp_invert = df_tmp.iloc[::-1] # инвертированный вариант для модели
            df_tmp_invert['year'] = df_tmp_invert['date'].dt.year
            df_tmp_invert['month'] = df_tmp_invert['date'].dt.month
            df_tmp_invert['day'] = df_tmp_invert['date'].dt.day # ЗАМЕНИТЬ на номер недели .dt.isocalendar().week

            df_tmp_invert['date'] = pd.to_datetime(df_tmp_invert['date']).dt.date # избавляемся от времени

            df_tmp.to_csv(PATH_DATA + timeframe + "_" + ticker + ".csv", index=False) # сохраняем в датасеты сервера = w1_aapl.csv
            df_tmp_invert.to_csv(PATH_DATA + "invert/i_" + timeframe + "_" + ticker + ".csv", index=False) # сохраняем инвертированный
            i += 1
            load_bar.progress((i * 1)/len(tickers_list), text="Data Loading... " + ticker.upper())
        load_bar.empty()
        getData() # после обновления файлов датасетов снова формируем df из них
        st.sidebar.success("Update success!")

    updateAssetsData(tickers_list)

# Загрузка данных из файлов - происходит после обновления данных
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

# ================ не пускаем дальше без авторизации
if ("password_correct" not in st.session_state) or (not st.session_state["password_correct"]):
    st.warning("You need to log in!")
    st.markdown('<a href="/" target="_self">Authorization</a>', unsafe_allow_html=True)
    st.stop()

# ================ авторизация прошла

df_assets = st.session_state["df_assets"]

with st.sidebar:
    # выбор просматриваемого актива
    st.title("Assets:")
    select_asset = st.selectbox('Select assets',
        df_assets["menu_item"],
        label_visibility = "collapsed"
    )

    # выбор стартового и финишного года - пока не привязано к данным
    # st.title("Years:")
    # start_year, end_year = st.select_slider("Data years",
    #     options = range(2009, 2023+1, 1),
    #     value=(2009, 2023),
    #     label_visibility = "collapsed", disabled=True
    # )

    # ? добавить сравнение активов, применить st.expander
    # ? добавить график объемов, может быть выбор того, что именно отображать

asset_code = df_assets.loc[df_assets['menu_item'] == select_asset, 'code'].values[0]

st.subheader(select_asset)
df_current = st.session_state["df_"+asset_code]

df_current['date'] = pd.to_datetime(df_current['date']).dt.date # избавляемся от времени
df_current = df_current.set_index('date')

st.sidebar.markdown("""---""")
# показывать работу с источниками только админу
if st.session_state["current_username"] == "admin":
    # обновление данных из БД админом по кнопке Update
    st.sidebar.info("Admin Panel")
    st.sidebar.write("Update data from the database:")
    if st.sidebar.button("Update", use_container_width=True):
        updateFromDB()

# показывается всем авторизованным после загрузки данных (доделать)
st.sidebar.write("Last update: -")
st.sidebar.write("Start date: -")
st.sidebar.write("End date: -")

# ОСНОВНОЕ ОКНО ====================== свечной график
fig = go.Figure(data=[go.Candlestick(x=df_current.index,
                open=df_current['open'],
                high=df_current['high'],
                low=df_current['low'],
                close=df_current['close'])])

fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df_current, use_container_width=True)