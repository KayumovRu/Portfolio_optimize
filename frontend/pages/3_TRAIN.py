import streamlit as st
import requests
import torch
import pandas as pd
import json
import threading
import time
import glob
import shutil

import plotly.express as px

def run():

    if "settings_train" not in st.session_state:
        st.session_state.settings_train = {}
    if "settings_train_fix" not in st.session_state:
        st.session_state.settings_train_fix = {}
    if "reward_history" not in st.session_state:
        st.session_state.reward_history = {}
    if "save_model" not in st.session_state:
        st.session_state.save_model = 'Save model' # 

    button_train = st.sidebar.button("TRAIN", use_container_width=True)

    button_save = st.sidebar.button(st.session_state.save_model, use_container_width=True, disabled=(st.session_state.save_model!='Save model'))
    
    cont_progress = st.container()

    # функция отслеживания прогресса обучения - ПОПРАВИТЬ, чтобы или пропадал график после обучения ИЛИ дорисовывался до последнего значения
    def trainProgress():
        train_bar = cont_progress.progress(0, text="Train in progress... ")
        inter_reward_history = [0]
        with st.empty():
            while True:
                with open(progress_file) as f:
                    progress = json.load(f)
                if progress['in_process'] != True:
                    st.session_state.reward_history = progress['reward_history']
                    st.session_state.train_steps = progress["mean_steps_in_ep"]
                    train_bar.progress(1/1, text="") # бар на 100%
                    time.sleep(0.5) # даем время анимации бара добежать
                    train_bar.empty()
                    break
                train_bar.progress(progress['episode']/st.session_state.settings_train['episodes_num'],
                            text="Train in progress... episode {}, step {}".format(progress['episode'], progress['total_steps']))

                # промежуточный график истории reward (отрисовываем, если есть изменения в reward_history)
                if inter_reward_history != progress['reward_history']:
                    inter_reward_history = progress['reward_history'] # впереди ноль, чтобы динамику было видно с первого эпизода
                    st.caption("Reward History by Episodes")
                    fig = px.line(inter_reward_history,
                                x=[i+1 for i in range(len(inter_reward_history))],
                                y=inter_reward_history,
                        )
                    fig.update_layout(xaxis_rangeslider_visible = False,
                                    xaxis_title = 'Episodes',
                                    yaxis_title = 'Reward',
                                    plot_bgcolor = '#0E1117',
                                    xaxis_gridcolor = '#262730',
                                    yaxis_gridcolor = '#262730',
                                    height=300,
                                    showlegend=False,
                                    xaxis=dict(type='linear', tick0=1, dtick=1),
                        )
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                # /

                time.sleep(2)

    try:
        trainProgress() # сразу вызываем функцию проверки прогресса на случай, если обучение в процессе
    except: # на случай, если обучения еще не запускались
        pass 

    # костыль от перезагрузки при смене опции
    # https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11
    if "button_train" not in st.session_state: # начальная инициализация
        st.session_state.button_train = False
    if button_train: # здесь только работа с бэкендом и прогрессом, отрисовка главного экрана передается в PRESS TRAIN - BY CONDITION 
        st.session_state.settings_train_fix = st.session_state.settings_train # фиксируем настройки для предупреждения о смене настроек
        st.session_state.button_train = True
        # ОТПРАВКА ЗАПРОСА
        # (добавить защиту от отправки запросов без выбора хотя бы одной модели - возможно при нажатии кнопки) st.warning('Select at least one model and press APPLY!')
        # response_train = requests.post("http://127.0.0.1:8000/train", json=st.session_state.settings_train)

        def make_request(settings_train):
            requests.post("http://127.0.0.1:8000/train", json=settings_train)

        # запуск в другом потоке для отслеживания прогресса
        thread = threading.Thread(target=make_request, args=(st.session_state.settings_train,))
        thread.start()

        # перезаписываем файл прогресса на новый перед вызовом функции прогресса
        progress_json = {"in_process": True, "episode": 0, "mean_steps_in_ep": 1, "total_steps": 0, "reward_history": [0]}
        with open('../models/progress/DDPG_' + st.session_state['current_username'] + '.json', 'w') as f:
            json.dump(progress_json, f)

        st.session_state.save_model = 'Save model' # новую модель еще не сохраняли

        trainProgress()

        thread.join() # ждем окончания потока

        response_train_test = requests.post("http://127.0.0.1:8000/inference", json=st.session_state.settings_train_test)
        res_tmp = response_train_test.json() # результаты по последнему эпизоду
        st.session_state.train_test_result = res_tmp[list(res_tmp.keys())[0]] # сразу берем по первой модели, так как модель только одна

        # st.experimental_rerun() # при включении все равно интерфейс не перезагружается, при этом настройки обученияы сбрасываются

    if "button_save" not in st.session_state: # начальная инициализация
        st.session_state.button_save = False
    if button_save:
        st.session_state.button_save = True

    # контейнер отображения предупреждения об изменениях
    cont_warning_change = st.container()

    # контейнер гиперпараметров
    # cont_params = st.container().expander('Hyperparams')

    # SIDEBAR ===============================================================
    with st.sidebar:

        try: # проверка на доступность CUDA DEVICE
            cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
            st.success("CUDA: " + cuda_name)
        except:
            cuda_name = "NO"
            st.warning("CUDA: " + cuda_name)
        
        st.text('DATA: ' + date_scale.iloc[0].strftime('%Y/%m/%d') + ' - ' + date_scale.iloc[-1].strftime('%Y/%m/%d'))

        st.divider()

        select_train_ratio = st.slider("Train-test split:",
            5, 95, 80, step = 5,
            format="%.0f%%",
        ) / 100

        select_episodes_num = st.slider("Number of Episodes:", 1, 200, 3, step = 1)

        select_depo = st.number_input("Deposit", value = 10000, step=1000)

        # выбор признаков для обучения
        # st.title("Features:")
        select_assets = st.multiselect('Features ',
            features,
            features[:4],
            disabled=True
        )

        # выбор активов для обучения модели (не то же самое, что на странице INFERENCE)
        select_assets = st.multiselect('Assets ',
            df_assets["ticker"], df_assets["ticker"],
            disabled=True,
        )

    st.session_state.settings_train = {
        'username': st.session_state['current_username'], # передаем, чтобы сохранять от каждого юзера уникальные модели со своей нумерацией
        'deposit': select_depo,
        'train_ratio': select_train_ratio,
        'episodes_num': select_episodes_num,
    }

    # настройки для сразу же инференса на тесте
    st.session_state.settings_train_test = {
        'deposit': select_depo,
        'train_ratio': select_train_ratio, 
        'basemodels': [],
        'model_ddpg': 'DDPG_' + st.session_state['current_username'],
        'comission': 0.0,
    }

    # предупреждение о смене настроек до нажатия TRAIN
    if st.session_state.settings_train_fix != st.session_state.settings_train:
        cont_warning_change.warning("Settings have been changed, results are not relevant. Press TRAIN!")

    # Нажатие TRAIN - обработка по состоянию
    if st.session_state.button_train:     

        train_test_result = st.session_state.train_test_result

        # визуализуем блок с метриками теста по параметрам последнего эпизода
        col_title, col_return, col_mdd, col_sharpe = st.columns(4)
        col_return.caption("Return")
        col_mdd.caption("Max Drowdown")
        col_sharpe.caption("Sharpe Ratio")

        col_title, col_return, col_mdd, col_sharpe = st.columns(4)
        loc_total_return    = round(train_test_result['portfolio_value'] * 100, 1)
        loc_mdd             = round(train_test_result['mdd'] * 100, 1)
        loc_sharpe          = round(train_test_result['sharpe_ratio'], 2)
        col_title.markdown("<h3 style='text-align: center; color: #FAFAFA99;'>DDPG</h3>", unsafe_allow_html=True)
        col_return.metric("Return",  str(loc_total_return)+'%', str(round(loc_total_return - 100, 1))+'%', label_visibility = "collapsed")
        col_mdd.metric("Max Drowdown",   "-"+str(loc_mdd)+'%', "", label_visibility = "collapsed")
        col_sharpe.metric("Sharpe Ratio",   loc_sharpe, "", label_visibility = "collapsed")


        df_result_depo = pd.DataFrame()
        df_result_depo['ddpg'] = train_test_result['depo_results']
        # шкалу даты подстраиваем под число данных
        df_result_depo["date"]= date_scale.tail(df_result_depo.shape[0]).reset_index(drop=True)

        tab_test, tab_reward = st.tabs(["Test", "Reward History"])

        # финальный график тестирования обученной модели
        with tab_test:
            st.caption("Final Test")
            fig = px.line(df_result_depo, x="date", y=df_result_depo.columns,
                            hover_data={"date": "|%B %d, %Y"}
                            )
            fig.update_layout(xaxis_rangeslider_visible = False,
                            xaxis_title = None,
                            yaxis_title = 'Portfolio Return, $',
                            plot_bgcolor = '#0E1117',
                            xaxis_gridcolor = '#262730',
                            yaxis_gridcolor = '#262730',
                            showlegend=False,
                            height=500,
                            )
            fig.add_hline(y=10000, line_dash="dot", row=1, col=1, line_color="#FFFFFF", line_width=1)
            st.plotly_chart(fig, use_container_width=True, theme=None)

        # история reward по эпизодам
        with tab_reward:
            st.caption("Reward History by Episodes; *steps in each episode: " + str(st.session_state.train_steps))
            fig = px.line(st.session_state.reward_history,
                        x=[i+1 for i in range(len(st.session_state.reward_history))],
                        y=st.session_state.reward_history,
                )
            fig.update_layout(xaxis_rangeslider_visible = False,
                            xaxis_title = 'Episodes',
                            yaxis_title = 'Reward',
                            plot_bgcolor = '#0E1117',
                            xaxis_gridcolor = '#262730',
                            yaxis_gridcolor = '#262730',
                            height=500,
                            showlegend=False,
                            xaxis=dict(type='linear'),
                )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        st.caption("History Allocation (*averaged values for every 10 weeks)")
        df_result_actions = pd.DataFrame(train_test_result["action_results"],
                                            columns=[0, *list(train_test_result["assets_list"].values())]).iloc[:, 1:] # исключаем первую, там кэш
        # оставляем только те активы, у которых макс значение хоть раз было сколько-нибудь значимо
        mask = df_result_actions.max() < 1e-3
        df_result_actions = df_result_actions.drop(columns = mask[mask].index.tolist())
        # ресемплируем по 10 недель
        df_result_actions_mean = df_result_actions.groupby(df_result_actions.index // 10).mean().round(4)
        # добавляем дату для графика
        df_result_actions_mean['date'] = df_result_depo['date'].groupby(df_result_depo["date"].index // 10).max().reset_index(drop=True)
        df_result_actions_mean.set_index('date', inplace=True)
        df_result_actions_mean.index = pd.to_datetime(df_result_actions_mean.index).strftime('%Y-%m-%d') # костыль - перевод в datetime и обратно позволяет корректно отображать бары на графике
        # st.write(df_result_actions)
        st.bar_chart(df_result_actions_mean, height=500)

        # st.write(st.session_state.train_test_result)

    else:
        cont_warning_change.warning("Set the parameters and press TRAIN!")

    # PRESS SAVE - BY CONDITION
    if st.session_state.button_save:
        try:
            user_models = glob.glob('../models/DDPG_' + st.session_state['current_username'] + '_*')
            user_models = [model.replace('../models/DDPG_' + st.session_state['current_username'] + '_', '') for model in user_models] # удаляем все лишнее из моделей,кроме числе на конце
            user_models = list(map(int, user_models))
            if len(user_models) > 0:
                new_name = 'DDPG_' + st.session_state['current_username'] + '_' + str(max(user_models) + 1)
            else: # если еще нет сохраненных моеделей, то сохраняем с номером 1
                new_name = 'DDPG_' + st.session_state['current_username'] + '_1'
            shutil.copy2('../models/DDPG_' + st.session_state['current_username'], '../models/' + new_name)
            st.session_state.save_model = new_name
            st.session_state.button_save = False
            st.experimental_rerun() # что-то все еще не так с кнопкой, должна становится не активной, но этого не происходит
            # button_save.
        except:
            st.error('Saving error')

if __name__ == '__main__':

    st.set_page_config(page_title="DRL Portfolio Simulator • TRAIN", layout="wide")

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # не пускаем дальше без авторизации
    if ("password_correct" not in st.session_state) or (not st.session_state["password_correct"]):
        st.warning("You need to log in!")
        st.markdown('<a href="/" target="_self">Authorization</a>', unsafe_allow_html=True)
        st.stop()

    df_assets = st.session_state["df_assets"] 
    date_scale = st.session_state["df_"+df_assets["code"][0]]["date"] # получаем шкалу по первому инструменту
    date_scale = pd.to_datetime(date_scale).dt.date
    features = list(st.session_state["df_"+df_assets["code"][0]].columns[1:])
    progress_file = '../models/progress/DDPG_' + st.session_state['current_username'] + '.json'

    run()