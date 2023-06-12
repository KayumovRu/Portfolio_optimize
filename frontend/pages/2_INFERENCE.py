import streamlit as st
import requests
import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import glob
import numpy as np

# Шарп в годовом исчислении, по умолчанию для недельных данных
def sharpe(returns, freq=52, rfr=0):
    return (np.sqrt(freq) * np.mean(returns - rfr + EPS)) / np.std(returns - rfr + EPS)

# новые настройки применения модели срабатывают после нажатия кнопки APPLY
# + поддерживается предупреждение, что текущие графики построены по старым настройкам
# НУЖНО ДОБАВИТЬ ЗАЩИТУ ОТ ПОВТОРНЫХ ОТПРАВОК - например, вынести в функцию и поставить декоратор кеширования вида data и отправлять в нее data
def apply():
    pass

# ОСНОВНАЯ ФУНКЦИЯ - при загрузке страницы
def run():

    button_apply = st.sidebar.button("APPLY", use_container_width=True)

    if "settings" not in st.session_state:
        st.session_state.settings = {}
    if "settings_fix" not in st.session_state:
        st.session_state.settings_fix = {}
    if "inference_result" not in st.session_state:
        st.session_state.inference_result = {}

    # костыль от перезагрузки при смене опции
    # https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11
    if "button_apply" not in st.session_state: # начальная инициализация
        st.session_state.button_apply = False 
    if button_apply:
        st.session_state.settings_fix = st.session_state.settings # фиксируем настройки для предупреждения о смене настроек
        st.session_state.button_apply = True
        # ОТПРАВКА ЗАПРОСА
        # (добавить защиту от отправки запросов без выбора хотя бы одной модели - возможно при нажатии кнопки) st.warning('Select at least one model and press APPLY!')
        response = requests.post("http://127.0.0.1:8000/inference", json=st.session_state.settings)
        st.session_state.inference_result = response.json()

        if 'DJIA' in st.session_state.settings['basemodels']:
            st.session_state.inference_result['DJIA'] = { # здесь рассчитываем отдельно для индекса - пока отключено
                'depo_results':     [],
                'action_results':   [[0, 1]], # распределение активов, структура: [[a11, a21], [a12, a22], ...], где aNx - доля N актива в период x
                'sharpe_ratio':     0,
                'mdd':              0, # рассчитывать!!!
                'portfolio_value':  0,
                'assets_list':      ['DJIA'],
                }
            

    # контейнер отображения предупреждения об изменениях
    warning_change = st.container()        

    # САЙДБАР С ПАРАМЕТРАМИ
    with st.sidebar:

        # выбор модели
        st.title("Models:")
        select_DDPG = st.checkbox("DDPG") # здесь еще продолжить с запоминанием настроек value=(st.session_state.settings["model_ddpg"] != 'none')
        if select_DDPG: # ищем и выбираем DDPG модели
            if st.session_state['current_username'] != 'admin': # если не админ, то показываем модели только самого юзера
                files = glob.glob(path_models + 'DDPG_' + st.session_state['current_username'] + '*')
            else:
                files = glob.glob(path_models + 'DDPG_*')
            files = [file.replace(path_models, '') for file in files] # удаляем пути до моделей
            select_model_DDPG = st.selectbox('Select DDPG-model', files,)
        else:
            select_model_DDPG = 'none'

        # формирование листа используемых моделей (не DRL)
        basemodels_dict = {
            'EW': st.checkbox("Equal Weighting (baseline)"),
            'Markowitz': st.checkbox("Markowitz (baseline)", disabled=True),
            'DJIA': st.checkbox("DJIA (benchmark)", disabled=True)
        }
        basemodels_list = [key for key, value in basemodels_dict.items() if value == 1]

        st.title("Deposit:") # value синхронизировать с settings.deposit (использовать st.session_state.depo)
        st.session_state.depo = st.number_input("Deposit", value = 10000, label_visibility = "collapsed",
                                                 step=1000)
        st.title("Date Start:")
        select_test_start = st.date_input(
            "Start Test",
            datetime.date(2020, 6, 18),
            min_value = date_scale_start,
            label_visibility = "collapsed")
        st.text(select_test_start.strftime('%Y/%m/%d') + " - " + date_scale_end.strftime('%Y/%m/%d'))

        # вычисляем реальную дату начала (с учетом ts)
        start_test_true = date_scale.loc[date_scale >= select_test_start].iloc[0]
        date_scale_test = date_scale.loc[date_scale >= start_test_true]
        len_test = len(date_scale_test)
        test_ratio = round(len_test / len_scale, 3)
        st.text("Weeks: " + str(len_test) + " ("+ str(round(test_ratio*100, 1)) + "%)")
        train_ratio = 1 - test_ratio # для передачи на бэкенд

        # Комиссии при сделках, пока не подключено, поэтому нулевые
        st.title("Comission (%):")
        comission = st.number_input("Trading Cost", value = 0.0, label_visibility = "collapsed", step = 0.01, format='%.2f', max_value = 1.0, min_value = 0.0) / 100

        # если уже обращались хоть раз к БД (существует st.session_state["df_assets"]), то показываем настройки
        try:
            df_assets = st.session_state["df_assets"]

            # выбор активов, на которых применить модель - в дальнейшем брать их из меты модели
            # хотя при использовании пересчета можно брать максимум без ошибки
            st.title("Assets:")
            select_assets = st.multiselect('Select assets ',
                df_assets["ticker"], df_assets["ticker"],
                label_visibility = "collapsed", disabled=True
            )
        except:
            pass

    st.session_state.settings = {
        'deposit': st.session_state.depo,
        'train_ratio': train_ratio, 
        #'test_start': st.session_state.test_period[0].strftime('%Y-%m-%d'),
        #'test_period_end': st.session_state.test_period[1].strftime('%Y-%m-%d'),
        'basemodels': basemodels_list,
        'model_ddpg': select_model_DDPG,
        'comission': comission,
    }

    # предупреждение о смене настроек до нажатия APPLY
    if st.session_state.settings_fix != st.session_state.settings:
        warning_change.warning("Settings have been changed, charts are not relevant. Press APPLY!")

    if st.session_state.button_apply and len(st.session_state.inference_result) > 0:        

        inference_result = st.session_state.inference_result

        # ВИЗУАЛИЗАЦИЯ
        df_result_depo = pd.DataFrame()
        df_result_depo_coms = pd.DataFrame() # здесь будут отдельно комиссии
        df_result_returns = pd.DataFrame() # здесь отдельно по периодам относительные возвраты

        col_title, col_return, col_mdd, col_sharpe = st.columns(4)
        col_return.caption("Return")
        col_mdd.caption("Max Drowdown")
        col_sharpe.caption("Sharpe Ratio")
        
        for model in inference_result:

            df_result_depo[model] = inference_result[model]['depo_results']

            if model != 'DJIA': # 
                df_result_actions = pd.DataFrame(inference_result[model]["action_results"],
                                columns=[0, *list(inference_result[model]["assets_list"].values())]).iloc[:, 1:] # исключаем первую, там кэш
            else:
                df_result_actions = pd.DataFrame(inference_result[model]["action_results"],
                                columns=[0, *list(inference_result[model]["assets_list"])]).iloc[:, 1:]


            # расчет комиссий постфактум
            diff_df = df_result_actions.sub(df_result_actions.shift()) # получение таблицы приращений по активам
            diff_df.loc[0] = df_result_actions.loc[0]
            diff_df = diff_df.applymap(lambda x: x if x > 0 else 0) # оставляем только положительные приращения по активу
            diff_df.loc[len(diff_df)] = 0 # добавление последней строки, чтобы при сдвиге дальше не терялась
            diff_df = diff_df.shift(1).fillna(0) # сдвиг на 1 вниз
            df_coms = diff_df.sum(axis=1) * df_result_depo[model] * st.session_state.settings_fix['comission']
            df_coms = df_coms.cumsum()
            df_result_depo_coms[model] = df_coms * -1

            df_result_depo[model] = df_result_depo[model] + df_result_depo_coms[model] # учет комиссий

            df_result_returns[model] = df_result_depo[model].pct_change()
            df_result_returns[model].iloc[0] = 0

            # визуализируем блок с метриками
            col_title, col_return, col_mdd, col_sharpe = st.columns(4)
            loc_total_return    = round(100 * df_result_depo[model].iloc[-1] / df_result_depo[model].iloc[0], 1) 
            loc_mdd             = round(inference_result[model]['mdd'] * 100, 1) # БЕЗ УЧЕТА КОМИССИЙ
            loc_sharpe          = round(sharpe(df_result_returns[model]), 2) # round(inference_result[model]['sharpe_ratio'], 2)
            # col_title.subheader(model)
            col_title.markdown("<h3 style='text-align: center; color: #FAFAFA99;'>" + model + "</h3>", unsafe_allow_html=True)
            col_return.metric("Return",  str(loc_total_return)+'%', str(round(loc_total_return - 100, 1))+'%', label_visibility = "collapsed")
            col_mdd.metric("Max Drowdown",   "-"+str(loc_mdd)+'%', "", label_visibility = "collapsed")
            col_sharpe.metric("Sharpe Ratio",   loc_sharpe, "", label_visibility = "collapsed")

        # шкалу даты подстраиваем под число данных
        df_result_depo["date"] = df_result_depo_coms["date"] = date_scale.tail(df_result_depo.shape[0]).reset_index(drop=True)

        tab_equity, tab_coms = st.tabs(["Equity", "Comission"])
        with tab_equity:            
            # график портфелей по моделям
            fig = px.line(df_result_depo, x="date", y=df_result_depo.columns,
                            hover_data={"date": "|%B %d, %Y"})
            fig.update_layout(xaxis_rangeslider_visible = False,
                            xaxis_title = None,
                            yaxis_title = 'Portfolio Return, $',
                            # plot_bgcolor = '#0E1117',
                            # xaxis_gridcolor = '#262730',
                            # yaxis_gridcolor = '#262730',
                            legend_title = '',
                            height=500,
                            )
            fig.add_hline(y=10000, line_dash="dot", row=1, col=1, line_color="#FFFFFF", line_width=1)
            st.plotly_chart(fig, use_container_width=True, theme='streamlit')
        
        # КОМИССИИ
        with tab_coms:
            col_coms_total, col_coms_change = st.columns(2)

            with col_coms_total: # сравнение комиссий моделей
                st.caption("Comission, %")
                df_result_depo_coms_undate = df_result_depo_coms.drop("date", axis=1).iloc[-1]
                df_result_depo_coms_undate = df_result_depo_coms_undate / abs(df_result_depo.iloc[-1].drop("date") - df_result_depo.iloc[0].drop("date") - df_result_depo_coms_undate)
                # st.write(df_result_depo_coms_undate)
                fig = px.bar(df_result_depo_coms_undate,
                        y=df_result_depo_coms_undate.index + "   ",
                        x=df_result_depo_coms_undate.values,
                        text = [f'{val:,.2%}' for val in df_result_depo_coms_undate.values],
                        orientation='h',
                        color_discrete_sequence=['#FECB52'],
                        )
                fig.update_layout(xaxis_rangeslider_visible = False,
                        xaxis_title = None,
                        yaxis_title = None,
                        # plot_bgcolor = '#0E1117',
                        height=400,)
                st.plotly_chart(fig, use_container_width=True, theme='streamlit')


            with col_coms_change: # график комиссий (куммулятивный) по моделям                
                st.caption("Comission History, $ (cumulative)")
                fig = px.line(df_result_depo_coms, x="date", y=df_result_depo_coms.columns,
                                hover_data={"date": "|%B %d, %Y"})
                fig.update_layout(xaxis_rangeslider_visible = False,
                                xaxis_title = None,
                                yaxis_title = 'Comission, $',
                                plot_bgcolor = '#0E1117',
                                xaxis_gridcolor = '#262730',
                                yaxis_gridcolor = '#262730',
                                legend_title = '',
                                height=400,
                                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

        # ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ АКТИВОВ
        st.subheader("Asset Allocation")
        # формирование табов для размещения распределений по разным моделям
        tabs_allocation = [x for x in inference_result]
        tabs_allocation = st.tabs(tabs_allocation)

        i = 0 # счетчик для управления табами
        for model in inference_result:


            if model != 'DJIA': # временный костыль
                df_result_actions = pd.DataFrame(inference_result[model]["action_results"],
                                            columns=[0, *list(inference_result[model]["assets_list"].values())]).iloc[:, 1:] # исключаем первую, там кэш
            else:
                df_result_actions = pd.DataFrame(inference_result[model]["action_results"],
                                            columns=[0, *list(inference_result[model]["assets_list"])]).iloc[:, 1:] # исключаем первую, там кэш
                

            # оставляем только те активы, у которых макс значение хоть раз было сколько-нибудь значимо
            mask = df_result_actions.max() < 1e-3
            df_result_actions = df_result_actions.drop(columns = mask[mask].index.tolist())

            # сохраняем последнюю строчку распределения для текущих рекомендаций, удаляя все нулевые
            actions_current = df_result_actions.iloc[-1].round(3).replace(0, None).dropna(how='any').sort_values()
            # ресемплируем по 10 недель
            df_result_actions_mean = df_result_actions.groupby(df_result_actions.index // 10).mean().round(4)
            # добавляем дату для графика
            df_result_actions_mean['date'] = df_result_depo['date'].groupby(df_result_depo["date"].index // 10).max().reset_index(drop=True)
            df_result_actions_mean.set_index('date', inplace=True)
            df_result_actions_mean.index = pd.to_datetime(df_result_actions_mean.index).strftime('%Y-%m-%d') # костыль - перевод в datetime и обратно позволяет корректно отображать бары на графике

            with tabs_allocation[i]:
                col_allocate_bars, col_allocate_pie = st.columns(2)
                with col_allocate_bars: # текущее распределение - бары
                    st.caption("Current Allocation, $ (start deposit rel)")
                    actions_current_depo = actions_current * st.session_state.settings["deposit"]
                    fig = px.bar(actions_current_depo,  y=actions_current_depo.index + "   ",
                                                        x=actions_current_depo.values,
                                                        text = [f'${val:,.2f}' for val in actions_current_depo.values],
                                                        orientation='h',
                                                        # color_discrete_sequence=['#00CC96'],
                                                        )
                    fig.update_layout(xaxis_rangeslider_visible = False,
                          xaxis_title = None,
                          yaxis_title = None,
                          plot_bgcolor = '#0E1117',)
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                with col_allocate_pie: # текущее распределение - круговая
                    st.caption("Current Allocation, %")
                    fig = go.Figure(data=[go.Pie(   labels=actions_current.index,
                                                    values=actions_current.values,
                                                    hole=.3,)])
                    st.plotly_chart(fig, use_container_width=True, theme=None)

                st.caption("History Allocation (*averaged values for every 10 weeks)")
                st.bar_chart(df_result_actions_mean, height=500)

                st.caption("Asset Contibution (coming soon...)")
            i += 1

    else:
        warning_change.warning("Select at least one model and press APPLY!")


if __name__ == '__main__':

    st.set_page_config(page_title="DRL Portfolio Simulator • INFERENCE", layout="wide")

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

    timeframe = 'w1'
    EPS = 1e-8
    update = {"Model":[], "Interval": []} # настройки для кнопки Update

    # получаем серию date по одному из активов в качестве шкалы
    first_code = st.session_state["df_assets"]["code"][0]
    date_scale = st.session_state["df_"+first_code]["date"]
    date_scale = pd.to_datetime(date_scale).dt.date
    date_scale_start = date_scale.iloc[0]  # дата старта шкалы
    date_scale_end = date_scale.iloc[-1] # дата окончания шкалы
    len_scale = len(date_scale) # всего периодов в шкале

    path_models = "../models/"

    run()