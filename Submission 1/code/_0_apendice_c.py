import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from _1_pre_process import plot_energy_data, show_plots
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show


def faz_markov(df_, n_components=2):
    np.random.seed(42) 
    # Suponha que você tenha um vetor de leituras de energia, onde cada leitura é agregada
    data = np.array(df_['kw_total']).reshape(-1, 1)  # Dados de consumo agregados
    print('data shape' ,data.shape)

    # Definindo um HMM
    # O número de componentes corresponde ao número de estados de potência que você espera
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)

    # Treinamento do modelo
    model.fit(data)

    # Predição dos estados
    hidden_states = model.predict(data)

    # Inicializa as colunas para as máscaras
    for i in np.unique(hidden_states):
        df_[f'mask{i}'] = np.nan  # Inicializa com zero

    # Aplica os valores de 'kw_total' baseados nos estados
    for i in np.unique(hidden_states):
        mask = hidden_states == i
        df_.loc[mask, f'mask{i}'] = df_.loc[mask, 'kw_total']

    # Plotando os dados
    plot_energy_data(df_, ['mask0', 'mask1'], title='Consumo de Energia e Estados de Potência', plot_type = 'scatter')

    return hidden_states
def return_dfs_markov(path_):
    data = pd.read_csv(path_)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['date'] = data['timestamp'].dt.date
    data['hour'] = data['timestamp'].dt.hour
    data['weekday'] = data['timestamp'].dt.weekday
    data['season'] = data['timestamp'].dt.month % 12 // 3 + 1

    hh = faz_markov(data, n_components=2)

    mask_sem_ocupacao = hh == 0
    mask_com_ocupacao = hh == 1

    df_dia_trabalho = data[data.index.isin(data.index[mask_com_ocupacao])]
    df_fim_de_semana = data[~data.index.isin(data.index[mask_com_ocupacao])]

    print(f'df_dia_trabalho shape: {df_dia_trabalho.shape}')
    print(f'df_fim_de_semana shape: {df_fim_de_semana.shape}')
    print(f'data shape: {data.shape}')

    columns_to_plot = ['kw_total']
    plot_energy_data(df_dia_trabalho, columns_to_plot, title='Consumo de Energia em Dias de Trabalho')
    plot_energy_data(df_fim_de_semana, columns_to_plot, title='Consumo de Energia em Finais de Semana')

    return df_dia_trabalho, df_fim_de_semana
def calculate_multiplier_1(path_09, quantile_upper=0.95, quantile_lower=0.05, corr_threshold=0.7):
    df1,df2 = return_dfs_markov(path_09)
    df_fim_de_semana_09 = df1 if df1['kw_total'].mean() > df2['kw_total'].mean() else df2
    list_ = []
    for h in range(24):
        for s in range(1, 5):
            for y in df_fim_de_semana_09['year'].unique():
                df_fim_de_semana_09_ = df_fim_de_semana_09[(df_fim_de_semana_09['hour'] == h) & (df_fim_de_semana_09['season'] == s) & (df_fim_de_semana_09['year']==y) ].copy()
                # correlação com temperatura ou humidade
                corr = df_fim_de_semana_09_['kw_total'].corr(df_fim_de_semana_09_['air_temperature_at_2m(deg_C)'])
                if len(df_fim_de_semana_09_) < 20 or abs(corr) < corr_threshold:
                    continue

                quantil09 = df_fim_de_semana_09_['kw_total'].quantile(quantile_upper)
                df_fim_de_semana_09_['quantil09'] = quantil09
                print(f'Quantil 90% do consumo de energia em df_fim_de_semana_09: {quantil09}')
                quantil01 = df_fim_de_semana_09_['kw_total'].quantile(quantile_lower)
                df_fim_de_semana_09_['quantil01'] = quantil01
                print(f'Quantil 10% do consumo de energia em df_fim_de_semana_09: {quantil01}')
                print('amplitude quantil09', (quantil09 - quantil01)/quantil09)
                list_.append(1-((quantil09 - quantil01)/quantil09))
                # Calculando a média e o desvio padrão de 'kw_total'
                mean_kw_total = df_fim_de_semana_09_['kw_total'].mean()
                std_kw_total = df_fim_de_semana_09_['kw_total'].std()

                # Normalizando 'air_temperature_at_2m(deg_C)' para ter a mesma média e desvio padrão de 'kw_total'
                df_fim_de_semana_09_['air_temperature_at_2m(deg_C)_norm'] = (
                    (df_fim_de_semana_09_['air_temperature_at_2m(deg_C)'] - df_fim_de_semana_09_['air_temperature_at_2m(deg_C)'].mean()) / df_fim_de_semana_09_['air_temperature_at_2m(deg_C)'].std()
                ) * std_kw_total + mean_kw_total
                plot_energy_data(df_fim_de_semana_09_, ['kw_total', 'quantil09', 'quantil01', 'air_temperature_at_2m(deg_C)_norm'], title=f'h:{h} s:{s} corr:{corr} amp:{(quantil09 - quantil01)/quantil09}')

    print('Plug Global Power', np.mean(list_), np.std(list_), np.min(list_), np.max(list_))
    show_plots(ncols=4)
    return np.mean(list_), np.std(list_)

# model 1 - Busca pelos valores de multiplicador usando o quantil 90% e 10% do consumo de energia altamente correlacionado com a temperatura do ar
path_09 = 'D:/repositorios/REP_ADRENALIN/Submission 1/data/L09.B01_1H_cleaned.csv'
calculate_multiplier_1(path_09, quantile_upper=0.9, quantile_lower=0.1, corr_threshold=0.7)
path_10 = 'D:/repositorios/REP_ADRENALIN/Submission 1/data/L10.B01_1H_cleaned.csv'
calculate_multiplier_1(path_10, quantile_upper=0.99, quantile_lower=0.01, corr_threshold=0.85)

# model 2 procura por picos e valos para tentar encontrar o valor dos plugs e iluminação
def find_plugs_in_valeis_l9():
    path_09 = 'D:/repositorios/REP_ADRENALIN/Submission 1/data/L09.B01_5min_cleaned.csv'
    data = pd.read_csv(path_09)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # Filtrar dados entre '01-03-2021' e '01-09-2021'
    start_date = '2021-03-01'
    end_date = '2021-09-01'
    data = data.loc[(data['timestamp'] > start_date) & (data['timestamp'] < end_date)].reset_index(drop=True)

    hh = faz_markov(data, n_components=2)

    mask_sem_ocupacao = hh == 0
    mask_com_ocupacao = hh == 1

    data['dia_trabalho_cluster'] = 0
    data.loc[mask_com_ocupacao, 'dia_trabalho_cluster'] = 1

    # Passo 3: Filtrar o DataFrame original usando essas datas
    df_dia_trabalho = data[data.index.isin(data.index[mask_com_ocupacao])]
    df_fim_de_semana = data[~data.index.isin(data.index[mask_com_ocupacao])]

    print(f'df_dia_trabalho shape: {df_dia_trabalho.shape}')
    print(f'df_fim_de_semana shape: {df_fim_de_semana.shape}')
    print(f'data shape: {data.shape}')

    data = df_fim_de_semana.copy()
    data = data.reset_index(drop=True)
    # Suponha que x é sua série temporal
    x = data['kw_total']
    x = -x  # Invertendo o sinal

    # Encontrando os picos com proeminência relativa
    peaks, properties = find_peaks(x, height=(-80, -10), prominence=(60, 150), width=(0, 4))

    # Adicionar informações sobre os picos ao DataFrame
    data['is_peak'] = 0
    data.loc[peaks, 'is_peak'] = 1

    # Criando a figura
    p = figure(title="Detecção de Fundos com Proeminência e Largura Específica", x_axis_label='Index', y_axis_label='kw_total', width=800, height=400)

    # Adicionando a linha do sinal
    p.line(x.index, x, legend_label="kw_total", line_width=2)

    p.circle(peaks, x[peaks], size=10, color="blue", legend_label="peaks", fill_alpha=0.6)

    # Adicionando uma linha cinza horizontal em y=0
    p.line(x.index, np.zeros_like(x), line_dash="dashed", color="gray")

    # Ajustando a legenda
    p.legend.location = "top_left"

    # Mostrando o gráfico
    show(p)

    # Analisar características dos picos
    data.loc[peaks, 'peak_height'] = data['kw_total'][peaks]
    data.loc[peaks, 'peak_prominence'] = properties['prominences']
    data.loc[peaks, 'peak_width'] = properties['widths']
    data.loc[peaks, 'peack_base'] = properties['peak_heights'] - properties['prominences']
    properties['plugs']= properties['prominences']/((properties['peak_heights'] - properties['prominences'])*-1) 
    data.loc[peaks, 'plugs'] = properties['plugs']
    #print media plugs
    print(f'media dos plugs: {np.mean(properties["plugs"])}') 
    print(f'mediana dos plugs: {np.median(properties["plugs"])}') 
def find_plugs_in_peaks_l10():
    path10 = 'D:/repositorios/REP_ADRENALIN/Submission 1/data/L10.B01_15min_cleaned.csv'
    data = pd.read_csv(path10)
    # Filtrar dados entre '01-03-2021' e '01-09-2021'
    start_date = '2019-03-01'
    end_date = '2023-09-01'
    data = data.loc[(data['timestamp'] > start_date) & (data['timestamp'] < end_date)].reset_index(drop=True)
    # Suponha que x é sua série temporal
    x = data['kw_total']
    x = x  # Invertendo o sinal

    # Encontrando os picos com proeminência relativa
    peaks, properties = find_peaks(x, height=(50, 200), prominence=(55, 100), width=(0, 7))

    # Adicionar informações sobre os picos ao DataFrame
    data['is_peak'] = 0
    data.loc[peaks, 'is_peak'] = 1

    # Analisar características dos picos
    data.loc[peaks, 'peak_height'] = data['kw_total'][peaks]
    data.loc[peaks, 'peak_prominence'] = properties['prominences']
    data.loc[peaks, 'peak_width'] = properties['widths']
    data.loc[peaks, 'peack_base'] = properties['peak_heights'] - properties['prominences']
    properties['plugs']= properties['prominences']/(properties['peak_heights'] - properties['prominences']) 
    data.loc[peaks, 'plugs'] = properties['plugs']
    #print media plugs
    print(f'media dos plugs: {np.mean(properties["plugs"])}') 
    print(f'mediana dos plugs: {np.median(properties["plugs"])}') 

    # Criando a figura
    p = figure(title="Detecção de Fundos com Proeminência e Largura Específica", x_axis_label='Index', y_axis_label='kw_total', width=800, height=400)

    # Adicionando a linha do sinal
    p.line(x.index, x, legend_label="kw_total", line_width=2)

    # Adicionando os fundos detectados
    p.circle(peaks, x[peaks], size=10, color="blue", legend_label="peaks", fill_alpha=0.6)

    # Adicionando uma linha cinza horizontal em y=0
    p.line(x.index, np.zeros_like(x), line_dash="dashed", color="gray")

    # Ajustando a legenda
    p.legend.location = "top_left"

    # Mostrando o gráfico
    show(p)

find_plugs_in_valeis_l9()
find_plugs_in_peaks_l10()