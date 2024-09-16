parameters_dict = {  
        'L14.B01_1H': {
            'otimizacao': {
                'media_final': 28.34,
                'nulos_removidos': 288,
                'base_predict': 11,
                'altura_predict': 77,
            },
            'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 1.0,
                'stl':'stl_adjusted',
                'stl_trend': 169,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.05, 0.99, #week median
                                       0.08, 0.3, # selecionar 25% dos dados
                                       0.05, 0.99],
                'seasonal_trend_power' : [0.1, # força ajuste tendencia 
                                              'base_min'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'country': 'Noruega',
                'tipo': 'Escola',
                'year': [2017, 2018],
                'Area': 6_084, 
                'disaggregation': 'just heating',
                'clear_holidays': True,
                'interval': [
                    # ((6, 22), (8, 1), 'ferias_verao'),  # Férias de verão de meados de junho a meados de agosto
                    # ((12, 22), (1, 8), 'ferias_natal'),  # Férias de Natal do dia 23 de dezembro até o dia 7 de janeiro
                    # # ((10, 7), (10, 14), 'ferias_outono'),  # Férias de outono durante uma semana em outubro
                    # ((2, 11), (2, 21), 'ferias_inverno'),  # Férias de inverno de uma semana em fevereiro ou março
                    # ((3, 23), (3, 27), 'ferias_pascoa')  # Férias de Páscoa começando na Quinta-feira Santa e indo até a segunda-feira após a Páscoa
                ],
                'holidays': [
                    # (1, 1, 'ano_novo'),  # Ano Novo
                    # (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    # (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    # (5, 10, 'ascencao'),  # Quinta-feira da Ascensão
                    # (5, 20, 'pentecostes'),  # Domingo de Pentecostes
                    # (5, 21, 'segunda_pentecostes'),  # Segunda-feira de Pentecostes
                    # # dias estranhos nos dados que merecem ser removidos (não são feriados)
                    # (4, 19, '___'),                  
                    # (6, 1, '___'),
                    # (6, 2, '___'),
                    # (6, 3, '___'),
                    # (6, 4, '___'),
                    # (6, 9, '___'),
                    # (6, 10, '___'),
                    # (6, 16, '___'),
                    # (6, 17, '___'),
                ],
                
            },
        },
        'L14.B02_1H': {
            'otimizacao': {
                'media_final': 15.43,
                'nulos_removidos': 312,
                'base_predict': 10,
                'altura_predict': 43,
            },
            'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 1.0,
                'stl':'stl_adjusted',
                'stl_trend': 169,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.0001, 0.9999, #week median
                                       0.0000, 0.12, # selecionar 25% dos dados
                                      0.0001, 0.9999],
                'seasonal_trend_power' : [0, # força ajuste tendencia 
                                              'base_min'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'country': 'Noruega',
                'clear_holidays': False,
                'tipo': 'AsiloDeIdosos',
                'year': [2017, 2018],
                'Area': 2_700, 
                'disaggregation': 'just heating',
                'interval': [
                    ((12, 22), (12, 26), 'celebracoes_natal'),  # Celebrações de Natal
                    ((3, 20), (3, 27), 'semana_pascoa')  # Semana de Páscoa
                ],
                'holidays': [
                    # (1, 1, 'ano_novo'),  # Ano Novo
                    # (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    # (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    # (5, 10, 'ascencao'),  # Quinta-feira da Ascensão
                    # (5, 20, 'pentecostes'),  # Domingo de Pentecostes
                    # (5, 21, 'segunda_pentecostes')  # Segunda-feira de Pentecostes
                ]   
            },
        },
        'L14.B03_1H': {
            'otimizacao': {
                'media_final': 20.26,
                'nulos_removidos': 288,
                'base_predict': 10,
                'altura_predict': 80,
            },
            'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 0.9,
                'stl':'stl_adjusted',
                'stl_trend': 169,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.01, 0.99, #week median
                        0.08, 0.2, # selecionar 25% dos dados
                        0.01, 0.99],
                'seasonal_trend_power' : [0.1, # força ajuste tendencia 
                                              'base_min'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'country': 'Noruega',
                'clear_holidays': True,
                'tipo': 'Escola',
                'year': [2017, 2018],
                'Area': 6_546, 
                'disaggregation': 'just heating',
                'interval': [
                    # ((7, 5), (8, 1), 'ferias_verao'),  # Férias de verão de meados de junho a meados de agosto
                    # ((12, 22), (1, 3), 'ferias_natal'),  # Férias de Natal do dia 23 de dezembro até o dia 7 de janeiro
                    # ((10, 1), (10, 7), 'ferias_outono'),  # Férias de outono durante uma semana em outubro
                    # ((2, 14), (2, 22), 'ferias_inverno'),  # Férias de inverno de uma semana em fevereiro ou março
                    # ((3, 26), (4, 2), 'ferias_pascoa'),  # Férias de Páscoa começando na Quinta-feira Santa e indo até a segunda-feira após a Páscoa
                    # ((5, 14), (5, 23), 'ferias_constituicao_pentecostes')  # Férias de Pentecostes de 17 a 21 de maio
                ],
                'holidays': [
                    # (1, 1, 'ano_novo'),  # Ano Novo
                    # (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    # (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    # (5, 10, 'ascencao'),  # Quinta-feira da Ascensão
                    # (5, 20, 'pentecostes'),  # Domingo de Pentecostes
                    # (5, 21, 'segunda_pentecostes')  # Segunda-feira de Pentecostes
                ]
            },
        },
        'L14.B04_1H': {
            'otimizacao': {
                'media_final': 107.96,
                'nulos_removidos': 288,
                'base_predict': 105,
                'altura_predict': 285,
            },
            'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 0.9,
                'stl':'stl_adjusted',
                'stl_trend': 169*5,
                'stl_seasonal': 169*5,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.00, 0.99, #wquantiles selelcao plugs
                                       0.0000, 0.25, # selecionar 25% dos dados
                                       0,1], # quantiles normalização
                'seasonal_trend_power' : [0, # força ajuste tendencia 
                                              'base_min'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'clear_holidays': True,                
                'country': 'Noruega',
                'tipo': 'office',
                'year': [2017, 2018],
                'Area': 11_502, 
                'disaggregation': 'just heating',
                'interval': [
                    # ((12, 22), (1, 2), 'ferias_natal'),  # Férias de Natal de 22 de dezembro a 2 de janeiro
                    # ((10, 7), (10, 14), 'ferias_outono'),  # Férias de outono durante uma semana em outubro
                    # ((3, 24), (4, 2), 'ferias_pascoa')  # Férias de Páscoa de 24 de março a 2 de abril
                ],
                'holidays': [
                    # (1, 1, 'ano_novo'),  # Ano Novo
                    # (3, 29, 'quinta_feira_santa'),  # Quinta-feira Santa
                    # (3, 30, 'sexta_feira_santa'),  # Sexta-feira Santa
                    # (4, 1, 'domingo_pascoa'),  # Domingo de Páscoa
                    # (4, 2, 'segunda_pascoa'),  # Segunda-feira de Páscoa
                    # (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    # (5, 10, 'ascensao'),  # Quinta-feira da Ascensão
                    # (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    # (5, 20, 'pentecostes'),  # Domingo de Pentecostes
                    # (5, 21, 'segunda_pentecostes'),  # Segunda-feira de Pentecostes
                    # (12, 25, 'natal'),  # Natal
                    # (12, 26, 'segundo_dia_natal')  # Segundo dia de Natal
                ],
            },
        },
        'L14.B05_1H': {
            'otimizacao': {
                'media_final': 98.52,
                'nulos_removidos': 288,
                'base_predict': 2,
                'altura_predict': 18,
            },
            'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 1,
                'stl':'stl_adjusted',
                'stl_trend': 169,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.00005, 0.999995, #week median
                        0.0000, 0.3, # selecionar 25% dos dados
                        0.00005, 0.999995],
                'seasonal_trend_power' : [0, # força ajuste tendencia 
                                              'base_min'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'clear_holidays': True,                
                'country': 'Noruega',
                'tipo': 'Kindergarten',
                'year': [2017, 2018],
                'Area': 9980, 
                'disaggregation': 'just heating',
                'interval': [
                    # ((5, 13), (9, 11), 'ferias_verao'),  # Férias de verão tipicamente de 24 de junho a 15 de agosto
                    # ((12, 22), (1, 3), 'ferias_natal'),  # Férias de Natal de 22 de dezembro a 3 de janeiro
                    # ((10, 9), (10, 18), 'ferias_outono'),  # Férias de outono durante a primeira semana de outubro
                    # ((2, 19), (2, 25), 'ferias_inverno'),  # Férias de inverno na última semana de fevereiro
                    # ((3, 24), (4, 2), 'ferias_pascoa')  # Férias de Páscoa de 24 de março a 2 de abril
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    (5, 10, 'ascencao'),  # Quinta-feira da Ascensão
                    (5, 20, 'pentecostes'),  # Domingo de Pentecostes
                    (5, 21, 'segunda_pentecostes')  # Segunda-feira de Pentecostes
                ],
        },
    },
        'L06.B01_1H': {
            'otimizacao': {
                'media_final': 11.85,
                'nulos_removidos': 1053,
                'base_predict': 12,
                'altura_predict': 107,
            },
                'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 1,
                'stl':'stl_adjusted',
                'stl_trend': 169*1,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.001, 0.999, #week median
                                        0.0000, 0.3, # selecionar 25% dos dados
                                        0.001, 0.999],
                'seasonal_trend_power' : [0.4, # força ajuste tendencia 
                                              'base_min'], # tipo do calculda base da tendencia 
            },
            'clean_data_parameters':{
                'clear_holidays': False,                
                'Outliers_feriados_cuidados': True,
                'media_sazonal_horaria_alterando': True,
                'country': 'Noruega',
                'tipo': 'ShoppingCenter',
                'Area': 2_200, 
                'disaggregation': 'just heating',
                'year': [2021],
                'interval': [
                    ((11, 1), (12, 1), 'erro'),  
                    ((12, 24), (12, 26), 'ferias_natal'),
                    ((4, 1), (4, 6), 'ferias_pascoa')
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    (5, 13, 'ascencao'),  # Quinta-feira da Ascensão
                    (12, 25, 'natal'),  # Natal
                    (12, 26, 'segundo_dia_natal')  # Segundo dia de Natal
                ],
            },
        },
        'L03.B02_1H': {
            'otimizacao': {
                'media_final': 31.94,
                'nulos_removidos': 1296,
                'base_predict': 0,
                'altura_predict': 122,
            },
            'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 1,
                'stl':'stl_adjusted',
                'stl_trend': 169,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.01, 0.99, #week median
                                       0.0000, 0.1, # selecionar 25% dos dados
                                       0.01, 0.99],
                'seasonal_trend_power' : [0, # força ajuste tendencia 
                                              'base_min'], # tipo do calculda base da tendencia 
            },
            'clean_data_parameters':{
                'clear_holidays': True,                
                'country': 'Dinamarca',
                'tipo': 'office',
                'year': [2020, 2021],
                'Area': 8_685, 
                'disaggregation': 'just heating',
                'interval': [
                        # ((12, 1), (6, 1), 'pandemia'),  # Férias de verão de 24 de junho a 12 de agosto
                        ((12, 1), (1, 3), 'ferias_natal'),  # Férias de Natal de 18 de dezembro a 3 de janeiro (2022)
                        ((2, 15), (2, 21), 'ferias_inverno'),  # Férias de inverno de 15 a 21 de fevereiro
                        ((3, 29), (4, 5), 'ferias_pascoa')  # Férias de Páscoa de 29 de março a 5 de abril
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (4, 1, 'quinta_feira_santa'),  # Quinta-feira Santa
                    (4, 2, 'sexta_feira_santa'),  # Sexta-feira Santa
                    (4, 4, 'domingo_pascoa'),  # Domingo de Páscoa
                    (4, 5, 'segunda_pascoa'),  # Segunda-feira de Páscoa
                    (4, 30, 'dia_oracao'),  # Dia de Oração Geral
                    (5, 13, 'ascensao'),  # Quinta-feira da Ascensão
                    (5, 23, 'pentecostes'),  # Domingo de Pentecostes
                    (5, 24, 'segunda_pentecostes'),  # Segunda-feira de Pentecostes
                    (6, 5, 'dia_constituicao'),  # Dia da Constituição
                    (12, 24, 'vespera_natal'),  # Véspera de Natal
                    (12, 25, 'natal'),  # Natal
                    (12, 26, 'segundo_dia_natal')  # Segundo dia de Natal
                ],
            },
        },
        'L10.B01_1H': {
            'otimizacao': {
                'media_final': 108.36,
                'nulos_removidos': 221,
                'base_predict': 58,
                'altura_predict': 140,
            },
            'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 0.5,
                'stl':'stl_adjusted',
                'stl_trend': 168+1,
                'stl_seasonal': 25, #25
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.01, 0.99, #week median
                                       0.0000, 0.1, # selecionar 25% dos dados
                                       0.01, 0.99],
                'seasonal_trend_power' : [0.2, # força ajuste tendencia 
                                              'base'], # tipo do calculda base da tendencia 
            },
            'clean_data_parameters':{
                'clear_holidays': False,                
                'Outliers_feriados_cuidados': True,
                'country': 'Noruega',
                'tipo': 'office',
                'year': [2020],
                'Area': 18_000, 
                'disaggregation': 'heating and cooling',
                'interval': [
                    # ((12, 24), (12, 26), 'ferias_natal'),  # Férias de Natal de 22 de dezembro a 2 de janeiro
                    # ((4, 9), (4, 13), 'ferias_pascoa 2020'),
                    # ((3, 27), (4, 5), 'ferias_pascoa 2021'),  # Férias de Páscoa de 24 de março a 2 de abril
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    (5, 10, 'ascensao'),  # Quinta-feira da Ascensão
                    (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    (5, 20, 'pentecostes'),  # Domingo de Pentecostes
                    (5, 21, 'segunda_pentecostes'),  # Segunda-feira de Pentecostes
                    (12, 25, 'natal'),  # Natal
                    (12, 26, 'segundo_dia_natal'),
                    (7, 2, 'erro')# Segundo dia de Natal
                ],
        },
    },
        'L10.B01_30min': {
            'otimizacao': {
                'media_final': 108.17,
                'nulos_removidos': 221,
                'base_predict': 58,
                'altura_predict': 142,
            },
            'decomposition_parameter':{
                'seasonality': 168*2,
                'multiplier': 0.5,
                'stl':'stl_adjusted',
                'stl_trend': 168*2+1,
                'stl_seasonal': 25,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.01, 0.99, #week median
                                       0.0000, 0.1, # selecionar 25% dos dados
                                       0.01, 0.99],
                'seasonal_trend_power' : [0.2, # força ajuste tendencia 
                                              'base'], # tipo do calculda base da tendencia 
            },
            'clean_data_parameters':{
                'clear_holidays': False,                
                'Outliers_feriados_cuidados': True,
                'country': 'Noruega',
                'tipo': 'office',
                'year': [2020],
                'Area': 18_000, 
                'disaggregation': 'heating and cooling',
                'interval': [
                    # ((12, 24), (12, 26), 'ferias_natal'),  # Férias de Natal de 22 de dezembro a 2 de janeiro
                    # ((4, 9), (4, 13), 'ferias_pascoa 2020'),
                    # ((3, 27), (4, 5), 'ferias_pascoa 2021'),  # Férias de Páscoa de 24 de março a 2 de abril
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    (5, 10, 'ascensao'),  # Quinta-feira da Ascensão
                    (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    (5, 20, 'pentecostes'),  # Domingo de Pentecostes
                    (5, 21, 'segunda_pentecostes'),  # Segunda-feira de Pentecostes
                    (12, 25, 'natal'),  # Natal
                    (12, 26, 'segundo_dia_natal'),
                    (7, 2, 'erro')# Segundo dia de Natal
                ],
        },
    },
        'L10.B01_15min': {
            'otimizacao': {
                'media_final': 105.79,
                'nulos_removidos': 892,
                'base_predict': 60,
                'altura_predict': 143,
            },
            'decomposition_parameter':{
                'seasonality': 168*4,
                'multiplier': 0.5,
                'stl':'stl_adjusted',
                'stl_trend': 168*4+1,
                'stl_seasonal': 25,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.01, 0.99, #week median
                                       0.0000, 0.1, # selecionar 25% dos dados
                                       0.01, 0.99],
                'seasonal_trend_power' : [0.2, # força ajuste tendencia 
                                              'base'], # tipo do calculda base da tendencia 
            },
            'clean_data_parameters':{
                'clear_holidays': False,                
                'Outliers_feriados_cuidados': True,
                'country': 'Noruega',
                'tipo': 'office',
                'year': [2020],
                'Area': 18_000, 
                'disaggregation': 'heating and cooling',
                'interval': [
                    # ((12, 24), (12, 26), 'ferias_natal'),  # Férias de Natal de 22 de dezembro a 2 de janeiro
                    # ((4, 9), (4, 13), 'ferias_pascoa 2020'),
                    # ((3, 27), (4, 5), 'ferias_pascoa 2021'),  # Férias de Páscoa de 24 de março a 2 de abril
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (5, 1, 'dia_trabalhador'),  # Dia Internacional dos Trabalhadores
                    (5, 10, 'ascensao'),  # Quinta-feira da Ascensão
                    (5, 17, 'dia_constituicao'),  # Dia da Constituição
                    (5, 20, 'pentecostes'),  # Domingo de Pentecostes
                    (5, 21, 'segunda_pentecostes'),  # Segunda-feira de Pentecostes
                    (12, 25, 'natal'),  # Natal
                    (12, 26, 'segundo_dia_natal'),
                    (7, 2, 'erro')# Segundo dia de Natal
                ],
        },
    },
        'L09.B01_5min': {
            'otimizacao': {
                'media_final': 43.20,
                'nulos_removidos': 26297,
                'base_predict': 45,
                'altura_predict': 111,
            },
         'decomposition_parameter':{
                'seasonality': 168*12,
                'multiplier': 0.6,
                'stl':'stl_adjusted',
                'stl_trend': 168*12+1,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.05, 0.95, #week median
                                       0.0, 0.15, # selecionar 25% dos dados
                                       0.01, 0.99],
                'seasonal_trend_power' : [0.5, # força ajuste tendencia 
                                              'base'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'clear_holidays': True,                
                'country': 'Australia',
                'tipo': 'office',
                'year': [2021],
                'Area': -1, 
                'disaggregation': 'heating and cooling',
                'interval': [
                    ((9, 1), (10, 1), 'erro'),
                    ((1, 1), (1, 15), 'FERIAS INICIO DO ANO'),
                    ((12, 15), (1, 5), 'ferias_natal'),  # Férias de Natal de 25 de dezembro a 3 de janeiro
                    ((4, 2), (4, 6), 'ferias_pascoa'),  # Férias de Páscoa (incluindo Sexta-
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (1, 26, 'australia_day'),  # Dia da Austrália
                    (4, 2, 'good_friday'),  # Sexta-feira Santa
                    (4, 4, 'easter_sunday'),  # Domingo de Páscoa
                    (4, 5, 'easter_monday'),  # Segunda-feira de Páscoa
                    (6, 14, 'queen_birthday'),  # Aniversário da Rainha (varia por estado)
                    (12, 25, 'christmas_day'),  # Natal
                    (12, 26, 'boxing_day')  # Boxing Day
                ],
            },
        },
        'L09.B01_15min': {
            'otimizacao': {
                'media_final': 50.07,
                'nulos_removidos': 8717,
                'base_predict': 46,
                'altura_predict': 108,
            },
         'decomposition_parameter':{
                'seasonality': 168*4,
                'multiplier': 0.6,
                'stl':'stl_adjusted',
                'stl_trend': 168*4+1,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.05, 0.95, #week median
                                       0.0, 0.15, # selecionar 25% dos dados
                                       0.01, 0.99],
                'seasonal_trend_power' : [0.5, # força ajuste tendencia 
                                              'base'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'clear_holidays': True,                
                'country': 'Australia',
                'tipo': 'office',
                'year': [2021],
                'Area': -1, 
                'disaggregation': 'heating and cooling',
                'interval': [
                    ((9, 1), (10, 1), 'erro'),
                    ((1, 1), (1, 15), 'FERIAS INICIO DO ANO'),
                    ((12, 15), (1, 5), 'ferias_natal'),  # Férias de Natal de 25 de dezembro a 3 de janeiro
                    ((4, 2), (4, 6), 'ferias_pascoa'),  # Férias de Páscoa (incluindo Sexta-
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (1, 26, 'australia_day'),  # Dia da Austrália
                    (4, 2, 'good_friday'),  # Sexta-feira Santa
                    (4, 4, 'easter_sunday'),  # Domingo de Páscoa
                    (4, 5, 'easter_monday'),  # Segunda-feira de Páscoa
                    (6, 14, 'queen_birthday'),  # Aniversário da Rainha (varia por estado)
                    (12, 25, 'christmas_day'),  # Natal
                    (12, 26, 'boxing_day')  # Boxing Day
                ],
            },
        },
        'L09.B01_30min': {
            'otimizacao': {
                'media_final': 52.88,
                'nulos_removidos': 4328,
                'base_predict': 48,
                'altura_predict': 108,
            },
         'decomposition_parameter':{
                'seasonality': 168*2,
                'multiplier': 0.6,
                'stl':'stl_adjusted',
                'stl_trend': 168*2+1,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.05, 0.95, #week median
                                       0.0, 0.15, # selecionar 25% dos dados
                                       0.01, 0.99],
                'seasonal_trend_power' : [0.5, # força ajuste tendencia 
                                              'base'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'clear_holidays': True,                
                'country': 'Australia',
                'tipo': 'office',
                'year': [2021],
                'Area': -1, 
                'disaggregation': 'heating and cooling',
                'interval': [
                    ((9, 1), (10, 1), 'erro'),
                    ((1, 1), (1, 15), 'FERIAS INICIO DO ANO'),
                    ((12, 15), (1, 5), 'ferias_natal'),  # Férias de Natal de 25 de dezembro a 3 de janeiro
                    ((4, 2), (4, 6), 'ferias_pascoa'),  # Férias de Páscoa (incluindo Sexta-
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (1, 26, 'australia_day'),  # Dia da Austrália
                    (4, 2, 'good_friday'),  # Sexta-feira Santa
                    (4, 4, 'easter_sunday'),  # Domingo de Páscoa
                    (4, 5, 'easter_monday'),  # Segunda-feira de Páscoa
                    (6, 14, 'queen_birthday'),  # Aniversário da Rainha (varia por estado)
                    (12, 25, 'christmas_day'),  # Natal
                    (12, 26, 'boxing_day')  # Boxing Day
                ],
            },
        },
        'L09.B01_1H': {
            'otimizacao': {
                'media_final': 53.67,
                'nulos_removidos': 2155,
                'base_predict': 48,
                'altura_predict': 108,
            },
         'decomposition_parameter':{
                'seasonality': 168,
                'multiplier': 0.6,
                'stl':'stl_adjusted',
                'stl_trend': 168+1,
                'stl_seasonal': 3,
                'stl_low_pass': -1,
                'stl_robust': False,
                'week_plug_selection':[0.05, 0.95, #week median
                                       0.0, 0.15, # selecionar 25% dos dados
                                       0.01, 0.99],
                'seasonal_trend_power' : [0.5, # força ajuste tendencia 
                                              'base'], # tipo do calculda base da tendencia 

            },
            'clean_data_parameters':{
                'clear_holidays': True,                
                'country': 'Australia',
                'tipo': 'office',
                'year': [2021],
                'Area': -1, 
                'disaggregation': 'heating and cooling',
                'interval': [
                    ((9, 1), (10, 1), 'erro'),
                    ((1, 1), (1, 15), 'FERIAS INICIO DO ANO'),
                    ((12, 15), (1, 5), 'ferias_natal'),  # Férias de Natal de 25 de dezembro a 3 de janeiro
                    ((4, 2), (4, 6), 'ferias_pascoa'),  # Férias de Páscoa (incluindo Sexta-
                ],
                'holidays': [
                    (1, 1, 'ano_novo'),  # Ano Novo
                    (1, 26, 'australia_day'),  # Dia da Austrália
                    (4, 2, 'good_friday'),  # Sexta-feira Santa
                    (4, 4, 'easter_sunday'),  # Domingo de Páscoa
                    (4, 5, 'easter_monday'),  # Segunda-feira de Páscoa
                    (6, 14, 'queen_birthday'),  # Aniversário da Rainha (varia por estado)
                    (12, 25, 'christmas_day'),  # Natal
                    (12, 26, 'boxing_day')  # Boxing Day
                ],
            },
        },
}
