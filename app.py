import numpy as np
import pandas as pd
from dash import Dash, html, dcc,Input, Output, callback
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import pulp



# Inicializar la aplicación Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
app.title = 'Optimización del Uso de Energía Renovable'
df = pd.read_csv('datos.csv')


# Header
header = html.Div(
    className='header',
    children=[
        html.H1('Optimización del Uso de Energía Renovable en Ciudades Inteligentes', style={'textAlign': 'center'}),
        html.P('El proyecto consiste en analizar el consumo de energía en una ciudad y proponer estrategias para optimizar el uso de energías renovables (solar, eólica, etc.) en función de patrones de consumo, condiciones climáticas y disponibilidad de recursos. El objetivo es reducir la huella de carbono y mejorar la eficiencia energética.', style={'textAlign': 'justify'}),
        ],
    style={'padding': '50px'}
)

# Primera sección
titulo_seccion_1 = html.H3('ANÁLISIS DESCRIPTIVO DE CONSUMO DE ENERGÍA', style={'textAlign': 'center'})
figura_seccion_1 = px.line(df, x='Fecha', y=["Consumo", "Generación Solar"], title='Tendencia de Consumo y Generación Solar', line_shape="spline", labels={"value": "Energía (kWh)", "variable": "Tipo"})
figura_seccion_1.layout.update(showlegend=True)
grafica_seccion_1 = dcc.Graph(figure=figura_seccion_1, className='graph')
grafica_correlacion = px.imshow(df.corr(numeric_only=True), title='Correlación entre variables', labels={"x": "Variable", "y": "Variable", "color": "Correlación"})
grafica_seccion_1_2 = dcc.Graph(figure=grafica_correlacion, className='graph')
grafica_box = px.box(df, x='Consumo', y='Generación Solar', title='Distribución de Consumo y Generación Solar', labels={"value": "Energía (kWh)"})
grafica_seccion_1_3 = dcc.Graph(figure=grafica_box, className='graph')
grafica_scartter = px.scatter_matrix(df, dimensions=["Consumo", "Generación Solar", "Temperatura"], title='Matriz de Dispersión', labels={"value": "Energía (kWh)"})
grafica_seccion_1_4 = dcc.Graph(figure=grafica_scartter, className='graph', style={'height': '500px'})

contenido_seccion_1 = html.Div(
    className='row',
    children=[
        html.Div(
            className='container',
            children=[
                titulo_seccion_1,
                grafica_seccion_1
            ]
        ),
        html.Div(
            className='row',
            children=[
                 html.Div(
                    className='col-md-6',
                    children=[
                        grafica_seccion_1_2
                    ]       
                ),
                html.Div(
                    className='col-md-6',
                    children=[
                       grafica_seccion_1_4
                    ]       
                ),
                ]
        )
    ],
    style={'padding': '25px', 'margin': 'auto'}
)

# Segunda sección


@callback(
            Output('grafica_1', 'figure'),
            Output('grafica_2', 'figure'),
            Output('grafica_3', 'figure'),
            Input('selector-columna', 'value'))
def grafica_descomposicion_estacional(columna):
    result = seasonal_decompose(df[columna], model='multiplicative', period=12)
    if columna == 'Temperatura':
        labels = {"value": "Temperatura (°C)"}
    else:
        labels = {"value": "Energía (kWh)"}
    grafica_1=px.line(df[columna], title=f'Datos Originales {columna}', labels=labels)
    grafica_2=px.line(result.trend, title=f'Tendencia {columna}', labels=labels)
    grafica_3=px.line(result.seasonal, title=f'Estacionalidad {columna}', labels=labels)
    return grafica_1, grafica_2, grafica_3


@callback(Output('resultado', 'children'), Input('selector-columna', 'value'))
def verificar_estacioalidad(columna):
    result = adfuller(df[columna])
    if result[1] > 0.05:
        resultado =f'La serie de tiempo {columna} no es estacionaria y que su valor de p-value es {result[1]:.8f}'
    else:
        resultado= f'La serie de tiempo {columna} es estacionaria ya que su valor de p-value es {result[1]:.8f}'
    return resultado
    


selector_columna = html.Div(
        children=[
            html.H6('Seleccion columna analisis'),
            dcc.Dropdown(
                id='selector-columna',
                placeholder='Seleccion columna analisis',
                options=[
                    {'label': 'Consumo', 'value': 'Consumo'},
                    {'label': 'Generación Solar', 'value': 'Generación Solar'},
                    {'label': 'Temperatura', 'value': 'Temperatura'},
                ],
                value='Consumo',
            )
        ],
        style={'width': '30%', 'margin': 'auto', 'textAlign': 'center'},
    )

grafica_1 = dcc.Graph( id='grafica_1')
grafica_2 = dcc.Graph( id='grafica_2')
grafica_3 = dcc.Graph( id='grafica_3')


contenido_seccion_2 = html.Div(
    className='row',
    children=[
        html.H3('ANÁLISIS DE DESCOMPOSICIÓN ESTACIONAL', style={'textAlign': 'center'}),
        selector_columna,
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='container',
                    children=[
                        grafica_1
                    ]       
                )
            ]
        ),
        html.Div(
            className='row',
            children=[
                 html.Div(
                    className='col-md-6',
                    children=[
                        grafica_2
                    ]       
                ),
                html.Div(
                    className='col-md-6',
                    children=[
                        grafica_3,
                        html.H6(id='resultado', style={'textAlign': 'center'}),
                    ]       
                ),
                ]
        )
    ],
    style={'margin': 'auto', 'padding': '25px', 'textAlign': 'center'}
)


# Tercera sección

@callback(Output('grafica_prediccion', 'figure'), [Input('selector-columna-1', 'value'), Input('num-multi', 'value')])
def grafica_prediccion(columna, periodos):
    df = pd.read_csv('datos.csv')
    df=df.rename(columns={'Fecha':'ds', columna:'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periodos, freq = 'd')
    forecast = model.predict(future)
    df.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)
    viz_df = df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
    viz_df['Predicción'] = (viz_df['yhat'])
    viz_df['Valor Real'] = (viz_df['y'])
    if columna == 'Temperatura':
        labels = {"value": "Temperatura (°C)", "ds": "Fecha",  'y_exp': 'Valor Real', 'yhat_exp': 'Predicción'}
    else:
        labels = {"value": "Energía (kWh)","ds": "Fecha",  'y_exp': 'Valor Real', 'yhat_exp': 'Predicción'}
    fig=px.line(viz_df, x=viz_df.index, y=['Valor Real', 'Predicción'], title=f'Predicción {columna}', labels=labels, line_shape="spline", )
    return fig


grafica_pred = dcc.Graph(id='grafica_prediccion', className='graph')


periodos = html.Div(
    className='controler-row',
    children=[
        html.H6('Número de periodos a predecir'),
        dcc.Input(
            id='num-multi',
            type='number',
            min=1,
            max=30,
            value=7
        )
    ],
    style={'width': '30%', 'margin': 'auto', 'textAlign': 'center'},
)

selector_columna_1 = html.Div(
        children=[
            html.H6('Seleccion columna analisis'),
            dcc.Dropdown(
                id='selector-columna-1',
                placeholder='Seleccion columna analisis',
                options=[
                    {'label': 'Consumo', 'value': 'Consumo'},
                    {'label': 'Generación Solar', 'value': 'Generación Solar'},
                    {'label': 'Temperatura', 'value': 'Temperatura'},
                ],
                value='Consumo',
            )
        ],
        style={'width': '30%', 'margin': 'auto','textAlign': 'center'},
    )

controls = html.Div(
    className='controler-row',
    children=[
        periodos,
        selector_columna_1
    ]
)

contenido_seccion_3 = html.Div(
    className='row',
    children=[
        html.H3('ANÁLISIS PREDICTIVO', style={'textAlign': 'center'}),
        controls,
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='container',
                    children=[
                        grafica_pred
                    ]       
                )
            ]
        ),
    ],
    style={'margin': 'auto', 'padding': '25px'}
)

# Análisis Prescriptivo: Optimización simple (maximizar uso de energía solar)Función de optimización
def optimizar_uso_solar(generacion, demanda):
    problema = pulp.LpProblem("Maximizar_Uso_Solar", pulp.LpMaximize)
    uso_solar = pulp.LpVariable.dicts("Uso_Solar", range(len(generacion)), lowBound=0)
    problema += pulp.lpSum([uso_solar[i] for i in range(len(generacion))])
    for i in range(len(generacion)):
        problema += uso_solar[i] <= generacion[i]
        problema += uso_solar[i] <= demanda[i]
    problema.solve()
    return [uso_solar[i].varValue for i in range(len(generacion))]

# Aplicar la optimización
df["Uso Óptimo Solar"] = optimizar_uso_solar(df["Generación Solar"], df["Consumo"])
fig_prescriptivo = px.bar(df, x="Fecha", y="Uso Óptimo Solar", 
                              title="Prescriptivo: Uso Óptimo de Energía Solar",
                              labels={"Uso Óptimo Solar": "Energía Solar Usada (kWh)"})
grafica_prescriptivo = dcc.Graph(figure=fig_prescriptivo, className='graph')
insights_prescriptivo = f"""
    El uso óptimo de energía solar maximiza su aprovechamiento sin exceder la generación disponible ni la demanda de energía.
    Uso total de energía solar: {df["Uso Óptimo Solar"].sum():.2f} kWh.
    """


contenido_seccion_4 = html.Div(
    className='row',
    children=[
        html.H3('ANÁLISIS PRESCRIPTIVO', style={'textAlign': 'center'}),
        html.Div(
            className='row',
            children=[
                html.Div(
                    className='container',
                    children=[
                        grafica_prescriptivo,
                        html.H5(insights_prescriptivo, style={'textAlign': 'center'})
                    ]       
                )
            ]
        ),
    ],
    style={'margin': 'auto', 'padding': '25px'}
)


# Layout del dashboard
app.layout = html.Div(
    children=[
        header,
        contenido_seccion_1,
        contenido_seccion_2,
        contenido_seccion_3,
        contenido_seccion_4

])

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run_server(debug=True)