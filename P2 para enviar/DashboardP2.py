import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([

    html.H1('Herramienta para conocer tu probabilidad de graduarse en la universidad', 
            style={'textAlign': 'center', 'color': 'black', 'backgroundColor': '#0000FF', 'padding': '20px'}),

    dcc.Tabs(
        id='tabs',
        value='tab-1',
        children=[
            dcc.Tab(label='Tab 1', value='tab-1'),
            dcc.Tab(label='Tab 2', value='tab-2'),
            dcc.Tab(label='Tab 3', value='tab-3'),
            dcc.Tab(label='Tab 4', value='tab-4')
        ]
    ),
    html.Div(id='tab-content')
])

# Define el contenido para cada pestaña
tab1_content = html.Div([
    html.P('Luego de completar algunos datos conocerás tu probabilidad de desertar:', 
           style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    
    html.Div([
        html.Label('¿Cuál es tu género?'),
        dcc.Dropdown(id='genero', options=[{'label':'Masculino', 'value':0}, {'label':'Femenino', 'value':1}], placeholder='Género'),

    ], className='row'),
    html.Br(),

    html.Div([
        html.Label('¿Eres becado?'),
        dcc.Dropdown(id='beca', options=[{'label':'Si', 'value':1}, {'label':'No', 'value':0}], placeholder='Beca'),

    ], className='row'),
    html.Br(),

    html.Div([
        html.Label('¿Eres deudor?'),
        dcc.Dropdown(id='deuda', options=[{'label':'Si', 'value':1}, {'label':'No', 'value':0}], placeholder='Deudor'),

    ], className='row'),
    html.Br(),

    html.Div([
        html.Label('¿Cuál es la nota de tu primer semestre?'),
        dcc.Dropdown(id='nota1', options=[{'label':'[0,5]', 'value':0}, {'label':'[5,10]', 'value':1}, {'label':'[10,15]', 'value':2}, {'label':'[15,20]', 'value':3}], placeholder='Nota'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Cuántos créditos aprobaste en el primer semestre?'),
        dcc.Dropdown(id='creditos1', options=[{'label':'[0,6]', 'value':0}, {'label':'(6,12]', 'value':1}, {'label':'(12,18]', 'value':2}, {'label':'(18,24]', 'value':3}, {'label':'(24,28]', 'value':4}], placeholder='Créditos'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Cuál es la nota de tu segundo semestre?'),
        dcc.Dropdown(id='nota2', options=[{'label':'[0,5]', 'value':0}, {'label':'[5,10]', 'value':1}, {'label':'[10,15]', 'value':2}, {'label':'[15,20]', 'value':3}], placeholder='Nota'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Cuántos créditos aprobaste en el segundo semestre?'),
        dcc.Dropdown(id='creditos2', options=[{'label':'[0,6]', 'value':0}, {'label':'(6,12]', 'value':1}, {'label':'(12,18]', 'value':2}, {'label':'(18,24]', 'value':3}], placeholder='Créditos'),

    ], className='row'),

    html.Br(),

    html.Button('Enviar datos', id='boton', n_clicks=0),

    html.Br(),

    html.Div(id='output')
])

# El resto de tu código Dash continúa aquí

# Define las otras pestañas y su contenido (gráficos) de manera similar
tab2_content = ...
tab3_content = ...
tab4_content = ...

# Define una función de callback para actualizar el contenido de la pestaña cuando el usuario selecciona una pestaña diferente
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return tab1_content
    elif tab == 'tab-2':
        return tab2_content
    elif tab == 'tab-3':
        return tab3_content
    elif tab == 'tab-4':
        return tab4_content

# Definir tus callbacks aquí
@app.callback(
    Output('output', 'children'),
    [Input('boton', 'n_clicks')],
    [State('genero', 'value'),
    State('beca', 'value'),
    State('deuda', 'value'),
    State('nota1', 'value'),
    State('creditos1', 'value'),
    State('nota2', 'value'),
    State('creditos2', 'value')]
)
def predecir_probabilidad_desercion(n_clicks, genero, beca, deuda, nota1, creditos1, nota2, creditos2):
    if n_clicks > 0:
        ruta = "/Users/sebastianrodriguez/Downloads/data_mod_copia2.xlsx"
        datos = pd.read_excel(ruta)

        modelo = BayesianNetwork([("Gender", "Scholarship holder"), ("Scholarship holder", "Debtor"),
                               ("Scholarship holder", "Target"), ("Debtor","Target"),
                               ("Curricular units 1st sem (grade)","Target"),
                               ("Curricular units 1st sem (grade)", "Curricular units 1st sem (approved)"),
                               ("Curricular units 1st sem (approved)","Target"),
                               ("Curricular units 2nd sem (grade)","Target"),
                               ("Curricular units 2nd sem (grade)", "Curricular units 2nd sem (approved)"),
                               ("Curricular units 2nd sem (approved)","Target")])

        X = datos.drop(columns=['Target'])
        y = datos['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

        emv = MaximumLikelihoodEstimator(modelo, data=datos)
        modelo.fit(data=datos, estimator=MaximumLikelihoodEstimator)

        y_pred = modelo.predict(X_test)

        infer = VariableElimination(modelo)

        probabilidad_desercion = infer.query(["Target"], evidence={"Debtor": deuda, "Gender": genero, "Scholarship holder": beca, "Curricular units 1st sem (approved)": creditos1, "Curricular units 1st sem (grade)": nota1, "Curricular units 2nd sem (approved)": creditos2, "Curricular units 2nd sem (grade)": nota2})

        return f'Probabilidad de graduarse: {round(probabilidad_desercion.values[1] * 100, 2)}%'

if __name__ == '__main__':
    app.run_server(debug=True, port=8070)

