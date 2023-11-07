import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import psycopg2
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server



def grafica1():
    #Conexión a la base de datos
    engine = psycopg2.connect(
        dbname="estudiantes",
        user="postgres",
        password="EstudiantesSSC",
        host="p2-ssc.ccesjqcligmw.us-east-1.rds.amazonaws.com",
        port="5432"
    )

    # Consulta SQL
    consulta1 = """
    SELECT Gender,
           SUM(CASE WHEN Scholarship_holder = 1 AND Target = 1 THEN 1 ELSE 0 END) AS becados_graduados,
           SUM(CASE WHEN Scholarship_holder = 1 AND Target = 0 THEN 1 ELSE 0 END) AS becados_desertores,
           SUM(CASE WHEN Scholarship_holder = 0 AND Target = 1 THEN 1 ELSE 0 END) AS no_becados_graduados,
           SUM(CASE WHEN Scholarship_holder = 0 AND Target = 0 THEN 1 ELSE 0 END) AS no_becados_desertores
    FROM Estudiantes
    GROUP BY Gender;
    """

    #Crear Dataframe con los datos de la consulta
    df = pd.read_sql_query(consulta1, engine)

    #Lo siguiente es para obtener "Mujeres" y "Hombres" en vez de 0 y 1
    df['gender'] = df['gender'].map({0: 'Mujeres', 1: 'Hombres'})

    #Gráfica de barras apiladas
    fig = px.bar(df, x='gender', y=['becados_graduados', 'becados_desertores', 'no_becados_graduados', 'no_becados_desertores'],
                 labels={'Gender': 'Género', 'value': 'Número de Estudiantes'},
                 title='Éxito académico por género y estado de beca',
                 color_discrete_sequence=['#011f4b', '#005b96', '#6497b1', '#b3cde0'],
                 barmode='stack',
                 )

    #Dimensiones
    fig.update_layout(
        width=800,  # Ancho de la gráfica
        height=500,  # Altura de la gráfica
        margin=dict(l=50, r=50, b=50, t=50),  # Márgenes
    )

    #Actualizar nombres de los ejes
    fig.update_xaxes(title_text='Género')
    fig.update_yaxes(title_text='Número de Estudiantes')

    return dcc.Graph(figure=fig)

def grafica2():
    #Conexión a la base de datos
    engine = psycopg2.connect(
        dbname="estudiantes",
        user="postgres",
        password="EstudiantesSSC",
        host="p2-ssc.ccesjqcligmw.us-east-1.rds.amazonaws.com",
        port="5432"
    )

    #Consulta SQL
    consulta2 = """
    SELECT Gender,
           SUM(CASE WHEN Debtor = 1 AND Target = 1 THEN 1 ELSE 0 END) AS deudores_graduados,
           SUM(CASE WHEN Debtor = 1 AND Target = 0 THEN 1 ELSE 0 END) AS deudores_desertores,
           SUM(CASE WHEN Debtor = 0 AND Target = 1 THEN 1 ELSE 0 END) AS no_deudores_graduados,
           SUM(CASE WHEN Debtor = 0 AND Target = 0 THEN 1 ELSE 0 END) AS no_deudores_desertores
    FROM Estudiantes
    GROUP BY Gender;
    """

    #Crear Dataframe con los datos de la consulta
    df2 = pd.read_sql_query(consulta2, engine)

    #Lo siguiente es para obtener "Mujeres" y "Hombres" en vez de 0 y 1
    df2['gender'] = df2['gender'].map({0: 'Mujeres', 1: 'Hombres'})

    #Gráfica de barras
    fig = px.bar(df2, x='gender', y=['deudores_graduados', 'deudores_desertores', 'no_deudores_graduados', 'no_deudores_desertores'],
                 labels={'Gender': 'Género', 'value': 'Número de Estudiantes'},
                 title='Éxito académico por género y estado de deuda',
                 color_discrete_sequence=['#011f4b', '#005b96', '#6497b1', '#b3cde0'],
                 barmode='group'  # Cambiar a "group" en lugar de "stack"
                )

    #Dimensiones
    fig.update_layout(
        width=800,  # Ancho de la gráfica
        height=500,  # Altura de la gráfica
        margin=dict(l=50, r=50, b=50, t=50)  # Márgenes
    )

    #Actualizar nombres de los ejes
    fig.update_xaxes(title_text='Género')
    fig.update_yaxes(title_text='Número de Estudiantes')

    return dcc.Graph(figure=fig)

def grafica3 ():
    #Conexión a la base de datos
    engine = psycopg2.connect(
        dbname="estudiantes",
        user="postgres",
        password="EstudiantesSSC",
        host="p2-ssc.ccesjqcligmw.us-east-1.rds.amazonaws.com",
        port="5432"
    )

    #Consulta SQL
    consulta3 = """
    SELECT Gender, 
        SUM(CASE WHEN Curricular_units_1st_sem_grade = 0 THEN 1 ELSE 0 END) AS primersem_rangouno,
        SUM(CASE WHEN Curricular_units_1st_sem_grade = 1 THEN 1 ELSE 0 END) AS primersem_rangodos,
        SUM(CASE WHEN Curricular_units_1st_sem_grade = 2 THEN 1 ELSE 0 END) AS primersem_rangotres,
        SUM(CASE WHEN Curricular_units_1st_sem_grade = 3 THEN 1 ELSE 0 END) AS primersem_rangocuatro,
        SUM(CASE WHEN Curricular_units_2nd_sem_grade = 0 THEN 1 ELSE 0 END) AS segundosem_rangouno,
        SUM(CASE WHEN Curricular_units_2nd_sem_grade = 1 THEN 1 ELSE 0 END) AS segundosem_rangodos,
        SUM(CASE WHEN Curricular_units_2nd_sem_grade = 2 THEN 1 ELSE 0 END) AS segundosem_rangotres,
        SUM(CASE WHEN Curricular_units_2nd_sem_grade = 3 THEN 1 ELSE 0 END) AS segundosem_rangocuatro
    FROM Estudiantes
    GROUP BY Gender;
    """

    #Crear Dataframe con los datos de la consulta
    df3 = pd.read_sql_query(consulta3, engine)

    #Lo siguiente es para obtener "Mujeres" y "Hombres" en vez de 0 y 1
    df3['gender'] = df3['gender'].map({0: 'Mujeres', 1: 'Hombres'})

    #Gráfica de barras
    fig = px.bar(df3, y='gender', x=['primersem_rangouno', 'primersem_rangodos', 'primersem_rangotres', 'primersem_rangocuatro', 'segundosem_rangouno', 'segundosem_rangodos', 'segundosem_rangotres', 'segundosem_rangocuatro'],
                 labels={'value': 'Número de Estudiantes', 'Gender': 'Género'},
                 title='Notas 1er y 2do Semestre por género y rango de notas',
                 color_discrete_sequence=['#000080', '#0066CC', '#3399FF', '#6699FF', '#99CCFF', '#3366CC', '#CCE5FF', '#E5F2FF'],
                 barmode='group',
                 orientation="h"
                )

    #Dimensiones
    fig.update_layout(
        width=800,  # Ancho de la gráfica
        height=500,  # Altura de la gráfica
        margin=dict(l=50, r=50, b=50, t=50)  # Márgenes
    )

    #Actualizar nombres de los ejes
    fig.update_xaxes(title_text='Número de Estudiantes')
    fig.update_yaxes(title_text='Género')

    return dcc.Graph(figure=fig)

app.layout = html.Div([

    html.H1('Herramienta de apoyo para conocer su éxito acádemico en la universidad', 
            style={'textAlign': 'center', 'color': 'black', 'backgroundColor': '#0000FF', 'padding': '20px'}),

    #Crear 4 pestañas
    dcc.Tabs(
        id='tabs',
        value='tab-1',
        children=[
            dcc.Tab(label='Probabilidad de Grado', value='tab-1'),
            dcc.Tab(label='Gráfica Estado de Beca', value='tab-2'),
            dcc.Tab(label='Gráfica Estado de Deuda', value='tab-3'),
            dcc.Tab(label='Gráfica Notas 1er y 2do Semestre', value='tab-4')
        ]
    ),
    html.Div(id='tab-content')
])

#Contenido de las pestañas
tab1_content = html.Div([
    html.P('Luego de completar algunos datos conocerás tu probabilidad de graduarte:', 
           style={'textAlign': 'center', 'fontSize': '20px', 'fontWeight': 'bold', 'color': 'black'}),
    
    html.Div([
        html.Label('¿Con cuál género te identificas?'),
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
        html.Label('¿Dentro de cuál rango esta la nota de tu primer semestre?'),
        dcc.Dropdown(id='nota1', options=[{'label':'[0,5]', 'value':0}, {'label':'[5,10]', 'value':1}, {'label':'[10,15]', 'value':2}, {'label':'[15,20]', 'value':3}], placeholder='Nota'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Dentro de cuál rango están los créditos que aprobaste en el primer semestre?'),
        dcc.Dropdown(id='creditos1', options=[{'label':'[0,6]', 'value':0}, {'label':'(6,12]', 'value':1}, {'label':'(12,18]', 'value':2}, {'label':'(18,24]', 'value':3}, {'label':'(24,28]', 'value':4}], placeholder='Créditos'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Dentro de cuál rango esta la nota de tu segundo semestre?'),
        dcc.Dropdown(id='nota2', options=[{'label':'[0,5]', 'value':0}, {'label':'[5,10]', 'value':1}, {'label':'[10,15]', 'value':2}, {'label':'[15,20]', 'value':3}], placeholder='Nota'),

    ], className='row'),

    html.Br(),

    html.Div([
        html.Label('¿Dentro de cuál rango están los créditos que aprobaste en el segundo semestre?'),
        dcc.Dropdown(id='creditos2', options=[{'label':'[0,6]', 'value':0}, {'label':'(6,12]', 'value':1}, {'label':'(12,18]', 'value':2}, {'label':'(18,24]', 'value':3}], placeholder='Créditos'),

    ], className='row'),

    html.Br(),

    html.Button('Enviar datos', id='boton', n_clicks=0),

    html.Br(),

    html.Div(id='output')
])

tab2_content = html.Div([
    html.H2('Gráfica de Barras Apiladas sobre el Estado de Beca'),
    html.P('A continuación, se muestra una gráfica de barras que representa el éxito académico por género y estado de beca.'),
    grafica1()
])

tab3_content = html.Div([
    html.H2('Gráfica de Barras sobre el Estado de Deuda'),
    html.P('A continuación, se muestra una gráfica de barras que representa el éxito académico por género y estado de deuda.'),
    grafica2()
])

tab4_content = html.Div([
    html.H2('Gráfica de Barras Horizontales sobre las Notas del 1er y 2do Semestre por Género'),
    html.P('A continuación, se muestra una gráfica de barras que representa las notas del primer y segundo semestre por género. Tenga en cuenta que los rangos son los siguientes: Rango 1: [0-5], Rango 2: (5-10], Rango 3: (10-15] y Rango4: (15-20]'),
    grafica3()
])

#Callbacks para actualizar el contenido de la pestaña que el usuario escoja
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
        ruta = "D:/datos usuario/Documents/Universidad de los Andes/2023-20/Analitica/Proyecto/data_mod_copia2.xlsx"
        datos = pd.read_excel(ruta)

        modelo = BayesianNetwork([('Scholarship holder', 'Gender'), ('Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)'), ('Curricular units 2nd sem (approved)', 'Target'), ('Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)'), ('Curricular units 2nd sem (grade)', 'Curricular units 1st sem (grade)'), ('Target', 'Curricular units 2nd sem (grade)'), ('Target', 'Scholarship holder'), ('Target', 'Debtor'), ('Target', 'Gender'), ('Target', 'Curricular units 1st sem (grade)')])

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

