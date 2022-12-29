import streamlit as st
import pandas as pd
import numpy as np


st.title('Aplicación de predicción de la producción lechera')

siteHeader = st.container()
datainput = st.container()
output = st.container()
modelTraining = st.container()

with siteHeader:

      st.markdown('Esta aplicación permite predecir la producción por lactancia de vacas en condiciones de **pastoreo** '
            ', bajo un sistema de pastoreo rotacional, bajo condiciones de clima templado como en la Sierra del Ecuador.')

      st.markdown('La predicción se realiza con el algoritmo de **Machine Learning** de regresión lineal que ha sido entrenado '
                  'con datos de producción de la ganadería de la Hacienda el Prado del IASA.  ')

      st.markdown('En esta hacienda se ha mantenido el sistema de cruzamiento absorbente **Montbéliarde x Holstein** que corresponde'
                  'corresponde al componente genético de la ganaderia.')

df = pd.read_csv(r"https://github.com/jorgeord/stream/blob/main/base2.csv")
pro10 = pd.read_csv(r"https://github.com/jorgeord/stream/blob/main/pro1.csv")

X = df[['raza', 'dur_lactancia', 'servicios', 'natimuerto', 'aborto', 'produccion1',
             'pre_2antes', 'pre_1antes', 'pre_mesparto', 'pre_1pos', 'pre_2pos', 'hums_1antes',
             'hums_mesparto', 'hums_1pos', 'hums_2pos']]
y = df['produccion']

X = pd.get_dummies(data=X, drop_first=True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])

a = coeff_df._get_value(0, 0, takeable=True)
b= coeff_df._get_value(1, 0, takeable=True)
c= coeff_df._get_value(2, 0, takeable=True)
d= coeff_df._get_value(3, 0, takeable=True)
e= coeff_df._get_value(4, 0, takeable=True)
f= coeff_df._get_value(5, 0, takeable=True)
g= coeff_df._get_value(6, 0, takeable=True)
h= coeff_df._get_value(7, 0, takeable=True)
i= coeff_df._get_value(8, 0, takeable=True)
j= coeff_df._get_value(9, 0, takeable=True)
k= coeff_df._get_value(10, 0, takeable=True)
l= coeff_df._get_value(11, 0, takeable=True)
m= coeff_df._get_value(12, 0, takeable=True)
n= coeff_df._get_value(13, 0, takeable=True)
o= coeff_df._get_value(14, 0, takeable=True)
p= coeff_df._get_value(15, 0, takeable=True)
q= coeff_df._get_value(16, 0, takeable=True)



with datainput:

    st.header('Variables del modelo predictivo')

    st.markdown('Por favor ingrese los valores requeridos para la estimación:')

    st.subheader('Factores genéticos')
    raza = st.radio('Raza',['Holstein','F1 (50% Montbéliarde, 50% Holstein)','F2 (75% Montbéliarde, 25% Holstein)','F3 (87.5% Montbéliarde, 12.5% Holstein)'])

    st.subheader('Variables de producción')
    dur_lact = st.slider('Duración de la lactancia', 10, 500, 305, 5)
    produccion_1= st.slider('Lactancia en muestra día 10',7.0,23.0,13.0,0.5)

    st.subheader('Variables de reproducción')
    servicios = st.slider('Servicios por concepción', 1, 5, 2, 1)

    natimuerto = st.radio('Presencia de Natimorto',['Si','No'])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    st.subheader('Variables agroecológicas')
    pre_2antes= st.slider('Precipitación 2 meses antes del parto (mm al mes)', 1,275,78,1)
    pre_1antes=st.slider('Precipitación 1 mes antes del parto (mm al mes)', 1,275,78,1)
    pre_mesparto=st.slider('Precipitación en el mes del parto (mm al mes)', 1,275,78,1)
    pre_1pos=st.slider('Precipitación 1 mes después del parto (mm al mes)', 1,275,78,1)
    pre_2pos=st.slider('Precipitación 2 meses después del parto (mm al mes)', 1,275,78,1)
    hums_1antes=st.slider('Humedad del suelo 1 mes antes del parto (% mensual)', 6, 44, 26, 1)
    hums_mesparto=st.slider('Humedad del suelo en el mes de parto (% mensual)', 6, 44, 26, 1)
    hums_1pos=st.slider('Humedad del suelo 1 mes depués del parto (% mensual)', 6, 44, 26, 1)
    hums_2pos=st.slider('Humedad del suelo 2 mese después del parto (% mensual)', 6, 44, 26, 1)

if raza == 'Holstein':
    w=1
    y=0
    z=0

elif raza== 'F2 (75% Montbéliarde, 25% Holstein)':
    w=0
    y=1
    z=0

elif raza== 'F3 (87.5% Montbéliarde, 12.5% Holstein)':
    w=0
    y=0
    z=1
elif raza== 'F1 (50% Montbéliarde, 50% Holstein)':
    w=0
    y=0
    z=0

if natimuerto== 'Si':
    s=1
elif natimuerto=='No':
    s=0

pro_in = pro10.set_index('produccion1')
loc_pro = pro_in.loc[[produccion_1]]
x1= loc_pro._get_value(0, 0, takeable=True)


pro_lact= (
-1054.97 +
a*dur_lact +
b*servicios +
c*s+
e*x1+
f*pre_2antes +
g*pre_1antes +
h*pre_mesparto +
i*pre_1pos +
j*pre_2pos +
k*hums_1antes +
l*hums_mesparto +
m*hums_1pos +
n* hums_2pos +
o*y +
p*z +
q*w
)

with output:
    st.header('Resultado del modelo predictivo')

    st.subheader(f'La producción estimada por lactancia es: {pro_lact:.2f} kilogramos')

