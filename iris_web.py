
import streamlit as st

import pandas as pd
from PIL import Image

import pandas as pd
from io  import StringIO

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from streamlit.elements import selectbox
image4 = Image.open('inicio.png')

st.image(image4, width=3000,use_column_width='auto')

#st.image(image, caption='Sunrise by the mountains')


options = st.multiselect(
    'Bienvenido escoja la opcion que desea ',
        ['Inicio', 'Tendencia de Covid por pais'])


st.write('You selected:', options)


video_file = open('efe.mp4', 'rb')
video_bytes = video_file.read()



def Inicio(icon="☄️"):

    st.video(video_bytes,start_time=1)
    st.markdown( '####  CORONAVIRUS DATA ANALYSIS WITH MACHINE LEARNING es una aplicación desarollada con el unico  proposito de analizar a detalle   el virus COVID-19'  )















def Tendencia_Covid_Pais():
    #image_tendencia = Image.open('tendenciaa.png')
    #st.image(image_tendencia, caption='Sunrise by the mountains' ,width=00,clamp=600)

    image4 = Image.open('tendencia_por_pais.png')

    st.image(image4, width=1200,use_column_width='auto')

    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        st.info("escoja Los campos que considere nescesarios para realizar la Tendencia de casos por pais")

        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo casos  o confirmited ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])


        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la prediccion")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')


        pais_Escogido=[pais]
        st.markdown('# Pais escogido:'+pais)

        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        st.write(casos_pais)

        tamanio=casos_pais[var1].__len__()

        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)




        X=np.asarray(arreglo).reshape(-1,1)


        Y=casos_pais[var1]
        st.set_option('deprecation.showPyplotGlobalUse', False)




        reg = LinearRegression()
        reg.fit(X, Y)
        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X, Y, color='red')

        plt.title("TENDENCIA DE CASOS DEL PAIS:"+pais)
        plt.ylabel('CASOS_COVID')
        plt.xlabel('#')


        plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()
        image7 = Image.open('tendenciaa.png')

        st.image(image7, width=1200,use_column_width='auto')
        st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        st.info(reg.coef_)
        if reg.coef_ < 0:
            st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que  debido a las restricciones  los casos de covid 19 han ido disminuyendo considerablemente ')
        else:
            st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa   los casos en este pais  han ido aumentando considerablemente  alo largo de los ultimos reportes ')


        st.markdown('## Grafica Polinomial  ')
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  ')
        st.write('El grado seria ', number)
        X2=np.asarray(arreglo)
        Y2=casos_pais[var1]

        X2=X2[:,np.newaxis]
        Y2=Y2[:,np.newaxis]

        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X2)

        model= LinearRegression()
        model.fit(X_TRANSF,Y2)

        Y_NEW = model.predict(X_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y2,Y_NEW))

        r2=r2_score(Y2,Y_NEW)
        x_new_main=0.0
        x__new_max=50.0

        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()
        plt.xlim(x_new_main,x__new_max)
        plt.ylim(0,1000)
        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Tendecia de casos de COVID-19 en el pais "+pais+title)
        plt.xlabel('#')
        plt.ylabel('Casos de COVID-19')
        plt.show()
        st.pyplot()








def Prediccion_Infectados_Pais(icon='pred.svg'):
    image2 = Image.open('prediccion_pais .png')

    st.image(image2,width=1200,use_column_width='auto')


    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)


        st.info("escoja Los campos que considere nescesarios para realizar la Prediccion")

        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo casos  o confirmited ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la prediccion")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')

        pais_Escogido=[pais]
        st.markdown('# Pais escogido:'+pais)
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')
        st.write('El grado seria ', number)

        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        st.write(casos_pais)
        tamanio=casos_pais[var1].__len__()

        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X=np.asarray(arreglo)
        Y=casos_pais[var1]

        X=X[:,np.newaxis]
        Y=Y[:,np.newaxis]
        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X)

        model= LinearRegression()
        model.fit(X_TRANSF,Y)

        Y_NEW = model.predict(X_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y,Y_NEW))

        r2=r2_score(Y,Y_NEW)
        x_new_main=0.0
        x__new_max=50.0


        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='coral',linewidth=4)
        plt.grid()
        plt.xlim(x_new_main,x__new_max)

        plt.ylim(0,1000)
        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de Infectados en el pais "+pais+title)
        plt.xlabel('#')
        plt.ylabel('Infectados por COVID-19')
        plt.show()
        st.pyplot()







def Prediccion_Muertes_Departamento ():




    image10 = Image.open('prediccion_muerte_departamento.png')

    st.image(image10,width=1200,use_column_width='auto')


    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        st.info("escoja Los campos que considere nescesarios para realizar la Prediccion")

        var = st.selectbox(
        'Seleccione el campo Departamento o Estado ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo Muertes   ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])
        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la prediccion")
        departamento = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')

        departamento_Escogido=[departamento]
        st.markdown('# Departamento escogido:'+departamento)
        number = st.number_input('Inserte el grado  del que desea hacer la grafica prediccion')
        st.write('El grado seria ', number)
        muertes_departamento=dataframe[dataframe[var].isin(departamento_Escogido)]
        st.write(muertes_departamento)
        tamanio=muertes_departamento[var1].__len__()
        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X=np.asarray(arreglo)
        Y=muertes_departamento[var1]

        X=X[:,np.newaxis]
        Y=Y[:,np.newaxis]
        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X)
        model= LinearRegression()
        model.fit(X_TRANSF,Y)

        Y_NEW = model.predict(X_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y,Y_NEW))

        r2=r2_score(Y,Y_NEW)
        x_new_main=0.0
        x__new_max=50.0

        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='green',linewidth=4)
        plt.grid()
        plt.xlim(x_new_main,x__new_max)

        plt.ylim(0,1000)
        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de Muertes  en el Departamento  "+departamento+title)
        plt.xlabel('#')
        plt.ylabel('Muertes por COVID-19')
        plt.show()
        st.pyplot()








def Analisis_Muertes_por_Pais():
    image22 = Image.open('analisis_muerte.png')

    st.image(image22,width=1200,use_column_width='auto')


    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)


        st.info("escoja Los campos que considere nescesarios para realizar el analisis")



        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo muertes  o deaths ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la prediccion")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')


        pais_Escogido=[pais]
        st.markdown('# Pais escogido:'+pais)



        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        st.write(casos_pais)
        muertes_pais=casos_pais[var1].sum()

        st.markdown('## Las muertes en este pais ascienden a la cantidad de :')
        st.write(muertes_pais)

        tamanio=casos_pais[var1].__len__()

        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)




        X=np.asarray(arreglo).reshape(-1,1)


        Y=casos_pais[var1]
        st.set_option('deprecation.showPyplotGlobalUse', False)




        reg = LinearRegression()

        reg.fit(X, Y)
        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X, Y, color='red')

        plt.title("Analisis DE muertes por COVID-19 en el  PAIS:"+pais)
        plt.ylabel('Muertes por COVID-19')
        plt.xlabel('#')



        plt.plot(prediction_space, reg.predict(prediction_space))
        plt.legend(('Muertes por covid-19','Linear Regression'), loc='upper right')

        plt.show()
        st.pyplot()
        st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        ecuacion='Y='+str(reg.coef_)+'X+'+str(reg.intercept_)
        st.info(ecuacion)
        st.info(reg.coef_)

        if reg.coef_ < 0:
            st.info('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que  debido a las restricciones  las muertes por COVID-19  en ese pais han ido  disminuyendo considerablemente ')
        else:
            st.error('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa   las muertes por COVID-19 en este pais han ido en aumento  ')





def Tendencia_casos_Departamento():



    image4 = Image.open('tendencia_covid_departamento.png')

    st.image(image4, width=1200,use_column_width='auto')




    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        st.info("escoja Los campos que considere nescesarios para realizar la Tendencia de casos por departamento")

        var = st.selectbox(
        'Seleccione el campo departamentos o estado segun sea el caso ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo casos  o confirmited ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la prediccion")
        departamento = st.text_input('',placeholder='Escriba al departamento o el estado  al que quiere realizar el analisis')
        deparamento_escogido=[departamento]
        st.markdown('# Estado  escogido:'+departamento)

        casos_departamento=dataframe[dataframe[var].isin(deparamento_escogido)]
        st.write(casos_departamento)
        tamanio=casos_departamento[var1].__len__()
        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)






        X=np.asarray(arreglo).reshape(-1,1)


        Y=casos_departamento[var1]
        st.set_option('deprecation.showPyplotGlobalUse', False)
        reg = LinearRegression()
        reg.fit(X, Y)
        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X, Y, color='cyan')
        plt.title("TENDENCIA DE CASOS DEL ESTADO:"+departamento)
        plt.ylabel('CASOS_COVID')
        plt.xlabel('#')
        plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()
        st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        st.info(reg.coef_)
        if reg.coef_ < 0:
            st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que  debido a las restricciones  los casos de covid 19 han ido disminuyendo considerablemente en este Estado  ')
        else:
            st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa   los casos en este pais  han ido aumentando considerablemente  en este Estado alo largo de los ultimos reportes ')


        st.markdown('## Grafica Polinomial  ')
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  ')
        st.write('El grado seria ', number)
        X2=np.asarray(arreglo)
        Y2=casos_departamento[var1]

        X2=X2[:,np.newaxis]
        Y2=Y2[:,np.newaxis]

        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X2)

        model= LinearRegression()
        model.fit(X_TRANSF,Y2)

        Y_NEW = model.predict(X_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y2,Y_NEW))

        r2=r2_score(Y2,Y_NEW)
        x_new_main=0.0
        x__new_max=50.0

        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()
        plt.xlim(x_new_main,x__new_max)
        plt.ylim(0,1000)
        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Tendecia de casos de COVID-19 en el Estado "+departamento+title)
        plt.xlabel('#')
        plt.ylabel('Casos de COVID-19')
        plt.show()
        st.pyplot()


    image = Image.open('tendenciaa.png')

    st.image(image, caption='Prediccion de Infectados por pais',width=200,use_column_width='auto')



def Prediccion_Muertes_Pais():
    image11 = Image.open('prediccion_muerte_pais.png')

    st.image(image11,width=1200,use_column_width='auto')


    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)


        st.info("escoja Los campos que considere nescesarios para realizar la Prediccion")

        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo muertes  o deaths ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la prediccion")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')

        pais_Escogido=[pais]
        st.markdown('# Pais escogido:'+pais)
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')
        st.write('El grado seria ', number)

        muertes_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        st.write(muertes_pais)
        tamanio=muertes_pais[var1].__len__()

        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X=np.asarray(arreglo)
        Y=muertes_pais[var1]

        X=X[:,np.newaxis]
        Y=Y[:,np.newaxis]
        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X)

        model= LinearRegression()
        model.fit(X_TRANSF,Y)

        Y_NEW = model.predict(X_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y,Y_NEW))

        r2=r2_score(Y,Y_NEW)
        x_new_main=0.0
        x__new_max=50.0

        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()
        plt.xlim(x_new_main,x__new_max)
        plt.ylim(0,1000)
        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de Muertes por Covid en el pais "+pais+title)
        plt.xlabel('#')
        plt.ylabel('Muertes por COVID-19')
        plt.show()
        st.pyplot()



def Tendencia_Vacunancion_Pais():
    image5 = Image.open('tendencia_vacuna_pais.png')

    st.image(image5, width=1200,use_column_width='auto')

    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        st.info("escoja Los campos que considere nescesarios para realizar la Tendencia de Vacunacion por pais")
        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo vacunacion  o vacunated ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])


        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la tendencia de vacunacion")
        pais_v = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')
        pais_Escogido_v=[pais_v]
        st.markdown('# Pais escogido:'+pais_v)

        casos_pais_v=dataframe[dataframe[var].isin(pais_Escogido_v)]
        st.write(casos_pais_v)
        tamanio=casos_pais_v[var1].__len__()
        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)




        X=np.asarray(arreglo).reshape(-1,1)


        Y=casos_pais_v[var1]
        st.set_option('deprecation.showPyplotGlobalUse', False)
        reg = LinearRegression()
        reg.fit(X, Y)
        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X, Y, color='blue')

        plt.title("TENDENCIA DE VACUNACION  DEL PAIS:"+pais_v)
        plt.ylabel('VACUNACION en '+pais_v)
        plt.xlabel('#')
        plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()
        image7 = Image.open('tendenciaa.png')
        st.image(image7, width=1200,use_column_width='auto')
        st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        st.info(reg.coef_)
        if reg.coef_ < 0:
            st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que   el pais  '+pais_v+' no ha logrado mantender una tendencia ascendente con respecto a su cadena de vacunacion   ')
        else:
            st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa      este pais ha logrado  mantener el ritmo en su  programa de vacunacion')


        st.markdown('## Grafica Polinomial de la vacunacion del pais : '+pais_v)
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  ')
        st.write('El grado seria ', number)
        X2=np.asarray(arreglo)
        Y2=casos_pais_v[var1]

        X2=X2[:,np.newaxis]
        Y2=Y2[:,np.newaxis]

        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X2)

        model= LinearRegression()
        model.fit(X_TRANSF,Y2)

        Y_NEW = model.predict(X_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y2,Y_NEW))

        r2=r2_score(Y2,Y_NEW)
        x_new_main=0.0
        x__new_max=50.0

        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()
        plt.xlim(x_new_main,x__new_max)
        plt.ylim(0,1000)
        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Tendecia de Vacunacion de COVID-19 en el pais "+pais_v+title)
        plt.xlabel('#')
        plt.ylabel('Vacunacion de COVID-19')
        plt.show()
        st.pyplot()




def Comparacion_Vacunacion_Pais():
    image12 = Image.open('Comparacion_vacunas.png')

    st.image(image12, width=1200,use_column_width='auto')

    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        st.info("escoja Los campos que considere nescesarios para realizar la Comparacion de Vacunacion entre 2 paises ")
        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo vacunacion  o vacunated ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])
        st.info(" si escogio los campos correctamente  proceda a escoger los paises  para  realizar la comparacion")

        options = st.multiselect(
            'Escoja los paises',
            dataframe[var].drop_duplicates())

        st.write('You selected:', options)

        pais_Escogido=[options[0]]
        pais_Escogido2=[options[1]]
        #st.markdown('##'+pais+"vs"+pais2)

        info_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        info_pais2=dataframe[dataframe[var].isin(pais_Escogido2)]
        #st.write(casos_pais[var],casos_pais[var1],casos_pais[var3])
        vacunacion_pais1=info_pais[var1].sum()

        vacunacion_pais2=info_pais2[var1].sum()


        grafica=pd.DataFrame({
            'vacunacion':[vacunacion_pais1,vacunacion_pais2]
        },  index=[options[0],options[1]]

        )

        st.write(grafica)



        st.bar_chart(grafica)
        col_pais1,col_pais2=st.columns(2)
        with col_pais1:
            st.write("Tendencia de Vacunacion de  "+options[0])
            tamanio=info_pais[var1].__len__()
            arreglo=[]
            for i in range (0,tamanio):
                arreglo.append(i)
                X=np.asarray(arreglo).reshape(-1,1)


            Y=info_pais[var1]
            st.set_option('deprecation.showPyplotGlobalUse', False)
            reg = LinearRegression()
            reg.fit(X, Y)
            prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
            plt.scatter(X, Y, color='blue')

            plt.title("TENDENCIA DE VACUNACION  DEL PAIS:"+options[0])
            plt.ylabel('VACUNACION en '+options[0])
            plt.xlabel('#')
            plt.plot(prediction_space, reg.predict(prediction_space))
            plt.show()
            st.pyplot()
            image7 = Image.open('tendenciaa.png')
            st.image(image7, width=1200,use_column_width='auto')
            st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
            st.info(reg.coef_)
            if reg.coef_ < 0:
                st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que   el pais  '+options[0]+' no ha logrado mantender una tendencia ascendente con respecto a su cadena de vacunacion   ')
            else:
                st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa      este pais ha logrado  mantener el ritmo en su  programa de vacunacion')



        with col_pais2:
            st.write("Tendencia de Vacunacion de"+options[1])
            tamanio=info_pais2[var1].__len__()
            arreglo=[]
            for i in range (0,tamanio):
                arreglo.append(i)
                X=np.asarray(arreglo).reshape(-1,1)


            Y=info_pais2[var1]
            st.set_option('deprecation.showPyplotGlobalUse', False)
            reg = LinearRegression()
            reg.fit(X, Y)
            prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
            plt.scatter(X, Y, color='blue')

            plt.title("TENDENCIA DE VACUNACION  DEL PAIS:"+options[0])
            plt.ylabel('VACUNACION en '+options[0])
            plt.xlabel('#')
            plt.plot(prediction_space, reg.predict(prediction_space))
            plt.show()
            st.pyplot()
            image7 = Image.open('tendenciaa.png')
            st.image(image7, width=1200,use_column_width='auto')
            st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
            st.info(reg.coef_)
            if reg.coef_ < 0:
                st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que   el pais  '+options[0]+' no ha logrado mantender una tendencia ascendente con respecto a su cadena de vacunacion   ')
            else:
                st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa      este pais ha logrado  mantener el ritmo en su  programa de vacunacion')



def Analisis_Comparativo_entre2_pais_contienente():
    image51 = Image.open('comparacion_pais_contienente.png')

    st.image(image51, width=1200,use_column_width='auto')

    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        st.info("escoja Los campos que considere nescesarios para realizar la Tendencia de Vacunacion por pais")
        genre = st.radio(
        "What's your favorite movie genre",
        ('Comedy', 'Drama', 'Documentary'))

        var = st.selectbox(
        'Seleccione el campo pais o continente ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo vacunacion  o vacunated ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])
        var3 = st.selectbox(
        'Seleccione el campo pruebas  ',
        (dataframe.columns))
        opcion2=var3.upper()
        st.write(opcion2)
        st.write(dataframe[var3])
        st.info(" si escogio los campos correctamente proceda a escoger los paises o contienentes para el analisis")

        options = st.multiselect(
            'Escoja los paises  o Continentes',
            dataframe[var].drop_duplicates())

        st.write('You selected:', options)
        pais_Escogido=[options[0]]
        pais_Escogido2=[options[1]]
        pais_Escogido3=[options[2]]
        pais_Escogido4=[options[3]]
        #st.markdown('##'+pais+"vs"+pais2)

        info_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        info_pais2=dataframe[dataframe[var].isin(pais_Escogido2)]
        info_pais3=dataframe[dataframe[var].isin(pais_Escogido3)]
        info_pais4=dataframe[dataframe[var].isin(pais_Escogido4)]

        st.markdown('## Comparativa de Vacunacion entre  los paises o continentes')
        vacunacion_pais1=info_pais[var1].sum()

        vacunacion_pais2=info_pais2[var1].sum()
        vacunacion_pais3=info_pais3[var1].sum()
        vacunacion_pais4=info_pais4[var1].sum()






        grafica=pd.DataFrame({
            'vacunacion':[vacunacion_pais1,vacunacion_pais2,vacunacion_pais3,vacunacion_pais4]
        },  index=[options[0],options[1],options[2],options[3]]

        )
        st.write(grafica)

        st.markdown('## Comparativa de Muertes entre  los paises o continentes')
        muerte_pais1=info_pais[var3].sum()

        vacunacion_pais2=info_pais2[var].sum()
        vacunacion_pais3=info_pais3[var1].sum()
        vacunacion_pais4=info_pais4[var1].sum()
        grafica=pd.DataFrame({
            'vacunacion':[vacunacion_pais1,vacunacion_pais2,vacunacion_pais3,vacunacion_pais4]
        },  index=[options[0],options[1],options[2],options[3]]

        )
        st.write(grafica)




        st.bar_chart(grafica)


def app8():
    st.info('Hello from app 3')


def Comparacion_Infectados_Vacunados_Pais():


    image3 = Image.open('compracion_pais.png')

    st.image(image3, width=1200,use_column_width='auto')

    uploaded_file = st.file_uploader("Para realizar el  archivo  escoja un archivo de preferencia CSV")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()


    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
        string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

        st.info("escoja Los campos que considere nescesarios para realizar la Prediccion")

        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo casos  o confirmited ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        var3 = st.selectbox(
        'Seleccione el campo pruebas  ',
        (dataframe.columns))
        opcion2=var3.upper()
        st.write(opcion2)
        st.write(dataframe[var3])
        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la comparacion")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')

        pais_Escogido=[pais]
        st.markdown('# Pais escogido:'+pais)

        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        #st.write(casos_pais[var],casos_pais[var1],casos_pais[var3])
        casos_confirmados_pais=casos_pais[var1].sum()


        vacunacion_pais=casos_pais[var3].sum()



        grafica=pd.DataFrame({
            'CASOS-COVID-19 vs Pruebas COVID-19':[casos_confirmados_pais,vacunacion_pais]
        },  index=['CASOS DE COVID-19 ','Pruebas de COVID-19']

        )

        st.write(grafica)



        st.bar_chart(grafica)









if options[0] =='Inicio':
    Inicio()
elif options[0] =='Tendencia de Covid por pais':
    Tendencia_Covid_Pais()



