
from scipy.sparse import data
import streamlit as st

import pandas as pd
from PIL import Image

import pandas as pd
from io  import StringIO

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from streamlit.elements import selectbox
image4 = Image.open('inicio.png')

st.image(image4, width=3000,use_column_width='auto')

#st.image(image, caption='Sunrise by the mountains')




video_file = open('efe.mp4', 'rb')
video_bytes = video_file.read()



def Inicio():

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


def Prediccion_Muertes_dia():
    image111 = Image.open('prediccion_casos_dia.png')

    st.image(image111,width=1200,use_column_width='auto')


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


        dias = st.number_input('Inserte numero de dias  para poder realizar la prediccion')
        st.write('numero de dias ', dias)

        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')
        st.write('El grado seria ', number)

        muertes_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        st.write(muertes_pais)
        tamanio=int(dias)
        cont1=0;
        arreglo=[]
        muertes=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        df=pd.DataFrame({
                "muertes":muertes_pais[var1],
                })
        for i in df.itertuples():
            cont1=cont1+1
            if cont1 <= tamanio:
                muertes.append(i.muertes)





        X=np.asarray(arreglo)
        Y=np.asarray(muertes)

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
        x_new_main=0
        x__new_max=400



        X_NEW=np.linspace(x_new_main,x__new_max,200)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de Muertes por Covid en el pais "+pais+title)
        plt.xlabel('#dias')
        plt.ylabel('Muertes por COVID-19')
        plt.show()
        st.pyplot()

def Factores_Muertes():
    mage12 = Image.open('factores_muerte.png')

    st.image(mage12, width=1200,use_column_width='auto')

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
        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar el analisis")
        pais_v = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')
        pais_Escogido_v=[pais_v]
        pais_analisis=dataframe[dataframe[var].isin(pais_Escogido_v)]
        st.info(" proceda a escoger los factores  de muerte para su analisis")
        st.write(pais_analisis)

        options = st.multiselect(
            'Escoja los paises  o Continentes',
            dataframe.columns.drop_duplicates())
        st.write('opciones escogidas')
        st.write(options)

        tamanio_options =options.__len__()
        factor_Arr=[]
        cantidad_arr=[]
        for i in range (0,tamanio_options):
            alv=pais_analisis[options[i]].sum()

            factor_Arr.append(options[i])
            cantidad_arr.append(alv)
        factor=pd.DataFrame({"Factor":factor_Arr,
                "NumerodeMuertes":cantidad_arr,
                })
        st.table(factor)

        tamanio=factor_Arr.__len__()

        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X=np.asarray(arreglo).reshape(-1,1)
        Y=np.asarray(cantidad_arr).reshape(-1,1)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #reg = LinearRegression()
        #reg.fit(X, Y)

        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X[:,0], Y, color='black',cmap='rainbow')

        plt.title("Factores de muertes en el PAIS :"+pais_v)
        plt.ylabel('Numero de muertes  en el  '+pais_v)
        plt.xlabel('#Factores de muertes')
        #plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()


        #st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        # st.info(reg.coef_)
        #  if reg.coef_ < 0:
        #    st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que   el pais  '+pais+' no ha logrado mantender una tendencia ascendente con respecto a su cadena de vacunacion   ')
        #else:
        #    st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa      este pais ha logrado  mantener el ritmo en su  programa de vacunacion')












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

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de Muertes por Covid en el pais "+pais+title)
        plt.xlabel('#')
        plt.ylabel('Muertes por COVID-19')
        plt.show()
        st.pyplot()

def Muertes_Edad():
    image51 = Image.open('muertes_promedio_edad.png')

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

        var2 = st.selectbox(
        'Seleccione el campo muertes   ',
        (dataframe.columns))
        opcion3=var2.upper()
        st.write(opcion3)
        st.write(dataframe[var2])



        var3 = st.selectbox(
        'Seleccione el campo edad   ',
        (dataframe.columns))
        opcion4=var3.upper()
        st.write(opcion4)
        st.write(dataframe[var2])
        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar el promedio")
        pais_v = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')
        pais_Escogido_v=[pais_v]
        data_pais=dataframe[dataframe[var].isin(pais_Escogido_v)]
        casos=[]
        muertes=[]
        edad=[]
        promedio=[]

        st.write(data_pais)

        df_edad=data_pais[var3].drop_duplicates()
        edades=[]


        for i in df_edad.index:
            edades.append(df_edad[i])
        edades.sort(reverse=True)
        pd_Edades=pd.DataFrame({"edades":edades

                })
        for row in pd_Edades.itertuples():
            edad_Calc=[row.edades]
            calc=data_pais[data_pais[var3].isin(edad_Calc)]
            casos.append(calc[var1].sum())
            muertes.append(calc[var2].sum())
            edad.append(row.edades)
            calculo =calc[var2].sum()/calc[var1].sum()
            promedio.append(calculo)








        promedios = pd.DataFrame({"Muertes":muertes,
                "Casos":casos,"Edad":edad,"PromediodeMuertes":promedio
                })

        st.table(promedios)


        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')

        tamanio=edad.__len__()
        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X=np.asarray(promedios.Casos)
        Y=promedios.Edad

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

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Muertes por covid-19  en el pais  "+pais_v+title)
        plt.xlabel('#casos de covid 19')
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

        st.markdown('## Grafica Polinomial de la vacunacion del pais : '+pais_v)
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')

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

        var2 = st.selectbox(
        'Seleccione el campo casos  ',
        (dataframe.columns))
        opcion3=var2.upper()
        st.write(opcion3)
        st.write(dataframe[var2])
        var3 = st.selectbox(
        'Seleccione el campo pruebas  ',
        (dataframe.columns))
        opcion4=var3.upper()
        st.write(opcion4)
        st.write(dataframe[var3])

        var4 = st.selectbox(
        'Seleccione el campo muertes  ',
        (dataframe.columns))
        opcion5=var4.upper()
        st.write(opcion5)
        st.write(dataframe[var4])

        st.info(" si escogio los campos correctamente proceda a escoger los paises o contienentes para el analisis")

        options = st.multiselect(
            'Escoja los paises  o Continentes',
            dataframe[var].drop_duplicates())
        pais=[]
        casos=[]
        pruebas=[]
        muertes=[]
        vacunacion=[]
        tamanio=options.__len__()
        for i in range (0,tamanio):
            pais_Escogido=[options[i]]

            lugar=dataframe[dataframe[var].isin(pais_Escogido)]


            pais.append(options[i])
            casos.append(lugar[var2].sum())
            pruebas.append(lugar[var3].sum())
            muertes.append(lugar[var4].sum())
            vacunacion.append(lugar[var1].sum())

            compracion = pd.DataFrame({"Lugar":pais,
                "CasosCovid-19":casos,"Pruebas":pruebas,"Vacunas":vacunacion,"Muertes":muertes
                })


        st.write(compracion)
        st.info('Si usted lo desea puede ver la tendencia de vacunacion, tendencia muertes y la prediccion de casos  para el lugar que escogio ')
        pais_v = st.text_input('',placeholder='Escriba al pais o lugar al que quiere realizar el analisis')
        pais_Escogido_v=[pais_v]
        st.markdown('# Pais escogido:'+pais_v)

        casos_pais_v=dataframe[dataframe[var].isin(pais_Escogido_v)]

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

        plt.title("TENDENCIA DE VACUNACION  en:"+pais_v)
        plt.ylabel('VACUNACION en '+pais_v)
        plt.xlabel('#')
        plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()
        st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        st.info(reg.coef_)
        if reg.coef_ < 0:
            st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que   el lugar  '+pais_v+' no ha logrado mantender una tendencia ascendente con respecto a su cadena de vacunacion   ')
        else:
            st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa      este lugar ha logrado  mantener el ritmo en su  programa de vacunacion')








        X2=np.asarray(arreglo).reshape(-1,1)


        Y2=casos_pais_v[var4]
        st.set_option('deprecation.showPyplotGlobalUse', False)
        reg = LinearRegression()
        reg.fit(X2, Y2)
        prediction_space = np.linspace(min(X2), max(X2)).reshape(-1, 1)
        plt.scatter(X2, Y2, color='red')

        plt.title("TENDENCIA DE Muertes  en:"+pais_v)
        plt.ylabel('Muertes en '+pais_v)
        plt.xlabel('#')
        plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()
        st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        st.info(reg.coef_)
        if reg.coef_ < 0:
            st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que   el pais  '+pais_v+'  ha logrado reducir las muertes en gran manera    ')
        else:
            st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa     este lugar no ha logrado reducir las muertes ')

        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')
        st.write('El grado seria ', number)



        X3=np.asarray(arreglo)
        Y3=casos_pais_v[var2]

        X3=X3[:,np.newaxis]
        Y3=Y3[:,np.newaxis]
        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X3)

        model= LinearRegression()
        model.fit(X_TRANSF,Y3)

        Y_NEW = model.predict(X_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y3,Y_NEW))

        r2=r2_score(Y,Y_NEW)
        x_new_main=0.0
        x__new_max=50.0


        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='coral',linewidth=4)
        plt.grid()



        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de Infectados en  "+pais_v+title)
        plt.xlabel('#')
        plt.ylabel('Infectados por COVID-19')
        plt.show()
        st.pyplot()




        #st.bar_chart(grafica)
def indice_progresion():
    image33 = Image.open('indice_progresion.png')

    st.image(image33, width=1200,use_column_width='auto')

    uploaded_file = st.file_uploader("Para realizar    analisis el archivo  escoja un archivo de preferencia CSV")
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

        st.info("escoja Los campos que considere nescesarios para realizar la tasa de muertes por COVID-19")
        var = st.selectbox(
        'Seleccione el casos ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        dias = st.number_input('Inserte numero de dias  para poder  ver la progresion de la pandemia')
        st.write('numero de dias ', dias)
        tamanio=int(dias)
        arreglo=[]
        casosar=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        casos=dataframe[var]

        df=pd.DataFrame({
            "casos":dataframe[var],
                })
        st.write(df)
        cont1=0
        total_casos=0
        for i in df.itertuples():
            cont1=cont1+1
            if cont1 <= tamanio:
                casosar.append(i.casos)
                total_casos=int(i.casos)+total_casos

        st.info('En '+str(tamanio)+' dias han habido '+str(total_casos)+' casos de COVID-19')

        number = st.number_input('Inserte el grado  del que desea hacer la grafica de casos de covid-19')
        st.write('El grado seria ', number)

        X=np.asarray(arreglo)
        Y=np.asarray(casosar)

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
        x_new_main=0
        x__new_max=400



        X_NEW=np.linspace(x_new_main,x__new_max,200)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Progresion de la pandemia de COVID-19"+title)
        plt.xlabel('#dias')
        plt.ylabel('Casos de  COVID-19')
        plt.show()
        st.pyplot()



def Tasa_Mortalidad_Pais():
    image33 = Image.open('tasa_muerte_covid.png')

    st.image(image33, width=1200,use_column_width='auto')

    uploaded_file = st.file_uploader("Para realizar    analisis el archivo  escoja un archivo de preferencia CSV")
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

        st.info("escoja Los campos que considere nescesarios para realizar la tasa de muertes por COVID-19")
        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo casos   ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        var2 = st.selectbox(
        'Seleccione el campo muertes  o death ',
        (dataframe.columns))
        opcion3=var2.upper()
        st.write(opcion3)
        st.write(dataframe[var2])
        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la tasa de mortalidad por covid-19")
        pais_v = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')
        pais_Escogido_v=[pais_v]
        st.markdown('# Pais escogido:'+pais_v)
        casos_pais_v=dataframe[dataframe[var].isin(pais_Escogido_v)]

        casos_al_Dia=casos_pais_v[var1].sum()
        muertes_al_Dia=casos_pais_v[var2].sum()
        st.markdown('## Casos por COVID 19 en el  PAIS '+pais_v)
        st.write(casos_al_Dia)
        st.markdown('## Muertes por COVID 19 en el  PAIS '+pais_v)
        st.write(muertes_al_Dia)
        tasa=muertes_al_Dia/casos_al_Dia
        st.write(tasa*100)
        st.info('Segun los datos obtenidos durante el analisis se obtuvo que el numero de casos de COVID-19 en el pais '+pais_v+'asciende a la cifra de '+str(casos_al_Dia)+' y lastimosamente la cifra de fallecidos asciende a la cantidad de '+ str(muertes_al_Dia)+' y para calcular la tasa de mortalidad se hace uso de la fórmula de Tasa de Mortalidad por Infección (IFR) = Muertes / Casos  que en este caso   seria: '+str(round(tasa*100,2))+'%')

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
        plt.scatter(X, Y, color='cyan')
        plt.title("TENDENCIA DE CASOS DEL ESTADO:"+pais_v)
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



        tamanio2=casos_pais_v[var2].__len__()
        arreglo2=[]
        for i in range (0,tamanio2):
            arreglo2.append(i)






        X=np.asarray(arreglo2).reshape(-1,1)


        Y=casos_pais_v[var2]
        st.set_option('deprecation.showPyplotGlobalUse', False)
        reg = LinearRegression()
        reg.fit(X, Y)
        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X, Y, color='cyan')
        plt.title("TENDENCIA DE MUERTE DEL ESTADO:"+pais_v)
        plt.ylabel('MUERTE POR COVID-19')
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
def prediccion_mundial():
    image113 = Image.open('predicciones_mundiales.png')

    st.image(image113, width=1200,use_column_width='auto')

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
        'Seleccione el campo o cases ',
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


        st.markdown('## Predicion de casos de COVID-19 en todo el mundo')
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')
        st.write('El grado seria ', number)
        tamanio=dataframe[var].__len__()
        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X=np.asarray(arreglo)
        Y=dataframe[var]

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

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de   casos de COVID-19 alrededor del mundo "+title)
        plt.xlabel('#')
        plt.ylabel('Casos de COVID-19')
        plt.show()
        st.pyplot()



        st.markdown('## Predicion de muertes por de COVID-19 en todo el mundo')
        tamanio=dataframe[var1].__len__()
        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X2=np.asarray(arreglo)
        Y2=dataframe[var1]

        X2=X2[:,np.newaxis]
        Y2=Y2[:,np.newaxis]
        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF2=polynomial_features.fit_transform(X2)

        model= LinearRegression()
        model.fit(X_TRANSF2,Y2)

        Y_NEW2 = model.predict(X_TRANSF2)
        rmse=np.sqrt(mean_squared_error(Y2,Y_NEW2))

        r2=r2_score(Y2,Y_NEW2)
        x_new_main=0.0
        x__new_max=50.0

        X_NEW2=np.linspace(x_new_main,x__new_max,50)

        X_NEW2=X_NEW2[:,np.newaxis]
        X_NEW_TRANSF2 =polynomial_features.fit_transform(X_NEW2)

        Y_NEW2=model.predict(X_NEW_TRANSF2)

        plt.plot(X_NEW2,Y_NEW2,color='red',linewidth=4)
        plt.grid()

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de   muertes de COVID-19 alrededor del mundo "+title)
        plt.xlabel('#')
        plt.ylabel('Muertes de COVID-19')
        plt.show()
        st.pyplot()


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

        df=pd.DataFrame({"casos":casos_pais[var1].drop_duplicates(),
        "pruebas":casos_pais[var3],
                })
        input_data = []

        for item in df.itertuples():
        #new tests

            if pd.isnull(item.casos)==False:
                input_data.append([item.casos, item.pruebas])



        st.write(input_data)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X = np.array(input_data)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)

        plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
        plt.title("Compracion entre  el numero de casos detectados y el numero de pruebas del pais :"+pais)
        plt.ylabel('pruebas de COVID-19 ')
        plt.xlabel('Numero de contagios  de COVID-19')
#comment this line to get only the points
        plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black')

        plt.show()
        st.pyplot()
        st.info('Para este analisis se ha decidido dividir  los datos en 3 regiones   para mostrar de una mejor manera los datos  la region 1 es de color morado , la region 2 es colorrojo y la region 3 es de color cyan esto se hizo para observar de una mejor manera la comparacion entre los casos de covid 19 y las puebtas que ha realizao este pais ')





        casos_confirmados_pais=casos_pais[var1].sum()


        vacunacion_pais=casos_pais[var3].sum()



        grafica=pd.DataFrame({
            'CASOS-COVID-19 vs Pruebas COVID-19':[casos_confirmados_pais,vacunacion_pais]
        },  index=['CASOS DE COVID-19 ','Pruebas de COVID-19']

        )

        st.write(grafica)



        st.bar_chart(grafica)
def Tendencia_Infectados_dia():
    image1121 = Image.open('tendencia_dia.png')

    st.image(image1121,width=1200,use_column_width='auto')


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
        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        st.write(casos_pais)
        st.markdown('# Pais escogido:'+pais)
        dias = st.number_input('Inserte numero de dias  para poder realizar la prediccion')
        st.write('numero de dias ', dias)
        tamanio=int(dias)
        cont1=0;
        arreglo=[]
        casos=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        df=pd.DataFrame({
                "casos":casos_pais[var1],
                })
        for i in df.itertuples():
            cont1=cont1+1
            if cont1 <= tamanio:
                casos.append(i.casos)


        X=np.asarray(arreglo).reshape(-1,1)


        Y=np.asarray(casos).reshape(-1,1)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        reg = LinearRegression()
        reg.fit(X, Y)
        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X, Y, color='blue')

        plt.title("TENDENCIA DE VACUNACION  DEL PAIS:"+pais)
        plt.ylabel('VACUNACION en '+pais)
        plt.xlabel('#')
        plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()
        image7 = Image.open('tendenciaa.png')
        st.image(image7, width=1200,use_column_width='auto')
        st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        st.info(reg.coef_)
        if reg.coef_ < 0:
            st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que   el pais  '+pais+' no ha logrado mantender una tendencia ascendente con respecto a su cadena de vacunacion   ')
        else:
            st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa      este pais ha logrado  mantener el ritmo en su  programa de vacunacion')

        st.markdown('## Grafica Polinomial de la vacunacion del pais : '+pais)
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  ')
        st.write('El grado seria ', number)
        X2=np.asarray(arreglo)
        Y2=np.asarray(casos)

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
        plt.title("Tendecia de casos   de COVID-19 por dia en el pais "+pais+title)
        plt.xlabel('#')
        plt.ylabel('Casos de COVID-19')
        plt.show()
        st.pyplot()

def Comportamiento_Casos_Municipio():
    image44 = Image.open('comportamiento_personas_covid.png')

    st.image(image44,width=1200,use_column_width='auto')


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


        st.info("escoja Los campos que considere nescesarios para obtener el porcentaje")


        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo municipio ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        var2 = st.selectbox(
        'Seleccione el campo casos  o confirm ',
        (dataframe.columns))
        opcion3=var2.upper()
        st.write(opcion3)
        st.write(dataframe[var2])



        var3 = st.selectbox(
        'Seleccione el campo hombres ',
        (dataframe.columns))
        opcion4=var3.upper()
        st.write(opcion4)
        st.write(dataframe[var3])



        var4 = st.selectbox(
        'Seleccione el campo mujeres ',
        (dataframe.columns))
        opcion5=var4.upper()
        st.write(opcion5)
        st.write(dataframe[var4])


        st.info(" si escogio los campos correctamente  proceda a escoger el pais para   poder ver el comportamiento")
        pais = st.text_input('',placeholder='Escriba al pais   al que quiere realizar el analisis')
        pais_Escogido_v=[pais]
        data_pais=dataframe[dataframe[var].isin(pais_Escogido_v)]

        casos=[]
        mujeres=[]
        hombres=[]
        municipios=[]


        pd_municipios=pd.DataFrame({"municipios":data_pais[var1].drop_duplicates()

            })

        for row in pd_municipios.itertuples():
            municipio=[row.municipios]

            calc=data_pais[data_pais[var1].isin(municipio)]
            municipios.append(row.municipios)
            casos.append(calc[var2].sum())
            mujeres.append(calc[var4].sum())
            hombres.append(calc[var3].sum())


        comportamiento_muni = pd.DataFrame({"Municipio":municipios,
                "Casos":casos,"Hombres":hombres,"Mujeres":mujeres
                })

        st.table(comportamiento_muni)
        tamanio=municipios.__len__()
        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X=np.asarray(arreglo)
        Y=comportamiento_muni.Casos
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')

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

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Comportamiento de casos de COVID-19 "+pais+title)
        plt.xlabel('#Municipio')
        plt.ylabel('casos de COVID-19')
        plt.show()
        st.pyplot()










def prediccion_ultimo_dia():
    image111 = Image.open('prediccion_ultimo_dia.png')

    st.image(image111,width=1200,use_column_width='auto')


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
        'Seleccione el campo muertes  ',
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
        tamanio=365
        cont1=0;
        arreglo=[]
        casosar=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        df=pd.DataFrame({
                "casos":casos_pais[var1],
                })
        for i in df.itertuples():
            cont1=cont1+1
            if cont1 <= tamanio:
                casosar.append(i.casos)



        X=np.asarray(arreglo)
        Y=np.asarray(casosar)

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
        x_new_main=0
        x__new_max=400



        X_NEW=np.linspace(x_new_main,x__new_max,200)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de casos  en el primer año por COVID-19 "+pais+title)
        plt.xlabel('#dias')
        plt.ylabel('CASOS COVID-19')
        plt.show()
        st.pyplot()




def prediccion_casos_anio():
    image111 = Image.open('prediccion_casos_año.png')

    st.image(image111,width=1200,use_column_width='auto')


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
        'Seleccione el campo casos  ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  realizar la prediccion")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')

        pais_Escogido=[pais]

        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]

        st.markdown('# Pais escogido:'+pais)



        anio = st.number_input('Inserte el anio que desea ver  prediccion')
        st.write('El año seria ', anio)
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')
        st.write('El grado seria ', number)

        arreglo=[]
        casosar=[]
        df=pd.DataFrame({
                "casos":casos_pais[var1],
                })


        if anio > 1:
            cont1=365
            tamanio=casos_pais[var1].__len__()
            for i in range (365,tamanio):
                arreglo.append(i)

            for i in df.itertuples():
                cont1=cont1+1
                if cont1 <= tamanio:
                    casosar.append(i.casos)
        else:
            cont1=0
            tamanio=casos_pais[var1].__len__()
            for i in range (0,tamanio):
                arreglo.append(i)

            for i in df.itertuples():
                cont1=cont1+1
                if cont1 <= tamanio:
                    casosar.append(i.casos)





        X=np.asarray(arreglo)
        Y=np.asarray(casosar)

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
        x_new_main=0
        x__new_max=400



        X_NEW=np.linspace(x_new_main,x__new_max,200)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()

        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Prediccion de casos  en el año "+str(anio)+"para el pais "+pais+title)
        plt.xlabel('#dias')
        plt.ylabel('CASOS COVID-19')
        plt.show()
        st.pyplot()


def Porcentaje_Hombres_Covid():
    image44 = Image.open('porcentaje_hombres.png')

    st.image(image44,width=1200,use_column_width='auto')


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


        st.info("escoja Los campos que considere nescesarios para obtener el porcentaje")


        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo hombres  o men ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        var2 = st.selectbox(
        'Seleccione el campo casos  o confirm ',
        (dataframe.columns))
        opcion3=var2.upper()
        st.write(opcion3)
        st.write(dataframe[var2])


        st.info(" si escogio los campos correctamente  proceda a escoger el pais para  encontrar el porcentaje")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')


        pais_Escogido=[pais]
        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        st.write(casos_pais)
        st.markdown('# Pais escogido:'+pais)

        casos=casos_pais[var2].sum()
        hombres=casos_pais[var1].sum()
        porcentaje_hombres=hombres/casos
        st.info('Se analizaron los datos   y se encontro que la cantidad de muertes en el pais '+pais+' asciende a la cantidad de '+str(casos)+' y la cantidad de hombres infectados a la fecha asciende a la cantidad de '+str(hombres)+' eso nos da como resultado  que el porcentaje de hombres contagiados  respecto al total de casos del  pais escogido es de '+str(round(porcentaje_hombres*100,2))+'%')
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  ')
        st.write('El grado seria ', number)


        tamanio=casos_pais[var1].__len__()
        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)



        X2=np.asarray(arreglo)
        Y2=casos_pais[var2]
        prueba=casos_pais[var1]
        X2=X2[:,np.newaxis]
        Y2=Y2[:,np.newaxis]
        prueba=prueba[:,np.newaxis]


        nb_degree=int(number)
        polynomial_features=PolynomialFeatures(degree=nb_degree)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        X_TRANSF=polynomial_features.fit_transform(X2)
        prueba_TRANSF=polynomial_features.fit_transform(prueba)

        model= LinearRegression()
        model.fit(X_TRANSF,Y2)

        Y_NEW = model.predict(X_TRANSF)
        prueba=new=model.predict(prueba_TRANSF)
        rmse=np.sqrt(mean_squared_error(Y2,Y_NEW))

        r2=r2_score(Y2,Y_NEW)
        x_new_main=0.0
        x__new_max=tamanio

        X_NEW=np.linspace(x_new_main,x__new_max,50)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X_NEW)
        plt.scatter(X_NEW,Y_NEW, cmap='rainbow' )

        plt.grid()
        plt.xlim(x_new_main,x__new_max)
        plt.ylim(0,10000)
        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("CAsOS  de COVID-19 en el pais "+pais+title)
        plt.xlabel('#dias')
        plt.ylabel('CASOS de COVID-19')
        plt.show()
        st.pyplot()

def porcentaje_muertes_p():
    image442 = Image.open('porcentaje_muertes_region.png')

    st.image(image442,width=1200,use_column_width='auto')


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


        st.info("escoja Los campos que considere nescesarios para obtener el porcentaje")
        var = st.selectbox(
        'Seleccione el campo Country o pais ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])

        var1 = st.selectbox(
        'Seleccione el campo muertes  o death ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])

        var2 = st.selectbox(
        'Seleccione el campo casos  o confirm ',
        (dataframe.columns))
        opcion3=var2.upper()
        st.write(opcion3)
        st.write(dataframe[var2])
        st.info(" si escogio los campos correctamente  procesada escribir el pais, la region o el continente para  encontrar el porcentaje")
        pais = st.text_input('',placeholder='Escriba al pais,region o continente al que quiere realizar el analisis')
        pais_Escogido=[pais]
        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        st.write(casos_pais)
        st.markdown('# Usted escogio :'+pais)

        casos=casos_pais[var2].sum()
        muertes=casos_pais[var1].sum()
        porcentaje_hombres=muertes/casos
        st.info('  Al analizar los datos se encuentro que por desgracia la cantidad de muertes en  asicende a la cantidad  de '+str(muertes) +' y  la cantidad de casos en '+pais+'asciende a la cantidad de '+str(casos) +'eso quiere decir  que el porcentaje de muertes  con respecto a la cantidad de casos en ' + pais+ ' es  de '+str(round(porcentaje_hombres*100,2))+'%')


        X=np.asarray(casos_pais[var2]).reshape(-1,1)
        Y=casos_pais[var1]
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #reg = LinearRegression()
        #reg.fit(X, Y)

        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X[:,0], Y, color='blue',cmap='rainbow')

        plt.title("Muertes por COVID-19 vs el numero de casos por COVID-19 en:"+pais)
        plt.ylabel('Numero de muertes  en  '+pais)
        plt.xlabel('Casos por COVID-19')
        #plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()
def Tasa_Comportamiento_Muertes_Covid():
    image122 = Image.open('tasa_comportamiento_casos_muertes.png')

    st.image(image122, width=1200,use_column_width='auto')

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

        st.info("escoja Los campos que considere nescesarios para realizar la Tasa de comportamiento  ")
        var = st.selectbox(
        'Seleccione el campo Contienente  ',
        (dataframe.columns))
        opcion1=var.upper()
        st.write(opcion1)
        st.write(dataframe[var])
        var1 = st.selectbox(
        'Seleccione el campo casos ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])
        var2 = st.selectbox(
        'Seleccione el campo muertes  o death ',
        (dataframe.columns))
        opcion2=var2.upper()
        st.write(opcion2)
        st.write(dataframe[var2])
        st.info(" si escogio los campos correctamente  proceda a escoger el pais  analizar las regiones")
        continente = st.text_input('',placeholder='Escriba el contienente al que quiere realizar el analisis')

        continente_escogido=[continente]
        st.markdown('# Continente escogido:'+continente)

        data_continente=dataframe[dataframe[var].isin(continente_escogido)]
        st.write(data_continente)

        casos=data_continente[var1].sum()
        muertes=data_continente[var2].sum()
        tasa=casos/muertes
        st.markdown('### Casos Activos :'+str(casos))
        st.markdown('### Muertes  :'+str(muertes))
        st.info('Al analizar tanto los casos activos en el continente '+continente+'  como la cantidad de muertes  se ha logrado determinar que la tasa de comportamiento es '+str(round(tasa,2))+' al obtener esta tasa podemos sacar una unica conclusion que entre menor sea la tasa es peor para el pais ya que es mas probable que mueran mas personas mientras que si la tasa presenta un valor mayor  habran en teoria menos muertes en el continente ')
        st.markdown('## Grafica de comportamiento ')
        number = st.number_input('Inserte el grado  del que desea hacer la grafica  prediccion')
        st.write('El grado seria ', number)
        X=np.asarray(data_continente[var2])
        Y=data_continente[var1]

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
        x_new_main=0
        x__new_max=1000



        X_NEW=np.linspace(x_new_main,x__new_max,200)

        X_NEW=X_NEW[:,np.newaxis]
        X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

        Y_NEW=model.predict(X_NEW_TRANSF)

        plt.plot(X_NEW,Y_NEW,color='red',linewidth=4)
        plt.grid()


        title='Grado={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
        plt.title("Tasa de comportamiento de casos activos de COVID-19 en relacion a las muertes en el continente de  "+continente+title)
        plt.ylabel('#Casos Activos')
        plt.xlabel('Muertes por COVID-19')
        plt.show()
        st.pyplot()


def Muertes_por_Region():
    image12 = Image.open('muertes_region.png')

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
        'Seleccione el campo estado  o departamento ',
        (dataframe.columns))
        opcion2=var1.upper()
        st.write(opcion2)
        st.write(dataframe[var1])
        var2 = st.selectbox(
        'Seleccione el campo muertes  o death ',
        (dataframe.columns))
        opcion2=var2.upper()
        st.write(opcion2)
        st.write(dataframe[var2])
        st.info(" si escogio los campos correctamente  proceda a escoger el pais  analizar las regiones")
        pais = st.text_input('',placeholder='Escriba al pais al que quiere realizar el analisis')

        pais_Escogido=[pais]
        st.markdown('# Pais escogido:'+pais)

        casos_pais=dataframe[dataframe[var].isin(pais_Escogido)]
        df=pd.DataFrame({"regiones":casos_pais[var1].drop_duplicates(),
                "muertes":casos_pais[var2],
                })


        cont=0
        regiones=[]
        muertesAR=[]

        for i in df.itertuples():


            if pd.isnull(i.regiones)==False:
                alv=[i.regiones]
                regiones.append(i.regiones)


                alv2=dataframe[dataframe[var1].isin(alv)]
                muertes=alv2[var2].sum()
                muertesAR.append(muertes)


        Muertes_REGION=pd.DataFrame({"regiones":regiones,
                "muertes":muertesAR,
                })











        st.table(Muertes_REGION)
        tamanio=casos_pais[var1].__len__()

        arreglo=[]
        for i in range (0,tamanio):
            arreglo.append(i)
        X=np.asarray(arreglo).reshape(-1,1)
        Y=Muertes_REGION.muertes
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #reg = LinearRegression()
        #reg.fit(X, Y)

        prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
        plt.scatter(X[:,0], Y, color='blue',cmap='rainbow')

        plt.title("Muertes por regiones del PAIS:"+pais)
        plt.ylabel('Numero de muertes  en el  '+pais)
        plt.xlabel('#REGIONES')
        #plt.plot(prediction_space, reg.predict(prediction_space))
        plt.show()
        st.pyplot()

        #st.markdown('### Analizando la grafica  se encontro que la prendiente de la grafica mostrada es:')
        # st.info(reg.coef_)
        #  if reg.coef_ < 0:
        #    st.error('La pendiente de una recta es negativa cuando la recta es decreciente , es decir que   el pais  '+pais+' no ha logrado mantender una tendencia ascendente con respecto a su cadena de vacunacion   ')
        #else:
        #    st.info('La pendiente de una recta es positiva cuando la recta es creciente, es decir que a diferencia de una pendiente negativa      este pais ha logrado  mantener el ritmo en su  programa de vacunacion')






op = st.multiselect(
    'Bienvenido escoja la opcion que desea ',
        ['Inicio☄️', 'Tendencia de Covid por pais📈','Inidice de Progresión de la pandemia.🦠','Predicción de Infertados en un País🧮','Predicción de mortalidad por COVID en un Departamento🧮','Análisis del número de muertes por coronavirus en un País☠️'
            ,'Tendencia de la vacunación de en un País💉📈','Tendencia de casos confirmados de Coronavirus en un departamento de un País📈'
        ,'Predicción de mortalidad por COVID en un Pais🧮','Muertes promedio por casos confirmados y edad de covid 19 en un País☠️','Ánalisis Comparativo entres 2 o más paises o continentes🌎','Ánalisis Comparativo de Vacunación entre 2 paises💉'
        ,'Muertes según regiones de un país - Covid 19☠️','Predicciones de casos y muertes en todo el mundo🧮','Tasa de comportamiento de casos activos en relación al número de muertes en un continente☠️📈🦠','Tasa de mortalidad por coronavirus (COVID-19) en un país📈☠️'
        ,'Predicción de casos de un país para un año🧮','Predicción de casos confirmados por día🧮', 'Predicción de muertes en el último día del primer año de infecciones en un país.☠️','Comportamiento y clasificación de personas infectadas por COVID-19 por municipio en un País.🦠','Factores de muerte por COVID-19 en un país.☠️','Porcentaje de muertes frente al total de casos en un país, región o continente.%📶☠️🌎','Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo🙍🏻‍♂️' ,'Tendencia del número de infectados por día de un País.🗓️📈',
        'Comparación entre el número de casos detectados y el número de pruebas de un país 💊💉'])


st.write('You selected:', op)



#Tasa de comportamiento de casos activos en relación al número de muertes en un continente.

if len(op)>0:
    if op[0] =='Inicio☄️':
        Inicio()
    elif op[0] =='Tendencia de Covid por pais📈':
        Tendencia_Covid_Pais()
    elif op[0] =='Predicción de Infectados en un País🧮':
        Prediccion_Infectados_Pais()
    elif op[0]=='Predicción de mortalidad por COVID en un Departamento🧮':
        Prediccion_Muertes_Departamento()
    elif op[0]=='Predicción de mortalidad por COVID en un Pais🧮':
        Prediccion_Muertes_Pais()

    elif op[0]=='Análisis del número de muertes por no en un País☠️':
        Analisis_Muertes_por_Pais()

    elif op[0]=='Tendencia de la vacunación de en un País💉📈':
        Tendencia_Vacunancion_Pais()
    elif op[0]=='Ánalisis Comparativo de Vacunación entre 2 paises💉':
        Comparacion_Vacunacion_Pais()


    elif op[0]=='Tendencia de casos confirmados de Coronavirus en un departamento de un País📈':
        Tendencia_casos_Departamento()

    elif op[0]=='Ánalisis Comparativo entres 2 o más paises o continentes🌎':
        Analisis_Comparativo_entre2_pais_contienente()

    elif op[0]=='Muertes según regiones de un país - Covid 19☠️':
        Muertes_por_Region()

    elif op[0]=='Predicción de casos confirmados por día🧮':

        Prediccion_Muertes_dia()
    elif op[0]=='Tendencia del número de infectados por día de un País.🗓️📈':
        Tendencia_Infectados_dia()
    elif op[0]=='Comparación entre el número de casos detectados y el número de pruebas de un país 💊💉':
        Comparacion_Infectados_Vacunados_Pais()
    elif op[0]=='Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo🙍🏻‍♂️':
        Porcentaje_Hombres_Covid()
    elif op[0]=='Porcentaje de muertes frente al total de casos en un país, región o continente.%📶☠️🌎':
        porcentaje_muertes_p()

    elif op[0]=='Tasa de comportamiento de casos activos en relación al número de muertes en un continente☠️📈🦠':
        Tasa_Comportamiento_Muertes_Covid()
    elif op[0]=='Factores de muerte por COVID-19 en un país.☠️':
        Factores_Muertes()
    elif op[0]=='Predicciones de casos y muertes en todo el mundo🧮':
        prediccion_mundial()


    elif op[0]=='Muertes promedio por casos confirmados y edad de covid 19 en un País☠️':
        Muertes_Edad()
    elif op[0]=='Inidice de Progresión de la pandemia.🦠':
        indice_progresion()

    elif op[0]=='Comportamiento y clasificación de personas infectadas por COVID-19 por municipio en un País.🦠':
        Comportamiento_Casos_Municipio()
    elif op[0]=='Predicción de muertes en el último día del primer año de infecciones en un país.☠️':
        prediccion_ultimo_dia()

    elif op[0]=='Predicción de casos de un país para un año🧮':
        prediccion_casos_anio()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Anlisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()
    elif op[0]=='Análisis del número de muertes por coronavirus en un País☠️':
        Analisis_Muertes_por_Pais()


#Tasa de mortalidad por coronavirus (COVID-19) en un país.