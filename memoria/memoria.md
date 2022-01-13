# BodyPerfomance Classification
| Autores|email|
|-|-|
|Clara Daniela Sima|csima@ucm.es|
|Stiven Arias Giraldo|starias@ucm.es|

Resultados del proyecto final de la asignatura de **Aprendizaje Automático y Minería de Datos** del grado de **Desarrollo de videojuegos** de la **Universidad Complutense de Madrid**

# Descripción
La base de datos que hemos escogido está compuesta por múltiples atributos que describen características físicas de diversas personas. Por ser más concretos, contamos con 13393 ejemplos de entrenamiento, es decir, 13393 personas, de las cuales tenemos 11 atributos y un resultado para cada una. 

Esto quiere decir que nuestro dataset está compuesto por una matriz de (13393, 12) dimensiones, donde la última columna nos indica la calificación de un _entrenamiento físico_ de cada persona en función del resto de atributos, pudiendo ser **A, B, C, D** dicha calificación, siendo **A** el mejor resultado posible.

![image](https://user-images.githubusercontent.com/47497948/149372136-b6356d03-dd10-4deb-a039-9ddb2182d937.png)

![image](https://user-images.githubusercontent.com/47497948/149371625-e32eefcb-1069-4653-bc3d-71cfeb01bf75.png)

Así pues, la idea principal de este proyecto es aprovechar los diferentes algoritmos de aprendizaje automático realizados durante el curso, procesando los diferentes datos para poder determinar qué sistema de aprendizaje resulta más óptimo para clasificar correctamente el _dataset_.
Cada uno de los sistemas tendrán que predecir cuál ha sido el grado de eficiencia del usuario (A, B, C o D) para finalmente comparar dicha predicción con los datos reales.

<!--- Nombrar aquí los diferentes sistemas que se usarán -->

# Selección de los datos de entrenamiento

En primer lugar, hemos separado los datos en 3 grupos: entrenamiento, validación y _testing_ donde la proporción de cada grupo es 60%, 20% y 20% del número total de datos respectivamente.

```py
def selectingData(allX, allY):
    """
    Seleccion de los datos de entrenamiento, 
    de validación y de pruebas
    """
    # Se cogerá un 60% de los datos para entrenar
    X = normalizeMatrix(allX[:int(0.6 * np.shape(allX)[0])])
    y = allY[:int(0.6 * np.shape(allY)[0])]

    # Después se coge un 20% para evaluar
    Xval = normalizeMatrix(allX[int(0.6 * np.shape(allX)[0]) : int(0.8 * np.shape(allX)[0])])
    yval = allY[int(0.6 * np.shape(allY)[0]) : int(0.8 * np.shape(allX)[0])]

    # Por último, el 20% resrtante para testing
    Xtest = normalizeMatrix(allX[int(0.8 * np.shape(allX)[0]):])
    ytest = allY[int(0.8 * np.shape(allX)[0]):]

    return X, y, Xval, yval, Xtest, ytest

def normalizeMatrix(X):
    """
    Normaliza el array X
    xi = (xi - ui) / si
    """
    matriz_normal = np.empty_like(X)

    # La media y la varianza de cada columna
    u = np.mean(X, axis=0)
    s = np.std(X, axis=0)

    matriz_normal = (X - u) / s
    return matriz_normal
```

En los valores **_X_** guardamos los atributos de cada ejemplo de entrenamiento y en los valores **_Y_**, los resultados.

La idea de esta distinción de valores es utilizar los **datos de entrenamiento** para entrenar el sistema utilizado. A partir de los valores generados por el sistema, se utilizarán los **datos de validación** para verificar que los valores obtenidos son realmente óptimos y, finalmente, con los **datos de testing** se pone a prueba el sistema al completo.