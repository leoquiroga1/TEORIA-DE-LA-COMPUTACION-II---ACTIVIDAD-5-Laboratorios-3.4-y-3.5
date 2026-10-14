# Actividad 5: Laboratorios 3.4 y 3.5 — Amazon SageMaker

| Información del Proyecto | Detalles |
| :--- | :--- |
| **Estudiante** | Elbert Leonardo Quiroga Ballén |
| **Programa** | Ingeniería de Software |
| **Asignatura** | Teoría de la Computación II |
| **Repositorio GitHub** | https://github.com/leoquiroga1/TEORIA-DE-LA-COMPUTACION-II---ACTIVIDAD-5-Laboratorios-3.4-y-3.5.git |

---

# Laboratorio 3.4

Una vez se ha ingresado a la IA de Sagemaker se realiza la creación y clonación del repositorio en GitHub con el objetivo de mantener un control de versiones de los laboratorios y mantener el código en la nube. En los pantallazos se hace evidencia de los permisos de acceso al repositorio, la creación de la rama dev_equiroga92 y la creación de la carpeta de pantallazos.

![](Pantallazos/Screenshot%202026-10-12%20231133.png)
![](Pantallazos/Screenshot%202026-10-12%20235627.png)

Se realiza la instalación de las dependencias necesarias para el desarrollo del proyecto, que nos permiten el análisis y manejo de los datos dentro del laboratorio, puesto que el entorno base del cuaderno puede no contar con las librerías requeridas de machine learning y otras.

![](Pantallazos/Screenshot%202026-10-12%20222653.png)

En el siguiente paso importamos los archivos necesarios para hacer la exploración de los datos y asi poder manipularlos en el entorno. Se puede evidenciar los archivos importados al verlos en la parte izquierda del notebook en donde acaban de aparecer.

Inicialmente para explorar los datos, hacemos uso de los comandos df.shape y df.columns para obtener los datos de numero de filas y columnas y los nombres de las columnas con las que vamos a trabajar. De acuerdo a los nombres de las columnas se puede inicialmnete ver que son datos de los pacientes y podría corresponder a tipos de datos a etiquetar como 0 y 1, puesto que cada paciente puede tener o no una característica y esto evita inclinaciones adicionales hacia cualquier variable por el uso de números adicionales.

![](Pantallazos/Screenshot%202026-10-13%20004713.png)

Para el paso 2 que corresponde a la preparación de los datos debemos dividir estos datos en 3 conjuntos, inicialmente se dividen en conjunto y característica y luego en tres subconjuntos de cada uno , lo que da como resultado un total de seis conjuntos de datos. El propósito de esta división es permitir que el modelo aprenda de un grupo de datos, luego pase al proceso de validacion para mejorar la precision y finalmente se ponga a prueba con datos que nunca ha visto, evitando así que nuestro modelo simplemente memorice la información.

Con los comandos provistos, organizamos los datos moviendo la columna de Class al inicio: Este paso es porque el algoritmo de XGBoost espera por defecto que la variable objetivo que queremos predecir se encuentre siempre en la primera posición del archivo o de las columnas.

![](Pantallazos/Screenshot%202026-10-13%20005542.png)

Luego de ello se dividen los datos entre los datos que vamos a utilizar para probar y validar en dos partes iguales, y se hace la distribución de las clases, debemos verificar que hayan quedado aplicados los cambios.

![](Pantallazos/Screenshot%202026-10-13%20005741.png)

Pasamos a la carga de los datos a Amazon S3, por ese motivo se deben pasar a formato CSV y que es un archivo simple y separa los datos por comas, y S3 es el sistema de almacenamiento del cual SageMaker extrae los datos durante el entrenamiento. Durante la escritura se debe evitar que se reescriban los encabezados de las columnas, puesto que el modelo solo sabe interpretar datos numéricos y los textos de los títulos generarían un error.

![](Pantallazos/Screenshot%202026-10-13%20011544.png)

En el paso 3 tenemos el entrenamiento del modelo, allí utilizaremos los datos ya segmentados para entrenar nuestro modelo de predicción:

Iniciamos obteniendo la URI del contenedor de XGBoost, lo cual nos permite usar el algoritmo preconfigurado y optimizado por AWS. Se definen los parámetros iniciales para entrenar el modelo por primera vez.

![](Pantallazos/Screenshot%202026-10-13%20012118.png)

Pasamos a la aplicación de canales para el entrenamiento y la validación del modelo: Estos canales son las rutas que le indican al trabajo de entrenamiento exactamente de qué parte de nuestro bucket de S3 debe jalar la información.

![](Pantallazos/Screenshot%202026-10-13%20012449.png)

Finalmente enviamos a entrenamiento al modelo para que a partir de los datos provistos pueda hacer predicción de los datos de los pacientes y aprenda a identificar de forma autónoma los patrones que diferencian un caso normal de uno anormal.

![](Pantallazos/Screenshot%202026-10-13%20013456.png)

---

# Laboratorio 3.5

Nos conectamos al repositorio de Github y clonamos el repositorio, es importante re entrar la información de nombre e email que nos identifica y entrar a trabajar desde la rama asignada, luego movemos los archivos del nuevo laboratorio hacia la carpeta en donde tenemos nuestro repositorio y  realizamos el primer commit y push con la finalidad de guardar nuestro progreso**, cumpliendo con los requisitos de la actividad de evidenciar el trabajo desde el workspace y asegurar el control de versiones en la nube.**

![](Pantallazos/Screenshot%202026-10-13%20190242.png)

Accedemos a nuestro laboratorio y vamos completándolo haciendo push en git cada tanto para asegurarnos que todo queda documentado y guardado.

![](Pantallazos/Screenshot%202026-10-13%20190740.png)

Iniciamos el laboratorio y lo primero que hacemos es instalar la librerías necesarias para desarrollarlo de la misma manera en que se hizo en el laboratorio 3.4**, ya que al ser una instancia nueva, carece de las dependencias actualizadas requeridas por el código.**

![](Pantallazos/Screenshot%202026-10-13%20191219.png)

Importamos la librerías y entrenamos el modelo tal y como estaba preparado desde el laboratorio 3.4 para que podamos avanzar con el laboratorio 3.5**, dado que cada laboratorio se ejecuta en un entorno efímero y necesitamos recrear el modelo base antes de poder consumirlo.**

![](Pantallazos/Screenshot%202026-10-13%20191416.png)

Una vez se entrena el modelo, pasamos a alojarlo, esto significa que ya estará disponible para hacer predicciones en tiempo real o por lotes según decidamos que ruta tomar.

![](Pantallazos/Screenshot%202026-10-13%20191719.png)

Una vez alojado el modelo, pasamos al paso 2, en el que empezaremos a realizar predicciones. Para esto necesitamos obtener los datos de prueba y debemos verificar que contienen:

![](Pantallazos/Screenshot%202026-10-13%20192351.png)

El paso 2 tambien nos indica como convertir los datos de prueba a CSV para inicial las predicciones, que es un formato que podemos utilizar para enviar los datos al endpoint de XGBoost para procesar correctamente los datos.

![](Pantallazos/Screenshot%202026-10-13%20192602.png)

Hacemos la predicción con los datos provisto por el laboratorio y nos da como resultado una probabilidad que debemos configurar para que determine si esa probabilidad debe dar una respuesta de 0 o 1, ya que necesitamos datos binarios para que la salida pueda leerse como un resultado verdadero o falso.

![](Pantallazos/Screenshot%202026-10-13%20192739.png)

Debemos también si el resultado de la predicción es preciso o exacto, puesto que aun debemos determinar que tan calibrado se encuentra nuestro modelo con la realidad, si comparamos la fila 136 que usamos para la predicción, nos da b'0.9966071844100952' que es una probabilidad muy alta, por lo tanto en los datos de prueba nos debe indicar clase 1 que en nuestro caso seria anormal.

![](Pantallazos/Screenshot%202026-10-13%20193029.png)

Al traer los datos reales, podemos verificar que el modelo entrenado es preciso de acuerdo a las primeras predicciones sin embargo hay que hacer varias predicciones para comprobar su fiabilidad.

![](Pantallazos/Screenshot%202026-10-13%20193240.png)

Para responder al desafio del laboratorio, repetimos los pasos dados para cargar los datos, convertirlos a CSV y obtener la predicción para la fila 230  y 134, con el objetivo de validar empíricamente cómo reacciona el modelo ante diferentes conjuntos de datos.

![](Pantallazos/Screenshot%202026-10-13%20194829.png)
![](Pantallazos/Screenshot%202026-10-13%20194921.png)

Revisando nuevamente los datos de la tabla, podemos evidenciar que la precisión de las predicciones empiezan a ser 0 desde el 0.77 que es el resultado de la fila 230 que en los datos de prueban son class 0. La predicción de la fila 134 dio 0.99 y fue precisa porque en los datos de prueba la clase es 1:

![](Pantallazos/Screenshot%202026-10-13%20195033.png)

En el paso 3 terminamos el modelo que implementamos y eliminamos el punto de enlace para proceder con el paso 4 para la predicción por lotes, esto es importante para detener el consumo de recursos de la instancia y evitar el uso de los mismos cuando no se requiere el uso del modelo o la instancia para hacer predicciones.

![](Pantallazos/Screenshot%202026-10-13%20195531.png)

Primero obtenemos los datos de prueba y los transformamos en un archivo CSV. Sin incluir la primera columna que es la que nos dice la clase (Normal o anormal) para que el modelo no conozca la respuesta por anticipado y la infiera.

![](Pantallazos/Screenshot%202026-10-13%20200242.png)

Luego aplicamos el transformado para definir el tipo de instancia, archivo de entrada y la ubicación de salida. Esta transformación por lotes es ideal en algunos casos porque optimiza el procesamiento de grandes volúmenes de datos encendiendo la máquina, procesando todo el conjunto de datos y optimizando el uso de recursos.

![](Pantallazos/Screenshot%202026-10-13%20200527.png)

Luego de terminar la transformación, se pueden declarar los resultados de las predicciones de Amazon S3 para cargarla en un dataframe de Pandas. Como vemos en el pantallazo están los resultados de las predicciones, estos resultados de pueden acercar a 0 o 1 dependiendo de como lo configuremos para que se vea reflejado.

![](Pantallazos/Screenshot%202026-10-13%20201736.png)

Si volvemos a los datos de prueba, vemos que es un modelo muy preciso y desde el resultado 0.77 podemos convertir el resultado a 0 porque a partir de ese punto el paciente entra dentro de la clase “normal”

En el laboratorio en cambio se utilizó un tresshold de 0.65 para convertir las predicciones en 0 y 1, sin embargo un dato como el de la fila 230 debio convertirse en 0 teniendo en cuenta los datos de prueba, se debe ajustar el tresshold para para mejorar la precisión de los resultados, demostrando que la calibración del umbral define el balance entre "falso" "verdadero" según la sensibilidad requerida en nuestro modelo de datos.

![](Pantallazos/Screenshot%202026-10-13%20202229.png)

Siguiendo el desafio propuesto por el laboratorio se sube el tresshold a 0.78 dando los mismos resultados que en los datos de prueba mejorando la precisión de los resultados de nuestro modelo:

![](Pantallazos/Screenshot%202026-10-13%20203239.png)

Y finalmente subimos el tresshold a 0.99 que deja por fuera varios pacientes que podrían estar en la clase “anormales” y que los trata como “normales”:

![](Pantallazos/Screenshot%202026-10-13%20203327.png)

Para finalizar se realiza la predicción de los datos dispuestos en la actividad, dando como resultado una predicción de la clase de  b'0.01694011501967907' que nos genera una clase 0 “normal” para el paciente de acuerdo al tresshold mas apropiado que generamos de 0.78, lo que responde directamente a la validación requerida la actividad y demuestra de igualmanera la viabilidad del modelo desplegado.

![](Pantallazos/Screenshot%202026-10-13%20204634.png)