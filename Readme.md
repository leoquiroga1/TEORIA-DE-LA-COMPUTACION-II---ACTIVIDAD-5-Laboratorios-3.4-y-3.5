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