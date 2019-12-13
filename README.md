# CI 0129 - Predicción de viralidad usando Tensor Flow
Script e instrucciones de como preparar el ambiente para correr el modelo de memes en Tensor Flow, se incluye el csv del dataset, mostrando el formato utilizado para las columnas

# Preparación del ambiente

Antes de todo, tenemos que tener Python 3 instalados en nuestras máquinas, la mayoría de sistemas operativos lo tienen implementado, en cualquier caso se puede descargar e instalar [aquí](https://www.python.org/downloads/).

## Miniconda
Es un framework para Python que permite crear ambientes virtuales sin que se afecte los diferentes ambientes entre si, pueden descargarlo y ver la implementación para su respectivo sistema operativo [aquí](https://docs.conda.io/en/latest/miniconda.html).

Una vez instalado. nos vamos al command line correspiendente y creamos nuestro ambiente de trabajo:

```bash
$ conda create --name memes python=3
```

Para verificar que se creó, y además ver la lista de todos nuestros ambientes disponibles:

```bash
$ conda env list
```
Una vez listo, solo queda activar el ambiente:

```bash
$ conda activate memes
```

## Preparando las bibliotecas

Para este proyecto se utilizan varias de las bibliotecas de mayor uso en Machine Learning para Python, para instalarlas correctamente usando miniconda, solo tenemos que seguir el siguiente formato:

```bash
$ conda install scipy
$ conda install numpy
$ conda install matplotlib
$ conda install pandas
$ conda install scikit-learn
```
O bien en una sola línea:

```bash
$ conda install scipy numpy matplotlib pandas scikit-learn
```

Y finalmente, instalamos Tensor Flow, encargado de entrenar nuestro modelo (en este caso utilizamos [pip](https://pip.pypa.io/en/stable/) como andministrador de paquetes):

```bash
$ pip install--ignore-installed --upgrade tensorflow==1.9
```

Una vez listo, solo ejecutamos ```tensor-flow-ml-model.py```, recomendamos correr línea a línea para ver el comportamiento de cada agrupamiento y como se van adaptando a datos que el modelo puede utilizar para correr el modelo predictivo.

## Consultas

Cualquier consulta, pueden subir un Pull Request o bien, enviar un correo a ```pablo.arroyoporras@ucr.ac.cr```.
