# UAV_IR_Surveillance
En este repositorio se introduce un modelo basado en Faster-RCNN, capaz de detectar animales y humanos en vídeos IR filmados desde UAVs.

![](https://github.com/jesusBV20/UAV_IR_Surveillance/blob/main/thumnail.png)

## Descargar los datos utilizados para entrenar el modelo

El dataset empleado para entrenar nuestra versión de Faster-RCNN es BIRDSAI. En el caso de Linux, los datos se pueden descargar comprimidos con ```curl``` o ```wget``` del repositorio oficial:

```
# TrainReal: "https://storage.googleapis.com/public-datasets-lila/conservationdrones/v01/conservation_drones_train_real.zip"
# TestReal:  "https://storage.googleapis.com/public-datasets-lila/conservationdrones/v01/conservation_drones_test_real.zip"

# TrainSim:  "https://storage.googleapis.com/public-datasets-lila/conservationdrones/v01/conservation_drones_train_simulation.zip"
```

Es importante destacar que el conjuntos de datos generado de forma sintética cuenta también con imágenes RGB y con máscaras de segmentación, tanto en formato ```.png``` como ```.jpg```. En este proyecto únicamente vamos a trabajar con imágenes IR en formato ```.jpg```.

Para más información, visitar la documentación oficial del dataset: https://sites.google.com/view/elizabethbondi/dataset

# Paquetes necesarios

Todos los paquetes de Python necesarios, tanto para entrenamiento como para inferencia, se encuentran en ```requirements.txt```. Si el usuario usa el gestor de paquetes ```pip```, la instalación de todas estas dependencias se puede automatizar con

```
make init
```

o

```
pip install -r requirements.txt
```

Si además se desea realizar localmente un entrenamiento con GPU de NVIDIA, necesitará instalar la versión 11.7 de CUDA Toolkit: https://developer.nvidia.com/cuda-11-7-0-download-archive

En caso de que el usuario busque realizar inferencia en dispositivos que funcionen con NVIDIA JetPack, en ```requirements_jetson.txt``` se encuentran los paquetes de Python 3.6 necesarios, en sus versiones compatibles con JetPack 4.6 (L4T R32.6.1). Para la instalación de Pytorch y Torchvision recomendamos consultar la documentación que se encuentra en el siguiente enlace: https://qengineering.eu/install-pytorch-on-jetson-nano.html

**¡IMPORTANTE!** Se requieren al menos 4GB de RAM para poder cargar la versión original de Fast-RCNN en memoria, se ha comprobado que la Jetson Nano de 2GB es incapaz de realizar una inferencia.

Este proyecto utiliza una serie de scripts (situados en ```UAVIR_tools/detection```) que han sido adaptados del repositorio oficial de pytorchvision: https://github.com/pytorch/vision/tree/main/references/detection
