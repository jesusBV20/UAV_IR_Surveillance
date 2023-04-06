# UAV_IR_Surveillance
En este repositorio se introduce un modelo basado en Faster-RCNN, capaz de detectar animales y humanos en vídeos IR filmados desde UAVs.

TODO: imágen de portada

## Descargar los datos utilizados para entrenar el modelo

El dataset empleado para entrenar nuestra versión de Faster-RCNN es BIRDSAI. En el caso de Linux, los datos se pueden descargar comprimidos con ```curl``` o ```wget``` del repositorio oficial:

```
# TrainReal: "https://lilablobssc.blob.core.windows.net/conservationdrones/v01/conservation_drones_train_real.zip"
# TestReal:  "https://lilablobssc.blob.core.windows.net/conservationdrones-testset/conservation_drones_test_real.zip"

# TrainSim:   "https://lilablobssc.blob.core.windows.net/conservationdrones/v01/conservation_drones_train_simulation.zip"
```

Es importante destacar que el conjuntos de datos generado de forma sintética cuenta también con imágenes RGB y con máscaras de segmentación, tanto en formato ```.png``` como ```.jpg```. En este proyecto únicamente vamos a trabajar con imágenes en formato ```.jpg```.

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

Si el usuario además desea de realizar localmente un entrenamiento con GPU de NVIDIA, necesitará instalar la versión 11.7 de CUDA Toolkit: https://developer.nvidia.com/cuda-11-7-0-download-archive .

Este proyecto utiliza una serie de scripts (situados en ```UAVIR_tools/detection```) que han sido adaptados del repositorio oficial de pytorchvision: https://github.com/pytorch/vision/tree/main/references/detection .

# Referencias

(WIP)