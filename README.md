# UAV_IR_Surveillance
En este repositorio se introduce un modelo basado en Faster-RCNN, capaz de detectar animales y humanos en vídeos IR filmados desde UAVs.

## Descargar los datos utilizados para entrenar el modelo

El dataset empleado para entrenar nuestra versión de Faster-RCNN es BIRDSAI. En caso de Linux, los datos se pueden descargar comprimidos con ```curl``` o ```wget``` del repositorio oficial:

```
# TrainReal: "https://lilablobssc.blob.core.windows.net/conservationdrones/v01/conservation_drones_train_real.zip"
# TestReal:  "https://lilablobssc.blob.core.windows.net/conservationdrones-testset/conservation_drones_test_real.zip"
```

Para más información, visitar la documentación oficial del dataset: https://sites.google.com/view/elizabethbondi/dataset

# Paquetes necesarios

Todos los paquetes necesarios, tanto para realizar entrenamiento como para inferencia, se encuentran en ```requirements.txt```. Si el usuario usa el gestor de paquetes ```pip```, la instalación de todos los paquetes se puede automatizar con

```
make init
```

o directamente

```
pip install -r requirements.txt
```

Este proyecto utila una serie de scripts (situados en ```UAVIR_tools/detection```) que han sido adaptados del repositorio oficial de pytorchvision: https://github.com/pytorch/vision/tree/main/references/detection. Dichos scripts dependen de ```pycocotools```, un paquete que, como suele dar algunos problemas al instalarlo con ```pip```, hemos decidido incluirlo (con ciertas adaptaciones) en este repositorio (```UAVIR_tools/detection/pycocotools```).