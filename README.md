# UAV_IR_Surveillance
En este repositorio se introduce un modelo basado en Faster-RCNN, capaz de detectar animales y humanos en vídeos IR filmados desde UAVs.

## Descargar los datos utilizados para entrenar el modelo

El dataset empleado para entrenar nuestra versión de Faster-RCNN es BIRDSAI. En caso de Linux, los datos se pueden descargar comprimidos con ```curl``` o ```wget``` del repositorio oficial:

```
# TrainReal: "https://lilablobssc.blob.core.windows.net/conservationdrones/v01/conservation_drones_train_real.zip"
# TestReal:  "https://lilablobssc.blob.core.windows.net/conservationdrones-testset/conservation_drones_test_real.zip"
```

Para más información, visitar la documentación oficial del dataset: https://sites.google.com/view/elizabethbondi/dataset
