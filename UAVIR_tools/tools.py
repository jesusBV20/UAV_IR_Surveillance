import os
import torch 

from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes

def createDir(dir):
  try:
    os.mkdir(dir)
    print("¡Directorio '{}' creado!".format(dir))
  except:
    print("¡El directorio '{}' ya existe!".format(dir))

def imgBoxes(img, boxes, labels, scores = None, score_th = 0.5):
  if not boxes.tolist():
    return img # No hay "boxes" que dibujar

  # Definimos una transformación para convertir PIL 
  # image a Torch tensor (y viceversa)
  transform_toTensor = T.PILToTensor()  
  transform_toImg = T.ToPILImage()

  # Convertimos la PIL image a Torch tensor
  img_tensor = transform_toTensor(img)

  # Si se ha dado un score, aplicamos el umbral
  if scores is not None:
    boxes_mask = torch.ones(len(boxes))
    labels_mask = torch.ones(len(boxes))
    for i in range(boxes.shape[0]):
      if scores[i] < score_th:
        boxes_mask[i]  = 0
        labels_mask[i] = 0

    boxes = boxes[boxes_mask == 1]
    labels = labels[labels_mask == 1] 

  # Seleccionamos el color de cada "bounding box" en función de la etiqueta
  colors = []
  for i in range(boxes.shape[0]):
    if labels[i] == 1:
      colors.append((0,0,255)) # 1 Animal (blue)
    else:
      colors.append((255,0,0)) # 2 Human (red)

  # Generamos el título de cada "bounding box"
  labels_str = ["Animal" if lab == 1 else "Human" for lab in labels.tolist()]

  # Dibujamos las cajas sobre la imagen 
  img_box = draw_bounding_boxes(img_tensor, boxes, labels=labels_str, 
                                colors=colors, width=1)
  return transform_toImg(img_box)