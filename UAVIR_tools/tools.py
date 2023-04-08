import os
import numpy as np
import matplotlib.pylab as plt

import torch 
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes

FIGSIZE = [16, 9]
RES = 1920 # 720p

def createDir(dir):
  try:
    os.mkdir(dir)
    print("¡Directorio '{}' creado!".format(dir))
  except:
    print("¡El directorio '{}' ya existe!".format(dir))

def clean_dir(dir):
  for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

"""
Superposición de las cajas sobre la imágen original.
"""
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

"""
Visualización de un único frame.
"""
def frameVisualicer(dataset, movie_id, frame, 
                    figsize = [16, 9/2], res = 1920):
    # Recogemos información sobre el frame
    frame_init, frame_end, n_frames = dataset.getMovieBoxInfo(movie_id)

    # Recogemos la img del dataset
    img, target = dataset[frame_init + frame]

    # Mostramos el resultado
    img_box_real = imgBoxes(img, target["boxes"], target["labels"])

    fig = plt.figure(figsize=figsize, dpi=res/figsize[0])
    ax1, ax2 = fig.subplots(1,2)
    plt.subplots_adjust(wspace=-0.2)

    ax1.axis('off')
    ax2.axis("off")

    ax1.set_title("Imágen original")
    ax2.set_title("Imágen original etiquetada")

    ax1.imshow(img)
    ax2.imshow(img_box_real)
    plt.show()