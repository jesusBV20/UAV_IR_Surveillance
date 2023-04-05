from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes

def imgBoxes(img, boxes, labels):
  if not boxes.tolist():
    return img # No hay "boxes" que dibujar

  # Definimos una transformación para convertir PIL 
  # image a Torch tensor (y viceversa)
  transform_toTensor = T.PILToTensor()  
  transform_toImg = T.ToPILImage()

  # Convertimos la PIL image a Torch tensor
  img_tensor = transform_toTensor(img)

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