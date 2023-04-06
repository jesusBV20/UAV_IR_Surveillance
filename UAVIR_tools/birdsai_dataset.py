import os
import re
import numpy as np
import pandas as pd
import torch

from PIL import Image

# Funciones locales
def findRemove(str_list, string):
  try:
    str_list.pop(str_list.index(string))
  except:
    pass

def selectIRimages(img_file_names):
  file_names = [f for f in img_file_names if re.findall(".jpg+", f)]
  file_names = [f for f in file_names if not re.findall("^rgb+|^seg+", f)]
  return file_names

# Clase principal
class BIRDSAIDataset(torch.utils.data.Dataset):
  def __init__(self, root, transform = None):
    self.root = root
    self.transform = transform

    self.images_path = os.path.join(root, "images")
    self.annot_path = os.path.join(root, "annotations")

    # Cargamos todas las imágenes y anotaciones. Ordenamos todos los
    # archivos para asegurarnos de que se encuentran bien pareados
    self.movies_dirs = list(sorted(os.listdir(self.images_path)))
    self.annot = list(sorted(os.listdir(self.annot_path)))

    # Eliminamos '.DS_Store'
    findRemove(self.movies_dirs, '.DS_Store')

    # Eliminamos 'water_metadata_test.txt' y el directorio 'tracking'
    self.annot.pop(-1)
    self.annot.pop(-1)

    # Almacenamos el número total de frames por directorio (película)
    self.movies_frames = self.getFrames()
    self.movies_box_frames = self.getBoxFrames()

  def __len__(self):
    # La longitud del dataset corresponderá a la suma total de frames (ANOTADOS)
    return np.sum(self.movies_box_frames)

  def __getitem__(self, idx):
    # Calculamos el id de la película
    movie_id, box_frame = self.getBoxFrameInfo(idx)[0:2]

    # Cargamos el archivo de anotaciones
    annot_path = os.path.join(self.annot_path, self.annot[movie_id])
    annot_data = pd.read_csv(annot_path, header=None)

    # Seleccionamos columnas ['frame','box_id','x','y','w','h','class']
    annot_data = annot_data.iloc[:,0:7]
    frame = np.unique(np.unique(annot_data.iloc[:,0]))[box_frame]

    # Seleccionamos las anotaciones de nuestro frame 
    annot_data = annot_data.loc[annot_data[0] == frame]

    # - img -----
    # Cargamos la imagen
    movie_path = os.path.join(self.images_path, self.movies_dirs[movie_id])
    movie_imgs = list(sorted(selectIRimages(os.listdir(movie_path))))
    findRemove(movie_imgs, '.DS_Store') # eliminamos '.DS_Store'

    img_path = movie_imgs[frame]
    img = Image.open(os.path.join(movie_path, img_path)).convert("RGB")

    # - target ----
    # Generamos las 'boundig_box'
    num_boxs = len(annot_data)
    
    boxes_list = []
    labels_list = []
    for i in range(num_boxs):
        xmin = annot_data.iloc[i,2]
        xmax = xmin + annot_data.iloc[i,4]
        ymin = annot_data.iloc[i,3]
        ymax = ymin + annot_data.iloc[i,5]
        boxes_list.append([xmin, ymin, xmax, ymax])
        labels_list.append(annot_data.iloc[i,6]+1)
    #print(boxes_list)

    # Convertimos las listas en torch.Tensor
    boxes = torch.as_tensor(boxes_list,  dtype=torch.float32)
    labels = torch.as_tensor(labels_list, dtype=torch.int64)

    if not boxes_list: # no boxes  
      area = 0
    else:
      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # Suponemos que todos los objetos son 'not crowd'
    iscrowd = torch.zeros((num_boxs,), dtype=torch.int64)

    image_id = torch.tensor([idx])

    # Construimos el diccionario target
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    # if self.transform is not None:
    #     img, target = self.transform(img, target)
    if self.transform is not None:
        img = self.transform(img)
        
    return img, target

  """
  Función para extraer cualquier todos los frames del vídeo,
  se encuentre o no anotado.
  """
  def frame(self, idx):
    # Calculamos el id de la película
    movie_id, frame = self.getFrameInfo(idx)[0:2]

    # - img -----
    # Cargamos la imagen
    movie_path = os.path.join(self.images_path, self.movies_dirs[movie_id])
    img_path = list(sorted(selectIRimages(os.listdir(movie_path))))[frame]
    img = Image.open(os.path.join(movie_path, img_path)).convert("RGB")

    # - target ----
    # Cargamos el archivo de anotaciones
    annot_path = os.path.join(self.annot_path, self.annot[movie_id])
    annot_data = pd.read_csv(annot_path, header=None)

    # Seleccionamos frame y columnas ['frame','box_id','x','y','w','h','class']
    annot_data = annot_data.iloc[:,0:7].loc[annot_data[0] == frame]

    # Generamos las 'boundig_box'
    num_boxs = len(annot_data)

    boxes_list = []
    labels_list = []
    for i in range(num_boxs):
        xmin = annot_data.iloc[i,2]
        xmax = xmin + annot_data.iloc[i,4]
        ymin = annot_data.iloc[i,3]
        ymax = ymin + annot_data.iloc[i,5]
        boxes_list.append([xmin, ymin, xmax, ymax])
        labels_list.append(annot_data.iloc[i,6]+1)

    # Convertimos las listas en torch.Tensor
    boxes = torch.as_tensor(boxes_list,  dtype=torch.float32)
    labels = torch.as_tensor(labels_list, dtype=torch.int64)

    # Construimos el diccionario target
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels

    return img, target

  """
  Contamos todas las imágene, anotadas y no anotadas.
  """
  def getFrames(self):
    movie_frames = []
    for i in range(len(self.movies_dirs)):
      movie_path = os.path.join(self.images_path, self.movies_dirs[i])
      frames = list(sorted(selectIRimages(os.listdir(movie_path))))
      findRemove(frames, '.DS_Store')
      movie_frames.append(len(frames))
    return movie_frames

  """
  Contamos únicamente el número de imágenes anotadas.
  """
  def getBoxFrames(self): # TODO: WIP
    movie_box_frames = []
    for movie_id in range(len(self.annot)):
      # Cargamos el archivo de anotaciones
      annot_path = os.path.join(self.annot_path, self.annot[movie_id])
      annot_data = pd.read_csv(annot_path, header=None)

      # Seleccionamos la columna frame y contamos
      box_frames = np.unique(annot_data.iloc[:,0])
      movie_box_frames.append(len(box_frames))
    return movie_box_frames

  """
  Función para obtener información sobre cualquier frame.
  """
  def getFrameInfo(self, idx):
    # Calculamos el id y el frame de la película
    movie_id = 0
    frame = idx
    while (frame // self.movies_frames[movie_id] != 0):
      frame -= self.movies_frames[movie_id]
      movie_id += 1
    #print(movie_id, frame)

    # Retornamos toda la información relevante
    total_movie_frames = self.movies_frames[movie_id]
    return movie_id, frame, total_movie_frames

  """
  Función para obtener información únicamente sobre los frames anotados.
  """
  def getBoxFrameInfo(self, idx):
    # Calculamos el id y el frame de la película
    movie_id = 0
    box_frame = idx
    while (box_frame // self.movies_box_frames[movie_id] != 0):
      box_frame -= self.movies_box_frames[movie_id]
      movie_id += 1

    # Retornamos toda la información relevante
    total_movie_box_frames = self.movies_box_frames[movie_id]
    return movie_id, box_frame, total_movie_box_frames

  """
  Función para obtener información sobre el vídeo indicado.
  """
  def getMovieInfo(self, movie_id):
    # Calculamos la idexación global de los frames del vídeo
    frame_init = np.sum(self.movies_frames[0:movie_id])
    frame_end = frame_init + self.movies_frames[movie_id] - 1
    return frame_init, frame_end, self.movies_frames[movie_id]

  """
  Función para obtener información sobre el vídeo indicado, teniendo en cuenta
  únicamente los frames anotados.
  """
  def getMovieBoxInfo(self, movie_id):
    # Calculamos la idexación global de los frames del vídeo
    frame_init = np.sum(self.movies_box_frames[0:movie_id])
    frame_end = frame_init + self.movies_box_frames[movie_id] - 1
    return frame_init, frame_end, self.movies_box_frames[movie_id]