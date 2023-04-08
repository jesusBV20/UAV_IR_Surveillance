import os
import numpy as np
import matplotlib.pylab as plt

from tqdm import tqdm
from  moviepy.editor import *

import torch
from torchvision import transforms as T

from .tools import createDir, clean_dir, imgBoxes

FIGSIZE = [16, 9]
RES = 1920 # 720p
OUTPUT_IMGS_PATH = "tmp"
OUTPUT_MOVIE_PATH = "output"

"""
Generación de un vídeo basado en las imágenes y anotaciones de dataset.
"""
def genMovie(movie_id, dataset, model = None, score_th = None, device = None,
             figsize = FIGSIZE, res = RES, init_frame = None, end_frame = None,
             output_imgs_path = OUTPUT_IMGS_PATH,
             output_movie_path = OUTPUT_MOVIE_PATH):
  
  ## Creamos los directorios de salida ##
  try:
    os.mkdir(output_imgs_path)
    clean_dir(output_imgs_path)
  except:
    clean_dir(output_imgs_path)

  ## Renderizamos todos los frames (imágen original con las anotaciones) ##
  if model is not None:
    if device is None:
      # Si se encuentra disponible, seleccionamos GPU como dispositivo para entrenar
      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: {}\n".format(device))
    model.to(device)
    model = model.eval()

  # Indicamos los frames a mostrar
  if model is not None:
    movie_frame_init, movie_frame_end, n_frames = dataset.getMovieBoxInfo(movie_id)
  else:
    movie_frame_init, movie_frame_end, n_frames = dataset.getMovieInfo(movie_id)

  if end_frame is not None:
    if (end_frame < n_frames):
      movie_frame_end = movie_frame_init + end_frame

  if init_frame is not None:
    if init_frame < n_frames:
      movie_frame_init = movie_frame_init + init_frame

  n_frames = movie_frame_end - movie_frame_init + 1
  idxs = np.linspace(movie_frame_init, movie_frame_end, n_frames, dtype=int)

  print("Generando imágenes... ")
  for i in tqdm(range(len(idxs))):
    fig = plt.figure(figsize=figsize, dpi=res/figsize[0])
    ax = fig.add_subplot()
    ax.axis('off')
    
    # Extraemos la anotación y, si hay modelo, inferimos
    if model is not None:
      img, _ = dataset[idxs[i]]
      target = model([img.to(device)])
      target = target[0]
    else:
      img, target = dataset.frame(idxs[i])

    # Si se ha realizado una transformación a tensor de la imágen, la deshacemos
    if dataset.transform is not None:
      transform_toImg = T.ToPILImage()
      img = transform_toImg(img)

    # Si se indica un umbral para el score entonces lo aplicamos
    if score_th is not None:
      img_box = imgBoxes(img, target["boxes"], target["labels"], target["scores"], score_th)
    else:
      img_box = imgBoxes(img, target["boxes"], target["labels"])

    # Dibujamos la imágen final y la guardamos
    if model is not None:
      _, frame, total_movie_frames = dataset.getBoxFrameInfo(idxs[i])
    else:
      _, frame, total_movie_frames = dataset.getFrameInfo(idxs[i])

    img_frame = ax.imshow(img_box)
    frame_text = ax.text(5, 10, "Frame: {0:>4}/{1:<4}".format(frame, total_movie_frames), c="g")
    movie_text = ax.text(5, 25, "Movie: {}".format(movie_id), c="g")

    plt.savefig(os.path.join(output_imgs_path, "frame{}.{}".format(str(i).zfill(5),"png")), 
                bbox_inches='tight', pad_inches = 0)
    plt.close()

  ## Fusionamos todos los frames para generar un .mp4 ##
  buildMP4(movie_id, output_imgs_path, output_movie_path)

"""
Función que permite generar un vídeo en formato .mp4 en base a un conjunto de imágenes contenidas
de forma ordenada en un directorio.
"""
def buildMP4(movie_id, output_imgs_path, output_movie_path):
  # Creamos directorio de salida
  createDir(output_movie_path)
  
  # Generamos el clip
  image_files = list(sorted([os.path.join(output_imgs_path,img)
                for img in os.listdir(output_imgs_path)
                if img.endswith(".png")]))
  clip = ImageSequenceClip(output_imgs_path, fps=25)
  clip.write_videofile(os.path.join(output_movie_path, "movie{}.mp4".format(str(movie_id).zfill(3))))

  # Eliminamos el directorio temporal
  try:
    clean_dir(output_imgs_path)
    os.rmdir(output_imgs_path)
  except:
    pass