#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image


from ipywidgets import IntProgress
from IPython.display import display


# ## Analyse des données
# 
# - On a des fichier CSV et h5
#     - h5 contenant les images en 32x32 et 64x64
#     - CSV dataframe contenant des informations sur les images des fichier h5
# - Ici utilisation des images en 64x64
# - On a un dataframe contenant les inforamtions sur les images contenu dans le fichier h5
# - On a 6 types de valeurs :
#     + good
#     + color
#     + cut
#     + hole
#     + metal_contamination
#     + thread
# - Pour simplifier l'anomalie détection on regroupe les types de problèmes autre que good sous le même label "Damaged"
# - Les données ne sont plus équilibré :
#     + Au départ on à 8000 de chaque type
#     + Après le regroupement on a :
#         * 8000 good
#         * 40000 damaged
# - On va donc rééquilibré le set pour l'apprentissage

# In[ ]:


# load the `train64.csv` file
train_df64 = pd.read_csv("./data/train64.csv")
train_df64['indication_type'].value_counts().plot(kind='bar')
plt.show()

# change les labels autre au que good par damaged et les valeurs correspondante dans le dataframe
train_df64["indication_type"] = train_df64.indication_type.apply(lambda row: "damaged" if row!="good" else "good")
train_df64["indication_value"] = train_df64.indication_value.apply(lambda row: 1 if row!=0 else 0)
train_df64['indication_type'].value_counts().plot(kind='bar')
print(train_df64['indication_type'].value_counts())
print(train_df64.keys())


# ## Récupération des images a partir du fichier h5
# 
# (code récupérer depuis le notebook kaggle d'où est tirer le [dataset](https://www.kaggle.com/code/aadiadgaonkar/isolationforest-lof-gauss-svm-anomaly-detection))

# In[ ]:


# create an object that will take the dataset and produce the dataset in a format required for tensorflow dataset's API
class H5ToStorage:
  def __init__(self, hdf_path, ds_name="train"):
    self.path = hdf_path

    self.classes = []
    with h5py.File(self.path, 'r') as hf:
      for class_ in hf:
        self.classes.append(class_)

    self.name = ds_name

  # a generator to load the (img, class, angle)
  def generate_img_arr(self):
    for class_ in self.classes:
      with h5py.File(self.path, 'r') as hf:
        for angle in hf[class_]:
            for img in hf[class_][f"{angle}"]:
                yield img, class_, angle
  
  # utilize the generator to create new images and load it back to Storage
  def generate_train_dirs(self):
    # create the dataset's directories
    path = "data/working/train"
    os.makedirs(f"{path}/good/")
    os.makedirs(f"{path}/damaged/")

    # random_bright = tf.keras.layers.RandomBrightness(factor=0.05)
    random_flip = tf.keras.layers.RandomFlip("horizontal_and_vertical")

    gen = self.generate_img_arr()
    metadata = {}

    for i, data in enumerate(gen):
        img, label, angle = data
        if label == "good":
          for j in range(4):
            img_path = f"{path}/{label}/{i}_aug{j}.jpeg"
            img = random_flip(tf.expand_dims(np.squeeze(img), axis=2)*255., training=True)
            plt.imsave(img_path, np.squeeze(img), cmap="gray")
        else:
          img_path = f"{path}/damaged/{i}.jpeg"
          plt.imsave(img_path, np.squeeze(img)*255., cmap="gray")

        metadata[img_path] = angle
    return metadata

  def generate_test_dirs(self):
    # create the dataset's directories
    path = "data/working/test"
    os.makedirs(f"{path}/good/")
    os.makedirs(f"{path}/damaged/")

      
    gen = self.generate_img_arr()
    metadata = {}

    for i, data in enumerate(gen):
        img, label, angle = data
        if label == "good":
          img_path = f"{path}/{label}/{i}.jpeg"
          plt.imsave(img_path, np.squeeze(img)*255., cmap="gray")
        else:
          img_path = f"{path}/damaged/{i}.jpeg"
          plt.imsave(img_path, np.squeeze(img)*255., cmap="gray")

        metadata[img_path] = angle

    return metadata

  def to_storage(self):
    if self.name == "train":
      self.generate_train_dirs()

    elif self.name == "test":
      self.generate_test_dirs()


# In[ ]:


# train data & test data paths
test_dir = "data/working/test"
train_dir = "data/working/train"


# In[ ]:


# pour ne pas avoir a recréer les images si on les a deja
if not(os.path.isdir(test_dir) and os.path.isdir(train_dir)) :
    # generate train data
    train_gen = H5ToStorage("data/matchingtDATASET_train_64.h5", "train")
    train_dict = train_gen.to_storage()
    # generate train data
    test_gen = H5ToStorage("data/matchingtDATASET_test_64.h5", "test")
    test_dict = test_gen.to_storage()


# ### Note transformation image pour la création du dataframe d'apprentissage
# 
# - Modifications apportés aux images :
#     + Image transformée en nuance de gris via openCV2
#     + Image flatten via numpy.flatten()
#     + Ajout de l'image dans le dataframe pour pouvoir apprendre dessus en suite
#     
# - Séparation en 2 dataframes avec : 
#     + Les images labeled good (fichier TrainSetGood.csv)
#     + Les images labeled damaged (fichier TrainSetDamaged.csv)
# - Perte de la labellisation lors de la création des dataset puisque les données sont séparer dans 2 fichiers
#     + Labellisation des 2 dataset
#     + fusion des deux dataset
#     + sauvegarde des datasets dans ref_data.csv

# ## Création des fichiers CSV pour l'apprentissage
# #### [tuto progress bar](https://stackoverflow.com/questions/38861829/how-do-i-implement-a-progress-bar)

# In[ ]:


path = "./data/working/train/"
file_path = "./data/"
ext = ".csv"

dataTrainGood = np.empty([48000,4096], dtype=int)

progressGood = IntProgress(min=0, max=100) # instantiate the bar
print("Création du dataset labeled good")
display(progressGood) # display the bar

for i, name in enumerate(os.listdir(path+"good/")) :
    img = cv2.imread(path+"good/"+name,0)
    img = img.flatten()
    dataTrainGood[i] = img
    if i %(48000/100) == 0 :
        progressGood.value += 1

dataTrainGood = pd.DataFrame(dataTrainGood)
# dataTrainGood.to_csv(file_path+"Good"+ext, sep=',', encoding='utf-8')


# In[ ]:


dataTrainDamaged = np.empty([60000,4096], dtype=int)

progressDamaged = IntProgress(min=0, max=100) # instantiate the bar
print("Création du dataset labeled damaged")
display(progressDamaged) # display the bar
for i, name in enumerate(os.listdir(path+"damaged/")) :
    img = cv2.imread(path+"damaged/"+name,0)
    img = img.flatten()
    dataTrainDamaged[i] = img
    if i %(60000/100) == 0 :
        progressDamaged.value += 1

dataTrainDamaged = pd.DataFrame(dataTrainDamaged)
# dataTrainDamaged.to_csv(file_path+"Damaged"+ext, sep=',', encoding='utf-8')


# ### Todo List
# 
# - [X] Labelliser les deux dataset et les concat en 1 seul
# 
# - [X] nommé le CSV ref_data.csv

# In[ ]:


dataTrainGood["indication_type"] = "good"
dataTrainDamaged["indication_type"] = "damaged"
ref_data = pd.concat([dataTrainGood,dataTrainDamaged])
ref_data.to_csv(file_path+"ref_data"+ext, sep=',', encoding='utf-8')


# ## Code pour undersample les données damaged
# 
# ## Attention : ici le code utilise les données du fichier CSV original et non celui des fichiers créer

# In[ ]:


# Version 1
damagedData = train_df64[train_df64['indication_type'] == 'damaged']
goodData = train_df64[train_df64['indication_type'] == 'good']
num = min(len(damagedData),len(goodData))
print(damagedData['indication_type'].value_counts())
print()
print(goodData['indication_type'].value_counts())
data = pd.concat([damagedData.sample(num,random_state=2),goodData.sample(num, random_state=0)],keys= damagedData.keys())
data


# In[ ]:


# Version 2
num_dmgd, num_good = train_df64.indication_value.value_counts()
num = np.min((num_dmgd, num_good))
good = train_df64[train_df64['indication_value'] == 0]
dmgd = train_df64[train_df64['indication_value'] == 1]
df_good = good.sample(num)
df_dmgd = dmgd.sample(num)

undersampled_df = pd.concat([df_good,df_dmgd],axis=0)

