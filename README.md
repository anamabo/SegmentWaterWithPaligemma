# Segment Water in Satellite images with Paligemma
This repository has Python scripts to create a dataset for Paligemma to segment water in satellite images.
The blog post of this project can be found [here](TB ADDED).

## Prerequisites 
- Python 3.12 
- Pipenv

## General set up and activation of the environment 
Clone the repository first.

Once the project is cloned, you need to create and set up a virtual environment. To do so,  
open a terminal and type the following commands:

```
> cd <path to SegmentWaterWithPaligemma>
> pipenv install --dev
> pipenv shell
> pre-commit install 
```

This last plugin will facilitate and automate the code formatting.

***Note:*** If you use VS Code or Pycharm, make sure to set up your Python interpreter 
to the virtual environment created.

## Satellite images
The original dataset is in Kaggle. It can be downloaded from [this link](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies). 

Since some masks are not correct, I manually selected the correct images. You can read [this blog post]()
to see how I did it. In [this link](https://drive.google.com/drive/folders/1U9MkU1fU6uZQrbN5-SAvewOU4dITuhtj?usp=sharing) 
you can get the final set of images.  

* Download `Mask_cleaned.zip` and `Images_cleaned.zip` from the link above.
* Unzip the files and place them in the `data/` folder.

## Create a dataset to fine-tune Paligemma

Once the cleaned images and masks are added in `data/`, you can run the following command 
to create the dataset used by Paligemma:

```
> python convert.py --data_path=<absolute path to data/> --masks_folder_name=Masks_cleaned --images_folder_name=Images_cleaned
```

After running this command, a subfolder called `water_bodies/` will be created in `data/`.
This subfolder contains the images and the JSONL format needed as input for Paligemma. 

## Fine-tune Paligemma for image segmentation
In your Google Drive:
* Create a folder for this project.
* In that folder upload `water_bodies/` 
* Also copy [This notebook](https://colab.research.google.com/drive/1t64cFa9RxjLn8_gLib64rbW_LyYuBKgE?usp=sharing). 
* Make sure to set a proper running time. It can be T4 GPU or A 100 GPU. 

You are ready to fine-tune Paligemma.
