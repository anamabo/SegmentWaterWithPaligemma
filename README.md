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
> git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision
> pre-commit install 
```
The last two commands will install Big Vision and the pre-commit hooks. 
The pre-commit hooks will check the code before committing.


***Note:*** If you use VS Code or Pycharm, make sure to set up your Python interpreter 
to the virtual environment created.

## Satellite images
The original dataset is in Kaggle. It can be downloaded from [this link](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies). 

Since some masks are not correct, I manually selected the correct images. You can read [this blog post]()
to see how I did it. In the folder `data/` you can find the final set of images.

## Create a dataset to fine-tune Paligemma

To create the dataset used by Paligemma, run this command:

```
> python convert.py --data_path=<absolute path to data/> --masks_folder_name=Masks_cleaned --images_folder_name=Images_cleaned
```

After running this command, a subfolder called `water_bodies/` will be created in `data/`.
This subfolder contains the images and the JSONL format needed as input for Paligemma. 

## Fine-tune Paligemma for image segmentation
In your Google Drive:
* Create a folder for this project.
* In that folder upload `water_bodies/` 
* copy the notebook accompanying this repository.
* Make sure to set a proper running time. It can be T4 GPU or A 100 GPU. 

You are ready to fine-tune Paligemma for segmentation.
