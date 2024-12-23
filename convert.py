"""
Code to prepare the data for Paligemma.
The formatting of the bounding boxes and masks is based on
Robolow's blog on how to prepare data to fine-tune Paligemma2,
section: Extra: Preparing Data for PaliGemma 2 Instance Segmentation Training.
https://blog.roboflow.com/fine-tune-paligemma-2/
"""

import json
import logging
import os
import sys
import random
import shutil

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

if "big_vision_repo" not in sys.path:
  sys.path.append("big_vision_repo")

from big_vision_repo.big_vision.pp.proj.paligemma.segmentation import get_checkpoint
from big_vision_repo.big_vision.pp.proj.paligemma.segmentation import encode_to_codebook_indices

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
random.seed(123)

CHECKPOINT = get_checkpoint(model='oi')


def get_file_names(data_path: str, file_name: str) -> list:
    with open(os.path.join(data_path, file_name), "r") as file:
        return file.read().splitlines()


def reduce_contours(contours, epsilon: float):
    """Reduce the number of points in the contours"""
    approximated_contours = tuple()
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon * perimeter, closed=True)
        approximated_contours += (approx,)
    return approximated_contours


def get_bounding_box(contour):
    x1, y1, w, h = cv2.boundingRect(contour)
    x2, y2 = x1 + w, y1 + h
    return x1, y1, x2, y2


def get_contours_coordinates(ccontours) -> dict:
    reshaped_cnts = [cnt.reshape(len(cnt), 2) for cnt in ccontours]

    contours_coords = dict()
    for n, contour in enumerate(reshaped_cnts):
        flatten_cnt = contour.flatten()
        xvals = [
            flatten_cnt[x] for x in range(0, len(flatten_cnt), 2)
        ]  # even=x
        yvals = [
            flatten_cnt[y] for y in range(1, len(flatten_cnt), 2)
        ]  # odd=y
        contours_coords[n] = (xvals, yvals)
    return contours_coords


def plot_image_and_contours(image, contour, points=None):
    cnt_dict = get_contours_coordinates(contour)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image)
    for _, (x, y) in cnt_dict.items():
        ax.plot(x, y, "r-")
    if points is not None:
        for (xp, yp) in points:
            ax.plot(xp, yp, "bo")
    plt.show()


def format_bbox(y1, x1, y2, x2, h: int, w:int, bbox_tokens: tf.Tensor) -> tf.Tensor:
    bbox = np.array([y1, x1, y2, x2]) / np.array([h, w, h, w])
    binned_loc = tf.cast(tf.round(bbox * 1023), tf.int32)
    binned_loc = tf.clip_by_value(binned_loc, 0, 1023)
    loc_string = tf.strings.reduce_join(tf.gather(bbox_tokens, binned_loc))
    return loc_string


def get_mask_from_contour(h: int, w: int, cnt: np.ndarray) -> np.ndarray:
    new_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    cv2.drawContours(new_mask,
                     [cnt],
                     contourIdx=0,
                     color=255,
                     thickness=cv2.FILLED,
                     )
    # convert to bool
    new_mask = new_mask.astype(bool).copy()
    return new_mask


def format_mask(boolean_mask: np.ndarray, y1, x1, y2, x2, segment_tokens: tf.Tensor):
    tensor_mask = tf.convert_to_tensor(boolean_mask.astype(np.uint8), dtype=tf.uint8)
    yy1 = tf.cast(tf.round(y1), tf.int32)
    xx1 = tf.cast(tf.round(x1), tf.int32)
    yy2 = tf.cast(tf.round(y2), tf.int32)
    xx2 = tf.cast(tf.round(x2), tf.int32)

    tensor_mask = tf.image.resize(
        tensor_mask[None, yy1:yy2, xx1:xx2, None],
        [64, 64],
        method='bilinear',
        antialias=True,
    )
    mask_indices = encode_to_codebook_indices(CHECKPOINT, tensor_mask)[0]
    mask_string = tf.strings.reduce_join(tf.gather(segment_tokens, mask_indices))
    return mask_string


def create_output_for_paligemma(
    mask_path,
    mask_name: str,
    threshold: int,
    epsilon: float,
    cclass: str,
    prefix: str,
    npoints: int,
) -> dict:
    """Given an image, it creates a dict with the output for paligemma.
    IMPORTANT: This function assumes the same filename for both images and masks."""

    mask = cv2.imread(os.path.join(mask_path, mask_name))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    im_height, im_width = mask.shape

    if np.unique(mask).shape[0] == 1 and np.unique(mask)[0] == 0:
        # If the mask has no water, return an empty suffix
        final_output = {"image": mask_name, "prefix": prefix, "suffix": " "}

    else:
        # make the mask binary
        _, mask_binary = cv2.threshold(
            mask, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY
        )

        # Get the contours of the mask
        # tuple(ndarray(cnt points, 1, 2),...)
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Reduce the number of points in the contours
        reduced_contours = reduce_contours(contours, epsilon=epsilon)

        # filter out contours with less than  npoints
        contours_r = [cnt for cnt in reduced_contours if len(cnt) >= npoints]

        if len(contours_r) == 0:
            contours_r = [cnt for cnt in reduced_contours]

        # plot the image and the contours
        # plot_image_and_contours(mask, contours_r)

        # Define the tokens for the output
        loc_tokens = tf.constant(['<loc%04d>' % i for i in range(1024)])
        seg_tokens = tf.constant(['<seg%03d>' % i for i in range(128)])

        # For each contour, get the output for paligemma
        paligemma_output = []
        for counter, contour in enumerate(contours_r):

            # Get bounding box of the contour
            x1, y1, x2, y2 = get_bounding_box(contour)

            # Get formatted bbox
            bbox_loc_string = format_bbox(y1, x1, y2, x2, im_height, im_width, loc_tokens)

            # get the corresponding mask of the contour
            bool_mask = get_mask_from_contour(im_height, im_width, contour)

            # Get the formatted mask
            mask_loc_string = format_mask(bool_mask, y1, x1, y2, x2, seg_tokens)

            suffix = tf.strings.join([bbox_loc_string, mask_loc_string])

            paligemma_output.append(
                f"{suffix.numpy().decode('utf-8')} {cclass}"
            )

        paligemma_output = " ; ".join(paligemma_output)

        final_output = {
            "image": mask_name,
            "prefix": prefix,
            "suffix": paligemma_output,
        }

    return final_output


@click.command()
@click.option(
    "--data_path",
    required=True,
    type=str,
    help="The absolute path to the data folder.",
)
@click.option(
    "--masks_folder_name",
    required=True,
    type=str,
    help="The name of the folder with the corrected masks.",
)
@click.option(
    "--images_folder_name",
    required=True,
    type=str,
    help="The name of the folder with the corrected images.",
)
@click.option(
    "--output_folder_name",
    default="water_bodies",
    type=str,
    help="The name of the folder with the output for Paligemma.",
)
@click.option(
    "--threshold",
    default=150,
    type=int,
    help="Threshold for the binary mask. Values larger then this will be tagged as water (255, which is white)",
)
@click.option(
    "--epsilon",
    default=0.001,
    type=float,
    help="threshold used in the contour approximation. The smaller the value, the more points in the contour.",
)
@click.option(
    "--npoints",
    default=8,
    type=int,
    help="min no. points that a contour must have to be considered.",
)
@click.option(
    "--prefix",
    default="segment water",
    type=str,
    help="The prefix field in the output for Paligemma.",
)
@click.option(
    "--class_in_file",
    default="water",
    type=str,
    help="The class to be segmented.",
)
def main(
    data_path,
    masks_folder_name,
    images_folder_name,
    output_folder_name,
    threshold,
    epsilon,
    npoints,
    prefix,
    class_in_file,
):
    # # Code
    mask_path = os.path.join(data_path, masks_folder_name)
    image_path = os.path.join(data_path, images_folder_name)
    output_path = os.path.join(data_path, output_folder_name)

    os.makedirs(output_path, exist_ok=True)

    # Read the txt files with the list of images for train and test
    images_train_set = get_file_names(
        data_path=data_path, file_name="train_images.txt"
    )
    images_test_set = get_file_names(
        data_path=data_path, file_name="test_images.txt"
    )

    # create the Paligemma output for each dataset
    dataset_names = ["train", "test"]
    dataset_images = [images_train_set, images_test_set]

    for dataset, list_images in zip(dataset_names, dataset_images):
        logging.info(f"{len(list_images)} images in the {dataset} dataset.")

        paligemma_list = []
        for image_name in list_images:
            output_line = create_output_for_paligemma(
                mask_path=mask_path,
                mask_name=image_name,
                threshold=threshold,
                epsilon=epsilon,
                cclass=class_in_file,
                prefix=prefix,
                npoints=npoints,
            )
            paligemma_list.append(output_line)

        logging.info(
            f"{len(paligemma_list)} added files out of {len(list_images)}."
        )
        output_filename = dataset + ".jsonl"
        full_out_path = os.path.join(output_path, output_filename)
        logging.info(f"Writing the results to {full_out_path}.")
        with open(full_out_path, "w", encoding="utf-8") as file:
            for item in paligemma_list:
                json.dump(item, file)
                file.write("\n")

    # finally, copy the imagesin images_train_set to the output folder
    logging.info(f"Copying the images to {output_path}.")
    for dataset in dataset_images:
        for image_name in dataset:
            shutil.copy(os.path.join(image_path, image_name), output_path)

    logging.info("Done!")


if __name__ == "__main__":
    main()
