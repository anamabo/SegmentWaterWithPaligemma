import json
import logging
import os
import random
import shutil

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
random.seed(123)


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


def get_padded_bbox(
    bbox_coords: tuple,
    image_height: int,
    image_width: int,
    detection_factor: int,
    pad: int = 4,
) -> np.ndarray:
    """new_size is the size of the images in Paligemma"""
    x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox_coords
    new_bbox = np.array(
        [
            y1_bbox / image_height,
            x1_bbox / image_width,
            y2_bbox / image_height,
            x2_bbox / image_width,
        ]
    )
    new_bbox *= detection_factor  # needed for paliGemma
    new_bbox = new_bbox.astype(int)
    # pad with zeros to the left
    paligemma_bbox = np.char.zfill(new_bbox.astype(str), width=pad)
    return paligemma_bbox


def format_padded_info(info_array: np.ndarray, case: str) -> str:
    if case == "bbox":
        return "".join([f"<loc{element}>" for element in info_array])
    elif case == "seg":
        return "".join([f"<seg{element}>" for element in info_array])
    else:
        raise ValueError("Case not recognized. either bbox or seg")


def get_padded_seg_points(
    seg_points,
    image_height: int,
    image_width: int,
    segmentation_factor: int,
) -> np.ndarray:
    # For segmentation, we need coords = y,x
    scaled_points = np.array(
        [
            (
                coords[1] / image_height,
                coords[0] / image_width,
            )
            for coords in seg_points
        ]
    )

    scaled_points *= segmentation_factor
    scaled_points = np.round(scaled_points).astype(int).flatten()

    paligemma_output = np.char.zfill(scaled_points.astype(str), width=3)
    return paligemma_output


def select_points_in_one_contour(
    contour: np.ndarray, npoints: int = 8
) -> np.ndarray:
    """npoints= 8 is the requirement for Paligemma
    output: npoints x 2 array with the coordinates of the points (x, y)
    """
    random_indices = np.random.choice(
        contour.shape[0], size=npoints, replace=False
    )
    random_coordinates = contour[random_indices]
    random_coordinates = np.reshape(
        random_coordinates,
        shape=(random_coordinates.shape[0], random_coordinates.shape[2]),
    )
    return random_coordinates


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


def create_output_for_paligemma(
    mask_path,
    mask_name: str,
    threshold: int,
    epsilon: float,
    cclass: str,
    prefix: str,
    det_factor: int,
    seg_factor: int,
    npoints: int = 8,  # n segmentations points in Paligemma
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

        # For each contour, get the output for paligemma
        paligemma_output = []
        for counter, contour in enumerate(contours_r):

            # Get bounding box of the contour
            x1, y1, x2, y2 = get_bounding_box(contour)

            # scale bounding boxes
            padded_bbox = get_padded_bbox(
                bbox_coords=(x1, y1, x2, y2),
                image_height=im_height,
                image_width=im_width,
                detection_factor=det_factor,
            )

            # Format the bounding box for Paligemma
            line_bbox = format_padded_info(padded_bbox, case="bbox")

            # Get segmentation points (inside the bbox)
            points = select_points_in_one_contour(
                contour=contour, npoints=npoints
            )

            # Scale the points
            padded_points = get_padded_seg_points(
                points,
                image_height=im_height,
                image_width=im_width,
                segmentation_factor=seg_factor,
            )

            line_points = format_padded_info(padded_points, case="seg")

            paligemma_output.append(
                line_bbox + "" + line_points + " " + cclass
            )

        paligemma_output = "; ".join(paligemma_output)

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
    "--detection_factor",
    default=1024,
    type=int,
    help="Factor to scale the bounding boxes.",
)
@click.option(
    "--segmentation_factor",
    default=128,
    type=int,
    help="Factor to scale the segmentation points.",
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
    detection_factor,
    segmentation_factor,
    output_folder_name,
    threshold,
    epsilon,
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
                det_factor=detection_factor,
                seg_factor=segmentation_factor,
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
