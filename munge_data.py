# Import libraries
import os
import ast
import pandas as pd
import numpy as np
from sklearn import model_selection
from tqdm import tqdm
import shutil

# The DATA_PATH will be where your augmented images and annotations.csv files are.
# The OUTPUT_PATH is where the train and validation images and labels will go to.
DATA_PATH = '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/yolo11/train'
OUTPUT_PATH = '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11'


# Function for taking each row in the annotations file
def process_data(data: pd.DataFrame, data_type='train'):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row['image_id'][:-4]  # removing file extension .jpeg
        bounding_boxes = row['bboxes']
        yolo_data = []
        for bbox in bounding_boxes:
            category = bbox[0]
            x_center = bbox[1]
            y_center = bbox[2]
            w = bbox[3]
            h = bbox[4]
            yolo_data.append([category, x_center, y_center, w,
                              h])  # yolo formated labels
        yolo_data = np.array(yolo_data)

        np.savetxt(
            # Outputting .txt file to appropriate train/validation folders
            os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"])
        shutil.copyfile(
            # Copying the augmented images to the appropriate train/validation folders
            os.path.join(DATA_PATH, f"aug_images/{image_name}.jpg"),
            os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg"),
        )


def copy_folder_contents(source_folder, destination_folder):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Move each item in the source folder to the destination folder
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder,
                                   item)  # Full path of the item
        destination_item = os.path.join(destination_folder,
                                        item)  # Destination path

        # Copy the item
        if os.path.isdir(source_item):
            shutil.copytree(source_item,
                            destination_item)  # Copy entire directory
            print(f"Copied directory: {source_item} to {destination_item}")
        else:
            shutil.copy2(source_item, destination_item)  # Copy file
            print(f"Copied file: {source_item} to {destination_item}")


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(f'{DATA_PATH}', 'annotations.csv'))
    df.bbox = df.bbox.apply(
        ast.literal_eval)  # Convert string to list for bounding boxes
    df = df.groupby('image_id')['bbox'].apply(list).reset_index(name='bboxes')

    # splitting data to a 90/10 split
    df_train, df_valid = model_selection.train_test_split(df,
                                                          test_size=0.1,
                                                          random_state=42,
                                                          shuffle=True)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    print(df_train.head())

    process_data(df_train, data_type='train')
    process_data(df_valid, data_type='valid')

    copy_folder_contents(
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/yolo11/train/images',
        f'{OUTPUT_PATH}/images/train')

    copy_folder_contents(
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/yolo11/train/labels',
        f'{OUTPUT_PATH}/labels/train')

    copy_folder_contents(
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/yolo11/valid/images',
        f'{OUTPUT_PATH}/images/valid')

    copy_folder_contents(
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/yolo11/valid/labels',
        f'{OUTPUT_PATH}/labels/valid')

    copy_folder_contents(
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/yolo11/test/images',
        f'{OUTPUT_PATH}/images/test')

    copy_folder_contents(
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/yolo11/test/labels',
        f'{OUTPUT_PATH}/labels/test')
