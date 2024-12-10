1. the data (augmented and original) are in the `./data` folder.
2. the `./images` folder contains the images in writing.
3. `convert_yolo_to_coco.py` is the file that convert yolo format data to coco format data
4. `munge_data.py` contains the spliting of train, validate and test data
5. `processing_and_data_augmentation.ipynb` is the .ipynb file to do the augmentation
6. `fasterRCNN.ipynb` is the file do faster RCNN model experiment
7. `yolo11_model_inference.ipynb` is the file do yolo11 model experiment
8. `RT_DETR.ipynb` is the file do RT-DETR model experiment
9. `results.ipynb` is the file process the models' training and testing performance data and generates plots
10. `webcam_inference.ipynb` is the file that using your webcam to do the real time inference
11. `vedio_inference.ipynb` is the file that use a .mp4 vedio to inference and output a annotated .mp4 file
12. `asl.yaml` is the config file model yolo11 and RT-DETR from ultralytics used
13. the yolo11 and RT-DETR models will output its model weights and training(testing) meta data in `./runs/detect`
14. the faster RCNN model from detectron2 will output its model weights and training(testing) meta data in `./detectron_output`
15. if runs in colab, please make sure mount the google drive's `./data` folder or make it with the same folder that excuting current `.ipynb` file. Mounting google drive example:
```
import os
from google.colab import drive

drive.mount('/content/drive')

# Change to the folder containing your notebook
notebook_path = '/content/drive/My Drive/data mining/Project' # change to your google file path
os.chdir(notebook_path)

# Verify the current working directory
print("Current Directory:", os.getcwd())
```
and install ultralytics package via:
```
!pip install ultralytics
```
in the begining of every `.ipynb` file
