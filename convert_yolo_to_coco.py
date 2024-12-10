import os
import json
import cv2
import glob


def yolo_to_coco(yolo_folder, img_folder, output_json, output_path):
    # Initialize COCO format
    coco_format = {"images": [], "annotations": [], "categories": []}
    category_map = {}

    # Load images
    image_files = glob.glob(os.path.join(
        img_folder, "*.jpg"))  # Adjust extension if needed
    for img_id, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        height, width, _ = img.shape
        coco_format["images"].append({
            "id": img_id,
            "file_name": os.path.basename(image_file),
            "height": height,
            "width": width
        })

        # Load YOLO annotations
        yolo_file = os.path.join(
            yolo_folder,
            os.path.basename(image_file).replace('.jpg', '.txt'))
        if os.path.exists(yolo_file):
            with open(yolo_file, 'r') as f:
                for line in f:
                    class_id, x_yolo, y_yolo, w_yolo, h_yolo = map(
                        float, line.split())
                    category_id = int(class_id) + 1  # for detectron use

                    # Convert YOLO to COCO format
                    width_coco = w_yolo * width
                    height_coco = h_yolo * height
                    x_coco = x_yolo * width - (width_coco / 2)
                    y_coco = y_yolo * height - (height_coco / 2)

                    # Clip the bounding boxes to the image boundaries
                    x_coco = max(0, min(x_coco, width - 1))
                    y_coco = max(0, min(y_coco, height - 1))
                    width_coco = max(0, min(width_coco, width - x_coco))
                    height_coco = max(0, min(height_coco, height - y_coco))

                    if width_coco <= 0 or height_coco <= 0:
                        print(
                            f"Invalid bounding box found: [{x_coco}, {y_coco}, {width_coco}, {height_coco}] for image: {os.path.basename(image_file)}"
                        )
                        continue  # Skip this invalid bounding box

                    coco_format["annotations"].append({
                        "id":
                        len(coco_format["annotations"]),
                        "image_id":
                        img_id,
                        "category_id":
                        category_id,
                        "bbox": [x_coco, y_coco, width_coco, height_coco],
                        "area": (h_yolo * height) * (w_yolo * width),
                        "iscrowd":
                        0
                    })

    # Create category entries
    for idx, name in enumerate([
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]):
        coco_format["categories"].append({
            "id": idx + 1,  # for detectron use
            "name": name,
        })

    # Save COCO format
    with open(f"{output_path}{output_json}", 'w') as f:
        json.dump(coco_format, f)


if __name__ == "__main__":
    yolo_folder = "/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/labels/train"  # Update path
    img_folder = "/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/images/train"
    output_json = "annotations.json"
    yolo_to_coco(
        yolo_folder, img_folder, output_json,
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/images/train/'
    )

    yolo_folder = "/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/labels/test"  # Update path
    img_folder = "/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/images/test"
    output_json = "annotations.json"
    yolo_to_coco(
        yolo_folder, img_folder, output_json,
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/images/test/'
    )

    yolo_folder = "/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/labels/valid"  # Update path
    img_folder = "/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/images/valid"
    output_json = "annotations.json"
    yolo_to_coco(
        yolo_folder, img_folder, output_json,
        '/home/xiang/Desktop/Data_Mining/project/data/ASLv1/processed_yolo11/images/valid/'
    )
