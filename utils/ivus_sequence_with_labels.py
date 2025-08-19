import os
import cv2
import numpy as np
from tqdm import tqdm

# set the path to the folder containing your IVUS frames
SERIES_FOLDER_PATH = "/home/qzhang-server2/Documents/Abdullah_Data/Frame_Dataset/val/BASE-005-LCX_BL" 

# set the desired path for the output video file
OUTPUT_VIDEO_PATH = "results/videos/IVUS_labelled_video.mp4" 

# set the desired frame rate
OUTPUT_FPS = 20

# set the drawing style for the bounding boxes.
BOX_COLOR = (0, 255, 0)  # green colour
BOX_THICKNESS = 1


def create_video_with_bboxes(series_path, output_path, fps):
    """
    reads all .jpg images from a folder, draws the corresponding ground-truth
    bounding boxes, and compiles them into an mp4 video
    """
    try:
        image_files = sorted([f for f in os.listdir(series_path) if f.endswith('.jpg')])
    except FileNotFoundError:
        print(f"Error: The directory was not found at '{series_path}'")
        return

    if not image_files:
        print(f"Error: No .jpg images found in '{series_path}'")
        return

    # read the first frame to get video dimensions
    first_frame_path = os.path.join(series_path, image_files[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"Error: Could not read the first image file at '{first_frame_path}'")
        return
    height, width, _ = frame.shape
    print(f"Detected frame size: {width}x{height}")

    # ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # initialise the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating video from {len(image_files)} frames with bounding boxes...")
    # loop through all image files
    for filename in tqdm(image_files, desc="Processing frames"):
        frame_path = os.path.join(series_path, filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # --- find and draw corresponding bounding boxes ---
        # construct the expected path for the annotation file
        base_name = filename.replace('.jpg', '')
        annot_path = os.path.join(series_path, f"{base_name}_bboxes.npy")
        
        # check if the annotation file exists for this frame
        if os.path.exists(annot_path):
            # loading the bounding box 
            # allow_pickle=True is required because the npy file contains dicts
            gt_boxes_xywh = np.load(annot_path, allow_pickle=True)
            
            # loop through each box and draw it
            for box in gt_boxes_xywh:
                # extract coordinates and convert to integers
                x1 = int(box['x'])
                y1 = int(box['y'])
                w = int(box['width'])
                h = int(box['height'])
                
                # calculate the bottom-right corner
                x2 = x1 + w
                y2 = y1 + h
                
                # draw the rectangle on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        
        # write the frame (with or without boxes) to the video
        video_writer.write(frame)

    # finalising video file
    video_writer.release()
    print(f"\nSuccessfully created video! It is saved at: {output_path}")


if __name__ == '__main__':
    if not os.path.isdir(SERIES_FOLDER_PATH):
        print(f"Error: The source folder '{SERIES_FOLDER_PATH}' does not exist.")
    else:
        create_video_with_bboxes(
            series_path=SERIES_FOLDER_PATH,
            output_path=OUTPUT_VIDEO_PATH,
            fps=OUTPUT_FPS
        )