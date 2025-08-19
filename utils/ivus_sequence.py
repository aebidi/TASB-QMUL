import os
import cv2
from tqdm import tqdm

# set the path to the folder containing IVUS frames
# e.g., 'Frame_Dataset/val/BERN-001-LCX_FU'
SERIES_FOLDER_PATH = "/home/qzhang-server2/Documents/Abdullah_Data/Frame_Dataset/train/BERN-034-RCA_BL"

# set the desired path for the output video file
OUTPUT_VIDEO_PATH = "results/IVUS_sequence.mp4"

# set the desired frame rate for the video
OUTPUT_FPS = 20


def create_video_from_frames(series_path, output_path, fps):
    """
    reads all .jpg images from a folder and compiles them into an mp4 video

    args:
        series_path (str): The path to the folder containing the image frames.
        output_path (str): The path where the output video will be saved.
        fps (int): The frame rate for the output video.
    """
    # find all JPEG image files in the directory and sort them alphanumerically
    # sorting is crucial to ensure the frames are in the correct order
    try:
        image_files = sorted([f for f in os.listdir(series_path) if f.endswith('.jpg')])
    except FileNotFoundError:
        print(f"Error: The directory was not found at '{series_path}'")
        return

    if not image_files:
        print(f"Error: No .jpg images found in the directory '{series_path}'")
        return

    # read the first frame to determine the video dimensions (width, height)
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

    # initialise the VideoWriter object from OpenCV
    # 'mp4v' is a common codec for creating .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating video from {len(image_files)} frames...")
    # loop through all the image files and write them to the video
    for filename in tqdm(image_files, desc="Processing frames"):
        frame_path = os.path.join(series_path, filename)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video_writer.write(frame)

    # release the video writer to finalize the video file
    video_writer.release()
    print(f"\nSuccessfully created video! It is saved at: {output_path}")


if __name__ == '__main__':
    # check if the source folder exists before starting
    if not os.path.isdir(SERIES_FOLDER_PATH):
        print(f"Error: The source folder '{SERIES_FOLDER_PATH}' does not exist.")
    else:
        create_video_from_frames(
            series_path=SERIES_FOLDER_PATH,
            output_path=OUTPUT_VIDEO_PATH,
            fps=OUTPUT_FPS
        )