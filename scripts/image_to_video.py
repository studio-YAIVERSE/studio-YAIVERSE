"""
Script Description
    - This script converts a directory of PNG image frames into a video file.
Usage
    - $ python image_to_video.py -i <image_folder> -o <output_video>
Author
    - Yunsu Park
"""


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image_folder', type=str, required=True, help='directory containing image frames')
    parser.add_argument('-o', '--output_video', type=str, required=True, help='output video file name')
    parser.add_argument('-f', '--fps', type=int, default=15, help='frames per second')
    parser.add_argument('-t', '--image_type', type=str, default='png', help='image file type')
    parser.add_argument('-m', '--max_frames', type=int, default=166, help='maximum number of frames to process')
    return parser.parse_args()


def main(args):
    import cv2
    import os

    # Set the directory containing the image frames
    image_folder = args.image_folder

    # Set the output video file name
    output_video = args.output_video

    # Set the frames per second (fps)
    fps = args.fps

    # Set the image type
    image_type = args.image_type

    # Set the maximum number of frames to process
    max_frames = args.max_frames

    # Get the dimensions of the first image to set the video size
    img = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
    height, width, layers = img.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = None

    try:
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        # Loop through each image in the directory and add it to the video
        for i, filename in enumerate([
            f for f in sorted(os.listdir(image_folder))
            if f.lower().endswith(image_type.lower())
        ], start=1):
            print('processed number of images {}'.format(i))
            img = cv2.imread(os.path.join(image_folder, filename))
            video.write(img)

            if i == max_frames:
                print('Video generation complete !')
                break

    finally:
        if video is not None:
            # Release the video writer
            video.release()


if __name__ == '__main__':
    main(parse_args())
