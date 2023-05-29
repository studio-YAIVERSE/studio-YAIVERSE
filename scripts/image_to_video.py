"""
Code by ys.park
"""
import cv2
import os

if __name__ == "__main__":
    # Set the directory containing the image frames
    image_folder = ''

    # Set the output video file name
    video_name = ''

    # Set the frames per second (fps)
    fps = 15

    # Get the dimensions of the first image to set the video size
    img = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
    height, width, layers = img.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    i=0
    # Loop through each image in the directory and add it to the video
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png'):
            i += 1
            print('processed number of images {}'.format(i))
            img = cv2.imread(os.path.join(image_folder, filename))
            video.write(img)

        if i == 166:
            print('Video generation complete !')
            break


    # Release the video writer
    video.release()