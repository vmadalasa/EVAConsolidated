## Assignment 13 File

**Members:**

Madalasa Venkataraman

Syed Abdul Khader

Jahnavi Ramagiri

Sachin Sharma

### Part A - Open CV

Objective of the project is

A) OpenCV Yolo: SOURCE (Links to an external site.)

B)Run this above code on your laptop or Colab.

C)Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn).

D)Run this image through the code above.

E)Upload the link to GitHub implementation of this

F)Upload the annotated image by YOLO.

Please find the colab file associated at the following location.

https://colab.research.google.com/drive/1XrJGjBGwInFvLl4s1v7O7NjKyCTNe-DW

Annotated copy of the image present here

![ORIGINAL IMAGE](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS13/Assignment%20A%20openCV/IMG\_1940.JPG)

![ANNOTATED IMAGE](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS13/Assignment%20A%20openCV/Annotatedimage.png)

### Part B – Yolo3

Goal for Part B

1. Training Custom Dataset on Colab for YoloV3
  1. Refer to this Colab File: [LINK (Links to an external site.)](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
  2. Refer to this GitHub [Repo (Links to an external site.)](https://github.com/theschoolofai/YoloV3)
  3. Collect a dataset of 500 images and annotate them.  **Please select a class for which you can find a YouTube video as well. ** Steps are explained in the readme.md file on GitHub.
  4. Once done:
    1. [Download (Links to an external site.)](https://www.y2mate.com/en19) a very small (~10-30sec) video from youtube which shows your class.
    2. Use [ffmpeg (Links to an external site.)](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence) to extract frames from the video.
    3. Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
    4. Inter on these images using detect.py file. \*\*Modify\*\* detect.py file if your file names do not match the ones mentioned on GitHub.
 `python detect.py --conf-thres 0.3 --output output_folder_name`
    5. Use [ffmpeg (Links to an external site.)](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence) to convert the files in your output folder to video
    6. Upload the video to YouTube.
  5. Share the link to your GitHub project with the steps as mentioned above
  6. Share the link of your YouTube video

Please find the colab file at the location

The class chosen was &#39;GUITAR&#39;, which is not in the coco set.

500 images were picked out from google images and by converting videos into frames using ffmpeg (ffmpeg –i video.mp4 img-%03d.jpg). The images were in jpeg format and where needed, were resized using ffmpeg keeping aspect ratio constant.

The annotation tool was utilised to annotate the images for the class chosen. Custom.names, custom.data and custom.txt files were created along with the images and the labels in the annotation file.

The yolov3-spp.cfg was customised to account for the number of channels. The custom data was passed into the YOLOv3 and trained using these 500 images.

Training was run for 300 epochs. At the end of the training, about 70 images were converted into a video where guitar detection was shown as per the outcome of the YOLOv3.

Youtube link https://youtu.be/1PQSRCehMgw
