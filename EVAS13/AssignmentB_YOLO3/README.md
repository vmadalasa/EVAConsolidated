
 

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



Colab file link 
https://colab.research.google.com/drive/1e9BaZ4tzTgnjNbQqm8zvMOu7sCBpE3eo

Youtube Link 
https://youtu.be/1PQSRCehMgw
