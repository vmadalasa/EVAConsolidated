
Goal 

Training Custom Dataset on Colab for YoloV3
Refer to this Colab File: LINK (Links to an external site.)
Refer to this GitHub Repo (Links to an external site.)
Collect a dataset of 500 images and annotate them. Please select a class for which you can find a YouTube video as well. Steps are explained in the readme.md file on GitHub.
Once done:
Download (Links to an external site.) a very small (~10-30sec) video from youtube which shows your class. 
Use ffmpeg (Links to an external site.) to extract frames from the video. 
Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
Inter on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
`python detect.py --conf-thres 0.3 --output output_folder_name`
Use ffmpeg (Links to an external site.) to convert the files in your output folder to video
Upload the video to YouTube. 
Share the link to your GitHub project with the steps as mentioned above
Share the link of your YouTube video


Colab file link 


Youtube Link 
https://youtu.be/1PQSRCehMgw
