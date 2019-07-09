
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

#Quick Script to make that training video from .avis from the pig cameras.

videoPath = r'F:\\IACUC training videos'
fileNameLst = ['2018108_08142018(0800-1200).avi', '2018122_20181031(1200-1600).avi', '2018143_20190225(1800-2100).avi',
                '2018144_20190226(1800-2100).avi', '2018146_20190306(0800-1200).avi', '2018150_20190322(1200-1600).avi', '2018161_06042019.avi']

ClipIdxLst = [0, 1, 1, 2, 3, 3, 3, 4, 5, 6]

ClipStartTimeLst = ['0:23:39', '2:08:15', '2:09:49', '0:19:00', '0:34:45', '0:43:07', '1:00:32', '2:23:20', '0:13:30', '0:00:00']
ClipEndTimeLst = ['0:25:47', '2:09:40', '2:11:00', '0:20:00', '0:36:10', '0:44:40', '1:01:34', '2:25:01', '0:17:05', '0:03:02']

retval = os.getcwd()
print ("Current working directory %s", retval )
os.chdir( videoPath )
retval = os.getcwd()
print ("Current working directory %s", retval )
print(os.listdir(os.getcwd()))

ClipLst =[]

for idx, item in enumerate(ClipIdxLst):
    print(idx, item)
    print(ClipStartTimeLst[idx])
    print(ClipEndTimeLst[idx])
    ClipLst.append(VideoFileClip(fileNameLst[item]).subclip(t_start=ClipStartTimeLst[idx], t_end=ClipEndTimeLst[idx]   ))

concat_clip = concatenate_videoclips(ClipLst, method="compose")

concat_clip.write_videofile("GazmuriTrainingVideo.mp4", codec="libx264", audio=False)
print(ClipLst)



""""
Instructions from Jenna
2018108_08142018 (0800-1200) 23:39 - 25:47
2018122_2018031 (1200-1600), 2:08:15 - 2:09:40
2018122_20181031 (1200-1600), 2:09:49 - 2:11:00
2018143_20190225 (1800-2100), 19:00 - 20:00
2018144_20190226 (1800-2100), 34:45 - 36:10
2018144_20190226 (1800-2100), 43:07 - 44:40
2018144_20190226 (1800-2100), 1:00:32 - 1:01:34
2018146_20190306 (0800-1200),  2:23:20 - 2:25:01
2018150_20190322 (1200-1600), 13:30 - 17:05 
2018161 - entire video *no editing needed*
"""