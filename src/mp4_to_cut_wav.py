from moviepy.editor import *
import cv2, pprint 

start_time = []
end_time = []

video = VideoFileClip("test2.mp4", audio=True)

with open("label.txt") as file:
	my_list = file.readlines()
	print my_list[-1]
	my_list = [x.strip('\n') for x in my_list]
	for line in my_list:
		start_time.append(line.split()[0])
		end_time.append(line.split()[1])
		#print "start:",line.split()[0]
		#print "end:",line.split()[1]
	file.close()

total_len = len(start_time)
#print start_time[0],end_time[0]


'''clip = video.subclip(start_time[0],end_time[0])
print clip.fps
clip.write_videofile("cut_"+str(1)+".mp4",fps=25)'''

'''for idx in range(total_len):
	print start_time[idx],end_time[idx]
	clip = video.subclip(float(start_time[idx]),float(end_time[idx]))
	print clip.fps
	clip.audio.write_audiofile("cut_"+str(idx+1)+".wav")'''
	#clip.write_videofile("cut_"+str(idx+1)+".mp4",fps=25)
clip = video.subclip(float(start_time[1]),float(end_time[1]))
clip = clip.audio.write_audiofile("cut_"+str(2)+".wav", fps=16000, channel=1)
