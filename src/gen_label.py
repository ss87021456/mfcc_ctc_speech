# usage python3

from __future__ import print_function
from webvtt.parsers import WebVTTParser
import glob, sys, os

def getVideoFilesFromFolder(dirPath):
	types = (dirPath+os.sep+'*.avi', dirPath+os.sep+'*.mkv', dirPath+os.sep+'*.mp4', dirPath+os.sep+'*.mp3', dirPath+os.sep+'*.flac') # the tuple of file types
	files_grabbed = []
	#print(types)
	for files in types:
		files_grabbed.extend(glob.glob(files))
		#print(glob.glob(files))
	return files_grabbed

def create_label(filename):
	write_file = open(filename[:-4]+".txt","w")
	print("writing ",filename[:-4]+".txt")
	web = WebVTTParser()
	read = web.read(filename[:-4]+'.vtt').captions
	
	start_time = [0.0]
	lines = []
	length = len(read)
	
	for index in range(length):
		line = read[index].text
		if index < len(read)-1:
			timestamp = read[index+1].start_in_seconds
		else:
			timestamp = read[index].end_in_seconds
		original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
		#targets = original.replace(' ','  ')
		#targets = targets.split(' ')
		start_time.append(timestamp)
		lines.append(original)
	
	
	for index in range(length):
		time_interval = str(round(start_time[index],2))+" "+str(round(start_time[index+1],2))
		write_file.write(time_interval+" "+lines[index]+"\n")
		#print("%.2f %.2f\t%s" %(start_time[index],start_time[index+1],lines[index]))

	write_file.close()

def main(argv):
	files = getVideoFilesFromFolder(argv[1])	
	for file in files:
		create_label(file)


if __name__ == '__main__':
	main(sys.argv)

