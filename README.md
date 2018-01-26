# mfcc_ctc_speech
Using CTC loss function combine mfcc feature test on youtube dataset
#### Dependency:
- for create label (gen_label.py) <br>
python3 - 3.6.1 <br>
webvtt-py - 0.4.0 
- for cuting dataset (mp4_to_cut_wav.py) <br>
python2 - 2.7.14 <br>
moviepy - 0.2.3.2 <br>
cv2 - 3.3.0 
- for training (ctc_speech_recognition.py) <br>
python2 - 2.7.14 <br>
tensorflow - 1.4.0
#### Usage:
Step 1 : Download youtube vedio with cc subtitle <br>
Step 2 : python gen_label.py - to generate clear label <br>
Step 3 : python mp4_to_cut_wav.py - to generate wav dataset <br>
Step 4 : python ctc_speech_recognition.py - training <br>

#### Dataset description:
<img src="wav_example.jpg" height=200> <img src="mp4_example.jpg" height=200> <img src="label.jpg" height=200>

#### Training Process:
<img src="training.jpg" height=300>
