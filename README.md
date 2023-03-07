# Bias_in_COVID-19_Detection_from_Audio
This project will focus on looking for biases that exist in the COVID-19 detection based on coughing sounds.



* DataPreprocessing.ipynb

  * Filter the data with cough_detected below 0.9;
  * Divided the age into 4 groups: <=20, 20-40, 40-60, >60;
  * Balanced dataset across the three status classes (COVID-19, Healthy, and Symptomatic) by randomly picking 519 samples from each class;
  * Split the dataset to traning set, validation set, and test set
  * Analyze the dataset by gender and age

* dataset_split

  Saving files of spliting dataset. Could apply different dataset split methods here.

  * metadata_balanced_test.csv: 
  * metadata_balanced_train.csv
  * metadata_balanced_val.csv

* coughvid_20211012

  * .wav files: audios
  * /mfcc: mfcc images

* AudioProcessing.py

  It is for processing the audio. Now it consists of two methods. Run it by `python AudioProcessing.py [task]`.

  [task] can convert2wav / wav2mfcc.

  * convert2wav: convert all the ogg / webm audios files to wav files.
  * wav2mfcc: generate mfcc images from wav files, and save them to ./coughvid_20211012/mfcc/.
