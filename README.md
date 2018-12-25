# SLID
spoken language identification
The project is part of Data Mining Course, York University Fall 2018.
The main Objective here is to explore the efficacy of different acoustic features for the purpose of Spoken Language Identification.

Dataset used - https://www.kaggle.com/toponowicz/spoken-language-identification

Following are the scripts and python programs:


-------------------------------------------------------
Saved-Models=> [all .mdl files are trained saved model]
-------------------------------------------------------
MFCC-13-1dCNNmodel.mdl	
MFCC-40-1dCNNmodel.mdl	
MFCC-delta-1dCNNmodel.mdl	
RAW-std-1dCNNmodel.mdl	
https://drive.google.com/file/d/1W2P-BgjkadtSi1xuetELjkBQemszCKJa/view?usp=sharing (SVM saved model being too large is stored on google drive)

----------------------------------------------------------


----------------------------------------------------------
Python programs & scripts for extracting features, normalization and training the model 
--------------------------------------------------------
flac_mfcc.py
flac_mfcc_delta.py	
flac_raw.py	
normalize.sh	
only_train_1d_cnn.py	
train_raw_1d_cnn.py
my_first_data_generator_class.py	
evaluate_1d_cnn.py	
train_mfcc_svm.py	
--------------------------------------------------------


--------------------------------------------------------
CNN Model architecture using plot_model
--------------------------------------------------------
model_plot_MFCC.png	
model_plot_raw.png	
model_plot_raw64x10.png	
--------------------------------------------------------


--------------------------------------------------------
Log files showing each Epoch and the training accuracy and model summary etc.
--------------------------------------------------------
output_13_mfcc.log	
output_MFCC_delta1.log	
output_raw22.log	
output_svm_final.log	
o_mfcc40.log	
--------------------------------------------------------
