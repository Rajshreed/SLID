# SLID
spoken language identification
The project is part of Data Mining Course, York University Fall 2018.
The main Objective here is to explore the efficacy of different acoustic features for the purpose of Spoken Language Identification.

Dataset used - https://www.kaggle.com/toponowicz/spoken-language-identification

Steps for Spoken Language Identification and corresponding python program or script:
1) Volume Normalization - using shell script and Sox manipulation tool - Normalize.sh
2) Acoustic Feature Extraction - For MFCC, MFCC+Delta and Raw - flac_mfcc.py/flac_mfcc_delta.py/flac_raw.py
3) Training the model for mentioned features -
MFCC-13, MFCC-40, MFCC-13+delta_deltadelta - 1D-CNN architecture [only_train_1d_cnn.py]
MFCC-RAW - 1D-CNN architecture [train_raw_1d_cnn.py, my_first_data_generator_class.py]
MFCC-13 - SVM (RBF based) [train_mfcc_svm.py]
As the raw audio was processing heavy, keras fit_generator batch- multi processing was used.

4) Evaluation of the trained model on test data
Confusion matrix and classification report [evaluate_1d_cnn.py]
------------------------------------------------------------------------
Following are the various scripts and python programs listed:

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
normalize.sh	

flac_mfcc.py
flac_mfcc_delta.py	
flac_raw.py	

only_train_1d_cnn.py	
train_raw_1d_cnn.py
my_first_data_generator_class.py	
train_mfcc_svm.py	

evaluate_1d_cnn.py	
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


--------------------------------------------------------
CNN Model architecture using plot_model
--------------------------------------------------------
model_plot_MFCC.png	
model_plot_raw.png	
model_plot_raw64x10.png	
--------------------------------------------------------
