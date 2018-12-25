# SLID
Spoken Language Identification
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
1) MFCC-13-1dCNNmodel.mdl
2) MFCC-40-1dCNNmodel.mdl	
3) MFCC-delta-1dCNNmodel.mdl	
4) RAW-std-1dCNNmodel.mdl	
5) https://drive.google.com/file/d/1W2P-BgjkadtSi1xuetELjkBQemszCKJa/view?usp=sharing (SVM saved model being too large is stored on google drive)

----------------------------------------------------------


----------------------------------------------------------
Python programs & scripts for extracting features, normalization and training the model 
--------------------------------------------------------
1) normalize.sh	

2) flac_mfcc.py
3) flac_mfcc_delta.py	
4) flac_raw.py	

5) only_train_1d_cnn.py	
6) train_raw_1d_cnn.py
7) my_first_data_generator_class.py	
8) train_mfcc_svm.py	

9) evaluate_1d_cnn.py	
--------------------------------------------------------


--------------------------------------------------------
Log files showing each Epoch and the training accuracy and model summary etc.
--------------------------------------------------------
1) output_13_mfcc.log	
2) output_MFCC_delta1.log	
3) output_raw22.log	
4) output_svm_final.log	
5) o_mfcc40.log	
--------------------------------------------------------


--------------------------------------------------------
CNN Model architecture using plot_model
--------------------------------------------------------
1) model_plot_MFCC.png	
2) model_plot_raw.png	
3) model_plot_raw64x10.png	
--------------------------------------------------------

Summarization of the evaluation results - 


1) MFCC - 13 coefficients with Neural Network (1d conv)
Training Accuracy     - 99.32%
Testing Data Accuracy - 97.22%
F1 Score              - 0.97

2) MFCC - 40 coefficients with Neural Network (1d conv)
Training Accuracy     - 99.75%
Testing Data Accuracy - 86.11%
F1 Score              - 0.86

3) MFCC - 13 coefficients + Delta + Delta-Delta with Neural Network (1d conv)
Training Accuracy     - 99.73%
Testing Data Accuracy - 97.04%
F1 Score              - 0.97

4) MFCC - 13 coefficients with Long Short Term Memory 
Training Accuracy     - 96.57%
Testing Data Accuracy - 85.61%
F1 Score              - 0.86

5) MFCC - 13 coefficients with Support Vector Machine(RBF)
Training Accuracy     - 91.48%
Testing Data Accuracy - 85.75%
F1 Score              - 0.85

6) Spectrogram with 2d Convolutional Neural Network
Training Accuracy     - 99.61%
Testing Data Accuracy - 74.07%
F1 Score              - 0.74
