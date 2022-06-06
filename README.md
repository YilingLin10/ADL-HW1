---
tags: ADL
---
# README (HW1)
https://www.csie.ntu.edu.tw/~miulab/s110-adl/doc/A1_RNN.pdf
## Environment setup
* Python 3.8
```
pip install -r requirements.txt
```
## Data preprocessing
* Run the following script to preprocess data for both intent-classification model and slot-tagging model
```
bash preprocess.sh /path/to/intent_cls_data_directory /path/to/slot_tag_data_directory
```
* The preprocessed files will saved to the directory **./cache**.
## Intent-classification model
* Run the following script to train the intent-classification model
```
python3.8 train_intent.py
```
* The trained model will be output to the directory **./ckpt/intent/** with file name **model.ckpt**.
## Slot-tagging model
* Run the following script to train the slot-tagging model
```
python3.8 train_slot.py
```
* The trained model will be output to the directory **./ckpt/slot/** with file name **model.ckpt**.
