# Car Damage Assessment

## Overview

This project involves using deep computer vision techniques to detect car damages, classify the severity level, identify the damaged part, and estimate the repair costs based on images of damaged cars with varying severity degrees. This is often used for automated damage assessment in the automotive industry or insurance claims processing.


This work is realised by:

* Yosr Abid
* Oussema Louhichi

## Steps

* Check if the damaged car image is real or AI-generated
  To verify that the images are genuine using CNN and VGG16 for image classification.

* Damage Severity Assessment
  Used to classify car damage into different severity levels ( moderate, severe and minor) using a CNN.

* Damaged parts segmentation
  To detect which parts of the car are damaged and identify the type of damage using YOLOv8, Mask R-CNN.



## Getting Started


1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/Car-Damage-Assessment.git
2. Download pre-trained  Models: https://drive.google.com/drive/folders/1BK-ErOYJ9Yf-p0zmUEE5L-8sswf0JKb2?usp=drive_link
