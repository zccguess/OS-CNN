# This is a pytorch implementation of the paper Intelligent Bearing Fault Diagnosis Based on Open Set Convolutional Neural Network (OS-CNN)

## Environment
    tensorflow 1.15.0
    keras      2.3.1
    numpy      1.19.5
    python     3.6
    scikit-learn  
    libmr
    pillow

## Data Preparation
CRWU dataset 链接：https://pan.baidu.com/s/10daD_ro2polDmar1oNiMAQ 提取码：gfpc

JNU dataset 链接：https://pan.baidu.com/s/1hbJ_Mtyiu1cBTd2_XBY0og 提取码：s9ki 

SEU dataset 链接：https://pan.baidu.com/s/1e5arWFx6PhcUMZwEGMj0wQ 提取码：5yc3

PHM09 dataset 链接：https://pan.baidu.com/s/1QDGiK4Ve1y2HM6Tcees65w 提取码：vnqn

## Network Structure
![img.png](https://github.com/zccguess/OS-CNN/blob/master/readmeImages/test%20phase1.png)

## How to run the existing code
   #### Step 1: Train a CNN model for the dataset.<br>
   #### Step 2: Load the trained model,and load the training data you trained the trained model.<br>
   #### Step 3: The feature center and corresponding distance of each class center are calculated,and Weibull distribution is established according to the maximum                      distance.<br>
   #### Step 4: Evaluate the distribution of test samples based on the Weibull distribution of training data.<br>
   #### Step 5: Calculate the corresponding CDF probability based on different distributions of test samples.<br>
   #### Step 6: According to the CDF probability revise activation vectors, each sample is classified.<br>


## Note
For the Python interface to work, this requires preinstall Cython on the machine.
## Refer to the main.py for detail implementation


