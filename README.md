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
CRWU dataset 链接：https://pan.baidu.com/s/1L9NldqxIvYtSRm_Vb9RNdg 提取码:c52u

JNU dataset 链接:https://pan.baidu.com/s/1T1Eca5S0lnlzPw9HzCdcfw 提取码:421f

SEU dataset 链接:https://pan.baidu.com/s/1Z6LYSB0QomZrTkkDfFV0ng 提取码:455m

PHM09 dataset 链接:https://pan.baidu.com/s/1lPFHe3FXHzyjWEr8V9lOWQ 提取码:32xc

## Network Structure
![img.png](https://github.com/zccguess/OS-CNN/blob/master/readmeImages/test%20phase1.png)

## How to run the existing code
   #### Step 1: The training data set generation model passes train.py.<br>
   #### Step 2: Load the trained model passes main.py.<br>
   #### Step 3: Create a Weibull model for each known class.<br>
   #### Step 4: Distance modeling between test data and each known class Weibull model.<br>
   #### Step 5: Calculate the CDF probability corresponding to each test data.<br>
   #### Step 6: According to the CDF probability revise activation vectors, each sample is classified.<br>


## Note
For the Python interface to work, this requires preinstall Cython on the machine.
## Refer to the main.py for detail implementation


