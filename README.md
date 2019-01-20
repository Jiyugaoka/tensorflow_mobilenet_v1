# tensorflow_mobilenet_v1
This repo mainly use **tensorflow 1.8, python 3.6.2**, and if it can not work correctly in a higher version, please inform me with a issue, I will update relavent code. 
# What's new in this repo
1. The model defination of MobileNet-V1 I used is mainly based on the work of [timctho](https://github.com/timctho/mobilenet-v2-tensorflow). I just modified the varible scope of **models.py** a bit to make its varible scope consistant with the google's official checkpoint, so that we can use the ImageNet pre-trained model to initialize our model.

2. I add another scripts as follows, so you can use your own dataset to train the MobileNet-v1 model.
   + get_tensor_from_checkpoint.py
   + loaddata.py
   + train_network.py

For more details about MobileNet-V1, please refer to the paper:[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications](https://arxiv.org/pdf/1704.04861.pdf).

# Usage
1. Download the official pre-trained model [checkpoint](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html).
In this repo I use the pre-trained model with resolution = 224 and width Multiplier = 1.

2. Extract the downloaded file into ```./checkpoint```, the directory should have this structure:
``` checkpoint/mobilenet_v1_1.0_224_2017_06_14 ```

   After extraction you can use the following script to inspect all the variable name in the checkpoint file.
   
   ```python get_tensor_from_checkpoint.py -c "your checkpoint file path"```

3. Prepare your train/test dataset,and re-organize the directory as follows:
   
   **Because the resolusion of the model is 224, so you should resize your train/test images to 224 before you use them**. 
``` 
   dataset/
          |/train/
          |      |/class1
          |      |/class2
          |      |/class3
          |      |/class4
          |/trainAnno/
          
    ......
          |/test/
          |     |/class1
          |     |/class2
          |     |/class3
          |     |/class4
          |/testAnno/
          
    ...... 
  ```  
4. Load the dataset and compute the mean of the train dataset for pre-processing.
    
    ```python loaddata.py -d "Your dataset dir" -n "the number of classes you want to classify"```
    
   The last output in your command window will show you the pixel means ```train_dataset_mean``` of your train dataset which will use      again in the **train_network.py** 
    
5. Now you can train the MobileNet_v1 :```python train_network.py```

   ***Notice:*** if you don't have sufficient train samples, you can modify the var ```exclude_vars``` to freeze more layers to reuse more vars in the original checkpoint. But in my experiments unfreezing more layers may achieve better performence. 
   
# Contents updata late
+ add a ```predict.py``` script
+ replace the dataset preparation module using ```tf.data``` API. 
  As for why I don't replace it now, I still can't figure out one thing when I parse a tfrecord file.
  But I think it won't be long, I will update soon.
