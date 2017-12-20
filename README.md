# YOLO-Pi: Real Time Object Recognition on Raspberry Pi

The purpose of this project is to attach a USB camera to a Raspberri Pi and then automatically detect objects that the camera sees.  

To do this we take [yolo](https://pjreddie.com/darknet/yolo/) weigts and configuration and run it through [yad2k](https://github.com/allanzelener/YAD2K).  This in turn generates a [keras](https://keras.io/) model.  With the keras model we modify the test_yolo code from the yad2k project and add in opencv3 so we can get the camera real time.  

## Using existing Models

Grab weights and config file from main [YOLO site.](https://pjreddie.com/darknet/yolo/)

Use tiny-yolo-voc as this seems to be supported. Once you get the weights and the config run it through yad2k to create the tiny-yolo-voc.h5

```
./yad2k.py tiny-yolo-voc.cfg tiny-yolo-voc.weights model_data/tiny-yolo-voc.h5
```
Now we have the ```tiny-yolo-voc.h5```

I've simplified the ```yolo-pi.py``` script by not taking in command line arguments.  Instead you can just change a few lines that look like the below:

```python
model_path = 'model_data/tiny-yolo-voc.h5'
anchors_path = 'model_data/tiny-yolo-voc_anchors.txt'
classes_path = 'model_data/pascal_classes.txt'
```

Running the ```yolo-pi.py``` will give you output like the following:

![img](src/images/example.png)

As you can tell from the picture above the voc model is not super accurate but its pretty fast generating about 1 frame ever 2 seconds on a macbook pro. 

## Installation Notes
Installation sucks.  

```<python rant>``` 
While Python is super easy to write code it is extremely difficult to get the environment for the code to run in work.  
```</python rant>``` 

Presently working to make this eaiser.  Basically you need: 

* [x] Python 3 (we use anaconda)
* [x] OpenCV 3.0.0 (3.1 has bugs on Mac and crashes)
* [x] Keras
* [x] Pillow
* [x] Tensorflow
* [x] NumPy
 
opencv3 3.0.0 we installed using:

```
conda install -c jlaura opencv3
```
You could use 3.1.1 from the meno repo but it crashes, so we stick with 3.0.0 for now. 

## Running in Docker


```
docker run -it --rm --privileged \
	--device=/dev/video0:/dev/video0 \
	-v `pwd`:/app ashya/yolopi /bin/bash
```
