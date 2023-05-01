# condWand
[Demonstration Video](https://youtu.be/uQUrC3XMN-k)

Using an LSTM to detect the tempo of a conductor from accelerometer data. This is useful for conductors by helping them stay on tempo when leading an orchestra without using an invasive click track.

The project spans the following files:
* `lstm.py`: This is the main training script, and contains the LSTM Keras implementation
* `dataGathering`: This is a sketch for the Arduino 33 BLE, and is used to report accelerometer data to collect for training and inference
* `condWandPi.py`: Uses Tensorflow Lite to perform inference against the trained network
* `condWandNeo.py`: Runs on a Circuit Express, receives RGB values over USB and displays it on the built-in LEDs

Other files include conversion utilities, test scripts, etc.

# Teansformer Implementation
 
 A basic transformer encoder is implemented based on [these algorithms for computing multi-headed self attention and positional encoding:](https://www.kaggle.com/code/arunprathap/transformer-encoder-implementation/notebook)
