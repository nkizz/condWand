# condWand
[Demonstration Video](https://youtu.be/uQUrC3XMN-k)

Using an LSTM to detect the tempo of a conductor from accelerometer data. This is useful for conductors by helping them stay on tempo when leading an orchestra without using an invasive click track.

The project spans the following files:
* `lstm.py`: This is the main training script, and contains the LSTM Keras implementation
* `dataGathering`: This is a sketch for the Arduino 33 BLE, and is used to report accelerometer data to collect for training and inference
* `condWandPi.py`: Uses Tensorflow Lite to perform inference against the trained network
* `condWandNeo.py`: Runs on a Circuit Express, receives RGB values over USB and displays it on the built-in LEDs

Other files include conversion utilities, test scripts, etc.

# Transformer Implementation
 
 A basic transformer encoder is implemented based on [these algorithms for computing multi-headed self attention and positional encoding:](https://www.kaggle.com/code/arunprathap/transformer-encoder-implementation/notebook)

The self attention mechanism is adjusted to work on inertial data by feeding the model a 6-dimensional-embedded tokenized sequence of time-series data into a self-attention encoder (stacked with N Layers) followed by a multi-layer-perceptron output stage that classifies two seconds of self-attended time-samples into 13 tempo classes (e.g. 65 bpm or 70 bpm or 120 bpm, etc.)

Results:
When training with anywhere between 2 and 3 encoder layers, a model dimensionality between 6 (raw input) and 32 (brought into higher dimension using convolutional layers (with f = 32 filters at the input) and with output MLP layers of varying hidden sizes, model validation accuracy does not appear to rise higher than ~22% during training. We suspect more research and potentially a more complicated model might bee necessary to find a working transformer architecture (increasing model complexity generally found to increase performance)
