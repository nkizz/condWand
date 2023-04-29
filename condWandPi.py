TARGET_BPM = 100

import numpy as np
from tflite_runtime.interpreter import Interpreter
import serial
import time
import math
import joblib
def floatRgb(mag, cmin, cmax):
    """ Return a tuple of floats between 0 and 1 for R, G, and B. """
    # Normalize to 0-1
    try: x = float(mag-cmin)/(cmax-cmin)
    except ZeroDivisionError: x = 0.5 # cmax == cmin
    blue  = 1-min((max((4*(0.75-x), 0.)), 1.))
    red   = 1-min((max((4*(x-0.25), 0.)), 1.))
    green = 1-min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
    return red, green, blue

interpreter = Interpreter(model_path="lstm5.tflite", )
interpreter.allocate_tensors()
data = np.loadtxt("100bpm3.csv", skiprows=1, delimiter=',',
                  dtype=np.float32)[119*30:119*32, 1:]
accelIn = serial.Serial(
    '/dev/serial/by-id/usb-Arduino_Nano_33_BLE_8D45D25C2F535E4D-if00', 115200, timeout=10)
rgbOut = serial.Serial(
    '/dev/serial/by-id/usb-Adafruit_Industries_LLC_CircuitPlayground_Express_F6E705213633535020312E37251607FF-if00', 115200, timeout=10)

accelScaler = joblib.load("accel.scaler")
gyroScaler = joblib.load("gyro.scaler")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#for i in data:
#    for j in i:
#        print(j, end=",")
#print(input_details)
#print(output_details)
count = 0
lastTime = time.time()
while True:
    data = np.roll(data, -1, axis=0)
    #print(b"Line: " + accelIn.readline())
    curSample = np.fromstring(accelIn.readline().decode('utf-8'), sep=',')[1:]
    if len(curSample) != 6:
        continue
    data[-1] = curSample
    data[-1, :3] = accelScaler.transform([data[-1, :3]])[0]
    data[-1, 3:] = gyroScaler.transform([data[-1, 3:]])[0]
    count += 1
    if count % 40 == 0:
        print("Latency: ", time.time() - lastTime)
        lastTime = time.time()
        interpreter.set_tensor(input_details[0]['index'], [data])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        #print(output_data)
        #bpm = 0
        #times = 60
        #for i in output_data[0]:
        #    bpm += i * times
        #    times += 5
        bpm = 60 + 5 * np.argmax(output_data[0])
        print(bpm)
        rgb = floatRgb(bpm, TARGET_BPM-30, TARGET_BPM+30)
        rgbStr = str(int(rgb[0]*255)) + "," + str(int(rgb[1]*255)) + "," + str(int(rgb[2]*255)) + "\r\n"
        rgbOut.write(rgbStr.encode('utf-8'))
