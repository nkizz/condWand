/*
  Arduino LSM9DS1 - Accelerometer Application

  This example reads the acceleration values as relative direction and degrees,
  from the LSM9DS1 sensor and prints them to the Serial Monitor or Serial Plotter.

  The circuit:
  - Arduino Nano 33 BLE

  Created by Riccardo Rizzo

  Modified by Jose García
  27 Nov 2020

  This example code is in the public domain.
*/

#include <Arduino_LSM9DS1.h>


float ax, ay, az, gx,gy,gz;
bool aRead, gRead;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println("Hz");
}

void loop() {

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    aRead = true;
  }
  if (IMU.gyroscopeAvailable()){
    IMU.readGyroscope(gx, gy, gz);
    gRead = true;
  }
  if (aRead && gRead){
    aRead = false;
    gRead = false;
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ax,4);
    Serial.print(",");
    Serial.print(ay,4);
    Serial.print(",");
    Serial.print(az,4);
    Serial.print(",");
    Serial.print(gx,4);
    Serial.print(",");
    Serial.print(gy,4);
    Serial.print(",");
    Serial.print(gz,4);
    Serial.print("\n");
  }
}











