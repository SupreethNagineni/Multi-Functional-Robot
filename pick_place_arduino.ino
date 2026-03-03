#include <AFMotor.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver armController = Adafruit_PWMServoDriver();
AF_DCMotor motor1(2);
AF_DCMotor motor2(3);

#define Echo A0
#define Trig A1
#define S1 A2
#define S2 A3
#define Speed 150

#define servo1 0
#define servo2 1
#define servo3 2
#define servo4 3

void setup() {
  pinMode(S1, INPUT);
  pinMode(S2, INPUT);
  pinMode(Trig, OUTPUT);
  pinMode(Echo, INPUT);

  armController.begin();
  armController.setPWMFreq(60);
  armController.setPWM(servo1, 0, 340);
  armController.setPWM(servo2, 0, 150);
  armController.setPWM(servo3, 0, 300);
  armController.setPWM(servo4, 0, 290);

  motor1.setSpeed(Speed);
  motor2.setSpeed(Speed);
}

void loop() {
  int distance = obstacle();
  Serial.println(distance);

  if (distance <= 9) {
    motor1.run(RELEASE);
    motor2.run(RELEASE);
    motor1.run(BACKWARD);
    motor2.run(BACKWARD);
    delay(100);
    motor1.run(RELEASE);
    motor2.run(RELEASE);
    delay(100);
    robotArm();
    delay(100);
    motor1.run(FORWARD);
    motor2.run(FORWARD);
    delay(100);
  } else {
    linefollower();
  }
}

void linefollower() {
  bool value1 = digitalRead(S1);
  bool value2 = digitalRead(S2);

  if (value1 == 0 && value2 == 0) {
    motor1.run(FORWARD);
    motor2.run(FORWARD);
  } else if (value1 == 1 && value2 == 1) {
    motor1.run(RELEASE);
    motor2.run(RELEASE);
  } else if (value1 == 1 && value2 == 0) {
    motor1.run(BACKWARD);
    motor2.run(FORWARD);
  } else if (value1 == 0 && value2 == 1) {
    motor1.run(FORWARD);
    motor2.run(BACKWARD);
  }
}

int obstacle() {
  digitalWrite(Trig, LOW);
  delayMicroseconds(4);
  digitalWrite(Trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(Trig, LOW);
  long t = pulseIn(Echo, HIGH);
  int cm = t / 29 / 2;
  return cm;
}

void robotArm() {

  for (int S4value = 290; S4value <= 490; S4value++) {
    armController.setPWM(servo4, 0, S4value);
    delay(10);
  }

  for (int S3value = 300; S3value <= 450; S3value++) {
    armController.setPWM(servo3, 0, S3value);
    delay(10);
  }

  for (int S2value = 150; S2value <= 190; S2value++) {
    armController.setPWM(servo2, 0, S2value);
    delay(10);
  }

  for (int S4value = 490; S4value > 290; S4value--) {
    armController.setPWM(servo4, 0, S4value);
    delay(10);
  }

  for (int S3value = 450; S3value > 300; S3value--) {
    armController.setPWM(servo3, 0, S3value);
    delay(10);
  }

  for (int S2value = 190; S2value <= 320; S2value++) {
    armController.setPWM(servo2, 0, S2value);
    delay(10);
  }

  for (int S1value = 340; S1value >= 150; S1value--) {
    armController.setPWM(servo1, 0, S1value);
    delay(10);
  }

  for (int S3value = 300; S3value <= 410; S3value++) {
    armController.setPWM(servo3, 0, S3value);
    delay(10);
  }

  for (int S4value = 290; S4value <= 490; S4value++) {
    armController.setPWM(servo4, 0, S4value);
    delay(10);
  }

  for (int S4value = 490; S4value > 290; S4value--) {
    armController.setPWM(servo4, 0, S4value);
    delay(10);
  }

  for (int S3value = 410; S3value > 300; S3value--) {
    armController.setPWM(servo3, 0, S3value);
    delay(10);
  }

  for (int S2value = 320; S2value > 150; S2value--) {
    armController.setPWM(servo2, 0, S2value);
    delay(10);
  }

  for (int S1value = 150; S1value < 340; S1value++) {
    armController.setPWM(servo1, 0, S1value);
    delay(10);
  }
}