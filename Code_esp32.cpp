/*
 * =====================================================================
 * ESP32 + MPU6050: Real-Time Complementary Filter Gait & Peak Angle Tracker
 * With Hardware Handshake (WHO_AM_I) & Auto-Zero Calibration
 * Project: SIH 2026 - AI-Assisted Osteoarthritis Screening System
 * =====================================================================
 */
#include <Wire.h>
#define MPU_ADDR 0x68
#define WHO_AM_I_REG 0x75
// Complementary Filter & Timing Settings
const float ALPHA = 0.98;           // 98% Gyro, 2% Accel
const unsigned long DT_MS = 10;     // 10 ms = 100 Hz
const float DT_SEC = 0.01;
const float GYRO_SCALE = 65.5;      // For ±500 deg/s
const float ACCEL_SCALE = 4096.0;   // For ±8g
// Calibration offset
float gyro_z_bias = 0.0;
// Tracking Variables
float current_angle = 0.0;
float peak_angle = 0.0;
float peak_timestamp = 0.0;
unsigned long last_sample_time = 0;
float last_strike_time = 0.0;
int stride_count = 0;
void setup() {
  Serial.begin(115200);
  delay(1000); // Allow serial monitor to open
  Serial.println("\n==================================================");
  Serial.println("   ESP32 + MPU6050 Gait & Peak Angle Analyzer     ");
  Serial.println("==================================================");
  Wire.begin(21, 22); // ESP32 pins: SDA=GPIO 21, SCL=GPIO 22
  Wire.setClock(400000); // Fast I2C (400 kHz)
  // -------------------------------------------------------------
  // 1. HARDWARE DETECTION CHECK (WHO_AM_I Register 0x75)
  // -------------------------------------------------------------
  Serial.print("[1/3] Detecting MPU6050 on I2C bus... ");
  
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(WHO_AM_I_REG);
  byte error = Wire.endTransmission(false);
  
  if (error != 0) {
    Serial.println("\n[FAILED] I2C Connection Error!");
    Serial.println(">>> CHECK WIRING: VCC -> 3.3V, GND -> GND, SDA -> GPIO 21, SCL -> GPIO 22");
    while (1); // Halt execution
  }
  Wire.requestFrom(MPU_ADDR, 1);
  byte chip_id = Wire.read();
  if (chip_id == 0x68 || chip_id == 0x70 || chip_id == 0x72) {
    Serial.print("FOUND! (Device ID: 0x");
    Serial.print(chip_id, HEX);
    Serial.println(")");
  } else {
    Serial.print("\n[FAILED] Unknown Chip ID: 0x");
    Serial.println(chip_id, HEX);
    Serial.println(">>> Make sure AD0 pin is connected to GND.");
    while (1); // Halt execution
  }
  // -------------------------------------------------------------
  // 2. CONFIGURE REGISTERS
  // -------------------------------------------------------------
  // Wake up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);
  // Set Accelerometer to ±8g (Register 0x1C = 0x10)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x10);
  Wire.endTransmission(true);
  // Set Gyroscope to ±500 deg/s (Register 0x1B = 0x08)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x08);
  Wire.endTransmission(true);
  // -------------------------------------------------------------
  // 3. AUTO-CALIBRATE GYROSCOPE ZERO-BIAS (Keep leg still)
  // -------------------------------------------------------------
  Serial.println("[2/3] Calibrating Gyroscope... KEEP LEG STILL for 2 seconds!");
  float sum_gz = 0.0;
  const int CALIB_SAMPLES = 200;
  for (int i = 0; i < CALIB_SAMPLES; i++) {
    int16_t ax, ay, az, gx, gy, gz;
    readMPU(ax, ay, az, gx, gy, gz);
    sum_gz += ((float)gz / GYRO_SCALE);
    delay(10);
  }
  gyro_z_bias = sum_gz / CALIB_SAMPLES;
  Serial.print("      Calibration Complete! Gyro Bias: ");
  Serial.print(gyro_z_bias, 3);
  Serial.println(" deg/s");
  // Initial angle from gravity
  int16_t ax, ay, az, gx, gy, gz;
  readMPU(ax, ay, az, gx, gy, gz);
  current_angle = atan2((float)az, (float)-ax) * 180.0 / PI;
  Serial.println("[3/3] SYSTEM READY! You can start walking now.\n");
  Serial.println("-------------------------------------------------------------------------");
  Serial.println("Stride_#  |  Peak_Angle (deg)  |  Stride_Duration (s)  |  Peak_Time (s)");
  Serial.println("-------------------------------------------------------------------------");
  last_sample_time = millis();
  last_strike_time = millis() / 1000.0;
}
void loop() {
  unsigned long now = millis();
  if (now - last_sample_time >= DT_MS) {
    last_sample_time = now;
    float current_time_sec = now / 1000.0;
    int16_t raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz;
    readMPU(raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz);
    float ax_g = (float)raw_ax / ACCEL_SCALE;
    float ay_g = (float)raw_ay / ACCEL_SCALE;
    float az_g = (float)raw_az / ACCEL_SCALE;
    
    // Subtract calibrated zero-bias so gyro never drifts
    float gz_degs = ((float)raw_gz / GYRO_SCALE) - gyro_z_bias;
    // Pitch from gravity
    float acc_angle = atan2(az_g, -ax_g) * 180.0 / PI;
    // Complementary Filter
    current_angle = ALPHA * (current_angle + gz_degs * DT_SEC) + (1.0 - ALPHA) * acc_angle;
    float leg_angle = abs(current_angle);
    // Track the Peak Angle during swing
    if (leg_angle > peak_angle) {
      peak_angle = leg_angle;
      peak_timestamp = current_time_sec;
    }
    // Heel Strike Impact Shock
    float impact_mag_g = sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
    float time_since_last_strike = current_time_sec - last_strike_time;
    // Completed Stride Trigger
    if (impact_mag_g > 1.8 && peak_angle > 20.0 && time_since_last_strike > 0.65) {
      stride_count++;
      float stride_duration = time_since_last_strike;
      last_strike_time = current_time_sec;
      // Print clean, formatted stride result
      Serial.print("Stride ");
      if (stride_count < 10) Serial.print(" ");
      Serial.print(stride_count);
      Serial.print("   |  ");
      Serial.print(peak_angle, 2);
      Serial.print(" deg          |  ");
      Serial.print(stride_duration, 3);
      Serial.print(" s              |  ");
      Serial.print(peak_timestamp, 2);
      Serial.println(" s");
      // Reset for next stride
      peak_angle = 0.0;
    }
  }
}
void readMPU(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read(); // Skip temperature bytes
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
}