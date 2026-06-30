// Stepper motor controller for T-Capture turntable
// Receives 8-byte serial commands from the PC: [4-digit num_captures][4-digit steps]
// Steps the motor by <steps> pulses, then sends 'e' to signal completion.

#define STEP_PIN      3
#define DIRECTION_PIN 4
#define ENABLE_PIN    5

#define STEP_HIGH_US  2
#define STEP_LOW_MS   5

void setup() {
  Serial.begin(115200);
  pinMode(STEP_PIN,      OUTPUT);
  pinMode(DIRECTION_PIN, OUTPUT);
  pinMode(ENABLE_PIN,    OUTPUT);
  digitalWrite(DIRECTION_PIN, HIGH);
  digitalWrite(ENABLE_PIN,    HIGH);  // disabled until a command arrives
}

void loop() {
  if (Serial.available() > 7) {
    char buf[9] = {0};
    for (int k = 0; k < 8; k++) {
      buf[k] = (char)Serial.read();
    }

    // First 4 chars = num_captures (not used for motor timing, reserved for future use)
    // Last  4 chars = steps to advance
    String packet = String(buf);
    int steps = packet.substring(4, 8).toInt();

    digitalWrite(ENABLE_PIN, LOW);   // enable driver
    for (int i = 0; i < steps; i++) {
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(STEP_HIGH_US * 1000);
      digitalWrite(STEP_PIN, LOW);
      delay(STEP_LOW_MS);
    }
    digitalWrite(ENABLE_PIN, HIGH);  // disable driver between moves

    Serial.print('e');  // acknowledge completion to PC
  }
}
