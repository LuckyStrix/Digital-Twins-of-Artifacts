#define n_pin 2
#define e_pin 3
#define s_pin 4
#define w_pin 5
#define g_pin 6
#define b_pin 7
#define s1_pin 8
#define s2_pin 9
#define s3_pin 10
#define s4_pin 11
int pin_list[] = {n_pin, e_pin, s_pin, w_pin, g_pin, b_pin, s1_pin, s2_pin, s3_pin, s4_pin};
char temp = 'a';
String input = "";
void setup() {
  Serial.begin(115200);
  for(int item : pin_list) {
    //Serial.println(item);
    pinMode(item, OUTPUT);
    digitalWrite(item, LOW);
  }
  digitalWrite(g_pin, HIGH);
}
void loop() {
  //will only read data if there is data to read
  //will also proceed to carry out instructions
  if (Serial.available() >= 8) {
    char buf[9];
    Serial.readBytes(buf, 8);
    buf[8] = '\0';
    uint8_t value = strtol(buf, nullptr, 2);
    // GPIOs from bits 0-5
    for (int i = 0; i < 6; i++) {
      digitalWrite(pin_list[i], (value >> (7 - i)) & 1);
    }
    // Special controls
    bool flash = (((value >> 3) & 1) == 0) && (((value >> 2) & 1) == 0);
    if(flash == true) {
      for(int i = 0; i < 2; i++) {
        digitalWrite(g_pin, HIGH);
        delay(500);
        digitalWrite(g_pin, LOW);
        delay(500);
        digitalWrite(b_pin, HIGH);
        delay(500);
        digitalWrite(b_pin, LOW);
        delay(500);
      }
      digitalWrite(g_pin, HIGH);
    }
    bool motor_enable = (value >> 1) & 1;
    bool motor_dir = (value >> 0) & 1;
    if(motor_enable == true) {
      int phase = 1;
      for(int i=0; i < 2048; i++) {
        if(motor_dir == true) {
          if(phase == 1) {
            digitalWrite(s1_pin, HIGH);
            phase = 2;
          }
          else if(phase == 2) {
            digitalWrite(s2_pin, HIGH);
            phase = 3;
          }
          else if(phase ==3) {
            digitalWrite(s3_pin, HIGH);
            phase = 4;
          }
          else if(phase ==4) {
            digitalWrite(s4_pin, HIGH);
            phase = 1;
          }
        } else{
          if(phase == 1) {
            digitalWrite(s1_pin, HIGH);
            phase = 4;
          }
          else if(phase == 2) {
            digitalWrite(s2_pin, HIGH);
            phase = 1;
          }
          else if(phase ==3) {
            digitalWrite(s3_pin, HIGH);
            phase = 2;
          }
          else if(phase ==4) {
            digitalWrite(s4_pin, HIGH);
            phase = 3;
          }
        }
        delay(15);
        digitalWrite(s1_pin, LOW);
        digitalWrite(s2_pin, LOW);
        digitalWrite(s3_pin, LOW);
        digitalWrite(s4_pin, LOW);
      }
    }
    //tell pi "we done working!!"
    Serial.print("e");
  }
}  


