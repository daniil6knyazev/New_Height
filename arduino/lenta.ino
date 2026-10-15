#include <Ethernet.h>
#include <EthernetUdp.h>
#include <EEPROM.h>
#include <DxlMaster.h>

#define UDP_PORT 8888
#define CONVEYOR_ID 21
#define DXL_DIR_PIN 2

DynamixelDevice conveyor(CONVEYOR_ID);
DynamixelMotor motor(CONVEYOR_ID);

IPAddress MyIP(192, 168, 42, 49);
EthernetUDP udp;
char packetBuffer[UDP_TX_PACKET_MAX_SIZE + 1];
byte mac[6];

String usbInputString = "";

void executeAction(int mode) {
  uint16_t speedVal = 0;

  if (mode == 2) {
    speedVal = 500;        // Движение вперед (обычное значение от 0 до 1023)
  } else if (mode == 1) {
    speedVal = 1024 + 500; // Движение назад (реверс: 1024 + скорость)
  } else {
    speedVal = 0;          // Стоп
  }

  // Переводим пин DIR на передачу
  pinMode(DXL_DIR_PIN, OUTPUT);
  digitalWrite(DXL_DIR_PIN, HIGH);
  delay(1);

  // Отправляем команды в регистры Dynamixel ID 12
  conveyor.write(24, (uint8_t)1);         // Torque Enable
  conveyor.write(28, (uint16_t)speedVal);  // Регистр скорости 28
  conveyor.write(32, (uint16_t)speedVal);  // Регистр скорости 32

  // Дублируем через объект мотора
  motor.enableTorque();
  motor.wheelMode();
  motor.speed(speedVal);

  // Возвращаем пин DIR в режим приема
  digitalWrite(DXL_DIR_PIN, LOW);

  Serial.print("-> Конвейер переведен в режим команды: ");
  Serial.println(mode);
}

void setup() {
  Serial.begin(115200);
  pinMode(DXL_DIR_PIN, OUTPUT);
  digitalWrite(DXL_DIR_PIN, HIGH);
  DxlMaster.begin(1000000); 
  conveyor.init();
  motor.init();
  digitalWrite(DXL_DIR_PIN, LOW);

  mac[0] = 0x02;
  mac[1] = MyIP[1];
  mac[2] = MyIP[2];
  mac[3] = MyIP[3];
  mac[4] = MyIP[2];
  mac[5] = MyIP[3];
  Ethernet.begin(mac, MyIP);
  udp.begin(UDP_PORT);
  
  Serial.println("System Ready with Reverse Support!");
}

void loop() {
  // Неблокирующее чтение команд из USB
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '#') {
      usbInputString.trim();
      if (usbInputString.startsWith("c:")) {
        int state = usbInputString.substring(2).toInt();
        executeAction(state);
      }
      usbInputString = ""; 
    } else if (inChar != '\r' && inChar != '\n') {
      usbInputString += inChar;
    }
  }

  // Чтение по UDP
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, UDP_TX_PACKET_MAX_SIZE);
    if (len > 0) {
      packetBuffer[len] = '\0';
      String udpCmd = String(packetBuffer);
      udpCmd.trim();
      if (udpCmd.startsWith("c:")) {
        int state = udpCmd.substring(2).toInt();
        executeAction(state);
      }
    }
  }
}
