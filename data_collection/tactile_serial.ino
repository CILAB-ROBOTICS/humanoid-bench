const int S0 = 2;
const int S1 = 3;
const int S2 = 4;
const int S3 = 5;

const int SIG_PINS[3] = {A0, A1, A2};  // MUX 0, MUX 1, MUX 2
const int NUM_MUX = 3;
const int NUM_CHANNELS = 16;

int sensorValues[NUM_MUX * NUM_CHANNELS];

void setup() {
  Serial.begin(2000000);

  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);

  Serial.println("Starting Fast 48-Channel MUX Sampling...");
}

void loop() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    // 1. 모든 MUX에서 해당 채널을 동시에 선택
    selectMuxChannel(ch);

    // 2. 전환 안정화 시간 (필요시 조절)
    delayMicroseconds(100);

    // 3. MUX 0~2의 출력값 읽기
    for (int mux = 0; mux < NUM_MUX; mux++) {
      int index = mux * NUM_CHANNELS + ch;
      sensorValues[index] = analogRead(SIG_PINS[mux]);
    }
  }

  // 4. 값만 출력 (탭 구분, 한 줄)
  for (int i = 0; i < NUM_MUX * NUM_CHANNELS; i++) {
    Serial.print(sensorValues[i]);
    if (i < NUM_MUX * NUM_CHANNELS - 1)
      Serial.print('\t');  // 탭 구분
    else
      Serial.println();    // 마지막은 줄바꿈
  }
}

void selectMuxChannel(int channel) {
  digitalWrite(S0, bitRead(channel, 0));
  digitalWrite(S1, bitRead(channel, 1));
  digitalWrite(S2, bitRead(channel, 2));
  digitalWrite(S3, bitRead(channel, 3));
}
