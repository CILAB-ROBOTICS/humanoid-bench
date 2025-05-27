import serial
import platform
import matplotlib
from sympy.liealgebras.type_e import TypeE

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from collections import deque
from threading import Thread, Lock
import time

# ---------------- 기본 설정 ----------------
NUM_SENSORS = 48
BUFFER_LEN = 500
SERIAL_PORT = "/dev/tty.usbmodem2101"  # macOS 기준. Windows는 'COM3' 등
BAUD_RATE = 2000000

# ---------------- 버퍼 및 잠금 ----------------
data_buffers = [deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN) for _ in range(NUM_SENSORS)]
lock = Lock()

# ---------------- 그래프 설정 ----------------
plt.ion()
fig, axs = plt.subplots(8, 6, figsize=(16, 10))
axs = axs.flatten()
lines = []

for i in range(NUM_SENSORS):
    line, = axs[i].plot([], [], lw=1)
    axs[i].set_xlim(0, BUFFER_LEN)
    axs[i].set_ylim(0, 1023)
    axs[i].set_title(f"CH{i}", fontsize=8)
    axs[i].tick_params(labelsize=6)
    lines.append(line)

plt.tight_layout()

# ---------------- 시리얼 수신 쓰레드 ----------------
def serial_reader():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"[✓] Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    except Exception as e:
        print(f"[✗] Serial connection failed: {e}")
        return

    while True:
        try:
            raw = ser.readline()
            clean_line = raw.decode(errors='ignore').replace('\r', '').strip()
            values = list(map(int, clean_line.split('\t')))

            if len(values) != NUM_SENSORS:
                continue

            with lock:
                for i in range(NUM_SENSORS):
                    data_buffers[i].append(values[i])

        except ValueError:
            continue
        except Exception as e:
            print(f"[!] Read error: {e}")

# ---------------- 그래프 갱신 루프 ----------------
def plot_updater():
    while True:
        with lock:
            for i in range(NUM_SENSORS):
                lines[i].set_data(range(BUFFER_LEN), data_buffers[i])
                axs[i].set_xlim(0, BUFFER_LEN)
                axs[i].set_ylim(0, 1023)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)  # 20 FPS

# ---------------- 실행 ----------------
if __name__ == "__main__":
    reader_thread = Thread(target=serial_reader, daemon=True)
    reader_thread.start()

    plot_updater()  # 메인 루프는 그래프 갱신 전담
