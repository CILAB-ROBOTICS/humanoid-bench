"""
Multimodal Collector with Enhanced Graphs
────────────────────────────────────────────
• 1280x720 저장, 320x180 프리뷰 전송 (큐 maxsize=1)
• 센서 40채널 → 20·20 그래프 (가로 꽉 채움 + 세로/가로선 + 글자 확대)
• 카메라/센서 프로세스가 직접 HDF5로 기록
"""

import cv2, h5py, queue, serial, tkinter as tk, numpy as np
from PIL import Image, ImageTk
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Event, Process, Queue, Manager
from pathlib import Path
from typing import Dict, Tuple

# CONFIG
CAMERA_IDS      = [0, 1]
FRAME_SIZE      = (1280, 720)
DISPLAY_SCALE   = 0.25
DISPLAY_SIZE    = (int(FRAME_SIZE[0]*DISPLAY_SCALE), int(FRAME_SIZE[1]*DISPLAY_SCALE))
FPS             = 30
SAVE_FLUSH_EVERY= 500
PREVIEW_MAX     = 1
GUI_REFRESH_MS  = 66
H5_COMP_LVL     = 1
SERIAL_PORT     = "/dev/tty.usbmodem2101"
BAUD_RATE       = 2_000_000
NUM_SENSORS     = 48
DISPLAY_SENSORS = 40
GROUPS          = 2
SENS_PER_GRP    = DISPLAY_SENSORS // GROUPS
CANVAS_WIDTH    = 1280
BAR_PADDING     = 2
BAR_W           = (CANVAS_WIDTH - SENS_PER_GRP * BAR_PADDING) // SENS_PER_GRP
BAR_H           = 120
OUTPUT_DIR      = Path("recordings"); OUTPUT_DIR.mkdir(exist_ok=True)

# DATA CLASSES
@dataclass
class FramePacket:
    ts: float; frame: np.ndarray

@dataclass
class SensorPacket:
    ts: float; values: np.ndarray

# HELPERS
def utc_now_s():
    return datetime.utcnow().replace(tzinfo=timezone.utc).timestamp()

def create_cam_h5(path, cid, shape):
    f = h5py.File(path / f"cam{cid}.h5", "w")
    f.create_dataset("frames", (0,*shape), maxshape=(None,*shape), dtype=np.uint8,
                     chunks=(1,*shape), compression="gzip", compression_opts=H5_COMP_LVL)
    f.create_dataset("ts", (0,), maxshape=(None,), dtype=np.float64, chunks=True)
    f.attrs["camera_id"] = cid
    return f

def create_sensor_h5(path):
    f = h5py.File(path / "sensors.h5", "w")
    f.create_dataset("values", (0,NUM_SENSORS), maxshape=(None,NUM_SENSORS), dtype=np.int16,
                     chunks=(1024,NUM_SENSORS), compression="gzip", compression_opts=H5_COMP_LVL)
    f.create_dataset("ts", (0,), maxshape=(None,), dtype=np.float64, chunks=True)
    return f

# WORKERS
def camera_worker(cid, preview_q, stop, rec_flag, shared):
    cap = cv2.VideoCapture(cid, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened(): print(f"[Cam {cid}] open failed"); return

    h5 = None; count = 0
    shape = (FRAME_SIZE[1], FRAME_SIZE[0], 3)
    while not stop.is_set():
        ret, frame = cap.read()
        if not ret: continue
        ts = utc_now_s()

        if rec_flag.is_set() and h5 is None:
            sess = OUTPUT_DIR / shared["session"]
            sess.mkdir(exist_ok=True)
            h5 = create_cam_h5(sess, cid, shape); count = 0
        elif (not rec_flag.is_set()) and h5:
            h5.flush(); h5.close(); h5 = None

        if h5 is not None:
            h5["frames"].resize(count+1,0); h5["ts"].resize(count+1,0)
            h5["frames"][count] = frame; h5["ts"][count] = ts
            count += 1
            if count % SAVE_FLUSH_EVERY == 0: h5.flush()

        if not preview_q.full():
            preview_q.put_nowait(FramePacket(ts, cv2.resize(frame, DISPLAY_SIZE)))

    if h5: h5.flush(); h5.close()
    cap.release()


def sensor_worker(preview_q, stop, rec_flag, shared):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print("[Serial] open error:", e); return

    h5 = None; count = 0
    while not stop.is_set():
        try:
            raw = ser.readline()
            vals = list(map(int, raw.decode(errors="ignore").strip().split("\t")))
            if len(vals) != NUM_SENSORS: continue
        except: continue

        ts = utc_now_s()
        pkt = SensorPacket(ts, np.array(vals, np.int16))

        if rec_flag.is_set() and h5 is None:
            sess = OUTPUT_DIR / shared["session"]
            sess.mkdir(exist_ok=True)
            h5 = create_sensor_h5(sess); count = 0
        elif (not rec_flag.is_set()) and h5:
            h5.flush(); h5.close(); h5 = None

        if h5 is not None:
            h5["values"].resize(count+1,0); h5["ts"].resize(count+1,0)
            h5["values"][count] = pkt.values; h5["ts"][count] = ts
            count += 1
            if count % SAVE_FLUSH_EVERY == 0: h5.flush()

        if not preview_q.full():
            preview_q.put_nowait(pkt)

    if h5: h5.flush(); h5.close()
    ser.close()

# GUI
class Dashboard(tk.Tk):
    def __init__(self, cam_qs, sensor_q, stop, rec_flag, shared):
        super().__init__()
        self.cam_qs, self.sensor_q = cam_qs, sensor_q
        self.stop, self.rec_flag, self.shared = stop, rec_flag, shared
        self.title("Multimodal Collector")

        ph_black = ImageTk.PhotoImage(Image.fromarray(
            np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), np.uint8)))
        self.photos, self.labels = {}, {}
        for i in range(4):
            lbl = tk.Label(self, image=ph_black, bg="black"); lbl.image = ph_black
            lbl.grid(row=0, column=i, padx=5, pady=5)
            self.labels[i] = lbl

        self.canvases = []
        for g in range(GROUPS):
            cv = tk.Canvas(self, width=CANVAS_WIDTH, height=BAR_H+25, bg="white")
            cv.grid(row=1+g, column=0, columnspan=4)
            self.canvases.append(cv)
        self.curr_vals = np.zeros(DISPLAY_SENSORS, np.int16)

        ctrl = tk.Frame(self); ctrl.grid(row=3, column=0, columnspan=4, pady=5)
        self.btn = tk.Button(ctrl, text="▶ Start Recording", command=self.toggle)
        self.btn.pack(side="left", padx=10)

        self.after(GUI_REFRESH_MS, self.update_gui)

    def toggle(self):
        if self.rec_flag.is_set():
            self.rec_flag.clear(); self.btn.config(text="▶ Start Recording")
        else:
            self.shared["session"] = datetime.utcnow().strftime("session_%Y%m%d_%H%M%S")
            self.rec_flag.set(); self.btn.config(text="⏸ Pause Recording")

    def draw_bars(self):
        for g, cv in enumerate(self.canvases):
            cv.delete("all")
            start = g * SENS_PER_GRP; end = start + SENS_PER_GRP
            y_mid = BAR_H - int((512 / 1023) * BAR_H)
            cv.create_line(0, y_mid, CANVAS_WIDTH, y_mid, fill="#888", dash=(2, 2))

            for i, v in enumerate(self.curr_vals[start:end]):
                h = int((v / 1023) * BAR_H)
                x = i * (BAR_W + BAR_PADDING)
                cv.create_rectangle(x, BAR_H - h, x + BAR_W, BAR_H, fill="skyblue", outline="black")
                if i % 5 == 0:
                    cv.create_line(x, 0, x, BAR_H, fill="#ccc")
                    cv.create_text(x + BAR_W // 2, BAR_H + 4, text=str(start + i), anchor="n",
                                   font=("Arial", 10, "bold"))

    def update_gui(self):
        for cid, q in self.cam_qs.items():
            try:
                pkt = q.get_nowait()
                img = ImageTk.PhotoImage(Image.fromarray(
                      cv2.cvtColor(pkt.frame, cv2.COLOR_BGR2RGB)))
                self.photos[cid] = img
                self.labels[cid].configure(image=img)
            except queue.Empty: pass

        try:
            spkt = self.sensor_q.get_nowait()
            self.curr_vals = spkt.values[:DISPLAY_SENSORS]
        except queue.Empty: pass
        self.draw_bars()

        if not self.stop.is_set():
            self.after(GUI_REFRESH_MS, self.update_gui)

    def on_closing(self):
        self.stop.set(); self.quit()

# MAIN
def main():
    stop, rec_flag = Event(), Event()
    shared = Manager().dict(session="")
    cam_qs = {cid: Queue(PREVIEW_MAX) for cid in CAMERA_IDS}
    sensor_q = Queue(PREVIEW_MAX)

    procs = []
    for cid in CAMERA_IDS:
        p = Process(target=camera_worker, args=(cid, cam_qs[cid], stop, rec_flag, shared), daemon=True)
        p.start(); procs.append(p)

    sp = Process(target=sensor_worker, args=(sensor_q, stop, rec_flag, shared), daemon=True)
    sp.start(); procs.append(sp)

    app = Dashboard(cam_qs, sensor_q, stop, rec_flag, shared)
    app.protocol("WM_DELETE_WINDOW", app.on_closing); app.mainloop()

    stop.set()
    for p in procs: p.join()

if __name__ == "__main__":
    main()
