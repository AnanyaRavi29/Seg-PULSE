import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import einops
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from scipy.signal import find_peaks
import requests
import threading

# --- CONFIG ---
ONNX_MODEL_PATH = r"C:\Users\anany\Downloads\rppg_model_foreeeeee.onnx"
WIN_SIZE = 128
HEIGHT, WIDTH = 40, 140
FOREHEAD_LANDMARKS = [103, 104, 105, 66, 107, 55, 8, 285, 336, 296, 334, 333, 332, 297, 338, 10, 109, 67]
LEFT_CHEEK_LANDMARKS = [116,117,118,100,126,209,129,203,206,186,212,135,136,172,138,215,177]
RIGHT_CHEEK_LANDMARKS = [345, 352, 376, 401, 435, 367, 397, 365, 364, 432, 410, 426, 423, 358, 429, 355, 329, 347, 346]
FS = 30  # Approximate webcam FPS, adjust if needed

def normalize(sig):
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return frame.transpose(2, 0, 1).astype('float32')

def estimate_bpm(signal, fs):
    peaks, _ = find_peaks(signal, distance=fs*0.4)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        avg_rr = np.mean(rr_intervals)
        bpm = 60.0 / avg_rr if avg_rr > 0 else 0
        return int(bpm)
    return 0

def draw_face_rois(frame, forehead_points, left_cheek_points, right_cheek_points):
    # Forehead: cyan
    if len(forehead_points) > 2:
        pts = np.array(forehead_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0,220,220), thickness=3)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color=(0,220,220, 60))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    # Left cheek: green
    if len(left_cheek_points) > 2:
        pts = np.array(left_cheek_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=3)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color=(0,255,0, 60))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    # Right cheek: magenta
    if len(right_cheek_points) > 2:
        pts = np.array(right_cheek_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255,0,255), thickness=3)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color=(255,0,255, 60))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

def rounded_border_pil(img, radius=28, border=8, shadow=6):
    w, h = img.size
    total_w, total_h = w + 2*border + 2*shadow, h + 2*border + 2*shadow
    base = Image.new('RGBA', (total_w, total_h), (0,0,0,0))
    shadow_layer = Image.new('RGBA', (w+2*border, h+2*border), (30,30,30,100))
    mask = Image.new('L', (w+2*border, h+2*border), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0,0,w+2*border-1,h+2*border-1), radius=radius+border, fill=255)
    shadow_layer.putalpha(mask)
    base.paste(shadow_layer, (shadow,shadow), mask=shadow_layer)
    border_layer = Image.new('RGBA', (w+2*border, h+2*border), (0,220,220,255))
    mask = Image.new('L', (w+2*border, h+2*border), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0,0,w+2*border-1,h+2*border-1), radius=radius+border, fill=255)
    draw.rounded_rectangle((border, border, w+border-1, h+border-1), radius=radius, fill=0)
    border_layer.putalpha(mask)
    base.paste(border_layer, (shadow,shadow), mask=border_layer)
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0,0,w-1,h-1), radius=radius, fill=255)
    img.putalpha(mask)
    base.paste(img, (border+shadow, border+shadow), mask=img)
    return base

# --- Model and FaceMesh ---
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = ort_session.get_inputs()[0].name
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = None
roi_buffer = deque(maxlen=WIN_SIZE)
signal = np.zeros(WIN_SIZE)
running = False

# --- Tkinter GUI setup ---
root = tk.Tk()
root.title("Real-Time rPPG (Python GUI)")
root.configure(bg="#181c20")

# "Card" frame for webcam + controls
card_frame = tk.Frame(root, bg="#23272b", bd=0, relief=tk.FLAT)
card_frame.pack(side=tk.LEFT, padx=32, pady=32)

# Webcam frame label (will show the webcam image)
lmain = tk.Label(card_frame, bg="#23272b")
lmain.pack(padx=10, pady=16)

# Controls and HR display
control_frame = tk.Frame(card_frame, bg="#23272b")
control_frame.pack(pady=10)
start_btn = tk.Button(control_frame, text="Start", width=10, font=("Segoe UI", 12, "bold"), bg="#00cccc", fg="#fff", activebackground="#00aaaa", relief=tk.RAISED, bd=0)
stop_btn = tk.Button(control_frame, text="Stop", width=10, font=("Segoe UI", 12, "bold"), bg="#444", fg="#fff", activebackground="#222", relief=tk.RAISED, bd=0, state=tk.DISABLED)
hr_label = tk.Label(control_frame, text="Heart Rate: -- BPM", font=("Segoe UI", 16, "bold"), bg="#23272b", fg="#00ff99", pady=8)
start_btn.grid(row=0, column=0, padx=7)
stop_btn.grid(row=0, column=1, padx=7)
hr_label.grid(row=1, column=0, columnspan=2, pady=(16,0))

# --- Chatbot UI (Ollama) ---
chat_frame = tk.Frame(card_frame, bg="#23272b")
chat_frame.pack(pady=20)
chat_title = tk.Label(chat_frame, text="AI Chatbot (Local, Free)", font=("Segoe UI", 14, "bold"), bg="#23272b", fg="#00cccc")
chat_title.pack(pady=(0,5))
chat_history = tk.Text(chat_frame, height=10, width=40, bg="#181c20", fg="#fff", font=("Segoe UI", 11), wrap=tk.WORD, state=tk.DISABLED)
chat_history.pack(pady=5)
chat_entry = tk.Entry(chat_frame, width=30, font=("Segoe UI", 12))
chat_entry.pack(side=tk.LEFT, padx=5)
send_btn = tk.Button(chat_frame, text="Send", bg="#00cccc", fg="#fff", font=("Segoe UI", 12, "bold"))

def ollama_chat(user_msg):
    url = "http://localhost:11434/api/generate"
    data = {"model": "phi", "prompt": user_msg, "stream": False}
    try:
        response = requests.post(url, json=data, timeout=60)
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error: {e}"

def send_message():
    user_msg = chat_entry.get().strip()
    if not user_msg:
        return
    chat_entry.delete(0, tk.END)
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {user_msg}\n")
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)
    root.update_idletasks()

    def get_bot_reply():
        bot_msg = ollama_chat(user_msg)
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, f"Bot: {bot_msg}\n")
        chat_history.config(state=tk.DISABLED)
        chat_history.see(tk.END)

    threading.Thread(target=get_bot_reply, daemon=True).start()

send_btn.config(command=send_message)
send_btn.pack(side=tk.LEFT, padx=5)
chat_entry.bind("<Return>", lambda event: send_message())

# --- Plot title above the plot ---
plot_title = tk.Label(root, text="SEG-PULSE APPLICATION", font=("Segoe UI", 18, "bold"), fg="#00cccc", bg="#181c20")
plot_title.pack(side=tk.TOP, pady=(20, 0))

# --- Matplotlib Figure for rPPG signal ---
fig, ax = plt.subplots(figsize=(6, 2))
line, = ax.plot(signal, label="rPPG signal", color="#00cccc", linewidth=2.0)
ax.set_facecolor("#181c20")
fig.patch.set_facecolor("#181c20")
ax.set_ylim(0, 1)
ax.set_xlim(0, WIN_SIZE)
ax.tick_params(colors='#aaa')
ax.spines['bottom'].set_color('#00cccc')
ax.spines['top'].set_color('#00cccc')
ax.spines['right'].set_color('#00cccc')
ax.spines['left'].set_color('#00cccc')
ax.yaxis.label.set_color('#00cccc')
ax.xaxis.label.set_color('#00cccc')
ax.legend(facecolor="#23272b", edgecolor="#00cccc", labelcolor="#00cccc")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.configure(bg="#181c20")
canvas_widget.pack(side=tk.RIGHT, padx=32, pady=32)

def update():
    global signal, cap, running
    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        print("Webcam not found or disconnected.")
        root.destroy()
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face_mesh.process(rgb)
    forehead_points, left_cheek_points, right_cheek_points = [], [], []
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        h, w, _ = frame.shape
        forehead_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in FOREHEAD_LANDMARKS]
        left_cheek_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_CHEEK_LANDMARKS]
        right_cheek_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_CHEEK_LANDMARKS]
        draw_face_rois(frame, forehead_points, left_cheek_points, right_cheek_points)
        if len(forehead_points) > 2:
            x, y, w_, h_ = cv2.boundingRect(np.array(forehead_points))
            x, y = max(x, 0), max(y, 0)
            x2, y2 = min(x + w_, frame.shape[1]), min(y + h_, frame.shape[0])
            roi = frame[y:y2, x:x2]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                roi = cv2.resize(roi, (WIDTH, HEIGHT))
                roi_buffer.append(roi)

    # Show webcam in Tkinter with rounded border and shadow
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((340, 260), Image.LANCZOS)
    img = rounded_border_pil(img, radius=32, border=9, shadow=12)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    # ONNX inference when buffer is full
    if len(roi_buffer) == WIN_SIZE:
        tensors = np.stack([preprocess(roi) for roi in roi_buffer])
        tensors = einops.rearrange(tensors, "f c h w -> 1 c f h w").astype(np.float32)
        pred = ort_session.run(None, {input_name: tensors})[0].squeeze()
        signal = normalize(pred)
        line.set_ydata(signal)
        canvas.draw()

        # Heart rate estimation
        bpm = estimate_bpm(signal, FS)
        if bpm > 0:
            hr_label.config(text=f"Heart Rate: {bpm} BPM", fg="#00ff99", bg="#23272b")
        else:
            hr_label.config(text="Heart Rate: -- BPM", fg="#ff6666", bg="#23272b")

    root.after(10, update)

def start():
    global cap, running, roi_buffer
    if cap is None:
        cap = cv2.VideoCapture(0)
    roi_buffer.clear()
    running = True
    start_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)
    update()

def stop():
    global running
    running = False
    start_btn.config(state=tk.NORMAL)
    stop_btn.config(state=tk.DISABLED)
    hr_label.config(text="Heart Rate: -- BPM", fg="#ff6666", bg="#23272b")

def on_closing():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
    root.destroy()

start_btn.config(command=start)
stop_btn.config(command=stop)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
