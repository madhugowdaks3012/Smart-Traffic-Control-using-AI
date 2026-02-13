import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas, Frame, messagebox
from PIL import Image, ImageTk
import cv2
import random
import torch
import numpy as np
import sys, io

# ===== Load YOLOv5 Model =====
print("Loading YOLOv5 model...")
old_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
finally:
    sys.stdout = old_stdout
print("Model loaded successfully.")

# ===== Global Variables =====
current_view = None
green_timer_labels = {}
video_caps = {}
lane_counts = {}
traffic_lights = {}
update_delay = 100  # ms between frames

# ===== Utility Functions =====
def show_view(view):
    global current_view
    if current_view is not None:
        current_view.pack_forget()
    view.pack(expand=True, fill="both")
    current_view = view

# ===== Initial View =====
def create_initial_view(root):
    frame = Frame(root, bg="#f0f4f8")
    Label(frame, text="Smart Traffic Control using AI ", font=("Arial", 20, "bold"),
          bg="#f0f4f8", fg="#2c3e50").pack(pady=30)
    Label(frame, text="Select 4 videos, one for each lane (A,B,C,D)", font=("Arial", 12),
          bg="#f0f4f8", fg="#34495e").pack(pady=10)
    Button(frame, text="Select Videos", command=select_videos, font=("Arial", 12),
           bg="#3498db", fg="white", padx=10, pady=5).pack(pady=20)
    return frame

# ===== Results View =====
def create_results_view(root):
    frame = Frame(root, bg="#f0f4f8", padx=10, pady=10)

    # Junction image
    frame.junction_label = Label(frame, bg="#f0f4f8")
    frame.junction_label.pack(pady=10)

    # Traffic lights row under junction
    frame.lights_frame = Frame(frame, bg="#f0f4f8")
    frame.lights_frame.pack(pady=10, fill="x")

    green_timer_labels.clear()
    for lane in ["A", "B", "C", "D"]:
        lane_frame = Frame(frame.lights_frame, bg="#f0f4f8", bd=2,
                           relief="groove", padx=10, pady=10)
        lane_frame.pack(side="left", expand=True, fill="both", padx=5)

        lane_label = Label(lane_frame, text=f"Lane {lane} - Count: 0",
                           font=("Arial",12), bg="#f0f4f8")
        lane_label.pack(pady=5)

        c = Canvas(lane_frame, width=30, height=30, bg="#f0f4f8",
                   highlightthickness=0)
        c.create_oval(2,2,28,28, fill="red")
        c.pack(pady=5)

        # No idle timer shown initially
        timer_label = Label(lane_frame, text="", font=("Arial",12,"bold"),
                            bg="#f0f4f8", fg="green")
        timer_label.pack(pady=5)

        green_timer_labels[lane] = [0, lane_label, c, timer_label]  # [timer, label, canvas, timer_label]

    # Home button
    frame.home_button = Button(frame, text="Home", font=("Arial",12),
                               bg="#e74c3c", fg="white",
                               relief="raised", padx=15, pady=5,
                               command=stop_videos)
    frame.home_button.pack(pady=20)

    return frame

# ===== Video Selection =====
def select_videos():
    global video_caps, lane_counts, traffic_lights
    files = filedialog.askopenfilenames(title="Select 4 Videos",
                                        filetypes=(("MP4 files","*.mp4"),("All files","*.*")))
    if len(files)==4:
        lanes = ["A","B","C","D"]
        video_caps.clear()
        lane_counts.clear()
        traffic_lights.clear()
        for i, lane in enumerate(lanes):
            cap = cv2.VideoCapture(files[i])
            video_caps[lane] = cap
            lane_counts[lane] = 0
            traffic_lights[lane] = "RED"
        global results_view
        results_view = create_results_view(root)
        show_view(results_view)
        update_video_frame()
    else:
        messagebox.showinfo("Info","Please select exactly 4 videos.")

# ===== Video Frame Processing =====
def update_video_frame():
    global video_caps, lane_counts, traffic_lights
    frames = {}
    height, width = 240, 320
    for lane, cap in video_caps.items():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            ret, frame = cap.read()
        frame = cv2.resize(frame,(width,height))
        frames[lane] = frame

    # Count vehicles
    for lane, frame in frames.items():
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        lane_counts[lane] = 0
        for det in detections:
            x1,y1,x2,y2,conf,cls = det
            label = model.names[int(cls)]
            if label in ["car","truck","bus","motorcycle"]:
                lane_counts[lane] += 1
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(frame,label,(int(x1),int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # Dynamic green light
    max_count = max(lane_counts.values())
    for lane in traffic_lights:
        traffic_lights[lane]="RED"
    if max_count>0:
        max_lanes = [l for l,c in lane_counts.items() if c==max_count]
        chosen_lane = random.choice(max_lanes)
        traffic_lights[chosen_lane] = "GREEN"
        green_timer_labels[chosen_lane][0] = min(20, 5+lane_counts[chosen_lane]*2)

    # Merge frames into junction
    top = np.hstack((frames["A"], frames["B"]))
    bottom = np.hstack((frames["C"], frames["D"]))
    junction_img = np.vstack((top,bottom))
    junction_rgb = cv2.cvtColor(junction_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(junction_rgb)
    pil_img = pil_img.resize((800,600))
    img_tk = ImageTk.PhotoImage(pil_img)
    results_view.junction_label.configure(image=img_tk)
    results_view.junction_label.image = img_tk

    # Update lane lights and timers
    for lane in ["A","B","C","D"]:
        timer,label,canvas,timer_label = green_timer_labels[lane]
        label.config(text=f"Lane {lane} - Count: {lane_counts[lane]}")
        color = "green" if traffic_lights[lane]=="GREEN" else "red"
        canvas.itemconfig(1, fill=color)

        if traffic_lights[lane]=="GREEN" and timer>0:
            green_timer_labels[lane][0]-=1
            timer_label.config(text=f"{green_timer_labels[lane][0]}s")
        elif traffic_lights[lane]=="GREEN" and timer<=0:
            traffic_lights[lane]="RED"
            timer_label.config(text="")  # clear when green ends
        else:
            timer_label.config(text="")  # keep empty for red

    root.after(update_delay, update_video_frame)

def stop_videos():
    for cap in video_caps.values():
        cap.release()
    show_view(initial_view)

# ===== Main Window =====
root = tk.Tk()
root.title("Smart Traffic Control using AI")
initial_view = create_initial_view(root)
show_view(initial_view)
root.mainloop()
