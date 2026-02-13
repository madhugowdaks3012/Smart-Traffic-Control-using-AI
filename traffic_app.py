import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas, Frame
from PIL import Image, ImageTk
import cv2
import random
import torch
import numpy as np
import sys
import io

# ====== Load YOLOv5 Model ======
print("Loading YOLOv5 model...")
old_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
finally:
    sys.stdout = old_stdout
print("Model loaded successfully.")

# ====== Global Variables ======
current_view = None
lane_order = ["A", "B", "C", "D"]
green_lane_index = 0
green_time_remaining = 0
lane_counts = {}
traffic_lights = {}
results_frame_global = None

def show_view(view):
    global current_view
    if current_view is not None:
        current_view.pack_forget()
    view.pack(expand=True, fill="both")
    current_view = view

# ====== Initial View ======
def create_initial_view(root):
    initial_frame = Frame(root, bg="#f0f4f8")

    welcome_label = Label(initial_frame, text="Smart Traffic Control using AI",
                          font=("Arial", 20, "bold"), bg="#f0f4f8", fg="#2c3e50")
    welcome_label.pack(pady=30)

    instructions_label = Label(initial_frame,
                               text="Select 4 images, one for each lane (A, B, C, D).",
                               font=("Arial", 12), bg="#f0f4f8", fg="#34495e")
    instructions_label.pack(pady=10)

    select_button = Button(initial_frame, text="Select Images",
                           command=select_images, font=("Arial", 12),
                           bg="#3498db", fg="white", relief="raised", padx=10, pady=5)
    select_button.pack(pady=20)

    return initial_frame

# ====== Results View ======
def create_results_view(root, processed_image, lane_counts_input, traffic_lights_input):
    global results_frame_global, lane_counts, traffic_lights, green_lane_index, green_time_remaining
    results_frame = Frame(root, bg="#f0f4f8", padx=20, pady=20)
    results_frame_global = results_frame
    lane_counts = lane_counts_input
    traffic_lights = traffic_lights_input
    green_lane_index = next((i for i,lane in enumerate(lane_order) if traffic_lights[lane]=="GREEN"),0)
    green_time_remaining = max(5, 2*lane_counts[lane_order[green_lane_index]])

    # Junction image
    img_tk = ImageTk.PhotoImage(processed_image)
    image_label = Label(results_frame, image=img_tk, bg="#f0f4f8")
    image_label.image = img_tk
    image_label.pack(pady=10)

    # Traffic lights below image
    lights_frame = Frame(results_frame, bg="#f0f4f8")
    lights_frame.pack(pady=10)

    for lane in lane_order:
        lane_frame = Frame(lights_frame, bg="#f0f4f8")
        lane_frame.pack(side="left", padx=15)

        label = Label(lane_frame, text=f"{lane} - {lane_counts[lane]} vehicles",
                      font=("Arial", 12), bg="#f0f4f8", fg="#34495e")
        label.pack()

        canvas = Canvas(lane_frame, width=30, height=30, bg="#f0f4f8", highlightthickness=0)
        color = "green" if traffic_lights[lane]=="GREEN" else "red"
        oval = canvas.create_oval(2,2,28,28, fill=color)
        canvas.pack()

        timer_label = Label(lane_frame, text="", font=("Arial", 12, "bold"),
                            bg="#f0f4f8", fg="green")
        timer_label.pack()
        traffic_lights[lane] = {"canvas":canvas, "oval":oval, "timer_label":timer_label, "status":traffic_lights[lane]}

    # Home button
    home_button = Button(results_frame, text="Home Page",
                         font=("Arial", 12), bg="#3498db", fg="white",
                         relief="raised", padx=10, pady=5,
                         command=lambda: show_view(initial_view))
    home_button.pack(pady=20)

    # Start dynamic traffic light switching
    update_traffic_lights()

    return results_frame

# ====== Dynamic Traffic Light Switching ======
def update_traffic_lights():
    global green_lane_index, green_time_remaining, traffic_lights
    current_lane = lane_order[green_lane_index]
    green_time_remaining -= 1
    traffic_lights[current_lane]["timer_label"].config(text=f"{green_time_remaining}s")
    
    if green_time_remaining <= 0:
        # Switch green to next lane with vehicles
        prev_lane = current_lane
        traffic_lights[prev_lane]["status"] = "RED"
        traffic_lights[prev_lane]["canvas"].itemconfig(traffic_lights[prev_lane]["oval"], fill="red")
        traffic_lights[prev_lane]["timer_label"].config(text="")

        for i in range(1,5):
            next_index = (green_lane_index+i)%4
            next_lane = lane_order[next_index]
            if lane_counts[next_lane]>0:
                green_lane_index = next_index
                green_time_remaining = max(5, 2*lane_counts[next_lane])
                traffic_lights[next_lane]["status"] = "GREEN"
                traffic_lights[next_lane]["canvas"].itemconfig(traffic_lights[next_lane]["oval"], fill="green")
                break
    # Continue updating every second
    results_frame_global.after(1000, update_traffic_lights)

# ====== Image Selection ======
def select_images():
    file_paths = filedialog.askopenfilenames(
        title="Select 4 Images",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
    )
    if len(file_paths) == 4:
        processed_img_pil, lane_counts_input, traffic_lights_input = process_junction(file_paths)
        if processed_img_pil is not None:
            results_view = create_results_view(root, processed_img_pil, lane_counts_input, traffic_lights_input)
            show_view(results_view)
    else:
        tk.messagebox.showinfo("Info", "Please select exactly 4 images.")

# ====== Junction Processing ======
def process_junction(image_paths):
    lane_counts_local = {}
    traffic_lights_local = {}
    lane_positions_local = {}
    lane_order_local = ["A","B","C","D"]

    sample_img = cv2.imread(image_paths[0])
    if sample_img is None: return None, {}, {}
    h,w,_ = sample_img.shape
    junction_img = np.zeros((h*2, w*2,3), dtype=np.uint8)+50
    lane_images = {}
    
    for i,lane in enumerate(lane_order_local):
        lane_images[lane] = cv2.imread(image_paths[i])
        if lane_images[lane] is None:
            lane_images[lane] = np.zeros((h,w,3),dtype=np.uint8)
        else:
            lane_images[lane] = cv2.resize(lane_images[lane], (w,h))
    
    junction_img[0:h,0:w] = lane_images["A"]
    junction_img[0:h,w:2*w] = lane_images["B"]
    junction_img[h:2*h,0:w] = lane_images["C"]
    junction_img[h:2*h,w:2*w] = lane_images["D"]

    lane_positions_local = {"A":(0,0,w,h),"B":(w,0,w,h),"C":(0,h,w,h),"D":(w,h,w,h)}
    junction_img_with_det = junction_img.copy()
    
    for lane,(x,y,ww,hh) in lane_positions_local.items():
        roi = junction_img[y:y+hh,x:x+ww]
        old_stdout_det = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = model(roi)
        finally:
            sys.stdout = old_stdout_det
        detections = results.xyxy[0].cpu().numpy()
        lane_counts_local[lane]=0
        for det in detections:
            x1,y1,x2,y2,conf,cls = det
            label = model.names[int(cls)]
            if label in ['car','truck','bus','motorcycle']:
                lane_counts_local[lane]+=1
                color=(0,255,0)
                cv2.rectangle(junction_img_with_det,(int(x1)+x,int(y1)+y),(int(x2)+x,int(y2)+y),color,2)
                cv2.putText(junction_img_with_det,label,(int(x1)+x,int(y1)+y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    
    max_count = max(lane_counts_local.values())
    for lane in lane_order_local:
        traffic_lights_local[lane]="RED"
    if max_count>0:
        max_lanes = [lane for lane,c in lane_counts_local.items() if c==max_count]
        traffic_lights_local[random.choice(max_lanes)]="GREEN"

    # Draw traffic light boxes
    for lane,(x,y,ww,hh) in lane_positions_local.items():
        color=(0,255,0) if traffic_lights_local[lane]=="GREEN" else (0,0,255)
        cv2.rectangle(junction_img_with_det,(x,y),(x+ww,y+hh),color,5)
        cv2.putText(junction_img_with_det,f"{lane}: {traffic_lights_local[lane]} - {lane_counts_local[lane]}",
                    (x+50,y+50),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)

    junction_rgb = cv2.cvtColor(junction_img_with_det, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(junction_rgb)
    display_width = 800
    display_height = int(pil_img.height*(display_width/pil_img.width))
    pil_img_resized = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)

    return pil_img_resized, lane_counts_local, traffic_lights_local

# ====== Main Window ======
root = tk.Tk()
root.title("Smart Traffic Control using AI")
initial_view = create_initial_view(root)
show_view(initial_view)
root.mainloop()
