# file: main_app.py

import sys
import cv2
import threading
import time
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import queue
import numpy as np

# import our modular components
from gui.results_window import ResultsWindow
from computer_vision.pose_analyzer import PoseAnalyzer
from pose_definitions import POSE_LIBRARY

class TuroArnisGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # force a 16:9 window
        self.window_width = 1920
        self.window_height = 1080
        self.window.geometry(f"{self.window_width}x{self.window_height}")

        # center the window on screen
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - self.window_width) // 2
        y = (screen_height - self.window_height) // 2
        self.window.geometry(f"+{x}+{y}")

        # initialize our pose analyzer class
        self.analyzer = PoseAnalyzer()

        # video capture setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[critical error] cannot open webcam.")
            self.is_running = False
            return
        
        self.queue = queue.Queue(maxsize=1)
        self.target_form = None
        self.current_user = "Default User"

        # add variables for frame skipping
        self.frame_counter = 0
        self.processing_interval = 4 # process ai every 4 frames
        self.last_known_results = []
        self.last_known_tracked_persons = []

        # create and place gui widgets
        self.video_label = ttk.Label(self.window)
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.controls_panel = ttk.Frame(self.window, padding=15, bootstyle="dark")
        self.controls_panel.place(x=20, y=20)

        # widget creation
        ttk.Label(self.controls_panel, text="Controls", font=("-size 14 -weight bold"), bootstyle="inverse-dark").pack(pady=(0, 10), anchor=W)
        self.user_button = ttk.Menubutton(self.controls_panel, text=self.current_user, bootstyle="secondary")
        self.user_button.pack(fill=X, pady=5)
        self.user_menu = ttk.Menu(self.user_button)
        users = ["Default User", "John Doe", "Jane Smith"]
        for user_text in users:
            self.user_menu.add_command(label=user_text, command=lambda u=user_text: self.on_user_selected(u))
        self.user_button["menu"] = self.user_menu
        self.practice_stances = {
            "Crown Thrust": "crown_thrust_correct",
            "Left Temple Block": "left_temple_block_correct"
        }
        self.form_button = ttk.Menubutton(self.controls_panel, text="Choose Arnis Form", bootstyle="primary")
        self.form_button.pack(fill=X, pady=5)
        self.form_menu = ttk.Menu(self.form_button)
        for pretty_name in self.practice_stances.keys():
            self.form_menu.add_command(label=pretty_name, command=lambda p=pretty_name: self.on_action_selected(p))
        self.form_button["menu"] = self.form_menu
        ttk.Separator(self.controls_panel, orient=HORIZONTAL).pack(fill=X, pady=15)
        
        # --- this is the corrected line ---
        self.status_label = ttk.Label(self.controls_panel, text="Status: Select a form", font=("-size 12"), wraplength=220, bootstyle="inverse-dark")
        self.status_label.pack(fill=X, pady=5, anchor=W)
        
        self.view_all_results_button = ttk.Button(self.controls_panel, text="View All Results", command=self.open_results_window, bootstyle="info")
        self.view_all_results_button.pack(fill=X, pady=10, side=BOTTOM)

        # thread control and startup
        self.is_running = True
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()
        self.window.mainloop()

    '''def resize_and_pad(self, img, size, pad_color=0):
        h, w = img.shape[:2]
        sw, sh = size
        interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC
        aspect = w / h
        if aspect > sw / sh:
            new_w = sw
            new_h = np.round(new_w / aspect).astype(int)
            pad_vert = (sh - new_h) / 2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        else:
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = (sw - new_w) / 2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        padded_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[pad_color]*3)
        return padded_img '''

    def video_loop(self):
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                frame = cv2.flip(frame, 1)
                #
                # frame = self.resize_and_pad(frame, (self.window_width, self.window_height))

                self.frame_counter += 1
                processed_frame = frame.copy() 

                if self.frame_counter % self.processing_interval == 0:
                    _, analysis_results, tracked_persons = self.analyzer.process_frame(frame)
                    self.last_known_results = analysis_results
                    self.last_known_tracked_persons = tracked_persons
                else:
                    analysis_results = self.last_known_results
                    tracked_persons = self.last_known_tracked_persons

                for person in tracked_persons:
                    x1, y1, x2, y2, person_id = map(int, person)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(processed_frame, f"User {person_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

                for result in analysis_results:
                    person_id = result['id']
                    predicted_class = result['predicted_class']
                    confidence = result['confidence']
                    live_angles = result['live_angles']
                    x1, y1, x2, y2 = result['bbox']
                    
                    if self.target_form:
                        if predicted_class == self.target_form and confidence > 0.60:
                            ideal_pose = POSE_LIBRARY.get(self.target_form)
                            error_messages = []
                            is_correct = True
                            if ideal_pose and live_angles:
                                for joint, ideal_range in ideal_pose.items():
                                    live_angle = live_angles.get(joint)
                                    if live_angle is not None:
                                        min_angle, max_angle = ideal_range
                                        if not (min_angle <= live_angle <= max_angle):
                                            is_correct = False
                                            error_messages.append(f"{joint.replace('_', ' ')} {'bent' if live_angle < min_angle else 'straight'}")
                            
                            if is_correct:
                                cv2.putText(processed_frame, "Correct!", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                for i, msg in enumerate(error_messages[:2]):
                                    cv2.putText(processed_frame, msg, (x1, y2 + 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            pretty_form_name = self.form_button.cget('text')
                            if pretty_form_name != "Choose Arnis Form":
                                cv2.putText(processed_frame, f"Adjust to {pretty_form_name}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                if self.queue.full():
                    try: self.queue.get_nowait()
                    except queue.Empty: pass
                
                self.queue.put(processed_frame)
            
            except Exception as e:
                print(f"[critical error in video_loop]: {e}")
                time.sleep(0.5)

    def process_queue(self):
        try:
            frame = self.queue.get_nowait()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
        finally:
            self.window.after(20, self.process_queue)

    def on_action_selected(self, pretty_name):
        self.form_button.config(text=pretty_name)
        self.target_form = self.practice_stances[pretty_name]
        self.status_label.config(text=f"Status: Analyzing '{pretty_name}'")
        print(f"targeting model class: '{self.target_form}'")

    def open_results_window(self):
        ResultsWindow(self.window)

    def on_closing(self):
        print("closing application...")
        self.is_running = False
        time.sleep(0.5)
        if hasattr(self, 'analyzer'): self.analyzer.close()
        if hasattr(self, 'cap') and self.cap.isOpened(): self.cap.release()
        self.window.destroy()
        
    def on_user_selected(self, username):
        self.current_user = username
        self.user_button.config(text=username)
        print(f"current user set to: {username}")
    
    def reset_feedback(self):
        self.target_form = None
        self.form_button.config(text="Choose Arnis Form")
        self.status_label.config(text="Status: Select a form")

if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    app = TuroArnisGUI(root, "TuroArnis - Multi-User Form Corrector")