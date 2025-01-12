import cv2
import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
import threading
import time
from PIL import Image, ImageTk

class ASLRecognizerApp:
    def __init__(self):
        # Load the trained model
        self.model = load_model('Resnet50.h5')
        self.img_width, self.img_height = 32, 32
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                       'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Delete', 'Nothing']
        
        self.word = ""
        self.is_predicting = False
        self.predicted_label = "Waiting for prediction"
        self.predicted_count = 0
        self.last_predicted = None
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("ASL Recognition")
        self.root.geometry("800x600")

        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        self.word_label = tk.Label(self.root, text="Current Word: ", font=("Arial", 18))
        self.word_label.pack(pady=10)

        self.predicted_label_label = tk.Label(self.root, text="Prediction: ", font=("Arial", 18))
        self.predicted_label_label.pack(pady=10)

        self.start_button = tk.Button(self.root, text="Start Prediction", command=self.start_prediction)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Prediction", command=self.stop_prediction)
        self.stop_button.pack(pady=10)

        self.delete_button = tk.Button(self.root, text="Delete Last Letter", command=self.delete_letter)
        self.delete_button.pack(pady=10)

        self.clear_button = tk.Button(self.root, text="Clear Word", command=self.clear_word)
        self.clear_button.pack(pady=10)

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.mainloop()

    def predict_hand_sign(self):
        while self.is_predicting:
            success, frame = self.cap.read()
            if not success:
                break

            x1, y1, x2, y2 = 80, 80, 320, 320
            roi = frame[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (self.img_width, self.img_height))
            roi_normalized = roi_resized / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=0)

            predictions = self.model.predict(roi_expanded)
            predicted_class = np.argmax(predictions)
            self.predicted_label = self.labels[predicted_class]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if self.predicted_label == self.last_predicted:
                self.predicted_count += 1
            else:
                self.predicted_count = 1
                self.last_predicted = self.predicted_label

            if self.predicted_count >= 15:
                if self.predicted_label == 'Space':
                    pass
                elif self.predicted_label == 'Delete':
                    pass
                elif self.predicted_label == 'Nothing':
                    self.word += " "
                else:
                    self.word += self.predicted_label
                self.last_predicted = None

            self.update_display(frame)
            time.sleep(0.1)

    def update_display(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img = Image.fromarray(frame)

        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

        if self.predicted_label not in ['Space', 'Delete']:
            self.predicted_label_label.configure(text=f"Prediction: {self.predicted_label}")
        else:
            self.predicted_label_label.configure(text="Prediction: ")

        self.update_word_display()

    def start_prediction(self):
        self.is_predicting = True
        threading.Thread(target=self.predict_hand_sign, daemon=True).start()

    def stop_prediction(self):
        self.is_predicting = False

    def delete_letter(self):
        if self.word:
            self.word = self.word[:-1]
        self.update_word_display()

    def clear_word(self):
        self.word = ""
        self.update_word_display()

    def update_word_display(self):
        self.word_label.configure(text=f"Current Word: {self.word}")

    def on_key_press(self, event):
        if event.keysym == "BackSpace":
            self.delete_letter()

    def on_closing(self):
        self.stop_prediction()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    ASLRecognizerApp()
