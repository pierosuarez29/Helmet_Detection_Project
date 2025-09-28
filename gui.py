import tkinter as tk
from tkinter import filedialog, messagebox
from yolo_utils import load_model_auto, run_detection

# Automatically load the latest available model (best.pt or prompt user)
try:
    model, stride, names, device = load_model_auto()
except Exception as e:
    messagebox.showerror("Model loading error", str(e))
    raise SystemExit

def open_file():
    # Open a dialog to select a video file for detection
    fp = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
    )
    if fp:
        run_detection(fp, model, stride, names, device)

def open_camera():
    # Webcam index 0 is the default camera
    run_detection(0, model, stride, names, device)

def main():
    root = tk.Tk()
    root.title("Helmet Detector - YOLOv5")
    root.geometry("420x220")

    tk.Button(root, text="Open video", command=open_file, width=30, height=2).pack(pady=10)
    tk.Button(root, text="Use webcam", command=open_camera, width=30, height=2).pack(pady=10)
    tk.Button(root, text="Exit", command=root.quit, width=30, height=2).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main()
