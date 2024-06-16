import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import threading
import os

import src.get_plates as get_plates
import src.visualize as vi
import src.process as process

import multiprocessing
multiprocessing.freeze_support()

def show_waiting_popup():
    popup = tk.Toplevel()
    popup.title("Processing")
    tk.Label(popup, text="Please wait...").pack(pady=10)
    progress = ttk.Progressbar(popup, orient="horizontal", mode="indeterminate", length=200)
    progress.pack(pady=10)
    progress.start()
    return popup

def upload_video():
    global file_path
    file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=(("Video Files", "*.mp4 *.avi"), ("All Files", "*.*")))
    if file_path:
        messagebox.showinfo("Success", "Video uploaded successfully.")

def get_video_data():
    if os.path.exists("final.csv"):
          messagebox.showerror("Error", "Video already processed.")
          return
    if not file_path:
        messagebox.showerror("Error", "Please upload video.")
        return
    def task():
        popup = show_waiting_popup()
        process.process_video(file_path)
        popup.destroy()
        messagebox.showinfo("Success", "Video processed successfully.")
    threading.Thread(target=task).start()

def visualize_video():
    if not os.path.exists("final.csv"):
        messagebox.showerror("Error", "Please process the video first.")
        return
    if not file_path:
        messagebox.showerror("Error", "Please upload video.")
        return
    def task():
        popup = show_waiting_popup()
        vi.process_video(file_path)
        popup.destroy()
        messagebox.showinfo("Success", "Visualized video created: out.mp4")
    threading.Thread(target=task).start()

def export_excel():
    if not os.path.exists("final.csv"):
        messagebox.showerror("Error", "Please process the video first.")
        return
    if not file_path:
        messagebox.showerror("Error", "Please upload video.")
        return
    popup = show_waiting_popup()
    df = pd.DataFrame(get_plates.get_plate_data(file_path))
    df.to_excel("plates_data.xlsx")
    popup.destroy()
    messagebox.showinfo("Success", "Excel file created: plates_data.xlsx")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("License Plate Recognition")
    root.geometry('350x200')

    file_path = None  

    tk.Button(root, text="Upload Video", command=upload_video).pack(pady=10)
    tk.Button(root, text="Process Video", command=get_video_data).pack(pady=10)
    tk.Button(root, text="Export Plates to Excel", command=export_excel).pack(pady=10)
    tk.Button(root, text="Visualize Video", command=visualize_video).pack(pady=10)

    root.mainloop()