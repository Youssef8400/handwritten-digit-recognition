import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)
    class_names = [str(i) for i in range(10)]
    return x_test, y_test, class_names

MODEL_PATH = "cnn_m.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle '{MODEL_PATH}' introuvable.")
model = tf.keras.models.load_model(MODEL_PATH)

x_test, y_test, class_names = load_data()

def show_custom_predictions():
    try:
        count = int(num_images_entry.get())
        if count < 1 or count > len(x_test):
            raise ValueError(f"Entrer un nombre entre 1 et {len(x_test)}")
        cols = 5
        rows = int(np.ceil(count / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        for i in range(count):
            idx = np.random.randint(0, len(x_test))
            img = x_test[idx].reshape(28, 28)
            true_label = y_test[idx]
            pred_label = np.argmax(model.predict(x_test[idx:idx+1], verbose=0))
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Réel: {class_names[true_label]}\nPré: {class_names[pred_label]}", fontsize=9)
            axes[i].axis('off')
        for j in range(count, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


def predict_by_index():
    try:
        idx = int(index_entry.get())
        if idx < 0 or idx >= len(x_test):
            raise ValueError("Indice invalide.")
        img_arr = x_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = np.argmax(model.predict(x_test[idx:idx+1], verbose=0))

        win = tk.Toplevel(root)
        win.title(f"Prédiction - Indice {idx}")
        win.configure(bg="#ffffff")

        img = (img_arr * 255).astype(np.uint8)
        pil = Image.fromarray(img).resize((200, 200))
        tk_img = ImageTk.PhotoImage(pil)

        frame = tk.Frame(win, bg="#ffffff", bd=2, relief="ridge")
        frame.pack(padx=20, pady=20)
        lbl_img = tk.Label(frame, image=tk_img, bg="#ffffff")
        lbl_img.image = tk_img
        lbl_img.pack(pady=10)
        lbl_text = tk.Label(frame, text=f"Réel: {class_names[true_label]}\nPrédit: {class_names[pred_label]}", font=("Arial", 14), bg="#ffffff")
        lbl_text.pack(pady=5)
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


def load_external_image():
    path = filedialog.askopenfilename()
    if not path:
        return
    try:
        orig = Image.open(path).convert('L')
        proc = orig.resize((28, 28))
        arr = np.array(proc).astype('float32') / 255.0
        _, bin_img = cv2.threshold((arr*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(bin_img) > 127:
            bin_img = 255 - bin_img
        inp = bin_img.astype('float32') / 255.0
        inp = inp.reshape(1, 28, 28, 1)
        pred = np.argmax(model.predict(inp, verbose=0))

        for widget in result_frame.winfo_children():
            widget.destroy()
        tk.Label(result_frame, text="Original", font=("Arial", 12, "bold"), bg="#f8f9fa").grid(row=0, column=0, padx=10)
        tk.Label(result_frame, text="Prétraitée", font=("Arial", 12, "bold"), bg="#f8f9fa").grid(row=0, column=1, padx=10)
        orig_disp = orig.resize((128, 128))
        tk_orig = ImageTk.PhotoImage(orig_disp)
        left = tk.Label(result_frame, image=tk_orig, bg="#f8f9fa")
        left.image = tk_orig
        left.grid(row=1, column=0, padx=10)

        disp_proc = Image.fromarray(bin_img).resize((128, 128))
        tk_proc = ImageTk.PhotoImage(disp_proc)
        right = tk.Label(result_frame, image=tk_proc, bg="#f8f9fa")
        right.image = tk_proc
        right.grid(row=1, column=1, padx=10)

        result_text.config(text=f"Prédiction: {class_names[pred]}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de charger l'image: {e}")

root = tk.Tk()
root.title("🧠 Application CNN MNIST")
root.geometry("1000x700")
root.configure(bg="#f8f9fa")

tk.Label(root, text="🧠 Prédiction CNN sur MNIST", font=("Helvetica", 22, "bold"), bg="#f8f9fa", fg="#343a40").pack(pady=20)

btn_frame = tk.Frame(root, bg="#f8f9fa")
btn_frame.pack(pady=10)
btn_opts = {"font": ("Arial", 12), "bg": "#007acc", "fg": "white", "padx": 10, "pady": 5, "bd": 0}

tk.Label(btn_frame, text="Nombre d'images :", font=("Arial", 12), bg="#f8f9fa").grid(row=0, column=0)
num_images_entry = tk.Entry(btn_frame, width=5, font=("Arial", 12))
num_images_entry.insert(0, "20")
num_images_entry.grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Afficher prédictions", command=show_custom_predictions, **btn_opts).grid(row=0, column=2, padx=10)

tk.Label(btn_frame, text="Indice [0-9999] :", font=("Arial", 12), bg="#f8f9fa").grid(row=1, column=0, pady=10)
index_entry = tk.Entry(btn_frame, width=10, font=("Arial", 12))
index_entry.grid(row=1, column=1)
tk.Button(btn_frame, text="Prédire", command=predict_by_index, **btn_opts).grid(row=1, column=2, padx=10)

import_frame = tk.Frame(root, bg="#e9ecef", width=300, height=180, relief="groove", bd=2)
import_frame.pack(pady=20)
label_drop = tk.Label(import_frame, text="📁 Glissez-déposez ou cliquez pour importer", font=("Arial", 12), bg="#e9ecef", fg="#6c757d", justify="center")
label_drop.place(relx=0.5, rely=0.5, anchor="center")
import_frame.bind("<Button-1>", lambda e: load_external_image())
label_drop.bind("<Button-1>", lambda e: load_external_image())

result_frame = tk.Frame(root, bg="#f8f9fa")
result_frame.pack(pady=10)
result_text = tk.Label(root, text="", font=("Arial", 14), bg="#f8f9fa")
result_text.pack()

root.mainloop()
