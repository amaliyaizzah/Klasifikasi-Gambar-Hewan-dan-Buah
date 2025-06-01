import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 5  

def train_model():
    print("Melatih model dari awal...")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )
    
    train_data = datagen.flow_from_directory(
        'dataset',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        classes=['hewan', 'buah']
    )
    
    val_data = datagen.flow_from_directory(
        'dataset',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        classes=['hewan', 'buah']
    )

    print("Label mapping:", train_data.class_indices)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    steps_per_epoch = train_data.samples // BATCH_SIZE
    validation_steps = val_data.samples // BATCH_SIZE
    
    model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

    print("Model selesai dilatih.")
    return model

def predict_image(file_path, model):
    try:
        img = tf.keras.utils.load_img(file_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)[0][0]

        if prediction < 0.5:
            return f"Hewan 🐾 ({prediction:.2f})"
        else:
            return f"Buah 🍎 ({prediction:.2f})"
    except Exception as e:
        return f"Error prediksi: {e}"

def load_image():
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_path:
        return

    result = predict_image(file_path, model)
    label_result.config(text=f"Hasil Prediksi:\n{result}")

    try:
        img = Image.open(file_path)
        img = img.resize((300, 350))
        img_tk = ImageTk.PhotoImage(img)
        label_img.config(image=img_tk)
        label_img.image = img_tk
    except Exception as e:
        label_result.config(text=f"Gagal menampilkan gambar: {e}")

if __name__ == "__main__":
    model = train_model()

    # GUI
    window = tk.Tk()
    window.title("Klasifikasi Hewan vs Buah")

    window_width = 450
    window_height = 550
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_pos = (screen_width // 2) - (window_width // 2)
    y_pos = (screen_height // 2) - (window_height // 2)
    window.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
    window.resizable(False, False)
    window.configure(bg="#e3f2fd")

    frame = tk.Frame(window, bg="#e3f2fd")
    frame.pack(pady=20)

    btn_load = tk.Button(
        frame, text="📷 Pilih Gambar", command=load_image,
        font=("Arial", 12, "bold"),
        bg="#81c784", fg="white",
        activebackground="#66bb6a", activeforeground="white"
    )
    btn_load.pack(pady=10)

    label_img = tk.Label(frame, bg="#e3f2fd")
    label_img.pack(pady=10)

    label_result = tk.Label(
        frame, text="Hasil Prediksi:\n-", font=("Arial", 14),
        bg="#e3f2fd", fg="#2e2e2e"
    )
    label_result.pack(pady=10)

    window.mainloop()
