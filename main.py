import tkinter as tk
from tkinter import ttk
import win32gui
from keras.models import load_model
import numpy as np
from PIL import ImageGrab
from keras.utils import load_img, img_to_array


model = load_model("DigitRecognizer.h5")


def preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size, color_mode='grayscale')
    image = img_to_array(image)
    image = image / 255.0  # Нормализация так как в коде есть нормализация
    image = np.expand_dims(image, axis=0)
    return image


class PaintApp:
    def __init__(self, root):
        self.root = root
        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black', bd=3, relief=tk.SUNKEN)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setup_navbar()
        self.setup_tools()
        self.setup_events()
        self.prev_x = None
        self.prev_y = None

    def setup_navbar(self):
        self.navbar = tk.Menu(self.root)
        self.root.config(menu=self.navbar)

        # File menu
        self.file_menu = tk.Menu(self.navbar, tearoff=False)
        self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)

    def setup_tools(self):
        self.selected_tool = "pen"
        self.colors = ["white", "red", "green", "blue", "yellow", "orange", "purple"]
        self.selected_color = self.colors[0]
        self.brush_sizes = [4, 6, 8, 10, 15, 20, 30, 50]
        self.selected_size = self.brush_sizes[0]
        self.selected_pen_type = "round"

        self.tool_frame = ttk.LabelFrame(self.root, text="Tools")
        self.tool_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)

        self.pen_button = ttk.Button(self.tool_frame, text="Pen", command=self.select_pen_tool)
        self.pen_button.pack(side=tk.TOP, padx=5, pady=5)

        self.eraser_button = ttk.Button(self.tool_frame, text="Eraser", command=self.select_eraser_tool)
        self.eraser_button.pack(side=tk.TOP, padx=5, pady=5)

        self.label = tk.Label(self.root, text="Думаю..", font=("Helvetica", 15))
        self.label.pack(side=tk.TOP, padx=5, pady=5)

        self.classify_btn = tk.Button(self.root, text="Распознать", command=self.classify_handwriting)
        self.classify_btn.pack(side=tk.TOP, padx=5, pady=5)

        self.brush_size_label = ttk.Label(self.tool_frame, text="Brush Size:")
        self.brush_size_label.pack(side=tk.TOP, padx=5, pady=5)

        self.brush_size_combobox = ttk.Combobox(self.tool_frame, values=self.brush_sizes, state="readonly")
        self.brush_size_combobox.current(0)
        self.brush_size_combobox.pack(side=tk.TOP, padx=5, pady=5)
        self.brush_size_combobox.bind("<<ComboboxSelected>>", lambda event: self.select_size(int(self.brush_size_combobox.get())))

        self.color_label = ttk.Label(self.tool_frame, text="Color:")
        self.color_label.pack(side=tk.TOP, padx=5, pady=5)

        self.color_combobox = ttk.Combobox(self.tool_frame, values=self.colors, state="readonly")
        self.color_combobox.current(0)
        self.color_combobox.pack(side=tk.TOP, padx=5, pady=5)
        self.color_combobox.bind("<<ComboboxSelected>>", lambda event: self.select_color(self.color_combobox.get()))

        self.clear_button = ttk.Button(self.tool_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack(side=tk.TOP, padx=5, pady=5)

    def setup_events(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.release)

    def select_pen_tool(self):
        self.selected_tool = "pen"

    def select_eraser_tool(self):
        self.selected_tool = "eraser"

    def select_size(self, size):
        self.selected_size = size

    def select_color(self, color):
        self.selected_color = color

    def select_pen_type(self, pen_type):
        self.selected_pen_type = pen_type

    def draw(self, event):
        if self.selected_tool == "pen":
            if self.prev_x is not None and self.prev_y is not None:
                if self.selected_pen_type == "round":
                    x1 = event.x - self.selected_size
                    y1 = event.y - self.selected_size
                    x2 = event.x + self.selected_size
                    y2 = event.y + self.selected_size
                    self.canvas.create_oval(x1, y1, x2, y2, fill=self.selected_color, outline=self.selected_color)
            self.prev_x = event.x
            self.prev_y = event.y
        else:
            if self.prev_x is not None and self.prev_y is not None:
                if self.selected_pen_type == "round":
                    x1 = event.x - self.selected_size
                    y1 = event.y - self.selected_size
                    x2 = event.x + self.selected_size
                    y2 = event.y + self.selected_size
                    self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
            self.prev_x = event.x
            self.prev_y = event.y

    def release(self, event):
        self.prev_x = None
        self.prev_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.configure(text="Думаю..")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)  # получаем координату холста
        im = ImageGrab.grab(rect)
        im.save('00.jpg')
        str1 = ''
        arg1 = []
        for i in range(0, 10):
            try:
                image_path = f"{i}{i}.jpg"
                input_shape = (28, 28)
                image = preprocess_image(image_path, target_size=input_shape)
                predictions = model.predict(image)
                print(np.argmax(predictions, axis=1))
                predictions = predictions[0]
                n = 0
                for i in predictions:
                    pr = float(str(i * 100)[:6])
                    arg1.append(pr)
                    str1 += "{" + str(n) + "} " + str(i * 100)[:6] + "%" + "\n"
                    n += 1

            except:
                break
        str1 = str1 + "\n" + "-----------\n" + "Это: " + str(arg1.index(max(arg1)))
        self.label.configure(text=str1)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Paint Application")
    app = PaintApp(root)
    root.mainloop()