import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch
import rasterio
from rasterio.plot import adjust_band
from constants import DEVICE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import visual


class TiffViewerApp:
    def __init__(self, root):
        self.fpn = torch.jit.load("models_fpn/best_model_new.pt", map_location=DEVICE)
        self.unet = torch.jit.load("models_unet/best_model_new.pt", map_location=DEVICE)

        self.root = root
        self.root.title("GeoViewer")
        self.root.geometry("1200x500")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.image_frame = tk.Frame(root)
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(3):
            self.image_frame.grid_columnconfigure(i, weight=1)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=1, column=0, sticky="ew")
        self.button_frame.grid_columnconfigure(0, weight=1)

        self.open_button = tk.Button(
            self.button_frame, text="Открыть файл", command=self.open_file
        )
        self.open_button.grid(row=0, column=0, pady=10)

        self.canvases = []

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("TIFF files", "*.tiff *.tif")]
        )
        if not file_path:
            return

        try:
            for canvas in self.canvases:
                canvas.get_tk_widget().destroy()
            self.canvases.clear()

            image = rasterio.open(file_path)
            image = np.array([adjust_band(image.read(1))])
            image = image.astype("float32")
            orig, axes_orig = plt.subplots()
            axes_orig.imshow(image[0], cmap="grey")
            axes_orig.axis("off")
            axes_orig.set_xticks([])
            axes_orig.set_yticks([])
            axes_orig.set_frame_on(False)
            orig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            x_tensor = x_tensor

            predict_mask_1 = self.unet(x_tensor)
            predict_mask_2 = self.fpn(x_tensor)

            masks = []
            for i, predict_mask in enumerate([predict_mask_1, predict_mask_2]):
                predict_mask = predict_mask.squeeze().cpu().detach().numpy()
                predict_mask = np.argmax(predict_mask, axis=0)

                mask, axes_mask = plt.subplots()
                predict_mask, _ = visual.color_mask(predict_mask)
                axes_mask.imshow(image[0], cmap="grey")
                axes_mask.imshow(predict_mask, cmap="twilight", alpha=0.5)
                axes_mask.axis("off")
                axes_mask.set_xticks([])
                axes_mask.set_yticks([])
                axes_mask.set_frame_on(False)
                mask.subplots_adjust(left=0, right=1, top=1, bottom=0)
                masks.append(mask)

            images = [orig] + masks
            titles = ["Изображение", "U-Net", "FPN"]
            for i, (image, title) in enumerate(zip(images, titles)):
                label = tk.Label(
                    self.image_frame, text=title, font=("Arial", 12, "bold")
                )
                label.grid(row=0, column=i, padx=5, pady=(5, 0))

                canvas = FigureCanvasTkAgg(image, master=self.image_frame)
                canvas.draw()
                widget = canvas.get_tk_widget()
                widget.grid(row=1, column=i, sticky="nsew", padx=5, pady=(0, 5))

                def resize(event, fig=image, canvas=canvas):
                    width = event.width / canvas.figure.dpi
                    height = event.height / canvas.figure.dpi
                    fig.set_size_inches(width, height)
                    canvas.draw()

                widget.bind("<Configure>", resize)
                self.canvases.append(canvas)

        except Exception as e:
            print("Ошибка:", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = TiffViewerApp(root)
    root.mainloop()
