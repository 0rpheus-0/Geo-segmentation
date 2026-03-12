import customtkinter as ctk
import tkinter.filedialog
import rasterio
import numpy as np
from rasterio.plot import adjust_band
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SegmentationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Segmentation App")
        self.geometry("1000x500")
        self.model = self.load_model()
        self._create_tabs()

    def load_model(self):
        import torch
        from constants import DEVICE

        try:
            model = torch.jit.load(
                "models_unet_rgb/best_model_new.pt", map_location=DEVICE
            )
            print("Модель успешно загружена.")
            return model
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None

    def _create_tabs(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        # Tabs
        self.tab_segment = self.tabview.add("Сегментация")
        self.tab_train = self.tabview.add("Обучение")
        self.tab_test = self.tabview.add("Тестирование")

        # --- Сегментация ---
        self.segment_frame = ctk.CTkFrame(self.tab_segment)
        self.segment_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.load_button = ctk.CTkButton(
            self.segment_frame,
            text="Загрузить изображение",
            command=self.load_image,
            font=("Arial", 18, "bold"),
        )
        self.load_button.pack(pady=(0, 20))

        self.image_panel = ctk.CTkLabel(
            self.segment_frame,
            text="Исходное изображение появится здесь",
            anchor="center",
            font=("Arial", 18),
        )
        self.image_panel.pack(side="left", fill="both", expand=True, padx=10)

        self.mask_panel = ctk.CTkLabel(
            self.segment_frame,
            text="Маска появится здесь",
            anchor="center",
            font=("Arial", 18),
        )
        self.mask_panel.pack(side="right", fill="both", expand=True, padx=10)

        # --- Обучение ---
        self.train_frame = ctk.CTkFrame(self.tab_train)
        self.train_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.dataset_label = ctk.CTkLabel(
            self.train_frame, text="Выберите датасет:", font=("Arial", 16)
        )
        self.dataset_label.pack(anchor="nw", pady=(0, 5))
        self.dataset_option = ctk.CTkOptionMenu(
            self.train_frame, values=["rgb_dataset", "geo_dataset"], font=("Arial", 16)
        )
        self.dataset_option.pack(anchor="nw", pady=(0, 20))

        self.start_train_button = ctk.CTkButton(
            self.train_frame,
            text="Запустить обучение",
            command=self.start_training,
            font=("Arial", 16, "bold"),
        )
        self.start_train_button.pack(anchor="nw", pady=(0, 20))

        self.train_progress = ctk.CTkProgressBar(self.train_frame)
        self.train_progress.set(0)
        self.train_progress.pack(fill="x", pady=(0, 20))

        self.train_plot_panel = ctk.CTkLabel(
            self.train_frame,
            text="График обучения появится здесь",
            anchor="center",
            font=("Arial", 16),
        )
        self.train_plot_panel.pack(fill="both", expand=True, padx=10)

        # --- Тестирование ---
        self.test_frame = ctk.CTkFrame(self.tab_test)
        self.test_frame.pack(fill="both", expand=True, padx=20, pady=20)

        from constants import X_TEST_DIR, Y_TEST_DIR
        self.x_test_dir = X_TEST_DIR
        self.y_test_dir = Y_TEST_DIR
        self.test_dataset_label = ctk.CTkLabel(
            self.test_frame,
            text=f"X: {self.x_test_dir}\nY: {self.y_test_dir}",
            font=("Arial", 14),
            anchor="w"
        )
        self.test_dataset_label.pack(anchor="nw", pady=(0, 5))
        self.choose_test_dataset_button = ctk.CTkButton(
            self.test_frame,
            text="Выбрать папки X/Y",
            command=self.choose_test_dataset,
            font=("Arial", 14)
        )
        self.choose_test_dataset_button.pack(anchor="nw", pady=(0, 20))
    def choose_test_dataset(self):
        import os
        base_dir = tkinter.filedialog.askdirectory(title="Выберите папку с тестовым датасетом")
        if not base_dir:
            return
        x_dir = os.path.join(base_dir, "image")
        y_dir = os.path.join(base_dir, "mask")
        print(f"Выбран тестовый датасет: X={x_dir}, Y={y_dir}")
        if not (os.path.isdir(x_dir) and os.path.isdir(y_dir)):
            self.test_dataset_label.configure(text="Ошибка: нужны подпапки 'image' и 'mask'")
            return
        self.x_test_dir = x_dir
        self.y_test_dir = y_dir
        self.test_dataset_label.configure(text=f"X: {self.x_test_dir}\nY: {self.y_test_dir}")

        self.start_test_button = ctk.CTkButton(
            self.test_frame,
            text="Запустить тестирование",
            command=self.start_testing,
            font=("Arial", 16, "bold"),
        )
        self.start_test_button.pack(anchor="nw", pady=(0, 20))

        self.test_progress = ctk.CTkProgressBar(self.test_frame)
        self.test_progress.set(0)
        self.test_progress.pack(fill="x", pady=(0, 20))

        self.metrics_panel = ctk.CTkLabel(
            self.test_frame,
            text="Метрики появятся здесь",
            anchor="center",
            font=("Arial", 18),
        )
        self.metrics_panel.pack(fill="both", expand=True, padx=10)

    def start_training(self):
        # TODO: реализовать запуск обучения, обновление прогрессбара и графика
        pass

    def start_testing(self):
        import threading

        self.metrics_panel.configure(text="Тестирование...", image=None)
        self.test_progress.set(0)

        def run_test():
            from test_model import test_model
            try:
                def progress_callback(progress):
                    self.test_progress.set(progress)
                logs = test_model(self.model, self.x_test_dir, self.y_test_dir, progress_callback)
                text = f"Test Loss: {logs['dice_loss']:.4f}\nTest F-score: {logs['fscore']:.4f}\nTest IoU: {logs['iou_score']:.4f}"
                self.metrics_panel.configure(text=text)
                self.test_progress.set(1)
            except Exception as e:
                self.metrics_panel.configure(text=f"Ошибка тестирования: {e}")
                self.test_progress.set(0)

        threading.Thread(target=run_test, daemon=True).start()

    def load_image(self):
        import torch
        import visual
        from constants import DEVICE, INFER_HEIGHT, INFER_WIDTH

        file_path = tkinter.filedialog.askopenfilename(
            filetypes=[("TIFF files", "*.tiff *.tif"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            with rasterio.open(file_path) as src:
                image = np.array(adjust_band(src.read([1, 2, 3])))
                # Преобразование как в predict.py
                # image = adjust_band(np.exp(2 * image))
                image = image.astype("float32")

            # Для отображения исходника
            image_show = image.transpose(1, 2, 0)
            image_show = (np.clip(image_show, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(image_show)
            pil_img = pil_img.resize((400, 300))
            try:
                self._img_tk = ctk.CTkImage(light_image=pil_img, size=(400, 300))
            except Exception:
                self._img_tk = ImageTk.PhotoImage(pil_img)
            self.image_panel.configure(image=self._img_tk, text="")

            # Подготовка для модели (размеры должны делиться на INFER_HEIGHT, INFER_WIDTH)
            def pad_image(image, target_size):
                _, H, W = image.shape
                tile_h, tile_w = target_size
                pad_h = (tile_h - H % tile_h) % tile_h
                pad_w = (tile_w - W % tile_w) % tile_w
                padded_image = np.pad(
                    image, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric"
                )
                return padded_image

            padding_image = pad_image(image, (INFER_HEIGHT, INFER_WIDTH))
            x_tensor = torch.from_numpy(padding_image).to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                pr_mask_unet = self.model(x_tensor)
            pr_mask_unet = pr_mask_unet.squeeze().cpu().detach().numpy()
            mask = np.argmax(pr_mask_unet, axis=0)

            # Обрезаем маску до исходного размера
            mask = mask[: image.shape[1], : image.shape[2]]
            mask_rgb, _ = visual.color_mask(mask)

            # Наложение маски с прозрачностью на исходное изображение
            # Приводим оба изображения к uint8 и одинаковому размеру
            base_img = Image.fromarray(image_show).resize((400, 300)).convert("RGBA")
            mask_img = Image.fromarray(mask_rgb).resize((400, 300)).convert("RGBA")
            # Добавляем альфа-канал маске
            alpha = 128  # 0-255, 128 = 0.5
            mask_img.putalpha(alpha)
            # Накладываем маску поверх исходника
            blended = Image.alpha_composite(base_img, mask_img)
            try:
                self._mask_tk = ctk.CTkImage(light_image=blended, size=(400, 300))
            except Exception:
                self._mask_tk = ImageTk.PhotoImage(blended)
            self.mask_panel.configure(image=self._mask_tk, text="")
        except Exception as e:
            self.image_panel.configure(text=f"Ошибка загрузки: {e}")
            self.mask_panel.configure(text="Маска появится здесь", image=None)


if __name__ == "__main__":
    app = SegmentationApp()
    app.mainloop()
