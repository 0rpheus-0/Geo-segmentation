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

        self.train_dataset_info = ctk.CTkFrame(self.train_frame, fg_color="#23272e")
        self.train_dataset_info.pack(anchor="nw", fill="x", pady=(0, 10), padx=0)
        self.train_dataset_expanded = True

        def toggle_dataset_info():
            self.train_dataset_expanded = not self.train_dataset_expanded
            if self.train_dataset_expanded:
                self.train_dataset_toggle.configure(text="▲ Датасет выбран (показать)")
                for lbl in self.train_dataset_paths.values():
                    lbl.pack(anchor="nw", padx=20, pady=(0, 2))
            else:
                self.train_dataset_toggle.configure(text="▼ Датасет выбран (скрыть)")
                for lbl in self.train_dataset_paths.values():
                    lbl.pack_forget()

        self.train_dataset_toggle = ctk.CTkButton(
            self.train_dataset_info,
            text="▲ Датасет не выбран",
            font=("Arial", 15, "bold"),
            anchor="w",
            fg_color="#23272e",
            text_color="#00BFFF",
            hover_color="#1a1d22",
            command=toggle_dataset_info,
        )
        self.train_dataset_toggle.pack(anchor="nw", pady=(5, 0), padx=10, fill="x")
        self.train_dataset_paths = {}
        for key in ["train/image", "train/mask", "val/image", "val/mask"]:
            lbl = ctk.CTkLabel(
                self.train_dataset_info,
                text=f"{key}: -",
                font=("Arial", 13),
                anchor="w",
                text_color="#CCCCCC",
            )
            lbl.pack(anchor="nw", padx=20, pady=(0, 2))
            self.train_dataset_paths[key] = lbl
        self.choose_train_dataset_button = ctk.CTkButton(
            self.train_frame,
            text="Выбрать папку датасета",
            command=self.choose_train_dataset,
            font=("Arial", 14),
        )
        self.choose_train_dataset_button.pack(anchor="nw", pady=(0, 20))

        # --- Тестирование ---
        self.test_frame = ctk.CTkFrame(self.tab_test)
        self.test_frame.pack(fill="both", expand=True, padx=20, pady=20)

        from constants import X_TEST_DIR, Y_TEST_DIR

        self.x_test_dir = X_TEST_DIR
        self.y_test_dir = Y_TEST_DIR
        self.test_dataset_info = ctk.CTkFrame(self.test_frame, fg_color="#23272e")
        self.test_dataset_info.pack(anchor="nw", fill="x", pady=(0, 10), padx=0)
        self.test_dataset_expanded = True

        def toggle_test_dataset_info():
            self.test_dataset_expanded = not self.test_dataset_expanded
            if self.test_dataset_expanded:
                self.test_dataset_toggle.configure(
                    text="▲ Тестовый датасет выбран (показать)"
                )
                for lbl in self.test_dataset_paths.values():
                    lbl.pack(anchor="nw", padx=20, pady=(0, 2))
            else:
                self.test_dataset_toggle.configure(
                    text="▼ Тестовый датасет выбран (скрыть)"
                )
                for lbl in self.test_dataset_paths.values():
                    lbl.pack_forget()

        self.test_dataset_toggle = ctk.CTkButton(
            self.test_dataset_info,
            text="▲ Тестовый датасет не выбран",
            font=("Arial", 15, "bold"),
            anchor="w",
            fg_color="#23272e",
            text_color="#00BFFF",
            hover_color="#1a1d22",
            command=toggle_test_dataset_info,
        )
        self.test_dataset_toggle.pack(anchor="nw", pady=(5, 0), padx=10, fill="x")
        self.test_dataset_paths = {}
        for key in ["image", "mask"]:
            lbl = ctk.CTkLabel(
                self.test_dataset_info,
                text=f"{key}: -",
                font=("Arial", 13),
                anchor="w",
                text_color="#CCCCCC",
            )
            lbl.pack(anchor="nw", padx=20, pady=(0, 2))
            self.test_dataset_paths[key] = lbl
        self.choose_test_dataset_button = ctk.CTkButton(
            self.test_frame,
            text="Выбрать папки X/Y",
            command=self.choose_test_dataset,
            font=("Arial", 14),
        )
        self.choose_test_dataset_button.pack(anchor="nw", pady=(0, 20))

    def choose_train_dataset(self):
        import os

        base_dir = tkinter.filedialog.askdirectory(
            title="Выберите папку с обучающим датасетом"
        )
        if not base_dir:
            return
        x_train = os.path.join(base_dir, "train", "image")
        y_train = os.path.join(base_dir, "train", "mask")
        x_val = os.path.join(base_dir, "validation", "image")
        y_val = os.path.join(base_dir, "validation", "mask")
        if not (
            os.path.isdir(x_train)
            and os.path.isdir(y_train)
            and os.path.isdir(x_val)
            and os.path.isdir(y_val)
        ):
            self.train_dataset_toggle.configure(
                text="▼ Ошибка: нужны подпапки train/image, train/mask, validation/image, validation/mask"
            )
            for key in self.train_dataset_paths:
                self.train_dataset_paths[key].configure(text=f"{key}: -")
            return
        self.x_train_dir = x_train
        self.y_train_dir = y_train
        self.x_valid_dir = x_val
        self.y_valid_dir = y_val
        self.train_dataset_toggle.configure(text="▲ Датасет выбран (показать)")
        self.train_dataset_paths["train/image"].configure(
            text=f"train/image: {x_train}"
        )
        self.train_dataset_paths["train/mask"].configure(text=f"train/mask: {y_train}")
        self.train_dataset_paths["val/image"].configure(text=f"val/image: {x_val}")
        self.train_dataset_paths["val/mask"].configure(text=f"val/mask: {y_val}")

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

        self.train_metrics_panel = ctk.CTkLabel(
            self.train_frame,
            text="Метрики появятся здесь",
            anchor="center",
            font=("Arial", 16),
        )
        self.train_metrics_panel.pack(fill="x", padx=10, pady=10)

    def choose_test_dataset(self):
        import os

        base_dir = tkinter.filedialog.askdirectory(
            title="Выберите папку с тестовым датасетом"
        )
        if not base_dir:
            return
        x_dir = os.path.join(base_dir, "image")
        y_dir = os.path.join(base_dir, "mask")
        print(f"Выбран тестовый датасет: X={x_dir}, Y={y_dir}")
        if not (os.path.isdir(x_dir) and os.path.isdir(y_dir)):
            self.test_dataset_toggle.configure(
                text="▼ Ошибка: нужны подпапки 'image' и 'mask'"
            )
            for key in self.test_dataset_paths:
                self.test_dataset_paths[key].configure(text=f"{key}: -")
            return
        self.x_test_dir = x_dir
        self.y_test_dir = y_dir
        self.test_dataset_toggle.configure(text="▲ Тестовый датасет выбран (показать)")
        self.test_dataset_paths["image"].configure(text=f"image: {x_dir}")
        self.test_dataset_paths["mask"].configure(text=f"mask: {y_dir}")

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
        import threading
        import matplotlib

        matplotlib.use("Agg")
        from train import train_model
        from constants import (
            ENCODER,
            ENCODER_WEIGHTS,
            CLASSES,
            ACTIVATION,
            INIT_LR,
            EPOCHS,
            BATCH_SIZE,
            INFER_HEIGHT,
            INFER_WIDTH,
            LR_DECREASE_STEP,
            LR_DECREASE_COEF,
            INFER_CHANNEL,
        )

        self.train_progress.set(0)
        self.train_plot_panel.configure(
            text="График обучения появится здесь", image=None
        )
        self.train_metrics_panel.configure(text="Метрики появятся здесь")
        self.dataset_label = self.train_dataset_title
        # Пути к данным (выбраны пользователем)
        x_train_dir = getattr(self, "x_train_dir", None)
        y_train_dir = getattr(self, "y_train_dir", None)
        x_valid_dir = getattr(self, "x_valid_dir", None)
        y_valid_dir = getattr(self, "y_valid_dir", None)
        if not all([x_train_dir, y_train_dir, x_valid_dir, y_valid_dir]):
            self.train_dataset_label.configure(text="Сначала выберите папку датасета!")
            return

        def run_train():
            try:

                def progress_callback(epoch, total_epochs, loss_logs, metric_logs):
                    self.train_progress.set((epoch + 1) / total_epochs)
                    self.dataset_label.configure(
                        text=f"Эпоха {epoch + 1} из {total_epochs}"
                    )
                    # Отображение метрик
                    train_loss = loss_logs["train"][-1] if loss_logs["train"] else 0
                    val_loss = loss_logs["val"][-1] if loss_logs["val"] else 0
                    train_iou = metric_logs["train"][-1] if metric_logs["train"] else 0
                    val_iou = metric_logs["val"][-1] if metric_logs["val"] else 0
                    self.train_metrics_panel.configure(
                        text=f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\nTrain IOU: {train_iou:.4f} | Val IOU: {val_iou:.4f}"
                    )
                    # Рисуем график в памяти (увеличенный, темный стиль)
                    import matplotlib.pyplot as plt
                    import io

                    plt.style.use("dark_background")
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].plot(loss_logs["train"], label="train", color="#00BFFF")
                    axes[0].plot(loss_logs["val"], label="val", color="#FF69B4")
                    axes[0].set_title("Losses - Dice", color="#CCCCCC")
                    axes[1].plot(metric_logs["train"], label="train", color="#00BFFF")
                    axes[1].plot(metric_logs["val"], label="val", color="#FF69B4")
                    axes[1].set_title("IOU", color="#CCCCCC")
                    [ax.legend() for ax in axes]
                    for ax in axes:
                        ax.set_facecolor("#222222")
                        ax.tick_params(axis="x", colors="#CCCCCC")
                        ax.tick_params(axis="y", colors="#CCCCCC")
                        ax.title.set_color("#CCCCCC")
                        ax.spines["bottom"].set_color("#CCCCCC")
                        ax.spines["top"].set_color("#CCCCCC")
                        ax.spines["left"].set_color("#CCCCCC")
                        ax.spines["right"].set_color("#CCCCCC")
                    fig.patch.set_facecolor("#222222")
                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    from PIL import Image

                    img = Image.open(buf)
                    img = img.resize((600, 260))
                    try:
                        self._train_plot_img = ctk.CTkImage(
                            light_image=img, size=(600, 260)
                        )
                    except Exception:
                        self._train_plot_img = ImageTk.PhotoImage(img)
                    self.train_plot_panel.configure(image=self._train_plot_img, text="")
                    plt.close(fig)

                train_model(
                    x_train_dir=x_train_dir,
                    y_train_dir=y_train_dir,
                    x_valid_dir=x_valid_dir,
                    y_valid_dir=y_valid_dir,
                    encoder=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    classes=CLASSES,
                    activation=ACTIVATION,
                    init_lr=INIT_LR,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    infer_height=INFER_HEIGHT,
                    infer_width=INFER_WIDTH,
                    lr_decrease_step=LR_DECREASE_STEP,
                    lr_decrease_coef=LR_DECREASE_COEF,
                    infer_channel=INFER_CHANNEL,
                    progress_callback=progress_callback,
                )
                self.train_progress.set(1)
                self.dataset_label.configure(text="Обучение завершено")
            except Exception as e:
                self.train_plot_panel.configure(
                    text=f"Ошибка обучения: {e}", image=None
                )
                self.train_progress.set(0)
                self.dataset_label.configure(text="Ошибка обучения")

        threading.Thread(target=run_train, daemon=True).start()

    def start_testing(self):
        import threading

        self.metrics_panel.configure(text="Тестирование...", image=None)
        self.test_progress.set(0)

        def run_test():
            from test_model import test_model

            try:

                def progress_callback(progress):
                    self.test_progress.set(progress)

                logs = test_model(
                    self.model, self.x_test_dir, self.y_test_dir, progress_callback
                )
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
