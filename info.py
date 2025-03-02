import rasterio
import sys

def print_tiff_info(file_path):
    try:
        with rasterio.open(file_path) as src:
            print(f"Файл: {file_path}")
            print(f"Формат: {src.driver}")
            print(f"Размер: {src.width} x {src.height}")
            print(f"Количество каналов: {src.count}")
            print(f"Координаты: {src.bounds}")
            print(f"CRS: {src.crs}")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Используйте: python script.py <путь_к_tiff_файлу>")
    else:
        print_tiff_info(sys.argv[1])