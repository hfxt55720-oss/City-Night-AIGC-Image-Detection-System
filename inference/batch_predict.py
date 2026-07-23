from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}


def find_images(folder):
    folder = Path(folder)
    return [
        path for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def batch_predict(model, folder):
    return [model.predict(path) for path in find_images(folder)]
