import re
from pathlib import Path

def natural_sort_key(s: str | Path):
    """
    Key function for natural sorting of strings or Path objects.
    e.g. image_1.jpg, image_2.jpg, image_10.jpg
    """
    s = str(s)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def get_image_files(directory: str | Path, supported_extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list[Path]:
    """
    Returns a naturally sorted list of supported image files in a directory.
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []
    
    files = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in supported_extensions:
            files.append(f)
            
    return sorted(files, key=lambda x: natural_sort_key(x.name))
