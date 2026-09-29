# Save as explore_dataset.py
import os
import glob
from pathlib import Path
from datetime import datetime

path = "D:\\Tehseen"

print("=" * 50)
print("   DATASET EXPLORATION REPORT")
print("=" * 50)
print()

# 1. Basic info
print("[1] BASIC INFORMATION")
print(f"Path: {path}")
print(f"Last Modified: {datetime.now()}")
print()

# 2. Total size
total_size = 0
total_files = 0
total_folders = 0

for root, dirs, files in os.walk(path):
    total_folders += len(dirs)
    for file in files:
        file_path = os.path.join(root, file)
        total_size += os.path.getsize(file_path)
        total_files += 1

size_in_gb = total_size / (1024**3)
size_in_mb = total_size / (1024**2)

print("[2] TOTAL SIZE")
print(f"Total Size: {size_in_gb:.2f} GB ({size_in_mb:.2f} MB)")
print()

# 3. File and folder counts
root_items = len(os.listdir(path))
print("[3] FILE & FOLDER COUNTS")
print(f"Root Items: {root_items}")
print(f"Total Folders: {total_folders}")
print(f"Total Files: {total_files}")
print()

# 4. File types breakdown
print("[4] FILE TYPE BREAKDOWN")
extensions = {}
for root, dirs, files in os.walk(path):
    for file in files:
        ext = os.path.splitext(file)[1]
        ext = ext if ext else "(no extension)"
        extensions[ext] = extensions.get(ext, 0) + 1

sorted_ext = sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]
for ext, count in sorted_ext:
    print(f"  {ext}: {count} files")
print()

# 5. Largest files
print("[5] TOP 10 LARGEST FILES")
all_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        all_files.append((file_path, os.path.getsize(file_path)))

largest = sorted(all_files, key=lambda x: x[1], reverse=True)[:10]
for i, (file_path, size) in enumerate(largest, 1):
    size_mb = size / (1024**2)
    relative_path = file_path.replace(path, ".")
    print(f"  {i}. {relative_path}")
    print(f"     Size: {size_mb:.2f} MB")
print()

# 6. Recent files
print("[6] TOP 10 MOST RECENT FILES")
recent = sorted(all_files, key=lambda x: os.path.getmtime(x[0]), reverse=True)[:10]
for i, (file_path, _) in enumerate(recent, 1):
    relative_path = file_path.replace(path, ".")
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    print(f"  {i}. {relative_path}")
    print(f"     Modified: {mtime}")
print()

# 7. Check for dataset structures
print("[7] DATASET STRUCTURE CHECK")
common_folders = ["train", "test", "val", "images", "labels", "data", "annotations"]
for folder in common_folders:
    folder_path = os.path.join(path, folder)
    if os.path.exists(folder_path):
        items = len(os.listdir(folder_path))
        print(f"  ✓ Found '{folder}' folder with {items} items")
print()

# 8. Compressed files
print("[8] COMPRESSED FILES")
archive_types = [".tar", ".gz", ".zip", ".rar", ".7z", ".tar.gz", ".tgz"]
for ext in archive_types:
    count = sum(1 for f in all_files if f[0].endswith(ext))
    if count > 0:
        print(f"  {ext}: {count} files (may need extraction)")

# 9. DL files
print("\n[9] DEEP LEARNING FILE CHECK")
dl_extensions = [".npy", ".pt", ".pth", ".h5", ".hdf5", ".pb", ".onnx"]
for ext in dl_extensions:
    count = sum(1 for f in all_files if f[0].endswith(ext))
    if count > 0:
        print(f"  {ext}: {count} files")

# 10. Sample files
print("\n[10] SAMPLE FILE NAMES (First 20)")
for i, (file_path, _) in enumerate(all_files[:20], 1):
    relative_path = file_path.replace(path, ".")
    print(f"  {relative_path}")
if total_files > 20:
    print(f"  ... and {total_files - 20} more files")
print()

print("=" * 50)
print("   EXPLORATION COMPLETE!")
print("=" * 50)