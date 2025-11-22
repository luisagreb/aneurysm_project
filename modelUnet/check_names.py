import os

BASE = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/Data"
RAW = os.path.join(BASE, "raw_ch1_nucleus")
LBL = os.path.join(BASE, "label_nucleus")

print("\n=== RAW FILES ===")
for f in sorted(os.listdir(RAW)):
    print(f)

print("\n=== LABEL FILES ===")
for f in sorted(os.listdir(LBL)):
    print(f)