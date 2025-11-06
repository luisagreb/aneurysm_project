from pathlib import Path
from aicsimageio import AICSImage

p = Path("data/raw/01Asc-180/+coll") / "01ASC-0180 +coll 60x DMSO48h-Zstack cell1.oir"
print(p.exists())           # should be True
img = AICSImage(str(p))     # AICSImage expects a str path
print(img.shape)            # e.g. (1, 3, Z, Y, X) or similar
print(img.dims)             # e.g. "TCZYX"
