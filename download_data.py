# Just run this once so you can have access to the data if you want you can downaload it from the kaggle directly and put it on Dataset/raw folder
import kagglehub, shutil, os

SAVE_TO = "./Dataset/raw"

print("Downloading WaRP dataset...")
path = kagglehub.dataset_download("parohod/warp-waste-recycling-plant-dataset")
print(f"Downloaded to kaggle cache: {path}")

if not os.path.exists(SAVE_TO):
    shutil.copytree(path, SAVE_TO)
    print("Saved !!")
else:
    print("Already exists !!")