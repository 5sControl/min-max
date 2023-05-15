import yolov7
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv("settings.env")


path = os.environ.get("BOX_MODEL")

model = yolov7.load(path)
print(model)

