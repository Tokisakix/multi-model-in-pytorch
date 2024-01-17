import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm

TSC_ROOT  = R"L:/multi-model-in-pytorch/multimodal-retrieval/data"
TSC_FILES = [
    Rf"{TSC_ROOT}/MR_train_imgs.tsv",
    Rf"{TSC_ROOT}/MR_test_imgs.tsv",
    Rf"{TSC_ROOT}/MR_valid_imgs.tsv",
]
TSC_PATH  = [
    Rf"{TSC_ROOT}/train_image",
    Rf"{TSC_ROOT}/test_image",
    Rf"{TSC_ROOT}/valid_image",
]

for files, path in zip(TSC_FILES, TSC_PATH):
    imgs = open(files, "r").readlines()
    for img_info in tqdm(imgs):
        img_idx, img_base64 = img_info.split("\t")
        img = Image.open(BytesIO(base64.urlsafe_b64decode(img_base64)))
        img.save(f"{path}/{img_idx}.jpg")