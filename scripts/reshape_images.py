import os
import argparse
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--output', type=str, help='path to the file list')
args = parser.parse_args()
input_path = args.path
output_path = args.output

files = os.listdir(input_path)
for filename in tqdm(files):
    img = Image.open(f'{input_path}/{filename}')
    resized = img.resize((128, 128))
    resized.save(f'{output_path}/{filename}')
