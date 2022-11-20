import os
import cv2
import dlib
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the celeba img_align_celeba folder')
parser.add_argument('--output', type=str, help='path to the output folder')
args = parser.parse_args()

input_path = args.path
output_path = args.output

MODEL_PATH = 'pretrained_models/dlib_68_face_landmarks_predictor.dat'
predictor = dlib.shape_predictor(MODEL_PATH)
filenames = os.listdir(input_path)
landmarks_obj = dict()
for filename in tqdm(filenames):
    face = dlib.rectangle(0, 0, 128, 128)
    img = cv2.imread(f'{input_path}/{filename}')
    landmarks = predictor(img, face)
    img_landmarks = dict()
    for i in range(0, 68):
        img_landmarks[str(i)] = {
            'x': landmarks.part(i).x,
            'y': landmarks.part(i).y,
        }
    landmarks_obj[filename] = img_landmarks
    with open(os.path.join(output_path), 'w') as f:
        f.write(json.dumps(landmarks_obj, indent=2))
