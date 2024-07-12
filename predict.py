import gc
import pathlib

import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.windows import Window

import torch
from torchvision import transforms as T

from utils import set_seeds, rle_numba_encode, make_grid
set_seeds()
from models import get_model

identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

def predict(
    device,
    model_path,
    data_path,
    new_size,
    window_size,
):
    model = get_model().to(device)
    trfm = T.Compose([
        T.ToPILImage(),
        T.Resize(new_size),
        T.ToTensor(),
        T.Normalize([0.625, 0.448, 0.688],
                    [0.131, 0.177, 0.101]),
    ])

    p = pathlib.Path(data_path)
    subm = {}
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i, filename in enumerate(p.glob('test/*.tiff')):
        if filename.stem in ["aa05346ff","d488c759a","57512b7f1"]:
            subm[i] = {'id':filename.stem, 'predicted': rle_numba_encode(np.zeros((10,10)))}
            continue
        print("loading", filename)
        dataset = rasterio.open(filename.as_posix(), transform = identity)
        slices = make_grid(dataset.shape, window=window_size)
        preds = np.zeros(dataset.shape, dtype=np.uint8)
        for (x1,x2,y1,y2) in slices:
            image = dataset.read([1,2,3], window=Window.from_slices((x1,x2),(y1,y2)))
            image = np.moveaxis(image, 0, -1)
            image = trfm(image)
            with torch.no_grad():
                image = image.to(device)[None]
                score = model(image)['out'][0][0]

                score2 = model(torch.flip(image, [0, 3]))['out']
                score2 = torch.flip(score2, [3, 0])[0][0]

                score3 = model(torch.flip(image, [1, 2]))['out']
                score3 = torch.flip(score3, [2, 1])[0][0]

                score_mean = (score + score2 + score3) / 3.0
                score_sigmoid = score_mean.sigmoid().cpu().numpy()
                score_sigmoid = cv2.resize(score_sigmoid, (window_size, window_size))
                
                preds[x1:x2,y1:y2] = (score_sigmoid > 0.5).astype(np.uint8)
                
        subm[i] = {'id':filename.stem, 'predicted': rle_numba_encode(preds)}
        del preds
        gc.collect()

    submission = pd.DataFrame.from_dict(subm, orient='index')
    submission.to_csv('submission.csv', index=False)

if __name__=="__main__":
    opt={
        "device":'cuda' if torch.cuda.is_available() else 'cpu',
        "model_path":"model_best.pth",
        "data_path":"data",
        "new_size":256,
        "window_size":1024,
    }
    predict(**opt)
    pass