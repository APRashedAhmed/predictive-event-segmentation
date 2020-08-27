"""Script that resizes the Schapiro fractals to be the CoxLab PN input size."""
import logging

import numpy as np
from PIL import Image, ImageOps

import prevseg.index as index

def main():
    # Borrowing some code from ``prevseg/schapiro/resnet_embedding.py``

    sch_shape = (128, 128)
    pn_shape = (128, 160)

    # Load and Resize
    paths_fractals = list(index.DIR_SCH_FRACTALS.iterdir())
    list_fractals = [Image.open(str(path)) for path in paths_fractals]
    _ = [img.load() for img in list_fractals]

    # Remove the alpha channel
    list_fractals_no_alpha = [Image.new("RGB", pn_shape, (0,0,0))
                              for _ in range(len(list_fractals))]
    _ = [bk.paste(img, (0, int((pn_shape[1]-sch_shape[1])/2 - 1)),
                  mask=img.split()[3])
         for bk, img in zip(list_fractals_no_alpha, list_fractals)]

    list_arrays = [np.moveaxis(np.array(img), (0,1,2), (2,1,0))
                   for img in list_fractals_no_alpha]    

    # Save the embeddings
    save_path = index.DIR_SCH / 'abstract_discs_resized_128_160/'
    if not save_path.exists():
        save_path.mkdir()

    for arr, path in zip(list_arrays, paths_fractals):
        # Turn to an array and reorder the dims
        np.save(str(save_path / path.stem), arr)
        arr2 = np.load(str(save_path / (path.stem + '.npy')))    

if __name__ == '__main__':
    main()
