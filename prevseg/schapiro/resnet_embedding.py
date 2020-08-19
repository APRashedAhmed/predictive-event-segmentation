"""Script to create resnet embeddings from the fractal images. See wb-2.0.2"""
import logging
from pathlib import Path

import torch
import torchvision
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import transforms
from torch.autograd import Variable

import prevseg.constants as const
import prevseg.index as index

def main():
    EMBED_SHAPE = (224,224)
    
    # Load and Resize
    paths_fractals = list(index.DIR_SCH_FRACTALS.iterdir())
    list_fractals = [Image.open(str(path)).resize(EMBED_SHAPE)
                     for path in paths_fractals]
    _ = [img.load() for img in list_fractals]

    # Remove the alpha channel
    list_fractals_no_alpha = [Image.new("RGB", EMBED_SHAPE, (0,0,0))
                              for _ in range(len(list_fractals))]
    _ = [bk.paste(img, mask=img.split()[3])
         for bk, img in zip(list_fractals_no_alpha, list_fractals)]    

    # Normalize the data
    normalize = transforms.Normalize(mean=const.IMAGENET_NORM_MEAN,
                                     std=const.IMAGENET_NORM_STD)
    to_tensor = transforms.ToTensor()

    # Load resnet
    resnet = torchvision.models.resnet101(pretrained=True, progress=False)
    resnet.eval()

    # The layer we are after
    layer = resnet._modules.get('avgpool')
    vector_len = 2048

    # Function to extract the features
    def get_vector(image):
       # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(normalize(to_tensor(image)).unsqueeze(0))
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros((1,vector_len,1,1))
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        h = layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        resnet(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return my_embedding

    # Create the embeddings and put them into an array
    list_vector_embeddings = [get_vector(arr).reshape(vector_len) 
                              for arr in list_fractals_no_alpha]
    array_vector_embeddings = np.array([np.array(vec)
                                        for vec in list_vector_embeddings])

    # Save the embeddings
    save_path = index.DIR_SCH / 'abstract_discs_embedded/'
    if not save_path.exists():
        save_path.mkdir()

    for arr, path in zip(array_vector_embeddings, paths_fractals):
        np.save(str(save_path / path.stem), arr)    

if __name__ == '__main__':
    main()
