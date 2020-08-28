# 2.0 Working with the Schapiro Data

Place to bring together the notes on working with Schapiro data.

## Environment Creation
Not as trivial as expected. Packages had to be installed in a particular order 
to retain full `ipython` tab-completion functionality. Simple approach would be
to start from the `yml` file:

	$ conda env create --file environment_schapiro.yml

From there, install the development packages:

	$ conda install --file dev-requirements.txt -c conda-forge

And then add the 	`predictive-event-segmentation` repo:

	$ python setup.py develop

## Fractals Data
The fractals data are provided at the very bottom of the 
[Schapiro Lab resources website](https://www.schapirolab.org/resources), and 
[here](https://cb17cd36-5a57-45de-9d66-0b98a3dc5be9.filesusr.com/archives/b37d16_de0969200fd14018bde6e9db80984783.zip?dn=abstract_discs.zip) is a direct download link. The repo assumes the data will be in a directory called `datasets` that is adjacent to the `predicitive-event-segmentation` repo. See `predicitive-event-segmentation/prevseg/index.py` 
for the most updated expected location of the data.

### Resnet-101 Embeddings
Rather than working with the images directly, `resnet-101` embeddings were 
created to reduce the dimentionality. The generate them, run the following 
script using the Schapiro environment:

	$ python predictive-event-segmentation/prevseg/schapiro/resnet_embedding.py

## Papers

[Neural representations of events arise from temporal community structure](https://www.nature.com/articles/nn.3331)
- Human fMRI showing the brain learns community-like representations when 
presented with sequences that have a community structure.

[Complementary learning systems within the hippocampus: a neural network modelling approach to reconciling episodic memory with statistical learning](https://royalsocietypublishing.org/doi/10.1098/rstb.2016.0049)
- Emergent model of the hippocampus can perform the task used above, and 
makes predictions for an intermediate representation between full localist and 
distributed representations.
