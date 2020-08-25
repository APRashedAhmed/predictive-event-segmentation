# Running the CoxLab PredNet

Turns out running PredNet is much harder than you'd think. The 
[public facing repo](https://github.com/coxlab/prednet) has conflicting 
documentation on what versions of packages to use, and while I have, at some 
point in the past gotten it to work, I wasn't successful this (08/24/2020) time.
So I created this note to keep track of what I try.

## Notes 2020-08-24

There is a [pull request](https://github.com/coxlab/prednet/pull/64) that seems
to make things a lot better. However, it seems to deprecate several scripts:

- download_data.sh 
- download_models.sh

So the setup that might work, is to clone the public repo for those scripts, and
then [dHannasch's fork](https://github.com/dHannasch/prednet) as well. The 
useful changes are in the `add-small-video-test` branch.

Using his/her branch, the env can be successfully created:

```
conda create -n prednet --file conda-requirements.txt -c conda-forge -c anaconda
```

Then the repo can be installed into the env by running:

```
python setup.py develop
```

While the model weights can be downloaded using the public 
`download_models.sh` script. The specific weights 
(`prednet_kitti_weights.hdf5` for example) can then be passed to the 
`prednet` executable as follows:

```
prednet --model-file /home/apra/work/prednet_master/model_data_keras2/tensorflow_weights/prednet_kitti_weights.hdf5 ...
```

Discovered also, an additional requirement is `ffmpeg`.

The issue currently is the new model doesn't take original data, but asks for
video files. As in, running the following command:

```
prednet --model-file /home/apra/work/prednet_master/model_data_keras2/tensorflow_weights/prednet_kitti_weights.hdf5 predict /home/apra/work/prednet_master/kitti_data
```

Leads to errors where it needs an additional dictionary to determine video 
information. Need to investigate the tests to see what the new expected format 
is.
