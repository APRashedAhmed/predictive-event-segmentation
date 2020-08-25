# Running the CoxLab PredNet

Turns out running PredNet is much harder than you'd think. The 
[public facing repo](https://github.com/coxlab/prednet) has conflicting 
documentation on what versions of packages to use, and while I have, at some 
point in the past gotten it to work, I wasn't successful this (08/24/2020) time.
So I created this note to keep track of what I try.

## Successful Conda Install:

```
# conda install --name tensorflow1 --file conda-requirements.txt
tensorflow-gpu>=1.13.1,<2.0
Keras>=2.2.4
requests
beautifulsoup4 # bs4 when installing using pip
imageio
scipy>=1.2.0
pillow
matplotlib
# hickle
pytest
# conda does not install jupyterlab correctly; it fails to launch with ImportError: cannot import name 'ensure_dir_exists' ModuleNotFoundError: No module named 'jupyter_server'
```

And then install `hickle` using `pip`:

	$ pip install hickle==2.1.0


## Notes 2020-08-25

Rather than running the code in the updated branch, I opted to try running the
public code using the branch-defined env. Initial error was:

	$ AttributeError: module 'tensorflow.python.keras.api._v1.keras.backend' has no attribute '_BACKEND'

Which was solved by changing line `168` in `prednet.py` from: 
```
168         if K._BACKEND == 'theano':
```

to:

```
168         if K.backend() == 'theano':
```

It appears to initialize the model with CUDA properly but another issue arises 
with `hickle`:

	$ ValueError: Provided argument 'file_obj' does not appear to be a valid hickle file! (HDF5-file does not have the proper attributes!)

It seems it cannot read the `.hkl` file. Checking the versions, the installed
`hickle` version is `4.0.1` while the one listed on the public repo is 
`2.1.0`. Installing via `conda` doesn't work, but `pip` installing does.

Next issue seems to be a `python 2` vs `python 3` problem where `hickle.py`
is using a deprecated datatype, `file`. Below is the path to `hickle.py`:

	$ ~/miniconda3/envs/prednet/lib/python3.7/site-packages/hickle.py
	
And the exact traceback when loading the `.hkl` file in `ipython`:
```
In [2]: hkl.load(str(test_file))
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-2-98a1710fdaed> in <module>
----> 1 hkl.load(str(test_file))

~/miniconda3/envs/prednet/lib/python3.7/site-packages/hickle.py in load(fileobj, path, safe)
    614 
    615     try:
--> 616         h5f = file_opener(fileobj)
    617 
    618         h_root_group = h5f.get(path)

~/miniconda3/envs/prednet/lib/python3.7/site-packages/hickle.py in file_opener(f, mode, track_times)
    146     """
    147     # Were we handed a file object or just a file name string?
--> 148     if isinstance(f, file):
    149         filename, mode = f.name, f.mode
    150         f.close()

NameError: name 'file' is not defined

In [3]: file
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-3-046c168df224> in <module>
----> 1 file

NameError: name 'file' is not defined
```

Simplest approach seemed to remove the lines using `file`:

```
    # Were we handed a file object or just a file name string?
    if isinstance(f, file):
        filename, mode = f.name, f.mode
        f.close()
        h5f = h5.File(filename, mode)
```

And then updating the `elif`  below it to `if`.

These changes (yesterday's included) seemed to result in a successful run of 
`kitti_evaluate.py`.

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
