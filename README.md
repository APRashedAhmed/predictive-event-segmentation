<h1 align="center">Predictive Event Segmentation</h1>
<!-- Pulled from the readme of pcdsdevices https://github.com/pcdshub/pcdsdevices -->
<h4 align="center">Unsupervised temporal event segmentation by predictive deep networks</h4>

Repo is a collection of smaller smaller projects that are meant to be somewhat
interdependent. Currently, bodies of work pertaining to the breakfast dataset
and the Schapiro task reside here.

## Quick Links

-   [Workbooks](https://github.com/apra93/predictive-event-segmentation/tree/master/docs/source/workbooks)
-	[Notes](https://github.com/apra93/predictive-event-segmentation/tree/master/docs/source/notes)

## Requirements

Clone the repo first:

    $ git clone https://github.com/apra93/predictive-event-segmentation.git

Create the appropriate `conda` environment according to the specific project 
within the repo you're interested in. For example, to create the breakfast 
environment:

    $ conda env create -f env/environment_breakfast.yml
	
Then, in the newly created environment, install the development requirements:

    $ conda install -f dev-requirements.txt -c conda-forge -c anaconda

Followed by adding the repo as a module:

    $ python setup.py develop
   
   
## Directory Structure

This repo is based on two cookiecutter templates. See the following github pages for more info:

-   [cookiecutter-data-science-pp](https://github.com/apra93/cookiecutter-data-science-pp)
-   [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science) 
