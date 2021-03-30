# Modeling protein-RNA interactions with GCNN

[![Build Status](https://travis-ci.org/AGregorc/protein-RNA-iteractions.svg?branch=master)](https://travis-ci.org/AGregorc/protein-RNA-iteractions)

 ... *description* ...


# Installation

Clone the code

```
git clone https://github.com/AGregorc/protein-RNA-iteractions.git
```

Now there is an option to use anaconda or pip to install all the requirements.
The preferred one is using anaconda since it provides cudatoolkit.

#### Using Anaconda
 
Check conda installation guides. 
Here is just one example how to install miniconda on linux

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source miniconda3/bin/activate

conda create -n torch python=3.7
conda activate torch
```
Move to the protein-RNA-iteractions directory eg. `cd protein-RNA-iteractions` and then
```
conda install --file requirements.txt -c pytorch -c dglteam
```

# Runing processes

#### Flask server

```shell script
cd protein-RNA-iteractions
export FLASK_APP=src/web/flask/api.py
flask run --host=0.0.0.0 --port=7777
```




## Useful tips
```
watch -n 0.5 nvidia-smi
```