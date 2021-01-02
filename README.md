# Modeling protein-RNA interactions with GCNN

 ... *description* ...


# Installation

Clone the code

```
git clone https://github.com/AGregorc/protein-RNA-iteractions.git
```

Now there is an option to use anaconda or pip to install all the requirements.
The preferred one is using anaconda since it provides cudatoolkit.

#### a) Using Anaconda
 
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

#### b) Using pip
Not preferred installation but if you'll not use GPU it is still fine.
You should probably also change some packages inside `requirements.txt` eg. `dgl-cu102` into `dgl`.

Create new python environment `python3 -m venv torch` or `virtualenv torch` and install all requirements

```
source torch/bin/activate

cd protein-RNA-iteractions/
python3 -m pip install -r requirements.txt
```




## Useful tips
```
watch -n 0.5 nvidia-smi
```