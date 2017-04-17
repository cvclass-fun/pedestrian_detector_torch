# Pedestrian detector for Lua/Torch7

This repo contains example code to train/test/benchmark a pedestrian detector using lua/torch7. This detector uses a modified [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn) network + a pedestrian oriented roi proposal generator for detection.

The following datasets are available for train/test:

- [Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)


## Installation

### Requirements

- NVIDIA GPU with compute capability 3.5+ (2GB+ ram)
- [Torch7](http://torch.ch/docs/getting-started.html)
- [Fast R-CNN module](https://github.com/farrajota/fast-rcnn-torch)
- [dbcollection](https://github.com/farrajota/dbcollection)

### Packages/dependencies installation

To use this example code, some packages are required for it to work: `fastrcnn` and `dbcollection`.


#### fastrcnn

To install the Fast R-CNN package do the following:

- step 1: install all the necessary dependencies.

```bash
luarocks install tds
luarocks install cudnn
luarocks install inn
luarocks install matio
luarocks install torchnet
```

- setp 2: download and install the package.

```bash
git clone https://github.com/farrajota/fast-rcnn-torch
cd fast-rcnn-torch && luarocks make rocks/*
```

> For more information about the fastrcnn package see [here](https://github.com/farrajota/fast-rcnn-torch).


#### dbcollection

To install the dbcollection package do the following:

- step 1: download the git repository to disk.
```
git clone https://github.com/farrajota/dbcollection
```

- step 2: install the Python module.
```
cd dbcollection/ && python setup.py install
```

- step 3: install the Lua package.
```
cd APIs/lua && luarocks make
```

> For more information about the dbcollection package see [here](https://github.com/farrajota/dbcollection).



## Data setup

### Download model results for benchmark



### RoI Proposals



### Dataset

To run the code, first it is necessary to setup the dataset's data.
This code uses the `dbcollection` package for data setup/management.

To setup a dataset do the following:

```lua
dbc = require 'dbcollection.manager'
dbc.load{name='caltech_pedestrian', data_dir='path/to/dataset'}
```

This will download and pre-process the dataset's data and store the all files to the selected path. If `data_dir` is not defined or left empty, the data files will be stored in the `dbcollection/` folder in your home directory in a folder with the dataset's name.

In case you already have the necessary data files, you can manually set the directory path of the files by doing the following commands:

```lua
dbc = require 'dbcollection.manager'
dbc.add{name='caltech_pedestrian', data_dir='path/to/dataset', task={}, file_path={}}
dbc.load('caltech_pedestrian')
```

> Note: All available datasets can be downloaded via `dbcollection`.


To download and extract the relevant data, please run the following scripts: `th download_extract_dataset.lua -save_dir <store_dir_path>` and `download_extract_algorithms.lua -save_dir <store_dir_path>`. The `-save_dir <store_dir_path>` allows for the user to save the downloaded data into another directory than the root dir of the code.


## Train and test a model using the example code

### Training a network

### Testing a network (mAP accuracy)

### Detecting persons

### Benchmark evaluation


## License

MIT license (see the LICENSE file)


## Acknowledges

The evaluation/benchmarking code is an adaptation of @pdollar [toolbox](https://github.com/pdollar/toolbox)/[evaluation](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip) code.