Lux
========================
Lux is a a distributed multi-GPU system that achieves
fast graph processing by exploiting the aggregate memory
bandwidth of multiple GPUs and taking advantage of locality
in the memory hierarchy of multi-GPU clusters.

Organization
------------
The code for the Lux runtime is located in the lux/ direcory.
The code for the applications 

Prerequisites
-------------
* [CUDA](https://developer.nvidia.com/cuda-zone) is used to implemented Lux.

* [CUB](http://nvlabs.github.io/cub/) is used as extenral submodules for Lux's tasks.

* [Legion](http://legion.stanford.edu/) is the underlying runtime for launching tasks and managing data movement.

* [GASNet](http://gasnet.lbl.gov)(Optional) is used for networking.

After you have cloned Lux, use the following command lines to clone CUB and Legion:
```
git submodule init
git submodule update
```

Compilation
-----------
* Download Lux source code:
```
# Using git to download Lux
git clone --recursive https://github.com/LuxGraph/Lux
```
* Compiling a Lux application (e.g., PageRank)
```
cd pagerank
make clean; make -j 4
```
* To build a distributed version of Lux, change pagerank/Makefile
```
USE_GASNET = 1
```

Input Format
------------


Publication
-----------
Zhihao Jia, Yongkee Kwon, Galen Shipman, Pat McCormick, Mattan Erez, and Alex Aiken. [A Distributed Multi-GPU System for Fast Graph Processing](http://www.vldb.org/pvldb/vol11/p297-jia.pdf). PVLDB 11(3), 2017.
