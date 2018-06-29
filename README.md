Lux
========================
A distributed multi-GPU system for fast graph processing.

Prerequisites
-------------
* [CUDA](https://developer.nvidia.com/cuda-zone) is used to implemented Lux.

* [CUB](http://nvlabs.github.io/cub/) is used as an external submodule for Lux's tasks.

* [Legion](http://legion.stanford.edu/) is the underlying runtime for launching tasks and managing data movement.

* (Optional) [GASNet](http://gasnet.lbl.gov) is used for multi-node executions. (see [installation instructions](http://legion.stanford.edu/gasnet/))

After you have cloned Lux, use the following command lines to clone CUB and Legion. 
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
* Compile a Lux application (e.g., PageRank):
```
cd pagerank
make clean; make -j 4
```
* To build a distributed version of Lux, set `USE_GASNET` flag and rebuild:
```
make clean
USE_GASNET=1 make -j 4
```

Running code
------------
The applications take an input graph as well as several runtime flags starting with `-ll:`. For example:
```
./pagerank -ll:gpu 4 -ll:fsize 12000 -ll:zsize 20000 -file twitter-2010.lux -ni 10
./components -ll:gpu 1 -ll:fsize 6000 -ll:zsize 6000 -file indochina.lux
./sssp -ll:gpu 2 -ll:fsize 12000 -ll:zsize 20000 -file twitter-2010.lux -start 0
```
* `-ll:gpu`: number of GPU processors to use in an execution 
* `-ll:fsize`: size of framebuffer memory for each GPU (in MB) 
* `-ll:zsize`: size of zero-copy memory (pinned DRAM with direct GPU access) on each node (in MB)
* `-file`: path to the input graph
* `-ni`: number of iterations to perform
* `-start`: root vertex for SSSP

Graph Format
------------
Lux uses compressed sparse column (CSC) graph in binary format. The specific format is as follows in binary:
```
<nv>
<ne>
<c0>
<c1>
...
<c(nv-1)>
<e0>
<e1>
...
<e(ne-1)>
```
`<nv>` and `<ne>` denote the number of vertices and direct edges in the graph.
The CSC format starts with a sequence of offsets for the vertices in the CSC format, followed by a sequence of directed edges ordered by their target vertex.
The offset `<c(i-1)` and `<c(i)>` refer to the start and end points of a contiguous block of in-edges for vertex `i`.
More specifically, `<e(c(i-1))>`...`<e(c(i)-1)>` is the collection of all in-neighbors for vertex `i`.
For weighted graphs, the weights are stored after the edge sources.

Some example graphs in Lux format are as follows.

| **Graph**                                                       |  **NV**     | **NE**        | **Weighted** |
|-----------------------------------------------------------------|-------------|---------------|--------------|
| [Hollywood](http://sapling.stanford.edu/~zhihao/hollywood.lux)  | 1,139,905   | 57,515,616    |              |
| [Indochina](http://sapling.stanford.edu/~zhihao/indochina.lux)  | 7,414,866   | 194,109,311   |              |
| [Twitter-2010](http://sapling.stanford.edu/~zhihao/twitter.lux) | 41,652,230  | 1,468,365,182 |              |
| [RMAT27](http://sapling.stanford.edu/~zhihao/rmat.lux)          | 134,217,728 | 2,147,483,648 |              |
| [Amazon](http://sapling.stanford.edu/~zhihao/amazon.lux)        | 3,376,972   | 11,676,082    | Yes          |
| [NetFlix](http://sapling.stanford.edu/~zhihao/netflix.lux)      | 497,959     | 200,961,014   | Yes          |


Publication
-----------
Zhihao Jia, Yongkee Kwon, Galen Shipman, Pat McCormick, Mattan Erez, and Alex Aiken. [A Distributed Multi-GPU System for Fast Graph Processing](http://www.vldb.org/pvldb/vol11/p297-jia.pdf). PVLDB 11(3), 2017.
