# Optimal Sequence Aligner allowing for Long Gaps (OSALG)

One of the main problems of bioinformatics is sequence alignment. Sometimes, as when aligning RNA sequences, it is necessary to find long deletion gaps which is something most classic alignment algorithms cannot do. To make sure such gaps get found, we use concave penalty function. Calculating value of concave function in specific point is complex and slow process so we approximate it with a certain number of affine functions. Most popular algorithm allowing for long gaps is Gotoh's algorithm. Vectorization is technology supported by most modern processors and it allows us to do simple operations such as addition on bigger data at the same time. Because of that, vectorization can efficiently speed up sequence alignment process. We developed tool called OSALG, an implementation of Gotoh's algorithm (and its variations).

## Installation

To install OSALG run commands:

```bash
git clone --recursive https://github.com/RJPenic/OSALG
cd OSALG
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Usage

After installation, executable file will be located in /build/bin folder. To use OSALG run:
```bash
cd bin
./OptSeqAlignmentLongGaps reads.fasta references.fasta
```
It is highly recommended to use '--vector' option for improved results and faster execution.

## Disclaimer

Laboratory for Bioinformatics and Computational Biology cannot be held responsible for any copyright infringement caused by actions of students contributing to any of its repositories. Any case of copyright infringement will be promptly removed from the affected repositories and reported to appropriate faculty organs.
