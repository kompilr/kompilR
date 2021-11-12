
```
So let it not look strange 
if I claim
that it is much easier to explain 
the movement of the giant celestial bodies 
than to interpret in mechanical terms 
the origination of just a single caterpillar
or a tiny grass.
``` 
- Immanuel Kant, *Natural History and the Theory of Heaven*, 1755

## TPU

[A Domain Specific Supercomputer for Training Deep Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3360307)

* 128x128- or 256x256-element systolic arrays of multipliers per core
* [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
* TPUv2 uses a 16x16 2D torus network topology.
* XLA compiler optimizations 

> Benchmarks suggests the TPUv3 chip performs similarly to the contemporary Volta GPU chip, but parallel scaling for production applications is stronger for the TPUv3 supercomputer

## [Bfloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

Designed to be used in hardware accelerating machine learning algorithms. 

Same formate as IEEE 754 single precision floating point BUT truncates
mantissa from 23 -> 7 bits. (floating point is made up of sign, exponent, and
mantissa) 

> Neural networks are far more sensitive to the size of the exponent than that of the mantissa
[source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

> Deep Learning models are known to tolerate lower numerical precision...the network can accomplish a task with the same accuracy using a lower precision approximation.
[source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus1)

> Surprisingly some models can even reach a higher accuracy with lower precision [source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

Has exactly same range as float32 - ~1e-38 to ~3e38

* 1 sign bit
* 8 exponent bits
* 7 mantissa bits 

SEEEEEEEEMMMMMMM

> Of deep learning models...the bfloat16 format works as well as the FP32 format while delivering increased performance and reducing memory usage. [source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) 

> The physical size of a hardware multiplier scales with the square of the mantissa width. With fewer mantissa bits than FP16, the bfloat16 multipliers are about half the size in silicon of a typical FP16 multiplier, and they are eight times smaller than an FP32 multiplier!
[source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) 

> We typically recommend keeping weights and gradients in FP32 but converting activations to bfloat 16 [source](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

## XLA

The CPU and GPU backends included with XLA use [LLVM](http://llvm.org) for low-level IR, optimization, and code-generation.

[XLA - HLO](https://www.tensorflow.org/xla/architecture#how_does_xla_work)

## MLIR

[MLIR - HLO](https://github.com/tensorflow/mlir-hlo)

MLIR is:

> intended for easy expression and optimization of computations involving deep loop nests and dense matrices of high dimensionality.

It was presented in [MLIR: A Compiler Infrastructure for the End ofMoore’s Law](https://arxiv.org/pdf/2002.11054.pdf)

MLIR is a Static Single Assignment form (SSA) compiler:

* SSA is a property of an intermediate representation (IR), which requires that
  * each variable be assigned exactly once
  * every variable be defined before it is used
* In an SSA, existing variables in the original IR are split into versions, new variables typically indicated by the original name with a subscript, so that every definition gets its own version.

MLIR introduces concepts from the [polytype model](https://en.wikipedia.org/wiki/Polytope_model) for loop optimization.

(In compiler construction, a [basic block](https://en.wikipedia.org/wiki/Basic_block) is a straight-line code sequence with no branches in except to the entry and no branches out except at the exit)

MLIR has no fixed/built-in list of globally known operations (no “intrinsics”)


## LLVM

> the LLVM libraries have many capabilities, but they don't actually do anything by themselves. It is up to the designer of the client of the libraries (e.g., the Clang C compiler) to decide how to put the pieces to best use. 

> This careful layering, factoring, and focus on subset-ability is also why the LLVM optimizer can be used for such a broad range of different applications in different contexts. 

[Architecture of Open Source Applications LLVM Chapter](http://www.aosabook.org/en/llvm.html)

* Global identifiers (functions, global variables) begin with the '@' character
* Local identifiers (register names, types) begin with the '%' character


# References

(2021) [A MLIR Dialect for Quantum Assembly Languages](https://arxiv.org/pdf/2101.11365.pdf)

(2021) [Ten Lessons From Three Generations Shaped Google’s TPUv4i](https://conferences.computer.org/iscapub/pdfs/ISCA2021-4ghucdBnCWYB7ES2Pe4YdT/333300a001/333300a001.pdf)

(2020) [MLIR: A Compiler Infrastructure for the End ofMoore’s Law](https://arxiv.org/pdf/2002.11054.pdf)

(2017) [In-Datacenter Performance Analysis of a Tensor Processing Unit​](https://drive.google.com/file/d/0Bx4hafXDDq2EMzRNcy1vSUxtcEk/view?resourcekey=0-ulCsvFTNky29UIPJ3pHyCw)

(2012) [A Systolic Array-Based FPGA Parallel Architecture for the BLAST Algorithm](https://www.hindawi.com/journals/isrn/2012/195658/)

(1982) [Why Systolic Architectures?](https://course.ece.cmu.edu/\~ece740/f13/lib/exe/fetch.php?media=kung_-_1982_-_why_systolic_architectures.pdf)

(1981) [Trace Scheduling: A Technique for Global Microcode Compaction](https://people.eecs.berkeley.edu/\~kubitron/courses/cs252-S12/handouts/papers/TraceScheduling.pdf)


### Other Links

Intel [Foundry Services FactSheet](https://newsroom.intel.com/wp-content/uploads/sites/11/2021/03/intel-foundry-services-fact-sheet-229940.pdf)

Amazon [EC2 F1 Instances](https://aws.amazon.com/ec2/instance-types/f1/) - FPGA accelerator development and deployment in the cloud
