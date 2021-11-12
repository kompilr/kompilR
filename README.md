
```
So let it not look strange 
if I claim
that it is much easier to explain 
the movement of the giant celestial bodies 
than to interpret in mechanical terms 
the origination of just a single caterpillar
or a tiny grass.

Immanuel Kant, Natural History and the Theory of Heaven, 1755
``` 

# kompilR: Domain Specific Compilers for Drug Discovery


# Folding

 > Structure-function relationships are the fundamental object of knowledge in protein chemistry; they allow us to rationally design drugs, engineer proteins with new functions, and understand why mutations cause disease. [- On The Origin of Proteins](https://www.chemistryworld.com/features/on-the-origin-of-proteins/3004719.article)

> There is now a testable explanation for how a protein can fold so quickly: A protein solves its large global optimization problem as a series of smaller local optimization problems, growing and assembling the native structure from peptide fragments, local structures first. [- The Protein Folding Problem](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2443096/)

![Simpsons](simpsons.gif)

The protein folding problem consists of three closely related puzzles:
* (a) What is the folding code? 
* (b) What is the folding mechanism?
* (c) Can we predict the native structure of a protein from its amino acid sequence? [source](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2443096/)


## Data Sources

![Prometheus](prometheus.gif)

[CATH/Gene3D](https://www.cathdb.info/) - 151 Million Protein Domains Classified into 5,481 Superfamilies

[NCBI Conserved Domains Database](https://www.ncbi.nlm.nih.gov/Structure/cdd/wrpsb.cgi) - resource for the annotation of functional units in proteins

[Protein Data Bank](https://www.rcsb.org/)

[Scop 2](https://scop.mrc-lmb.cam.ac.uk/) - Structural Classification of Proteins

[UniProt](https://www.uniprot.org/) -  comprehensive, high-quality and freely accessible resource of protein sequence and functional information.

[Fold@Home](https://foldingathome.org/about/)

## Deep Learning Protein Folding

### [AlphaFold 14](https://www.predictioncenter.org/casp14/doc/presentations/2020_12_01_TS_predictor_AlphaFold2.pdf)

  * [:computer: Code](https://github.com/deepmind/alphafold)
  * [:book: Paper 2](https://www.nature.com/articles/s41586-021-03819-2_reference.pdf)
  * [:book: Paper](https://www.nature.com/articles/s41586-019-1923-7.epdf?author_access_token=Z_KaZKDqtKzbE7Wd5HtwI9RgN0jAjWel9jnR3ZoTv0MCcgAwHMgRx9mvLjNQdB2TlQQaa7l420UCtGo8vYQ39gg8lFWR9mAZtvsN_1PrccXfIbc6e-tGSgazNL_XdtQzn1PHfy21qdcxV7Pw-k3htw%3D%3D)
  * [:newspaper: article](https://deepmind.com/blog/article/AlphaFold-Using-AI-for-scientific-discovery)
  * [AlpahFold 14 Results Discussion](https://dasher.wustl.edu/bio5357/discussion/oxford-alphafold2.pdf)
  * [What AlphaFold means for Structural BioInformatics](https://ammiellewb.medium.com/what-alphafold-means-for-structural-bioinformatics-78117adb7d11)
  * [AlphaFold 2 Explained](https://youtu.be/B9PL__gVxLI) - Yanick Video
  * [Illustrated Transformer](kjalammar.github.io/illustrated-transformer/)
  * [Transformers from Scratch](http://peterbloem.nl/blog/transformers)

### [AlphaFold 13](https://www.predictioncenter.org/CASP13/doc/presentations/Pred_CASP13-Structure-AlphaFold-Jumper.pdf)

  * [:floppy_disk: Code](https://github.com/deepmind/deepmind-research/tree/master/alphafold_casp13)
  * [:floppy_disk: Code](https://github.com/dellacortelab/prospr) - Prospr - Open Source Implementation
  * [:book: Prospr Paper](https://www.biorxiv.org/content/10.1101/830273v1) 
  * [AlphaFold @ Casp13: What Just Happened?](https://moalquraishi.wordpress.com/2018/12/09/alphafold-casp13-what-just-happened/) 

### [MiniFold](https://github.com/hypnopump/MiniFold) - Open Source toy example of AlphaFold 13 algorithm 

> The DeepMind work presented @ CASP was not a technological breakthrough (they did not invent any new type of AI) but an engineering one: they applied well-known AI algorithms to a problem along with lots of data and computing power and found a great solution through model design, feature engineering, model ensembling and so on...

> Based on the premise exposed before, the aim of this project is to build a model suitable for protein 3D structure prediction inspired by AlphaFold and many other AI solutions that may appear and achieve SOTA results.

![MiniFold](minifold.png)

> Two different residual neural networks (ResNets) are used to predict angles between adjacent aminoacids (AAs) and distance between every pair of AAs of a protein. For distance prediction a 2D Resnet was used while for angles prediction a 1D Resnet was used.

### PDNet

> As deep learning algorithms drive the progress in protein structure prediction, a lot remains to be studied at this merging superhighway of deep learning and protein structure prediction. Recent findings show that inter-residue distance prediction, a more granular version of the well-known contact prediction problem, is a key to predicting accurate models. However, deep learning methods that predict these distances are still in the early stages of their development. To advance these methods and develop other novel methods, a need exists for a small and representative dataset packaged for faster development and testing. In this work, we introduce protein distance net (PDNET), a framework that consists of one such representative dataset along with the scripts for training and testing deep learning methods. The framework also includes all the scripts that were used to curate the dataset, and generate the input features and distance maps.

[:desktop: Github](https://github.com/ba-lab/pdnet/)

[:book: Paper](https://www.nature.com/articles/s41598-020-70181-0) 

[:vhs: YouTube](https://youtu.be/uAIuA1O7iE8)

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

# You Don't Know [Jax](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html)

or maybe you do. either way.

Obviously you know:

* jax arrays are [immutable](https://stackoverflow.com/questions/4828080/how-to-make-an-immutable-object-in-python) 

## High Level Jax 

### Using GRAD to calculate gradiants 

**Derivative of a Squared**: (the derivative of <img src="https://render.githubusercontent.com/render/math?math=x^2"> is always <img src="https://render.githubusercontent.com/render/math?math=2x">)


```python
import jax
import jax.numpy as jnp

def sum_of_squares(x):
  return jnp.sum(x**2)

sum_of_squares(6)

>> DeviceArray(36, dtype=int32)

squares_derivative = jax.grad(sum_of_squares)
x = jnp.asarray([6., 8.])
print(squares_derivative(x))

>> [12. 16.]
```
Jax's version of numpy outputs `DEVICEARRAY` format (which is dtype=int32 rather than Numpys int64). 

Taking the gradiant of the square we end up with: 

 <img src="https://render.githubusercontent.com/render/math?math=[[ 6 * 2] , [ 8 * 2 ]] = [ 12, 16]">

---

**Derivative of a Cubed**:  (the derivative of <img src="https://render.githubusercontent.com/render/math?math=x^2"> is always <img src="https://render.githubusercontent.com/render/math?math=2x">)

```python
import jax
import jax.numpy as jnp

def sum_of_cubes(x):
  return jnp.sum(x**3)

sum_of_cubes(6)

>> DeviceArray(216, dtype=int32)

cubes_derivative = jax.grad(sum_of_cubes)
x = jnp.asarray([6., 8.])
print(cubes_derivative(x))

>> [108. 192.]
```

Taking the gradiant of the cube we end up with: 

 <img src="https://render.githubusercontent.com/render/math?math=[[ 3(6 * 6)] , [ 3( 8 * 8) ]] = [ 108, 192]">

---

**Derivative to the Fourth Power**:  (the derivative of <img src="https://render.githubusercontent.com/render/math?math=x^4"> is always <img src="https://render.githubusercontent.com/render/math?math=4x^3">)


```python
import jax
import jax.numpy as jnp

def sum_of_fourth(x):
  return jnp.sum(x**4)

sum_of_fourth(6)

>> DeviceArray(1296, dtype=int32)

fourth_derivative = jax.grad(sum_of_fourth)
x = jnp.asarray([6., 8.])
print(fourth_derivative(x))

>> [864. 2048.]
```

Taking the gradiant to the power of four we end up with: 

 <img src="https://render.githubusercontent.com/render/math?math=[[ 4(6 * 6 * 6)] , [ 4( 8 * 8 * 8) ]] = [ 864, 2048]">

---

**Simple Loss Function** - taking from the documentation how the above can be applied to a simple loss function: 

```python
def squared_error(x, y):
  return jnp.sum((x-y)**2)

squared_error(20,10)

>> DeviceArray(100, dtype=int32)

squared_error_derivative = jax.grad(squared_error)

x = jnp.asarray([14., 10., 8.])

y = jnp.asarray([10., 8, 5.])

print(squared_error_derivative(x, y))

>> [8. 4. 6.]

```

Given the arrays we end with:

 <img src="https://render.githubusercontent.com/render/math?math=[2(14 -10), 2(10 - 8), 2(8 - 5) ] = [8, 4, 6]">


---

**Cubed Loss Function** - taking from the documentation how the above can be applied to a simple loss function: 

```python
def cubed_error(x, y):
  return jnp.sum((x-y)**3)

cubed_error(20,10)

>> DeviceArray(1000, dtype=int32)

cubed_error_derivative = jax.grad(cubed_error)

x = jnp.asarray([14., 10., 8.])

y = jnp.asarray([10., 8, 5.])

print(cubed_error_derivative(x, y))

>> [48. 12. 27.]

```
Given the arrays we end with:

 <img src="https://render.githubusercontent.com/render/math?math=[3((14 -10)^2)), 3((10 - 8)^2)), 3((8 - 5)^2) ] = [48,  12,  27]">
 
 ---
 

Is what is happening in all of these grad examples the type of auto differentiation that generative "derivative evaluations rather than derivative expressions (as in [symbolic differentiation](https://www.cs.utexas.edu/users/novak/asg-symdif.html#:~:text=A%20symbolic%20differentiation%20program%20finds,numeric%20calculations%20based%20on%20formulas.))?

But what is auto differentiation really? It is said to be: 

"a non-standard interpretation of a computer program where this interpretation involves augmenting the standard computation with the calculation of various derivatives." [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/pdf/1502.05767.pdf)

It is *"is code augmentation where code is provided for derivatives of your functions free of charge."* [source](https://rlhick.people.wm.edu/posts/mle-autograd.html)

It is seperate from **symbolic differentiation** & **numeric differentiation** 

> We would like to stress that AD as a technical term refers to
a specific family of techniques that compute derivatives through accumulation of values
during code execution to generate numerical derivative evaluations rather than derivative
expressions. [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/pdf/1502.05767.pdf)

> In contrast with the effort involved in arranging code as closed-form expressions under the syntactic and semantic constraints of symbolic differentiation, AD can be applied to regular code with minimal
change, allowing branching, loops, and recursion.  [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/pdf/1502.05767.pdf)

It has to be imagined that `jax.grad` works by " keeping track of derivative values as opposed to the resulting
expressions" given that what it returns is not the formula of the derivation but the result. The result must be derived by automatically taking the input and putting them through the function that is expected by the [chain rule](https://youtu.be/H-ybCx8gt-8). 


To have any understanding of how its working we should look at the output of JAXPR:

## JAXPR

> If one wants to understand how JAX works internally, or to make use of the result of JAX tracing, it is useful to understand jaxprs"

Jaxprs are JAX’s internal intermediate representation of programs.

A Jaxpr is a data structure that can be evaluated like a mini functional programming language and thus Jaxprs are a useful intermediate representation for function transformation.

JAXPR's are:
*  explicitly typed
*  [functional](https://web.archive.org/web/20131010134641/http://www-formal.stanford.edu/jmc/recursive/recursive.html)
*  first-order 
*  in ANF form.

```python

def sum_cubed_error(x, y):
  return jnp.sum((x-y)**3)

print(jax.make_jaxpr(sum_cubed_error)(20,10))

>> { lambda  ; a b.
>>    let c = sub a b
>>    d = integer_pow[ y=3 ] c
>>    e = convert_element_type[ new_dtype=int32
>>                          weak_type=False ] d
>>    f = reduce_sum[ axes=(  ) ] e
>>    in (f,) }
```

# [JaxCore](https://github.com/google/jax/blob/b884cb4ce2ca5ad4f0080545e294ce2561b89138/jax/core.py#L694)

```python

import operator
from operator import attrgetter
from contextlib import contextmanager
from collections import namedtuple
from functools import total_ordering
import itertools as it
from weakref import ref
import threading
import types
from typing import (Any, Callable, ClassVar, Dict, Generator,
                    Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple,
                    Type, Union, cast, Iterable, Hashable)

import numpy as np
```

[`Operator`](https://docs.python.org/3/library/operator.html#module-operator) module 'exports a set of **efficient functions corresponding to the intrinsic operators of Python**"


[`attrgetter`](https://docs.python.org/3/library/operator.html#operator.attrgetter) returns an attribute or tuple of attributes, useful for fast extractoros such as maps and groupby's. 

---

[`contextlib's`]() [`contextmanager`](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager) is a decorator `@contextmanager` for factory functions that can be created without needed to create `__enter__` & `__exit__` dunder methods. 

---

[`collections`](https://docs.python.org/3/library/collections.html) [`namedtuple()`](https://docs.python.org/3/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields) assign meaning to each position in a tuple - for readability and self-documenting code - allowing ability to access fields by name rather than position index. Documenation exampes:

```python

Nets = namedtuple('Nets',['KD', 'Kyrie', 'Harden'])
print(Nets)
>>> <class '__main__.Nets'>

jerseys = Nets(7,11,13)  # instantiate with positional or keyword
print(jerseys)

>>> Nets(KD=7, Kyrie=11, Harden=13)

>>> jerseys[0] + jerseys[1]             # indexable like the plain tuple 
>>> 18

x, y, z = jerseys                # unpack like a regular tuple
print(x, y, z)

>>> (7, 11, 13)

>>> jerseys.KD + jerseys.Harden      # fields also accessible by name
>>> 20 

```

---

[`functools`](https://docs.python.org/3/library/functools.html) is pretty important in that it allows functions to return other functions - as in decorators like `@cache`. 

`Core` uses [`total_ordering`](https://docs.python.org/3/library/functools.html#functools.total_ordering) to allow a class to define only one rich comparison ordering method and the `__eq__`, and the decorator supplies the other four.

Drawbacks of `@total_ordering`: 
* creates slower execution
* mpore complex stack traces for comparison methods 
* implementing all six comparison methods for the class instead will give a speed boost
  * `==` : `__eq__`
  * `!=` : `__ne__`
  * `<` : `__lt__`
  * `>` : `__gt__`
  * `<=` : `__le__`
  * `>=` : `__ge__`

---

[`itertools`](https://docs.python.org/3/library/itertools.html) creates [iterator](https://docs.python.org/3/glossary.html#term-iterator) building  blocks as a core set of fast memory efficient tools. Similar to [C++ Standard Library](https://www.cplusplus.com/reference/stl/)? 

---

[`weakrefs`](https://docs.python.org/3/library/weakref.html) [`ref`](https://docs.python.org/3/library/weakref.html#weakref.ref) can retreieve an object it is still alive, but returns `None` if it is not. Useful for caches/mappings that have large objects - the object will not be kept alive solely because it appears in the cache/map. Garbage collection can delete the object when dead.

---

[`threading`](https://docs.python.org/3/library/threading.html)


![threading](https://media.giphy.com/media/YP9WadrYt8dz2/giphy.gif)


## JAX [Primatives](https://github.com/google/jax/blob/b884cb4ce2ca5ad4f0080545e294ce2561b89138/jax/core.py#L244)

JAX comes with an implementation of numpy functions in terms of JAX primitives.

JAX primatives are found in the [`jax.lax`](https://github.com/google/jax/blob/main/jax/_src/lax/lax.py)

## JAX [Tracers](https://github.com/google/jax/blob/b884cb4ce2ca5ad4f0080545e294ce2561b89138/jax/core.py#L464)

## PyTree

# JAX in use: AlphaFold2

AlphaFold2 is Google's state of the art protein structure prediction model.

AF2 predicts 3D coordinates of all atoms of a protein, [using the amino acid sequence and aligned sequences homology.](https://github.com/b0mTrady/awesome-structural-bioinformatics)

![image](https://user-images.githubusercontent.com/64801585/126504747-281b12dd-4157-4d73-a7f2-107c26494f1c.png)



* PreProcessing
  * Input Sequence 
  * Multiple Sequence Alignments
  * Structural Templates  
* Transformer (EvoFormer)
* Recycling
* Structure Module -> 3D coordinates 

![image](https://user-images.githubusercontent.com/64801585/127316142-126458b5-edf4-4bc0-8aeb-d42a24d01750.png)

![Screenshot from 2021-07-28 07-58-02](https://user-images.githubusercontent.com/64801585/127318851-d3c5f87e-75ba-4632-aa13-7b68eee2f2f8.png)

![Screenshot from 2021-07-28 07-58-54](https://user-images.githubusercontent.com/64801585/127318883-b049f5c5-9415-40b6-9de0-9eac288dcb34.png)



```python

def softmax_cross_entropy(logits, labels):
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.asarray(loss)
  
```
If you didn't know jax's [nn.logsoftmax](https://github.com/google/jax/blob/890a41f7191fa468e2f638ba4efb9e32ad26adaa/jax/_src/nn/functions.py#L264) AF2's implemenation would not mean much to you. 

So going down the rabbit hole in Jax's nn we have the softmax function:

  (The `LogSoftmax` function, rescales elements to the range <img src="https://render.githubusercontent.com/render/math?math=(-\infty, 0)">)


```python
def log_softmax(x: Array, axis: Optional[Union[int, Tuple[int, ...]]] = -1) -> Array:  
  shifted = x - lax.stop_gradient(x.max(axis, keepdims=True))
  return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis, keepdims=True))
  ```

The accepted arguments are: 
* **x** : input array
* **axis**: the axis or axes along which the `log_softmax` should be computed. Either an integer or a tuple of integers.

and an array is returned.

Inside this function we go further down the lane to:
* [`lax.stop_gradient`](https://github.com/google/jax/blob/890a41f7191fa468e2f638ba4efb9e32ad26adaa/jax/_src/lax/lax.py#L1661) - is the identity function, that is, it returns argument `x` unchanged. However, ``stop_gradient`` prevents the flow of
  gradients during forward or reverse-mode automatic differentiation.
```python
def stop_gradient(x):
  def stop(x):
    if (dtypes.issubdtype(_dtype(x), np.floating) or
        dtypes.issubdtype(_dtype(x), np.complexfloating)):
      return ad_util.stop_gradient_p.bind(x)
    else:
      return x  # only bind primitive on inexact dtypes, to avoid some staging
  return tree_map(stop, x)
```
This in turn relies upon [`tree_map`](https://github.com/google/jax/blob/890a41f7191fa468e2f638ba4efb9e32ad26adaa/jax/_src/tree_util.py#L144)

```python 
def tree_map(f: Callable[..., Any], tree: Any, *rest: Any,
                    is_leaf: Optional[Callable[[Any], bool]] = None) -> Any:
  
  leaves, treedef = tree_flatten(tree, is_leaf)
  all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
  return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))
```

* `jnp.log`
* `jnp.sum`
* `jnp.exp`

# References

(2018) [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/pdf/1802.01528.pdf)

(2018) [The Simple Essence of Automatic Differentiation](http://conal.net/papers/essence-of-ad/essence-of-ad-icfp.pdf)

(2015) [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/pdf/1502.05767.pdf)


[Automatic Differentiation Lecture Slides](https://www.cs.ubc.ca/~fwood/CS340/lectures/AD1.pdf)

[Gans in Jax](https://github.com/lweitkamp/GANs-JAX)

[Jax MD](https://github.com/google/jax-md)
