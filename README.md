# FRU

Implementation of Fourier Recurrent Unit

by Jiong Zhang, Yibo Lin, Zhao Song, Inderjit S. Dhillon

## How to run 

~~~
source ./test/run.sh
~~~

## Experimental Environment

1. Tensorflow 1.2.1
2. TFLearn 0.3.2 (modified to integrate FRU)
3. Nvidia GTX 1080 GPU 

Different runs may have different results due to uncertainty of GPU parallelism. 
Sometimes if a model cannot converge, we try 3 times and report the best convergence. 
