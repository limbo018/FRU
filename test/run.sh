##
# @file   run.sh
# @author Yibo Lin
# @date   Feb 2018
#
#!/bin/bash

###################### MNIST #####################
mkdir -p repeat/mnist
echo ./test/mnist/FRU.40.10.json
python code/main.py ./test/mnist/FRU.40.10.json > repeat/mnist/FRU.40.10.log
echo ./test/mnist/FRU.60.10.json
python code/main.py ./test/mnist/FRU.60.10.json > repeat/mnist/FRU.60.10.log
echo ./test/mnist/SRU.json
python code/main.py ./test/mnist/SRU.json       > repeat/mnist/SRU.log
echo ./test/mnist/LSTM.json
python code/main.py ./test/mnist/LSTM.json      > repeat/mnist/LSTM.log
echo ./test/mnist/RNN.json
python code/main.py ./test/mnist/RNN.json       > repeat/mnist/RNN.log

###################### MNIST permute #####################
mkdir -p repeat/mnist_permute
echo ./test/mnist_permute/FRU.json
python code/main.py ./test/mnist_permute/FRU.json  > repeat/mnist_permute/FRU.log
echo ./test/mnist_permute/SRU.json
python code/main.py ./test/mnist_permute/SRU.json  > repeat/mnist_permute/SRU.log
echo ./test/mnist_permute/LSTM.json
python code/main.py ./test/mnist_permute/LSTM.json > repeat/mnist_permute/LSTM.log
echo ./test/mnist_permute/RNN.json
python code/main.py ./test/mnist_permute/RNN.json  > repeat/mnist_permute/RNN.log

###################### polynomial synthetic #####################
mkdir -p repeat/poly_synthetic/degree5_npoly5
echo ./test/poly_synthetic/degree5_npoly5/FRU.json
python code/main.py ./test/poly_synthetic/degree5_npoly5/FRU.json  > repeat/poly_synthetic/degree5_npoly5/FRU.log
echo ./test/poly_synthetic/degree5_npoly5/SRU.json
python code/main.py ./test/poly_synthetic/degree5_npoly5/SRU.json  > repeat/poly_synthetic/degree5_npoly5/SRU.log
echo ./test/poly_synthetic/degree5_npoly5/LSTM.json
python code/main.py ./test/poly_synthetic/degree5_npoly5/LSTM.json > repeat/poly_synthetic/degree5_npoly5/LSTM.log
echo ./test/poly_synthetic/degree5_npoly5/RNN.json
python code/main.py ./test/poly_synthetic/degree5_npoly5/RNN.json  > repeat/poly_synthetic/degree5_npoly5/RNN.log

mkdir -p repeat/poly_synthetic/degree10_npoly5
echo ./test/poly_synthetic/degree10_npoly5/FRU.json
python code/main.py ./test/poly_synthetic/degree10_npoly5/FRU.json  > repeat/poly_synthetic/degree10_npoly5/FRU.log
echo ./test/poly_synthetic/degree10_npoly5/SRU.json
python code/main.py ./test/poly_synthetic/degree10_npoly5/SRU.json  > repeat/poly_synthetic/degree10_npoly5/SRU.log
echo ./test/poly_synthetic/degree10_npoly5/LSTM.json
python code/main.py ./test/poly_synthetic/degree10_npoly5/LSTM.json > repeat/poly_synthetic/degree10_npoly5/LSTM.log
echo ./test/poly_synthetic/degree10_npoly5/RNN.json
python code/main.py ./test/poly_synthetic/degree10_npoly5/RNN.json  > repeat/poly_synthetic/degree10_npoly5/RNN.log

mkdir -p repeat/poly_synthetic/degree15_npoly5
echo ./test/poly_synthetic/degree15_npoly5/FRU.json
python code/main.py ./test/poly_synthetic/degree15_npoly5/FRU.json  > repeat/poly_synthetic/degree15_npoly5/FRU.log
echo ./test/poly_synthetic/degree15_npoly5/SRU.json
python code/main.py ./test/poly_synthetic/degree15_npoly5/SRU.json  > repeat/poly_synthetic/degree15_npoly5/SRU.log
echo ./test/poly_synthetic/degree15_npoly5/LSTM.json
python code/main.py ./test/poly_synthetic/degree15_npoly5/LSTM.json > repeat/poly_synthetic/degree15_npoly5/LSTM.log
echo ./test/poly_synthetic/degree15_npoly5/RNN.json
python code/main.py ./test/poly_synthetic/degree15_npoly5/RNN.json  > repeat/poly_synthetic/degree15_npoly5/RNN.log

###################### sine synthetic #####################
mkdir -p repeat/sine_synthetic
echo ./test/sine_synthetic/FRU.json
python code/main.py ./test/sine_synthetic/FRU.json  > repeat/sine_synthetic/FRU.log
echo ./test/sine_synthetic/SRU.json
python code/main.py ./test/sine_synthetic/SRU.json  > repeat/sine_synthetic/SRU.log
echo ./test/sine_synthetic/LSTM.json
python code/main.py ./test/sine_synthetic/LSTM.json > repeat/sine_synthetic/LSTM.log
echo ./test/sine_synthetic/RNN.json
python code/main.py ./test/sine_synthetic/RNN.json  > repeat/sine_synthetic/RNN.log

###################### IMDb #####################
mkdir -p repeat/imdb
echo test/imdb/FRU.1.10.json
python code/tflearn/main_imdb.py ./test/imdb/FRU.1.10.json #> repeat/imdb/FRU.1.10.log
echo test/imdb/FRU.5.10.json
python code/tflearn/main_imdb.py ./test/imdb/FRU.5.10.json #> repeat/imdb/FRU.5.10.log
echo test/imdb/SRU.json
python code/tflearn/main_imdb.py ./test/imdb/SRU.json #> repeat/imdb/SRU.log
echo test/imdb/LSTM.json
python code/tflearn/main_imdb.py ./test/imdb/LSTM.json #> repeat/imdb/LSTM.log
echo test/imdb/RNN.json
python code/tflearn/main_imdb.py ./test/imdb/RNN.json #> repeat/imdb/RNN.log
