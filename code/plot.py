##
# @file   plot.py
# @author Yibo Lin
# @date   Jan 2018
#

import re 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import ticker

def read(filename): 
    print "read ", filename 
    train_data = []
    test_data = []
    with open(filename, "r") as f: 
        for line in f:
            if line.startswith("Epoch"): 
                iterations = re.search(r"iterations = ([-+]?\d+)", line)
                iterations = int(iterations.group(1))

                train_mse = re.search(r"training \w+ = (\d+\.\d+|\d+)", line)
                train_mse = float(train_mse.group(1))
                train_data.append([iterations, train_mse])

                test_mse = re.search(r"testing \w+ = (\d+\.\d+|\d+)", line)
                test_mse = float(test_mse.group(1))
                test_data.append([iterations, test_mse])
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # truncate loss after zero 
    for i in range(len(train_data)):
        if train_data[i][1] == 0.0:
            for j in range(i, len(train_data)): 
                train_data[j][1] = 0.0
                test_data[j][1] = test_data[i][1]
            break 

    return train_data, test_data

def read_tflearn(filename): 
    print "read ", filename 
    train_data = []
    test_data = []
    with open(filename, "r") as f: 
        for line in f:
            line = line.strip()
            if line.startswith("| Adam"): 
                epoch = re.search(r"epoch: ([-+]?\d+)", line)
                epoch = int(epoch.group(1))
                iters = re.search(r"iter: ([-+]?\d+)/([-+]?\d+)", line)
                iter_ratio = int(iters.group(1))/float(iters.group(2))
                epoch = epoch + iter_ratio

                test_mse = re.search(r"val_acc: (\d+\.\d+|\d+)", line)
                test_mse = float(test_mse.group(1))
                test_data.append([epoch, test_mse])
    test_data = np.array(test_data)

    return test_data

#dataset = "mnist.784"
#dataset = "mnist.permute.784"
#dataset = "imdb.300"
#dataset = "poly_synthetic"
dataset = "sine_synthetic.nfreqs15_npoly5"

if dataset == "mnist.784": 
    num_samples = 70000
    time_step = 784
    batch_size = 256
    num_batch = float(num_samples*0.8/batch_size) # convert iterations to epochs approximately 

    fig1 = plt.figure(1, figsize=(4, 3))

    train_data, test_data = read("./log/mnist.784.FRU.freqs.log.unit200.phi10.r60.dropout100.batch256.decay0.9.lr0.005.34.txt")
    data = test_data
    plt.plot(data[:, 0]/num_batch, data[:, 1]*100, label='FRU', linestyle='-', color='red', linewidth = 0.5)

    train_data, test_data = read("./log/mnist.784.SRU.256.decay0.8.txt")
    data = test_data
    plt.plot(data[:, 0]/num_batch, data[:, 1]*100, label='SRU', linestyle='dashed', color='blue', marker = 'v', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    train_data, test_data = read("./log/mnist.784.LSTM.256.decay0.8.lr0.01.epoch100.txt")
    data = test_data
    plt.plot(data[:, 0]/num_batch, data[:, 1]*100, label='LSTM', linestyle=':', color='green', marker = 'o', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    train_data, test_data = read("./log/mnist.784.RNN.256.txt")
    data = test_data
    plt.plot(data[:, 0]/num_batch, data[:, 1]*100, label='RNN', linestyle='-.', color='black', marker = 'x', markersize=2.2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    plt.legend(loc='best', fontsize=8)

    plt.ylim(10, 100)
    #plt.yticks([50, 60, 70, 80, 90, 100])
    #plt.gca().yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))
    #plt.gca().yaxis.set_major_formatter(ticker.LogFormatter())

    plt.title("MNIST with %d Time Steps" % (time_step), fontsize=8)
    plt.xlabel('Epoch', fontsize=8)
    plt.ylabel('Testing Accuracy (%)', fontsize=8)
    plt.tight_layout()
    plt.savefig('./log/mnist.784.test.pdf', bbox_inches='tight', pad_inches=0.0)
elif dataset == "mnist.permute.784": 
    num_samples = 70000
    time_step = 784
    batch_size = 256
    num_batch = float(num_samples*0.8/batch_size) # convert iterations to epochs approximately 

    fig1 = plt.figure(1, figsize=(4, 3))

    train_data, test_data = read("./log/mnist.permute.784.FRU.256.decay0.9.lr0.005.epoch100.txt")
    data = test_data
    plt.plot(data[:, 0]/num_batch, data[:, 1]*100, label='FRU', linestyle='-', color='red', linewidth = 0.5)

    train_data, test_data = read("./log/mnist.permute.784.SRU.256.decay0.8.lr0.001.epoch100.txt")
    data = test_data
    plt.plot(data[:, 0]/num_batch, data[:, 1]*100, label='SRU', linestyle='dashed', color='blue', marker = 'v', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    train_data, test_data = read("./log/mnist.permute.784.LSTM.256.decay0.9.lr0.001.epoch100.txt")
    data = test_data
    plt.plot(data[:, 0]/num_batch, data[:, 1]*100, label='LSTM', linestyle=':', color='green', marker = 'o', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    train_data, test_data = read("./log/mnist.permute.784.RNN.256.decay0.9.lr0.001.epoch100.txt")
    data = test_data
    plt.plot(data[:, 0]/num_batch, data[:, 1]*100, label='RNN', linestyle='-.', color='black', marker = 'x', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    plt.legend(loc='best', fontsize=8)

    plt.ylim(10, 100)
    #plt.yticks([50, 60, 70, 80, 90, 100])
    #plt.gca().yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))
    #plt.gca().yaxis.set_major_formatter(ticker.LogFormatter())

    plt.title("Permuted MNIST with %d Time Steps" % (time_step), fontsize=8)
    plt.xlabel('Epoch', fontsize=8)
    plt.ylabel('Testing Accuracy (%)', fontsize=8)
    plt.tight_layout()
    plt.savefig('./log/mnist.permute.784.test.pdf', bbox_inches='tight', pad_inches=0.0)
elif dataset == "imdb.200" or dataset == "imdb.300": 
    time_step = int(dataset[len("imdb."):])

    fig1 = plt.figure(1, figsize=(4, 3))

    test_data = read_tflearn("./tflearn/log/imdb.300.FRU.dropout80.decay0.2.freq5.log.0.25to100.phi10.test.txt")
    data = test_data
    plt.plot(data[:, 0], data[:, 1]*100, label=r'FRU$_{5, 10}$', linestyle='-', color='red', linewidth = 0.5)

    test_data = read_tflearn("./tflearn/log/imdb.300.FRU.dropout80.decay0.2.freq1.phi10.test.txt")
    data = test_data
    plt.plot(data[:, 0], data[:, 1]*100, label=r'FRU$_{1, 10}$', linestyle='-', color='red', linewidth = 0.5, marker = 'd', markersize=2, fillstyle='none', markeredgewidth = '0.1')

    test_data = read_tflearn("./tflearn/log/imdb.300.SRU.dropout80.decay0.2.test.txt")
    data = test_data
    plt.plot(data[:, 0], data[:, 1]*100, label='SRU', linestyle='dashed', color='blue', marker = 'v', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    test_data = read_tflearn("./tflearn/log/imdb.300.LSTM.dropout80.decay0.8.lr0.01.test.txt")
    data = test_data
    plt.plot(data[:, 0], data[:, 1]*100, label='LSTM', linestyle=':', color='green', marker = 'o', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    test_data = read_tflearn("./tflearn/log/imdb.300.RNN.dropout80.decay0.8.lr0.01.test.txt")
    data = test_data
    plt.plot(data[:, 0], data[:, 1]*100, label='RNN', linestyle='-.', color='black', marker = 'x', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    plt.legend(loc='best')

    plt.ylim(50, 100)
    #plt.yticks([50, 60, 70, 80, 90, 100])
    #plt.gca().yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))
    #plt.gca().yaxis.set_major_formatter(ticker.LogFormatter())

    plt.title("IMDB with %d Max Time Steps" % (time_step))
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy (%)')
    plt.tight_layout()
    plt.savefig("./tflearn/%s.test.pdf" % (dataset), bbox_inches='tight', pad_inches=0.0)
elif dataset.startswith("poly_synthetic"): 
    folders = ["degree5_npoly5", "degree10_npoly5", "degree15_npoly5"]
    #folder = dataset[len("poly_synthetic."):]
    num_samples = 4000
    time_step = 176
    batch_size = 32
    num_batch = float(num_samples*0.8/batch_size) # convert iterations to epochs approximately 

    fig1 = plt.figure(1, figsize=(4, 3))
    plt.subplots_adjust(hspace=0.1)

    ax = [None]
    for i, folder in enumerate(folders): 
        extract = re.search(r"degree(\d+)_npoly(\d+)", folder)
        degree = int(extract.group(1))
        npoly = int(extract.group(2))

        ax.append(plt.subplot(311+i, sharex=ax[-1], sharey=ax[-1]))

        train_data, test_data = read("./log/poly/%s/FRU.120.5" % (folder))
        data = test_data
        plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='FRU', linestyle='-', color='red', linewidth = 0.5)

        train_data, test_data = read("./log/poly/%s/SRU.200" % (folder))
        data = test_data
        plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='SRU', linestyle='dashed', color='blue', marker = 'v', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

        train_data, test_data = read("./log/poly/%s/LSTM" % (folder))
        data = test_data
        plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='LSTM', linestyle=':', color='green', marker = 'o', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

        train_data, test_data = read("./log/poly/%s/RNN" % (folder))
        data = test_data
        plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='RNN', linestyle='-.', color='black', marker = 'x', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

        #plt.gca().set_yscale('symlog', linthreshy=1e-7)
        plt.setp(ax[-1].get_xticklabels(), visible=False if i+1 < len(folders) else True, fontsize=8)
        plt.setp(ax[-1].get_yticklabels(), fontsize=8)
        plt.ylim(5*1e-7, 1e-1)
        plt.yticks([1e-6, 1e-4, 1e-2])

        plt.xlabel("")
        plt.title("")
        ax[-1].title.set_visible(False)
        if i == 0: 
            plt.legend(loc=9, bbox_to_anchor=(0.5, 1.5), ncol=4, fontsize=8)
        plt.tight_layout()
        #plt.subplots_adjust(hspace=0.0, bottom=0, top=0.1)

    #plt.yticks([50, 60, 70, 80, 90, 100])
    #plt.gca().yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))
    #plt.gca().yaxis.set_major_formatter(ticker.LogFormatter())

    #plt.title("Polynomial Data (%d, %d)" % (npoly, degree))
    #plt.title("Polynomial Data with Max Degree %d" % (degree))
    #plt.xlabel('Epoch')
    #plt.ylabel('Testing MSE')
    fig1.text(0.5, 0.0, 'Epoch', ha='center', fontsize=8)
    fig1.text(0.0, 0.5, 'Testing MSE', va='center', rotation='vertical', fontsize=8)
    fig1.text(0.25, 0.74, 'Mix-Poly Data (5, 5)', ha='center', fontsize=6)
    fig1.text(0.25, 0.433, 'Mix-Poly Data (5, 10)', ha='center', fontsize=6)
    fig1.text(0.25, 0.125, 'Mix-Poly Data (5, 15)', ha='center', fontsize=6)
    plt.tight_layout()
    plt.savefig("./log/%s.test.pdf" % (dataset), bbox_inches='tight', pad_inches=0.0)
elif dataset.startswith("sine_synthetic"): 
    folder = dataset[len("sine_synthetic."):]
    extract = re.search(r"nfreqs(\d+)_npoly(\d+)", folder)
    nfreqs = int(extract.group(1))
    npoly = int(extract.group(2))
    num_samples = 4000
    time_step = 176
    batch_size = 32
    num_batch = float(num_samples*0.8/batch_size) # convert iterations to epochs approximately 

    fig1 = plt.figure(1, figsize=(4, 2))

    train_data, test_data = read("./log/sin/sin_%s/FRU.120.5" % (folder))
    data = test_data
    plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='FRU', linestyle='-', color='red', linewidth = 0.5)

    train_data, test_data = read("./log/sin/sin_%s/SRU.200" % (folder))
    data = test_data
    plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='SRU', linestyle='dashed', color='blue', marker = 'v', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    train_data, test_data = read("./log/sin/sin_%s/LSTM" % (folder))
    data = test_data
    plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='LSTM', linestyle=':', color='green', marker = 'o', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    train_data, test_data = read("./log/sin/sin_%s/RNN" % (folder))
    data = test_data
    plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='RNN', linestyle='-.', color='black', marker = 'x', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    #plt.ylim(1e-5, 100)
    plt.legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=4, fontsize=8)

    #plt.yticks([50, 60, 70, 80, 90, 100])
    #plt.gca().yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))
    #plt.gca().yaxis.set_major_formatter(ticker.LogFormatter())

    #plt.title("Mix-Sin Data with %d Frequencies" % (nfreqs), fontsize=8)
    #plt.xlabel('Epoch')
    #plt.ylabel('Testing MSE')
    fig1.text(0.5, 0.0, 'Epoch', ha='center', fontsize=8)
    fig1.text(0.0, 0.5, 'Testing MSE', va='center', rotation='vertical', fontsize=8)
    plt.tight_layout()
    plt.savefig("./log/%s.test.pdf" % (dataset), bbox_inches='tight', pad_inches=0.0)
