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
from matplotlib import cm 
import matplotlib

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

def read_gradient(filename): 
    print "read ", filename 
    grad_data = []
    with open(filename, "r") as f: 
        count = 0
        for line in f:
            if line.startswith("initial_state_grads_value"): 
                l1 = re.search(r"l1 = (\d+\.\d+|\d+)", line)
                l1 = float(l1.group(1))

                l2 = re.search(r"l2 = (\d+\.\d+|\d+)", line)
                l2 = float(l2.group(1))

                linf = re.search(r"linf = (\d+\.\d+|\d+)", line)
                linf = float(linf.group(1))

                lninf = re.search(r"lninf = (\d+\.\d+|\d+)", line)
                lninf = float(lninf.group(1))

                grad_data.append([count, l1, l2, linf, lninf])

                count += 1

    grad_data = np.array(grad_data)

    return grad_data

def read_gradient_time(filename): 
    print "read ", filename 
    grad_data = {}
    with open(filename, "r") as f: 
        for line in f:
            if line.startswith("Epoch"):
                epoch = re.search(r"Epoch ([-+]?\d+)", line)
                epoch = int(epoch.group(1))
            if line.startswith("initial_state_grads_values"): 
                idx = re.search(r"initial_state_grads_values\[(\d+)\]", line)
                idx = int(idx.group(1))

                l1 = re.search(r"l1 = (\d+\.\d+|\d+)", line)
                l1 = float(l1.group(1))

                l2 = re.search(r"l2 = (\d+\.\d+|\d+)", line)
                l2 = float(l2.group(1))

                linf = re.search(r"linf = (\d+\.\d+|\d+)", line)
                linf = float(linf.group(1))

                lninf = re.search(r"lninf = (\d+\.\d+|\d+)", line)
                lninf = float(lninf.group(1))

                if epoch not in grad_data:
                    grad_data[epoch] = []
                grad_data[epoch].append([idx, l1, l2, linf, lninf])

    for key in grad_data:
        grad_data[key] = np.array(grad_data[key])

    return grad_data

#dataset = "mnist.784"
#dataset = "mnist.permute.784"
#dataset = "imdb.300"
dataset = "poly_synthetic.nfreqs5_npoly5"
#dataset = "sine_synthetic"

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
    #folders = ["degree5_npoly5", "degree10_npoly5", "degree15_npoly5"]
    folders = ["degree5_npoly5"]
    #folder = dataset[len("poly_synthetic."):]
    num_samples = 4000
    time_step = 176
    batch_size = 32
    num_batch = float(num_samples*0.8/batch_size) # convert iterations to epochs approximately 

    #fig1 = plt.figure(1, figsize=(4, 3))
    #plt.subplots_adjust(hspace=0.1)

    #ax = [None]
    #for i, folder in enumerate(folders): 
    #    extract = re.search(r"degree(\d+)_npoly(\d+)", folder)
    #    degree = int(extract.group(1))
    #    npoly = int(extract.group(2))

    #    ax.append(plt.subplot(311+i, sharex=ax[-1], sharey=ax[-1]))

    #    train_data, test_data = read("./repeat/poly_synthetic/%s/FRU.log" % (folder))
    #    data = test_data
    #    plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='FRU', linestyle='-', color='red', linewidth = 0.5)

    #    train_data, test_data = read("./repeat/poly_synthetic/%s/SRU.log" % (folder))
    #    data = test_data
    #    plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='SRU', linestyle='dashed', color='blue', marker = 'v', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    #    train_data, test_data = read("./repeat/poly_synthetic/%s/LSTM.log" % (folder))
    #    data = test_data
    #    plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='LSTM', linestyle=':', color='green', marker = 'o', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    #    train_data, test_data = read("./repeat/poly_synthetic/%s/RNN.log" % (folder))
    #    data = test_data
    #    plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='RNN', linestyle='-.', color='black', marker = 'x', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    #    #plt.gca().set_yscale('symlog', linthreshy=1e-7)
    #    plt.setp(ax[-1].get_xticklabels(), visible=False if i+1 < len(folders) else True, fontsize=8)
    #    plt.setp(ax[-1].get_yticklabels(), fontsize=8)
    #    plt.ylim(5*1e-7, 1e-1)
    #    plt.yticks([1e-6, 1e-4, 1e-2])

    #    plt.xlabel("")
    #    plt.title("")
    #    ax[-1].title.set_visible(False)
    #    if i == 0: 
    #        plt.legend(loc=9, bbox_to_anchor=(0.5, 1.5), ncol=4, fontsize=8)
    #    plt.tight_layout()
    #    #plt.subplots_adjust(hspace=0.0, bottom=0, top=0.1)

    ##plt.yticks([50, 60, 70, 80, 90, 100])
    ##plt.gca().yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))
    ##plt.gca().yaxis.set_major_formatter(ticker.LogFormatter())

    ##plt.title("Polynomial Data (%d, %d)" % (npoly, degree))
    ##plt.title("Polynomial Data with Max Degree %d" % (degree))
    ##plt.xlabel('Epoch')
    ##plt.ylabel('Testing MSE')
    #fig1.text(0.5, 0.0, 'Epoch', ha='center', fontsize=8)
    #fig1.text(0.0, 0.5, 'Testing MSE', va='center', rotation='vertical', fontsize=8)
    #fig1.text(0.25, 0.74, 'Mix-Poly Data (5, 5)', ha='center', fontsize=6)
    #fig1.text(0.25, 0.433, 'Mix-Poly Data (5, 10)', ha='center', fontsize=6)
    #fig1.text(0.25, 0.125, 'Mix-Poly Data (5, 15)', ha='center', fontsize=6)
    #plt.tight_layout()
    #plt.savefig("./log/%s.test.pdf" % (dataset), bbox_inches='tight', pad_inches=0.0)

    #for k, folder in enumerate(folders): 
    #    extract = re.search(r"degree(\d+)_npoly(\d+)", folder)
    #    degree = int(extract.group(1))
    #    npoly = int(extract.group(2))

    #    fig1 = plt.figure(1)
    #    grad_norms = ['L1', 'L2', 'LInf', 'LNInf']
    #    for i in range(3):
    #        plt.subplot(220+i+1)

    #        grad_data = read_gradient("./repeat/poly_synthetic/%s/FRU.log" % (folder))
    #        data = grad_data
    #        plt.semilogy(data[:, 0]/num_batch, data[:, i+1], label='FRU', linestyle='none', color='red', marker = '*', linewidth = 0.5, fillstyle='none', markeredgewidth = '0.1')

    #        grad_data = read_gradient("./repeat/poly_synthetic/%s/SRU.log" % (folder))
    #        data = grad_data
    #        plt.semilogy(data[:, 0]/num_batch, data[:, i+1], label='SRU', linestyle='none', color='blue', marker = 'v', markersize=2, linewidth = 0.4, fillstyle='none', markeredgewidth = '0.1')

    #        grad_data = read_gradient("./repeat/poly_synthetic/%s/LSTM.log" % (folder))
    #        data = grad_data
    #        plt.semilogy(data[:, 0]/num_batch, data[:, i+1], label='LSTM', linestyle='none', color='green', marker = 'o', markersize=2, linewidth = 0.4, fillstyle='none', markeredgewidth = '0.1')

    #        grad_data = read_gradient("./repeat/poly_synthetic/%s/RNN.log" % (folder))
    #        data = grad_data
    #        plt.semilogy(data[:, 0]/num_batch, data[:, i+1], label='RNN', linestyle='none', color='black', marker = 'x', markersize=2, linewidth = 0.4, fillstyle='none', markeredgewidth = '0.1')

    #        #plt.legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=4, fontsize=8)
    #        if i == 2: 
    #            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1, fontsize=8)
    #        plt.title("Gradient %s Norm" % (grad_norms[i]))
    #        plt.xlabel('Epoch', fontsize=8)

    #    plt.tight_layout()
    #    filename = "./repeat/poly_synthetic.%s.grads.png" % (folder)
    #    print filename
    #    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)

    markers = [None, 'v', 'x', 's']
    linestyles = ['-', 'dashed', ':', '-.']
    cmap = cm.get_cmap('plasma')
    #epochs = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    models = ['FRU', 'SRU', 'LSTM', 'RNN']
    norm = matplotlib.colors.Normalize(vmin=np.amin(epochs), vmax=np.amax(epochs))
    for k, folder in enumerate(folders): 
        extract = re.search(r"degree(\d+)_npoly(\d+)", folder)
        degree = int(extract.group(1))
        npoly = int(extract.group(2))

        fig1 = plt.figure(1, figsize=(18, 14))
        grad_norms = ['L1', 'L2', 'LInf', 'LNInf']
        data_fru = read_gradient_time("./repeat/poly_synthetic/%s/FRU.log" % (folder))
        data_sru = read_gradient_time("./repeat/poly_synthetic/%s/SRU.log" % (folder))
        data_lstm = read_gradient_time("./repeat/poly_synthetic/%s/LSTM.log" % (folder))
        data_rnn = read_gradient_time("./repeat/poly_synthetic/%s/RNN.log" % (folder))
        avg_time_step = 20
        lines = []
        for i in range(3):
            for ii, model in enumerate(models):
                plt.subplot(3, 4, i*4+ii+1)
                if ii == 0:
                    data = data_fru
                elif ii == 1:
                    data = data_sru 
                elif ii == 2:
                    data = data_lstm 
                else:
                    data = data_rnn
                for j, epoch in enumerate(epochs): 

                    line, = plt.semilogy(data[epoch-1][:, 0]*avg_time_step, data[epoch-1][:, i+1], 
                            label=model, linestyle=linestyles[ii], color=cmap(norm(epoch)), marker = markers[ii], linewidth = 0.5, fillstyle='none', markeredgewidth = '0.1')
                    lines.append(line)

                    #plt.legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=4, fontsize=8)
                    #if i == 2: 
                    #    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1, fontsize=8)
                    plt.title("%s Gradient %s Norm" % (model, grad_norms[i]))
                    plt.xlabel('t')
                    plt.gca().set_yscale('symlog', linthreshy=1e-5)
                    plt.ylim(-1e-5, 1e2)

        #plt.figlegend(lines[:len(epochs)], labels=['FRU', 'SRU', 'LSTM', 'RNN'], loc='lower center', ncol=4)
        cbar_ax = fig1.add_axes([0.3, 0.05, 0.4, 0.02])
        cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
        cbar.set_label('Epoch')
        plt.subplots_adjust()
        #plt.tight_layout()
        filename = "./repeat/poly_synthetic.%s.grads_time.pdf" % (folder)
        print filename
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)

elif dataset.startswith("sine_synthetic"): 
    num_samples = 4000
    time_step = 176
    batch_size = 32
    num_batch = float(num_samples*0.8/batch_size) # convert iterations to epochs approximately 

    #fig1 = plt.figure(1, figsize=(4, 2))
    #
    #train_data, test_data = read("./repeat/sine_synthetic/FRU.log")
    #data = test_data
    #plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='FRU', linestyle='-', color='red', linewidth = 0.5)

    #train_data, test_data = read("./repeat/sine_synthetic/SRU.log")
    #data = test_data
    #plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='SRU', linestyle='dashed', color='blue', marker = 'v', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    #train_data, test_data = read("./repeat/sine_synthetic/LSTM.log")
    #data = test_data
    #plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='LSTM', linestyle=':', color='green', marker = 'o', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    #train_data, test_data = read("./repeat/sine_synthetic/RNN.log")
    #data = test_data
    #plt.semilogy(data[:, 0]/num_batch, data[:, 1], label='RNN', linestyle='-.', color='black', marker = 'x', markersize=2, linewidth = 0.4,fillstyle='none', markeredgewidth = '0.1')

    ##plt.ylim(1e-5, 100)
    #plt.legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=4, fontsize=8)

    ##plt.yticks([50, 60, 70, 80, 90, 100])
    ##plt.gca().yaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))
    ##plt.gca().yaxis.set_major_formatter(ticker.LogFormatter())

    ##plt.title("Mix-Sin Data with %d Frequencies" % (nfreqs), fontsize=8)
    ##plt.xlabel('Epoch')
    ##plt.ylabel('Testing MSE')
    #fig1.text(0.5, 0.0, 'Epoch', ha='center', fontsize=8)
    #fig1.text(0.0, 0.5, 'Testing MSE', va='center', rotation='vertical', fontsize=8)
    #plt.tight_layout()
    #plt.savefig("./repeat/%s.test.pdf" % (dataset), bbox_inches='tight', pad_inches=0.0)

    fig1 = plt.figure(1)
    grad_norms = ['L1', 'L2', 'LInf', 'LNInf']
    for i in range(3):
        plt.subplot(220+i+1)

        grad_data = read_gradient("./repeat/sine_synthetic/FRU.log")
        data = grad_data
        plt.semilogy(data[:, 0]/num_batch, data[:, i+1], label='FRU', linestyle='none', color='red', marker = '*', linewidth = 0.5, fillstyle='none', markeredgewidth = '0.1')

        grad_data = read_gradient("./repeat/sine_synthetic/SRU.log")
        data = grad_data
        plt.semilogy(data[:, 0]/num_batch, data[:, i+1], label='SRU', linestyle='none', color='blue', marker = 'v', markersize=2, linewidth = 0.4, fillstyle='none', markeredgewidth = '0.1')

        grad_data = read_gradient("./repeat/sine_synthetic/LSTM.log")
        data = grad_data
        plt.semilogy(data[:, 0]/num_batch, data[:, i+1], label='LSTM', linestyle='none', color='green', marker = 'o', markersize=2, linewidth = 0.4, fillstyle='none', markeredgewidth = '0.1')

        grad_data = read_gradient("./repeat/sine_synthetic/RNN.log")
        data = grad_data
        plt.semilogy(data[:, 0]/num_batch, data[:, i+1], label='RNN', linestyle='none', color='black', marker = 'x', markersize=2, linewidth = 0.4, fillstyle='none', markeredgewidth = '0.1')

        #plt.legend(loc=9, bbox_to_anchor=(0.5, 1.23), ncol=4, fontsize=8)
        if i == 2: 
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1, fontsize=8)
        plt.title("Gradient %s Norm" % (grad_norms[i]))
        plt.xlabel('Epoch', fontsize=8)

    plt.tight_layout()
    filename = "./repeat/%s.grads.png" % (dataset)
    print filename
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
