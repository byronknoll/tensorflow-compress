# tensorflow-compress

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byronknoll/tensorflow-compress/blob/master/tensorflow-compress.ipynb)

Made by Byron Knoll. GitHub repository: https://github.com/byronknoll/tensorflow-compress

### Description

tensorflow-compress performs lossless data compression using neural networks in TensorFlow. It can run on GPUs with a large batch size to get a substantial speed improvement. It is made using Colab, which should make it easy to run through a web browser. You can choose a file, perform compression (or decompression), and download the result.

tensorflow-compress is open source and the code should hopefully be easy to understand and modify. Feel free to experiment with the code and create pull requests with improvements.

The neural network is trained from scratch during compression and decompression, so the model weights do not need to be stored. Arithmetic coding is used to encode the model predictions to a file.

Feel free to contact me at byron@byronknoll.com if you have any questions.

### Instructions:

Basic usage: configure all the fields in the "Parameters" section and select Runtime->Run All.

Advanced usage: save a copy of this notebook and modify the code.

### Related Projects
*   [NNCP](https://bellard.org/nncp/) - this uses a similar architecture to tensorflow-compress, but with transformers rather than LSTM. NNCP currently outperforms tensorflow-compress: running faster, on a worse GPU, while getting better compression rate.
*   [lstm-compress](https://github.com/byronknoll/lstm-compress) - uses LSTM for compression, but limited to running on a CPU with a batch size of one.
*   [cmix](http://www.byronknoll.com/cmix.html) - shares the same LSTM code as lstm-compress, but contains a bunch of other components to get better compression rate.
*   [DeepZip](https://github.com/mohit1997/DeepZip) - this also performs compression using TensorFlow. However, it has some substantial architecture differences to tensorflow-compress: it uses pretraining (using multiple passes over the training data) and stores the model weights in the compressed file.

### Benchmarks
These benchmarks were performed using tensorflow-compress v4 with the default parameter settings. Some parameters differ between enwik8 and enwik9 as noted below. Compute Engine VM was used with A100 GPU. Compression time and decompression time are approximately the same.
*   enwik8: compressed to 15,905,037 bytes in 32,048.55 seconds. NNCP preprocessing time: 206.38 seconds. Dictionary size: 65,987 bytes.
*   enwik9: compressed to 113,542,413 bytes in 289,632.17 seconds. NNCP preprocessing time: 1,762.28 seconds. Dictionary size: 79,876 bytes. The preprocessed enwik9 file was split into four parts using [this notebook](https://colab.sandbox.google.com/github/byronknoll/tensorflow-compress/blob/master/nncp-splitter.ipynb). The "checkpoint" option was used to save/load model weights between processing each part. For the first part, start_learning_rate=0.0005 and end_learning_rate=0.0002 was used. For the remaining three parts, a constant 0.0002 learning rate was used.

See the [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html) for more information about the test files and a comparison with other programs.

### Versions
* v4 - released August 10, 2022. Changes from v3:
  * Added embedding layer
  * Tuned parameters to run on A100 GPU
* v3 - released November 28, 2020. Changes from v2:
  * Parameter tuning
  * [New notebook](https://colab.sandbox.google.com/github/byronknoll/tensorflow-compress/blob/master/nncp-splitter.ipynb) for file splitting
  * Support for learning rate decay
* v2 - released September 6, 2020. Changes from v1:
  * 16 bit floats for improved speed
  * Weight updates occur at every timestep (instead of at spaced intervals)
  * Support for saving/loading model weights
* v1 - released July 20, 2020.
