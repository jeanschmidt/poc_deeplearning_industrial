# poc_deeplearning_industrial
A Prof Of Concept for Deep Learning as replacement for traditional machine vision application on industrial setting

This is just a Proof Of Concept, the goal here is to prove that Deep Learning and Convolutional Neural Network can be used as a replacement for traditional machine vision application techniques: use of expensive cameras, lights and filters.

The classification of the quality that cashew nuts for end user consumption is one of the hardest problem for machine vision on industrial setting. As they naturally differ a lot in shape, size, texture, color and are hard to keep a consistent orientation to the camera. With all this problems, the use of DL and CNN seems to be a perfect fit. But, to be usable in a industrial setting it must prove itself:

1. Quick: The processing time should be as low as possible, allowing more throughput per machine;
2. Reliable: Preventing waste of god quality material, and guarantee the quality of downstream processes;
3. Stable: Given same conditions, give same results in a confident manner;

# This project proved that:

1. Can be implemented faster: From data acquisition, to prototyping and training took only 2 days;
2. Can be way cheaper: Total estimated cost for the solution dropped in at least 1 order of magnitude _(10X)_;
3. Can run faster: Total processing time for each image is **8ms** in my _GM108M [GeForce 940MX]_, compared to **30ms** on a standard industrial camera;
4. Is way more precise: With accuracy of _99.57%_ on learning data and _100%_ on validation data, being a total average of _99.6%_. Being considerable stable and reliable for industrial settings;

# Startup

To startup this project make sure you have `virtualenv` and `ImageMagick` installed on your system, check the URL for pytorch on `config.sh` based on your system requirements. Then simply runs:

`$ ./configure.sh`

In order to run the learning process simply type:

`$ ./run.sh`
