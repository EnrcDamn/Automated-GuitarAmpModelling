*Note: originally forked from https://github.com/Alec-Wright/Automated-GuitarAmpModelling*

# Automated-Guitar Amplifier Modelling

## GuitarML Fork
This fork adds Colab Training, wav file pre-processing, results plotting, and some helpful printouts during training.
Use the example config "config/RNN3-acoustic1-pre.json" to train models for [NeuralPi](https://github.com/GuitarML/NeuralPi)<br>

Models for NeuralPi must have hidden_size=20 specified in the config file.

```
git clone https://github.com/GuitarML/Automated-GuitarAmpModelling.git
cd Automated-GuitarAmpModelling
git submodule update --init --recursive
```
```
# If Locally, install the required python packages 
# Using a python virtual enviroment is advisable
pip install -r ./requirements.txt
```
```
# Add your input.wav and output.wav to the top directory, and add your config to the config directory.
# Name the .wav files and config file appropriately in the following command
python prep_wav.py acoustic1-pre -s input.wav output.wav 
```
```
# Edit to use your config in the following command
# The model will be located in Results/
python dist_model_recnet.py -l "RNN3-acoustic1-pre"
```
```
# Edit your config name in the following command
# The plots will be generated in the Results/modelName/ directory
python plot.py acoustic1-pre
```

### Training Conditioned Models

Helper scipts have been added to aid in training conditioned models. These models are capable of reproducing the full range of one or more knobs, such as Gain/Drive/EQ. See the ```ExampleOfDifferentModelTraining.ipynb``` notebook for examples on how to train multi-parameter models. The ```prep_wav.py``` script has been updated to handle multi-parameter audio data.

You will still use mono, 32FP wav files for training conditioned models. See the ```Parameterization-Config.json``` and ```Parameterization-Config-2.json``` for examples on how to set up your audio data. Record 4-5 samples of the full range of a knob normalized as 0.0 to 1.0. For example: (0.0 0.33, 0.66, 1.0) or (0.0, 0.25, 0.5, 0.75, 1.0) for each knob or combination of knobs. 

Note: The more knobs you include in your model, the more .wav files you will need. For example, 2 knobs at five steps each will have 5 * 5 = 25 different combinations, for 100 individual wav files (1 train/validation in, 1 train/validation out, 1 test in, 1 test out) for each knob combination. You can train without separate test .wav files if desired, simply remove the "Test" entries from the config file. 

Note: [NeuralPi](https://github.com/GuitarML/NeuralPi) version 1.3 has the ability to run models conditioned on a single parameter as a real-time guitar plugin (the conditioned parameter is assumed to be gain/drive).

### Using Transfer Learning
You can improve training by starting with a pre-trained model of a similar amp/pedal. Simply start and stop the training script to generate the model file in the "Results" folder, then replace the "model.json" file with another trained model and restart training. This gives the training a head start, and can also reduce the amount of training data needed for a accurate model. 

For more information on using Transfer Learning for Guitar Effects check out this article published on [Towards Data Science](https://towardsdatascience.com/transfer-learning-for-guitar-effects-4af50609dce1)

##

This repository contains neural network training scripts and trained models of guitar amplifiers and distortion pedals. The 'Results' directory contains some example recurrent neural network models trained to emulate the ht-1 amplifier and Big Muff Pi fuzz pedal, these models are described in this [conference paper](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)

## Using this repository
It is possible to use this repository to train your own models. To model a different distortion pedal or amplifier, a dataset recorded from your target device is required, example datasets recorded from the ht1 and Big Muff Pi are contained in the 'Data' directory. For a set of examples of different trainings, please refer to the `./ExampleOfDifferentModelTraining.ipynb` file. 

### Cloning this repository

To create a working local copy of this repository, use the following command:

git clone --recurse-submodules https://github.com/Alec-Wright/NeuralGuitarAmpModelling

### Python Environment

Using this repository requires a python environment with the 'pytorch', 'scipy', 'tensorboard' and 'numpy' packages installed. A requirements.txt has been generated and is included, however your milage may vary based on your operating system and set up due to the nature of python dependency handling. 
Additionally this repository uses the 'CoreAudioML' package, which is included as a submodule. Cloining the repo as described in 'Cloning this repository' ensures the CoreAudioML package is also downloaded.

### Processing Audio

The 'proc_audio.py' script loads a neural network model and uses it to process some audio, then saving the processed audio. This is also a good way to check if your python environment is setup correctly. Running the script with no arguments:

python proc_audio.py

will use the default arguments, the script will load the 'model_best.json' file from the directory 'Results/ht1-ht11/' and use it to process the audio file 'Data/test/ht1-input.wav', then save the output audio as 'output.wav'
Different arguments can be used as follows

python proc_audio.py 'path/to/input_audio.wav' 'output_filename.wav' 'Results/path/to/model_best.json'

### Training Script

the 'dist_model_recnet.py' script was used to train the example models in the 'Results' directory. At the top of the file the 'argparser' contains a description of all the training script arguments, as well as their default values. To train a model using the default arguments, simply run the model from the command line as follows:

python dist_model_recnet.py

note that you must run this command from a python environment that has the libraries described in 'Python Environment' installed. To use different arguments in the training script you can change the default arguments directly in 'dist_model_recnet.py', or you can direct the 'dist_model_recnet.py' script to look for a config file that contains different arguments, for example by running the script using the following command:

python dist_model_recnet.py -l "ht11.json"

Where in this case the script will look for the file ht11.json in the the 'Configs' directory. To create custom config files, the ht11.json file provided can be edited in any text editor.

During training, the script will save some data to a folder in the Results directory. These are, the lowest loss achieved on the validation set so far in 'bestvloss.txt', as well as a copy of that model 'model_best.json', and the audio created by that model 'best_val_out.wav'. The neural network at the end of the most recent training epoch is also saved, as 'model.json'. When training is complete the test dataset is processed, and the audio produced and the test loss is also saved to the same directory.

A trained model contained in one of the 'model.json' or 'model_best.json' files can be loaded, see the 'proc_audio.py' script for an example of how this is done.

### Parameterization Configuration
The configuration file for a Parameterized training looks rather like the following.
```
{
    "Number of Parameters":2,
    "Data Sets":[
    {
	"Parameters": [ 0.0, 0.0 ],
	"TrainingClean": "./Recordings/DS1/20211105_LPB1_000_000_Training_Clean.wav",
	"TrainingTarget": "./Recordings/DS1/20211105_LPB1_000_000_Training_Dirty.wav",
	"TestClean": "./Recordings/DS1/20211105_LPB1_000_000_Test_Clean.wav",
	"TestTarget": "./Recordings/DS1/20211105_LPB1_000_000_Test_Dirty.wav"
    },
    {
	"Parameters": [ 0.0, 0.25 ],
	"TrainingClean": "./Recordings/DS1/20211105_LPB1_000_025_Training_Clean.wav",
	"TrainingTarget": "./Recordings/DS1/20211105_LPB1_000_025_Training_Dirty.wav",
	"TestClean": "./Recordings/DS1/20211105_LPB1_000_025_Test_Clean.wav",
	"TestTarget": "./Recordings/DS1/20211105_LPB1_000_025_Test_Dirty.wav"
    },
    ...
```
Note that the `"TestClean"` and `"TestTarget"` data sets are optional; if they are not included, the validation data set will be used for the training. However, this leads to a biased test results and is not advised for comparing model performance!

### Determinism

If determinism is desired, `dist_model_recnet.py` provides an option to seed all of the random number generators used at once. However, if NVIDIA CUDA is used, you must also handle the non-deterministic behavior of CUDA for RNN calculations as is described in the [Rev8 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/release-notes/rel_8.html). Because it is unadvisable to gloabaly configure the CUDA buffer size manually, it is recomended to launch jupyter with the CUDE buffer configuation as shown below for two buffers of size 4096.
```
CUBLAS_WORKSPACE_CONFIG=:4096:2 jupyter notebook
```
or for 8 buffers of 16:
```
CUBLAS_WORKSPACE_CONFIG=:16:8 jupyter notebook
```

### Tensorboard
The `./dist_model_recnet.py` has been implemented with PyTorch's Tensorboard hooks. To see the data, run:
```
tensorboard --logdir ./TensorboardData
```

### Feedback

This repository is still a work in progress, and I welcome your feedback! Either by raising an Issue or in the 'Discussions' tab 
