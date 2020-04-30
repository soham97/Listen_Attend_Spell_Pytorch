# Listen, Attend and Spell
This is my pytorch implementation of paper Listen, Attend and Spell (LAS) [1] which is a neural network that learns to transcribe speech utterances to characters. This implementation is used in this kaggle competition [here](https://www.kaggle.com/c/11-785-s20-hw4p2) and gets 11 levenshtein distance on leaderboard.

# Requirements
- Python3
- Pytorch
- Numpy
- Logging
- argparse
- [Levenshtein](https://pypi.org/project/python-Levenshtein/)

# Running
Download data from competition and store in 'data' folder. <br>
LAS can be used in three modes: <br>
- run_train.sh: Training model from scratch
- run_train_continue.sh: Continue training a model
- run_test.sh: Generate test submission <br>
Specify the parameters you want to run the experiments for in the shell scripts. The description of parameters is in main.py.

# Notes:
- The implementation uses greedy decoding. For random decoding, you can refer master branch. There is only a minor difference in performance between both.
- The Encoder using pBiLSTM in concat mode, where the forward and backward direction hidden state are concatenated. You can choose average, where the forward and backward direction hidden state are averaged. For switching to average in code, couple of changes need to done. They are: uncommenting line 103, 115 in model.py, commenting line 104 in model.py, the LSTM input size in 82, 83, 84 in model.py needs to be reduced by 4. This will save you compute and running time over little decreased performance
- Scheduler is initialised by not used, if used with patience on 5 on val_dist with switching to SGD in later epochs can help you reach in 10 Levenshtein distance on test
- Adding CNN layers before piBLSTM will also help in improving the score to 9.0 Levenshtein distance on test

# References
[1] W. Chan, N. Jaitly, Q. Le and O. Vinyals, "Listen, attend and spell: A neural network for large vocabulary conversational speech recognition," 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Shanghai, 2016, pp. 4960-4964.
