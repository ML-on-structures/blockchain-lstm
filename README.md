# blockchain-lstm
Multi level sequence learners with data from Bitcoin's blockchain to predict spending.

This repository contains the implementation of a module that uses multi level sequence learners to predict spending of addresses.

The sequence learners are implemented with Long Short-Term Memory recurrent neural networks.

See [https://sites.google.com/view/ml-on-structures](project page) for the overall project, papers, and data.

Running Code

In order to run the code, you need to invoke the method multi_layer in the blockchain_data_import.py file
In the local folder, at a directory called data there should be the blockchain csv files that can be found
at : [https://drive.google.com/open?id=0ByhyeddNVklEQ2FwOHFLbmE4alU](https://drive.google.com/open?id=0ByhyeddNVklEQ2FwOHFLbmE4alU)

Run python blockchain_data_import.py to run it.

Changing the DEPTH parameter will change the depth of tree unfoldings (for this dataset, it can range from 1 to 3)

Results are printed on the screen.
