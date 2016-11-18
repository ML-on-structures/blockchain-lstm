# blockchain-lstm
Multi level sequence learners with data from Bitcoin's blockchain to predict spending.

This repository contains the implementation of a module that uses multi level sequence learners to predict spending of addresses.

The sequence learners are implemented with Long Short-Term Memory recurrent neural networks.

See [project page](https://sites.google.com/view/ml-on-structures) for the overall project, papers, and data.

## Data

We obtain the dataset of addresses by using a slice of the blockchain.

In particular, we consider all the addresses where deposits happened in a range of 101 blocks, from 200,000 to 200,100 (included)

They contain 15,709 unique addresses where deposits took place.

Looking at the state of the blockchain after 50,000 blocks (which corresponds to roughly one year later as each block is mined on average every 10 minutes), 3,717 of those addresses still had funds sitting: we call these hoarding addresses'.

The goal is to predict which addresses are hoarding addresses, and which addresses had spent the funds (i.e. had a balance of zero at the later state)

We create a graph with addresses as nodes, and transactions as edges. 

Each edge was labeled with features of the transaction: its time, amount of funds transmitted, number of recipients, and so forth, for a total of 9 features.

## Running Code

In order to run the code, you need to invoke the method multi_layer in the blockchain_data_import.py file

In the local folder, at a directory called data there should be the blockchain csv files that can be found
at : [https://drive.google.com/open?id=0ByhyeddNVklEQ2FwOHFLbmE4alU](https://drive.google.com/open?id=0ByhyeddNVklEQ2FwOHFLbmE4alU)

Run python blockchain_data_import.py to run it.

Changing the DEPTH parameter will change the depth of tree unfoldings (for this dataset, it can range from 1 to 3)

Results are printed on the screen.
