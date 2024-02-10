# ML-AI-Kriti-24

Introduction to the project - 

This project addresses 'Automated Research Paper Categorization' which is a ML-AI related problem statement released by Techboard of IIT Guwahati for the interhostel competition named Kriti'24. The main focus of this project is to increase the automation of submission system that not only streamlines the process of classification and categorization but also enhances user experience by offering intelligent category suggestions based on the paper's content. 

Model used - 

We've used LSTM ( Long Short Term Memory) model. Long Short-Term Memory is an advanced version of recurrent neural network (RNN) architecture that was designed to model chronological sequences and their long-range dependencies more precisely than conventional RNNs.LSTM networks are an extension of recurrent neural networks (RNNs) mainly introduced to handle situations where RNNs fail.  

Structure of LSTM model used - 

The basic difference between the architectures of RNNs and LSTMs is that the hidden layer of LSTM is a gated unit or gated cell. It consists of four layers that interact with one another in a way to produce the output of that cell along with the cell state. These two things are then passed onto the next hidden layer. Unlike RNNs which have got only a single neural net layer of tanh, LSTMs comprise three logistic sigmoid gates and one tanh layer. Gates have been introduced in order to limit the information that is passed through the cell. They determine which part of the information will be needed by the next cell and which part is to be discarded. The output is usually in the range of 0-1 where ‘0’ means ‘reject all’ and ‘1’ means ‘include all’.  


