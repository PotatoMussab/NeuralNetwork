# Simple Backpropagated Neural Network

This program was made so I could practice Python while also learn the basics about Neural Networks. The neural network has a Rectified Linear Unit (ReLU) activation function on all the nodes. Every bias and weight is 1 by default. ML.py contains the Layer and Network classes.

# How to use Layer class:

- Layers are initialized as Layer(number of nodes)
- setWeights(weights) sets the weights between this layer and the one before it. The argument passed should be a two dimensional list containing the weights. The syntax for the weight indexes should be W[to][from]. As in, W[1][2] refers to the weight connecting node 2 of the previous layer to node 1 of the current layer. 
- setBiases(biases) sets the biases of the layer. The argument passed should be a one dimensional list containing all the biases of the layer
- getNodes() returns the number of nodes in the layer
- weights and biases can be directly accessed through the layerObj.weight and layerObj.bias attributes as well.

# How to use Network class
- Instantiate a network and add layers to it using addLayer(layer). The first layer added will become the input layer. The last one added will become the output layer.
- Add training data using addTrainingData(inputs,desiredOutputs) and train the network for one epoch using train(). Using train() will reset the training data
- Set an input using setInput(inputs) and receive an output using getOutput()
