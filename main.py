import ML
inLayer=ML.Layer(3)
midLayer=ML.Layer(3)
outLayer=ML.Layer(1)
network = ML.Network()
network.addLayer(inLayer)
network.addLayer(outLayer)
network.setInput([1,1,1])
print(network.getOutput())
for i in range(30):
    network.addTrainingData([1,1,1],[20])
    network.train()
    print(network.getOutput())
