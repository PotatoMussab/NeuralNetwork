class Layer:
    ##Weights are indexed [to][from]
    ##The weights on a layer are the ones connected between it and the prev layer
    ##The biases on a layer are the ones directly on it
    def __init__(this,nodes):
        this.nodes=nodes
        this.activation=[0]*nodes
        this.errorSum=0
        this.isOutputLayer=False
        this.learnRate=0.3
        this.cost=[0]*nodes
        
    def setPrevLayer(this,prevLayer):
        this.prevLayer=prevLayer

    def setNextLayer(this,nextLayer):
        this.nextLayer=nextLayer
        
    def setWeights(this,weight):
        this.weight=weight
    
    def setBiases(this,bias):
        this.bias=bias
        
    def propagate(this):
        for i in range(this.nodes):
            sum=0
            for j in range(len(this.weight[i])):
                sum=sum+this.weight[i][j]*this.prevLayer.getActivation()[j]
            sum=sum+this.bias[i]
            if(sum>0):
                this.activation[i]=sum
            else:
                this.activation[i]=0
        return 0

    def getActivation(this):
        return this.activation
    
    def setActivation(this,node,newVal):
        this.activation[node]=newVal
        
    def setActivation(this,newVals):
        this.activation=newVals
        
    def getNodes(this):
        return this.nodes

    def setCost(this,cost):
        this.cost=cost
        
    def backPropagate(this):
        nWeight=this.nextLayer.weight
        nBias=this.bias
        for j in range(this.nodes):
            cost=0
            for i in range(this.nextLayer.nodes):
                nWeight[i][j]=nWeight[i][j]-(this.learnRate * this.nextLayer.cost[i] * this.activation[j])
                #print("Adding {} to W{}{}".format((this.learnRate * this.nextLayer.cost[i] * this.activation[j]),i,j))
                if this.nextLayer.activation[i]>0:
                    nBias[i]=nBias[i]-(this.learnRate*this.nextLayer.cost[i])
                    #print("Adding {} to B{}".format((this.learnRate*this.nextLayer.cost[i]),i))
                    cost=cost-(this.nextLayer.cost[i]*this.nextLayer.weight[i][j])
            this.cost[j]=cost
        this.nextLayer.weight=nWeight
        this.bias=nBias
        
#=============================================================================      
class Network:
    
    def __init__(this):
        this.numOfLayers=0
        this.Layers=[None]
        this.numOfOutputs=0
        this.numOfTrainingData=0
        
    def addLayer(this,layer):
        this.numOfLayers=this.numOfLayers+1
        numOfLayers=this.numOfLayers
        if numOfLayers==1:
            this.Layers[numOfLayers-1]=layer
            this.Layers[numOfLayers-1].setBiases([1]*this.Layers[numOfLayers-1].getNodes())
        else:
            this.Layers.append(layer)
            this.Layers[numOfLayers-1].setBiases([1]*layer.getNodes())
            this.Layers[numOfLayers-1].setPrevLayer(this.Layers[numOfLayers-2])
            this.Layers[numOfLayers-1].setWeights([[1]*this.Layers[numOfLayers-2].getNodes()]*this.Layers[numOfLayers-1].getNodes())
            this.Layers[numOfLayers-2].setNextLayer(this.Layers[numOfLayers-1])
        this.numOfOutputs=this.Layers[this.numOfLayers-1].getNodes()
        
    def setWeights(this,layer,weights):
        this.Layers[layer].setWeights(weights)

    def setBiases(this,layer,biases):
        this.Layers[layer].setBiases(biases)

    def setInput(this,inputs):
        this.Layers[0].setActivation(inputs)

    def setLearnRate(this,lr):
        for i in this.Layers:
            i.learnRate=lr
    
    def getOutput(this):
        for n in range(1,this.numOfLayers):
            this.Layers[n].propagate()
        return this.Layers[this.numOfLayers-1].getActivation()
    
    def addTrainingData(this,inputs,desOutputs):
        this.setInput(inputs)
        outputs=this.getOutput()
        cost=[0]*this.numOfOutputs
        for i in range(this.numOfOutputs):
            cost[i]=cost[i]+2*(outputs[i]-desOutputs[i])
        this.Layers[this.numOfLayers-1].setCost(cost)
        this.numOfTrainingData=this.numOfTrainingData+1

    def train(this):
        for i in range(len(this.Layers[this.numOfLayers-1].cost)):
            this.Layers[this.numOfLayers-1].cost[i]=this.Layers[this.numOfLayers-1].cost[i]/this.numOfTrainingData
        numOfTrainingData=0
        for i in range(this.numOfLayers-2,-1,-1):
            this.Layers[i].backPropagate()
        this.Layers[this.numOfLayers-1].cost=[0]*this.numOfOutputs

    def getOutputCost(this):
        return this.Layers[this.numOfLayers-1].cost
