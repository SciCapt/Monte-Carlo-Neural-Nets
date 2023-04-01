from matplotlib import pyplot as plt
import numpy as np
import textwrap as txt

class AdvNet:
    ## Default Functions ## 
    def __init__(self, inputCount: int, hiddenCounts: list, outputCount: int, activations):
        """
        Create a deep neural network from the given counts of neurons requested per layer.

        Inputs:

        1 - inputCount
        - Type == int
        - Range == (0, inf]

        2 - hiddenCounts
        - Type == List of int
        - Range == (0, inf]

        3 - outputCount
        - Type == int
        - Range == (0, inf]

        4 - activations
        - Possible Values: "NONE", "ELU", "ATAN", "RELU"
            - NONE: No applied function
            - ELU: max(val, -0.4)
            - ATAN: max(val, -1) & min(val, 1)
            - RELU: max(val, 0)
        - A list of different functions per layer can be given, otherwise, all calculation
          steps will use the single given activation function.
        - Examples:
            - AdvNet(1, [10, 20, 10], 2, "RELU") 
                - 5-layer net / 4 in-between spaces
                - Uses the "RELU" activation function for all (4) in-between spaces.
            - AdvNet(2, [15, 15], 5, ["RELU", "ELU", "ATAN"]) 
                - 4-layer net / 3 in-between spaces
                - Uses RELU in between layers 1 & 2, ELU in between layers 2 & 3, and 
                ATAN in between layers 3 & 4.

        Operation:

        A deep neural network is generated from the requested sizes, with 
        the initial weights set to random parameters in the range [-1, 1].
        The values given are directly used to create the amount of neurons
        for the specific layer.

        The layer's heights are in the order given, that is:

        > inputCount, hiddenCounts[0], hiddenCounts[1], ... hiddenCounts[n-1], outputCount

        Where n is just the length of the given hiddenCounts list given.

        For each step of forward propogation (n-1 total steps / spaces) an activation function
        is referenced from the activation functions that the net has (use print(net) to see these).
        If only one type was provided in the nets creation, all possible spaces use just this 
        activation type.
        """

        # Matrix/Stuff Sizes
        self.inSize = int(inputCount)
        self.outSize = int(outputCount)
        self.hiddenSize = hiddenCounts

        # Setup weights
        self.sizes = [inputCount] + hiddenCounts + [outputCount]
        self.weights = []
        for i in range(len(self.sizes) - 1):
            # Initialize weights with some random weights in range [-0.5, 0.5]
            newWeightsArr = (np.random.rand(self.sizes[i], self.sizes[i+1]) - 0.5)
            self.weights.append(newWeightsArr)

        # Number of Parameters in the Weights
        weightSizes = [M.size for M in self.weights]
        self.parameters = sum(weightSizes)

        # Construct activation functions list (if only given a single function type)
        if type(activations) == str:
            self.activationFunction = [activations]*len(self.weights)
        elif type(activations) == list:
            # Check that enough activation functions are given
            if len(activations) >= len(self.weights):
                self.activationFunction = activations
            else:
                raise ValueError(f"Not enough activation functions are provided! ({len(activations)} given, {len(self.weights)} required)")

    def __str__(self):
        # String Forming and formating
        strlen = 64
        l1 = f"="
        l2 = f"Neural Net Characteristics:"

        l3 = f"1. Layer Sizes = {self.sizes}"
        l3 = txt.fill(l3, strlen-2)

        weightMedians = []
        for weights in self.weights:
            weightMedians.append(round(np.median(weights), 2))
        l4 = f"2. Weight Medians = {weightMedians}"
        l4 = txt.fill(l4, strlen-2)

        l6 = f"3. Number of Parameters: {self.parameters}"

        l7a = f"4. Activation Functions: {self.activationFunction}"
        l7a = txt.fill(l7a, strlen-2)

        # l7b = f"{self.activationFunction}"
        l5 = f"="

        # Resulting String
        full = (l1*strlen + '\n' + l2.center(strlen, ' ') + '\n' + l3 + 
                '\n' + l4 + '\n' + l6 + '\n' + 
                l7a + '\n' + l5*strlen)
        return full

    ## Internal Functions ##
    def TweakWeights(self, Amplitude: float, Selection='all'):
        """
        Adjust the weights of a neural net by a given amplitude.
        
        Inputs:

        1 - Amplitude
        - Type == float
        - Range == (0, inf]*

        2 - Selection
        - Type == List (or 'gates' or 'middle')
        - Used to select specific weight set(s) to train instead of training all weights.
        Useful for making sure the first and last sets of weights are trained correctly to
        avoid 'gate keeping'.
        - The string 'gates' can be used to select the first and last weight arrays to train
        while 'middle' can be used to select all weight arrays (if any) in between the 'gates'.

        
        Operation:

        For a weight array of a given size, an array of random values is
        generated of the same size. These values are random float values
        in the range [0,1].
        
        The range of these values is transformed to a middle of 0 by subtracting 0.5.
        By then multiplying by 2*Amplitude, the new range becomes [-Amplitude, Amplitude].

        *The net weights are clipped to the range [-1,1], therefore, an amplitude magnitude 
        above 1 is rather aggressive. Typical starting values include 0.2, 0.02, etc. and
        can be decreased to achieve finer tuning.

        Selection can be used to only tweak a specfic set of ranges, or even just a single one
        for even finer tuning of the parameters. Any indicies given in Selection that are out 
        of the range of weight arrays are ignored.

         """
        # Translate selection of weights reqested, if any
        if Selection == 'all':
            # Use default range (all weights)
            Selection = [*range(len(self.weights))]
        elif Selection == 'gates':
            # Select first and last arrays
            Selection = [0, -1]
        elif Selection == 'middle':
            # Select all non-gate arrays
            Selection = [*range(1, len(self.weights))]
        else:
            # Check selection for any non-valid indicies to call the weights
            Selection.sort()
            for i in range(-1, -len(Selection)-1, -1):
                # Positive indicies
                if Selection[i] >= len(self.weights):
                    Selection.pop(i)
                # Negative indicies
                if (Selection[i] < 0 and
                    Selection[i] + len(self.weights) < 0):
                    Selection.pop(i)

        # Adjust all the weights randomly, with a maximum change magnitude of Amplitude
        for i in Selection:
            # Negative Indicies
            if i < 0:
                # raise ValueError("Negative Indicies to select weights is not supported!")
                i += len(self.weights)

            # Generate arrays of values to tweak existing weights
            weightTweak = (np.random.rand(self.sizes[i], self.sizes[i+1]) - 0.5)*(2*Amplitude)

            # Add tweaks to existing weights
            self.weights[i] += weightTweak

            # Make sure values are in the range [-1, 1]
            self.weights[i] = np.clip(self.weights[i], -1, 1)

    def Calculate(self, inVector):
        """
        Returns the neural net's calculation for a given input vector. If the net used has an input size
        of one, a single float value can be given (array type not required).

        Inputs:

        1 - inVector
        - Type == NumPy Array
        - Size == (netInputCount, 1) or (1, netInputCount)
        - Note: 
            - If the net input size is 1, then a float can be given and a vector form is not required.
        
        2 - hiddenFunction
        - Type == String (NONE, ELU, ATAN, RELU)
        - Note:
            - If the net has an activation function for a particular layer (ie. not the defualt None
            value) this is used instead of the "hiddenFunction" provided. Similarly, if the net has a
            None value for a layer (as by default), the "hiddenFunction" provided is used.

        -- Lore --

        Simple forward propogation of the vector through the neural net that the function is called on via 
        matrix multiplication with the net's weights.

        The hidden function handles any desired non-linear behavior within the hidden layers. "NONE" can be
        selected if desired, although this provides a fully linear net. That is, if trying to curve fit
        using a net, it will only be capable of finding the best linear regression of the given data.
        
        Selecting 'ELU' would, instead, process each summed vector to the non-linear function max(val, -0.4). 
        
        Similarly, "RELU" applies the function max(val, 0), such that all values below zero are set to zero.
        
        "ATAN" instead limits values to the range [-1, 1] before being passed forward to the next set 
        of weights.       
        """
        # Handling if a single number/float was given not in an array form
        try:
            # Test if a vector form was given
            test = inVector.size
        except:
            # Not a vector -- check that net input size is one to confirm a single value works
            if self.inSize == 1:
                inVector = np.array(inVector).reshape(1, 1)
            
        # Check if the overall size is correct
        if inVector.size == np.size(inVector, 0) or inVector.size == np.size(inVector, 1):
            # Convert row vectors to column vectors
            if self.inSize != 1:
                if inVector.size == np.size(inVector, 1):
                    inVector = inVector.transpose()

            # Vector has the right column vector shape -- now check for correct size
            if inVector.size == self.inSize:
                # Vector is of the right shape and size so continue

                # Go through all of weights calculating the next vector
                ## in vector, w1 vec, w2 vec, ... out vec
                calcVec = inVector # Start the calcVector with the given input vector
                for i, weights in enumerate(self.weights):
                    # Forward propogation
                    calcVec = sum(calcVec*weights)

                    # Use the net's activation function for the current space
                    hiddenFunction = self.activationFunction[i]

                    # Apply the activation function
                    if hiddenFunction == 'NONE':
                        pass
                    elif hiddenFunction == 'ELU':
                        calcVec[calcVec < -0.4] = 0
                    elif hiddenFunction == 'ATAN':
                        calcVec[calcVec < -1] = -1
                        calcVec[calcVec > 1]  = 1
                    elif hiddenFunction == 'RELU':
                        calcVec[calcVec < 0] = 0
                    else:
                        raise ValueError("Given hidden function is not of the avalible types!")

                    # Triple-checking size throughout forward prop.
                    calcVec = calcVec.reshape(calcVec.size, 1)

                # Handling for single-number outputs (free it from the vector at the end)
                if self.outSize == 1:
                    calcVec = calcVec[0][0]

                return calcVec
            
            # Vector is not of the correct shape for the used neural net
            else:
                raise ValueError(f"inVector size ({inVector.size}) does not match the net input size ({self.inSize})")

    def CopyNet(self):
        """
        Copy a neural net object to a new variable. Used in similar
        situations as NumPy's copy method to make an isolated net
        with the same values and properties as the parent net.

        """
        # New Net Matrix/Stuff Sizes
        newInputCount = self.inSize
        newOutputCount = self.outSize
        newHiddenCounts = self.hiddenSize
        newActivations = self.activationFunction
        newSizes = [newInputCount] + newHiddenCounts + [newOutputCount]

        # Make net net shell
        newNet = AdvNet(newInputCount, newHiddenCounts, newOutputCount, newActivations)
        
        # Setup weights
        for i in range(len(newSizes) - 1):
            newNet.weights[i] = self.weights[i].copy()

        # Return the copied net
        return newNet

    def SaveNN(self, name: str):
        """
        A method to save the neural net system for later recall under the given nickname. The
        file names are saved under a specific name and should not be
        altered. If errors related to the files is created (i.e. 
        'permission denied') then the simplest solution is simply
        to move the files to a local (non-cloud) drive instead, if possible.
        
        Inputs:
        
        1 - Name
        - Type == String
        
        """
        # Save the various Characteristics and weights of the NN to load later
        name = str(name)
        np.savetxt(f"NN_{name}_Size", self.sizes)
        for i in range(len(self.sizes) - 1):
            # Redundant save cycles in case of 'onedrive' situations
            ## Limit save tries to 10 times, if thats not working
            ## something else is likely wrong that this won't fix
            for tries in range(10):
                try:
                    np.savetxt(f"NN_{name}_Weights{i+1}", self.weights[i])
                    # if made this far, it successfully saved so break the loop
                    break
                except:
                    # Something didn't work, try again
                    # BTW a 'onedrive' issue is when the local net files
                    # are saved on some cloud-like thing and if saved or
                    # accessed too quickly, will throw a 'permission denied'
                    # error that can be avoided if tried again with a short
                    # time delay.
                    continue
        
        
        # # Convert activation data from list to numerical array
        data = []
        for function in self.activationFunction:
            # Conversions
            if function == "NONE":
                data.append(0)
            elif function == "RELU":
                data.append(1)
            elif function == "ELU":
                data.append(2)
            elif function == "ATAN":
                data.append(3)
            else:
                data.append(0)

        # Save the activation function information
        for tries in range(10):
            try:
                np.savetxt(f"NN_{name}_Activations", np.array(data))
                break
            except:
                continue


## External Functions ##
def LoadNet(name):
    """
    Returns a nerual net object with the loaded characteristics from the
    given nickname. 
    
    Inputs:

    1 - Name
    - Type == String

    """
    # Load the various Characteristics and weights of the NN
    name = str(name)
    for tries in range(10): ## Redundant loading loop, reference SaveNN on 'onedrive issue'
        try:
            sizes = list(np.loadtxt(f"NN_{name}_Size"))
            break
        except:
            continue
    sizes = [int(i) for i in sizes]
    inSize = int(sizes[0])
    hiddenSize = sizes[1:len(sizes)-1]
    outSize = int(sizes[-1])

    # Load in the activation function data
    for tries in range(10): ## Redundant loading loop, reference SaveNN on 'onedrive issue'
        try:
            activations = list(np.loadtxt(f"NN_{name}_Activations"))
            break
        except:
            continue

    # Convert to str activation data
    strActivations = []
    for function in activations:
        # Conversions
        if function == 0:
            strActivations.append("NONE")
        elif function == 1:
            strActivations.append("RELU")
        elif function == 2:
            strActivations.append("ELU")
        elif function == 3:
            strActivations.append("ATAN")
        else:
            strActivations.append("NONE")

    # From the size, construct the net frame
    net = AdvNet(inSize, hiddenSize, outSize, strActivations)

    # Load in the saved net weights
    for i in range(len(net.sizes) - 1):
        for tries in range(10): ## Redundant loading loop, reference SaveNN on 'onedrive issue'
            try:
                weights = np.loadtxt(f"NN_{name}_Weights{i+1}")
                break
            except:
                continue
        weights = weights.reshape(net.sizes[i], net.sizes[i+1])
        net.weights[i] = weights

    # Return the generated neural net
    return net

def Forecast(Net, inputData, comparisonData=[], plotResults=True):
    """
    Test a net against a series of input values to get its current predictions which
    is then returned. Additionally, the predictions are plotted (by default) along
    with the comparison / validation data if provided.

    Inputs:

    1 - Net
    - Type == String or AdvNet
    - Looks for size and weights files with the given net name to load. If a 
    neural net is provided instead, this is simply the net used. If loading a 
    net, the size and weight files created from SaveNN(name) are gathered from
    their specific name made when generated. So pls don't change their names, thx.

    2 - inputData
    - Type == List of vectors or (Numpy) Array

    3 -  comparisonData
    - Type == List or single-column vector
    - Used to plot against the first outputs of the net's output vector. If a net only has a single
    output, this comparison data is of course just plotted against the net's only predictions.

    4 -  plotResults
    - Type == Bool
    - Decides if the resulting predictions are plotted for a visualization
    """

    # Load in the requested neural net
    if type(Net) not in [str, AdvNet]:
        raise TypeError(f"netName should be a string or AdvNet! Not {type(Net)} ")
    if type(Net) == str:
        net = LoadNet(Net)
    elif type(Net) == AdvNet:
        net = Net

    # Shape input data into matrix from the data vectors
    if type(inputData) == list:
        # Check that the given data have the same len()
        sizes = [np.size(i) for i in inputData]
        sizes.sort()
        if sizes != len(sizes)*[sizes[0]]:
            raise ValueError(f"Input Data does not all have the same size! Sizes: {sizes}")
        else:
            dataSize = sizes[0]

        # Check that there are the correct amount of inputs
        inSize = len(inputData)
        if inSize != net.inSize:
            raise ValueError(f"Given # of unique inputs {len(inputData)} does not match the net input size {net.inSize}!")
        
        # If sizes are correct, transform list into a matrix of vectors
        xData = np.zeros((sizes[0], len(inputData)))
        for i, data in enumerate(inputData):
            data = np.array(data).reshape(sizes[0])
            xData[:, i] = data
    # Check input data size if given as array
    elif type(inputData) == np.ndarray:
        # Check for the correct amount of inputs
        try:
            inSize = np.size(inputData, 1)
        except:
            inSize = 1
        dataSize = np.size(inputData, 0)
        if inSize != net.inSize:
            raise ValueError(f"Given # of unique inputs {inSize} does not match the net input size {net.inSize}!")
        else:
            xData = inputData
    else:
        raise TypeError("Unrecognized input data type!")

    # Calculate predictions for given inputs
    predictions = []
    if plotResults:
        forPlot = []
    for i in range(dataSize):
        # Generate net input vector
        if inSize > 1:
            invec = np.array(xData[i]).reshape(1,inSize)
        else:
            invec = np.array(xData[i]).reshape(1,1)

        # Get net prediciton
        val = net.Calculate(invec)
        predictions.append(val)

        # Add first type of net output to plot for later
        if plotResults:
            try:
                forPlot.append(val[0])
            except:
                forPlot.append(val)
    
    # Plot results if desired
    if plotResults:
        plt.cla() # Clear anything from ealier training plots if applicable

        # plt.plot([*range(len(forPlot))], forPlot)
        plt.plot(xData, forPlot)

        if len(forPlot) == np.size(comparisonData):

            # plt.plot([*range(np.size(comparisonData))], comparisonData, "--")
            plt.plot(xData, comparisonData, "--")

        plt.legend(["Forecasted Data", "Validation Data"])
        plt.title("Net Predictions")
        plt.show()

    # Return forecast values
    return predictions

def Extend(baseNet:AdvNet, d_height:int, imputeType:str = "zeros"):
    """
    Returns an AdvNet that has its hidden layers height increased by d_height,
    with the base weight values given from baseNet. 
    The input and output layer sizes are not changed.

    1. baseNet
        - AdvNet to use as a starting point for the parameters

    2. d_height
        - The integer change in height that should be added to the all hidden layers height

    3. imputeType
        - "median", "zeros" or "random"
        - Determines which value is used for filling the new weights created.
        - Default is zeros, as the others tend to not provide good results.
    """
    # Create an isolate version of net
    net = baseNet.CopyNet()

    # Loop
    for i, arr in enumerate(net.weights):
        # Get current weight array dimensions
        Y, X = np.size(arr, axis=0), np.size(arr, axis=1)

        # Determine new size
        if i == 0:
            newSize = (Y, X+d_height)
        elif i == len(net.weights)-1:
            newSize = (Y+d_height, X)
        else:
            newSize = (Y+d_height, X+d_height)

        # Get impute type
        if imputeType == "median":
            ogMedian = np.median(arr)
        elif imputeType == "zeros":
            ogMedian = 0
        elif imputeType == "random":
            ogMedian = np.random.rand(newSize[0], newSize[1])

        # Rules for extending array sizes=
        newArr = np.zeros(newSize) + ogMedian
        
        # Put in existing data into new extended array
        newArr[:Y, :X] = arr

        # Put new array size into new
        net.weights[i] = newArr

    # Update various size characteristics of the new net
    net.hiddenSize = [i+d_height for i in net.hiddenSize]
    net.sizes[1:-1] = net.hiddenSize
    net.parameters = sum([M.size for M in net.weights])

    # Finish
    return net

def netMetrics(net:AdvNet, xArr:np.ndarray, yArr:np.ndarray, returnAll:bool = False):
    """
    Returns the nets R^2 value for the given x input data and y validation data.

    returnAll:
        - If True, all errors are returned in the order of (R^2, ResidualError, RegressionError, TotalError).
          Otherwise, just the R^2 value is returned.
    """

    # Get predictions
    yHatList = Forecast(net, xArr, plotResults=False)

    ## Errors ##
    # Residual
    try:
        sizeY = np.size(yArr, axis=1)
    except:
        sizeY = 1
    sizeX = np.size(yArr, axis=0)
    SSres = np.sum((np.array(yHatList).reshape(sizeY, sizeX) - yArr) ** 2)

    # Regression
    yMean = np.mean(yArr)
    SSreg = np.sum((np.array(yHatList) - yMean) ** 2)

    # Total
    SStot = SSres + SSreg

    # R2-D2 Value
    if not returnAll:
        return SSreg / SStot
    else:
        return SSreg / SStot, SSres, SSreg, SStot

def genTrain(net:AdvNet, xArr:np.ndarray, yArr:np.ndarray, iterations:int = 1000, 
                    batchSize:int = 0, gamma:int = 50, weightSelection:str = None, 
                    R2Goal = 0.999, Silent:bool = False):
    """
    Returns a net and its R^2 value relative to the given X and Y data.
    
    Instead of always using the first net that has any improvement from the previous best net,
    a 'batch' of nets are generated from the parent net, then tested to find the best of those.

    Net:
        - The deep neural net to be trained.

    xArr:
        - Input data for the net.

    yArr:
        - Validation data used to train the net to.

    iterations:
        - Amount of attempts at improving the net's fit to the data.

    batchSize:
        - Amount of nets the be constructed and tested for improvements per iteration.
        - The default value (0) instead calculates a batch size based on the current iteration.
          The minimum batch size used in this case is 10.

    gamma:
        - Sets the learning factor exponential decay rate. That is, every 'gamma' number 
          of iterations, the learning rate will have decreased by 1/2.

    weightSelection: (all, gates, middle, None)
        - Which weight array(s) that are to be modified to train the net.
        - The default method (called by None), alternates between training
          the gates and middle arrays every iteration.

    R2Goal:
        - The target R2 value to train the net to. Training automatically stops once this goal is reached.

    Silent:
        - Turns off printing the current progress and best R^2 value during training.
    """

    # Verify data types
    if type(net) == AdvNet and type(xArr) == np.ndarray and type(yArr) == np.ndarray:
        # print("All data is good")
        pass
    else:
        print("A data type is not correct")

    # Get inital accuracy
    currentR2 = netMetrics(net, xArr, yArr)

    # Generational Training method
    for I in range(iterations):
        # Get current tweak amplitude
        twk = 2 ** (-I / gamma)        ## ~30-60 seems to be a good scale for this (found via Train())

        # exp. weight selection
        if weightSelection == None:
            if I%2 == 0:
                weightSelection = 'gates'
            else:
                weightSelection = 'middle'

        #### This doesn't seem worth the usage
        # Smart Batch Size handling if at 0 default
        if batchSize <= 0:
            depth = max(10, round(20 * 2**-(I / 1000)))
        else:
            depth = batchSize

        # Get new mutated test nets
        testNets = []
        for n in range(depth):
            newNet = net.CopyNet()
            newNet.TweakWeights(twk, Selection=weightSelection)
            testNets.append(newNet)

        # Get the offspring's scores
        newNetScores = []
        for mutNet in testNets:
            newR2 = netMetrics(mutNet, xArr, yArr)
            newNetScores.append(newR2)

        # See if the best score is an improvement
        newBestR2 = max(newNetScores)
        if newBestR2 > currentR2:
            # Actions for improvement
            bestIndex = newNetScores.index(newBestR2)
            net = testNets[bestIndex].CopyNet()
            currentR2 = newBestR2

        # Stop iterations if net achieved accuracy goal
        if currentR2 >= R2Goal and not Silent:
            print(f"R2: {currentR2:.6f} | Training: {'=='*20}", end='\r')
            print()
            return net, currentR2

        # Update fancy progress bar stuff
        if not Silent:
            pix1 = '='
            pix2 = '-'
            donePercent = I / (iterations - 1)
            barLength = 40
            pix1Len = round(barLength * donePercent)
            pix2Len = barLength - pix1Len
            print(f"R2: {currentR2:.6f} | Training: {pix1*pix1Len}{pix2*pix2Len}", end='\r')
    
    # Finish with returning best net and its R2 value
    if not Silent:
        print(f"R2: {currentR2:.6f} | Training: {'=='*20}", end='\r')
        print()
    return net, currentR2

def thinData(xData, yData, numPoints:int):
    """
    Returns three arrays for AdvNet curve fitting / plotting from the arrays of given data (this is 
    primarily meant for curve fitting using ATAN AdvNets, as fitting to all the data tends to be 
    more than what is neccessary).
    
    Array 1 is the thinned x-data for the net input data. This array will have a number of data points
    equal to numPoints that are seperated evenly.

    Array 2 is the thinned y-data for the net validation data. This array has the same number of data
    points as Array 1.

    Array 3 is the indicies sliced at -- this is useful for plotting the thinned y-data.

    Notes:
        -The start and ending points of the given data are forced to be included, so the gap between the last 
        and second to last data point in this array might not match the rest of the data points' spacing.
    """

    # Get data size
    try:
        # For multiple output columns
        dataLen = yData.size / np.size(yData, axis=1)
    except:
        # For single output columns
        dataLen = yData.size

    # Check that number of points requested is lower than amount of data points
    if numPoints > dataLen:
        print(f"numPoints > Length of given data! Reducing to 10% of the points instead.")
        numPoints = dataLen * 0.10

    # Calculate step size
    step = round(dataLen / (numPoints-1))

    # Get the thinned y data and their indicies (for graphing)
    yThinData = []
    xThinData = []
    xPlotData = []
    for i in range(0, dataLen, step):
        yThinData.append(yData[i])
        xThinData.append(xData[i])
        xPlotData.append(i)

    # Check that the last data point is included
    if yThinData[-1] != yData[-1]:
        yThinData.append(yData[-1])
        xThinData.append(xData[-1])
        xPlotData.append(yData.size)

    # # Thinned x data
    # ## This is what to use for small nets when curve fitting, as fitting to the true indicies is much harder
    # xShort = [*range(len(yShort))]

    # Finish
    return np.array(xThinData), np.array(yThinData), np.array(xPlotData)
   

## Old / Depreceated ##
def Train(Net, inputData, yData, startingTweakAmp=0.8, 
          plotLive=False, plotResults=False, normalizeData=False, 
          hiddenFunc="ELU", trainWeights='all', maxIterations=1000, 
          blockSize=30, Silent=False):
    """
    **** Depreceated since version 0.2.0 ****

    Train a specified neural net to the given validation data, using the 
    input data as the input vector to the net. Returns the trained net
    and the final average error after completion. Note the error is 
    found from the sum-square error divided amoung the number of data points.

    Inputs:

    1 - Net
    - Type == String or AdvNet
    - Looks for size and weights files with the given net name to load. If a 
    neural net is provided instead, this is simply the net used. If loading a 
    net, the size and weight files created from SaveNN(name) are gathered from
    their specific name made when generated. So pls don't change their names, thx.

    2 - inputData
    - Type == Numpy Array or List of (Numpy) Vectors
    - Size == # of unique data lists = Net input size
    - If an array is given, the columns should be the data points of that inputs

    3 - yData
    - Type == Numpy Array or List

    4 - startingTweakAmp
    - Type == Float
    - The starting amplitude used for tweaking the weight arrays, explained more in
    the tweakWeights() information.

    5 - plotLive
    - Type == Bool
    - Decides if pyplot is used to give a live visualization of the training.
    Currently just the first set of data/validation data are plotted against eachother

    6 - plotReults
    - Type == Bool
    - For the sake of speed, if a visualization is prefered but only needed after training
    this is the option to use. A plot of the net's predictions are plotted only after the
    trianing has finished.

    7 - normalizeData
    - Type == Bool
    - Decides of data is to be normalized according to its mean and standard
    deviation. This is particularly useful when there are multiple unique input
    data sets, with very different values.
    - If True, the mean and std. dev. of the validation data is used as the values
    to de-normalize the net's outputs

    8 - hiddenFunc
    - Type == String
    - Decides what hidden-layer processing should be used. 'NONE' simply passes
    through each summed up vector to the next layer with no processing. The default,
    'ELU', uses the function max(val, 0) before passing on to the next hidden layer.

    9 - trainWeights
    - Type == List (or 'gates' or 'middle')
    - Used to select specific weight set(s) to train instead of training all weights.
    Useful for making sure the first and last sets of weights are trained correctly to
    avoid 'gate keeping'.
    - The string 'gates' can be used to select the first and last weight arrays to train
    while 'middle' can be used to select all weight arrays (if any) in between the 'gates'.

    10 - maxIterations
    - Type == Int
    - Used to set an upper limit of training iterations. If no useful training is completed
    within a few hundred iterations (depending on tweak amplitude, 0.4 means 300 iterations),
    the training will end, assuming the maxIterations is greater than this limit.

    11 - blockSize
    - Type == Int
    - Used for determining how quickly the tweak amplitude is decreased from its starting value.
    Every blockSize # of iterations, if no improvements are made, the tweak amplitude is 
    divided by a factor of two. This progresses training and is what makes the automatic
    detection of how fine the net should be trained, possible. Training is ended once max 
    iterations are reached or the tweak amplitude gets to be very small (< 5e-4).

    12 - Silent
    - Type = Bool
    - Decides if the current error and iteration shouldn't be printed to the console.
    """

    # Load in the requested neural net
    if type(Net) not in [str, AdvNet]:
        raise TypeError(f"netName should be a string or AdvNet! Not {type(Net)} ")
    if type(Net) == str:
        net = LoadNet(Net)
    elif type(Net) == AdvNet:
        net = Net

    # Shape input data into matrix from the data vectors
    if type(inputData) == list:
        # Check that the given data have the same len()
        sizes = [np.size(i) for i in inputData]
        sizes.sort()
        if sizes != len(sizes)*[sizes[0]]:
            raise ValueError(f"Input Data does not all have the same size! Sizes: {sizes}")
        else:
            dataSize = sizes[0]

        # Check that there are the correct amount of inputs
        inSize = len(inputData)
        if inSize != net.inSize:
            raise ValueError(f"Given # of unique inputs {len(inputData)} does not match the net input size {net.inSize}!")
        
        # If sizes are correct, transform list into a matrix of vectors
        xData = np.zeros((sizes[0], len(inputData)))
        for i, data in enumerate(inputData):
            data = np.array(data).reshape(sizes[0])
            xData[:, i] = data
    # Check input data size if given as array
    elif type(inputData) == np.ndarray:
        # Check for the correct amount of inputs
        try:
            inSize = np.size(inputData, 1)
        except:
            inSize = 1
        dataSize = np.size(inputData, 0)
        if inSize != net.inSize:
            raise ValueError(f"Given # of unique inputs {inSize} does not match the net input size {net.inSize}!")
        else:
            xData = inputData
    else:
        raise TypeError("Unrecognized input data type!")

    # Shape validation data into matrix from the data vectors
    if type(yData) == list:
        # Check that the given data have the same len()
        ysizes = [np.size(i) for i in yData].sort()
        if ysizes != len(ysizes)*[ysizes[0]]:
            raise ValueError(f"Validation Data does not all have the same size! Sizes: {ysizes}")
        else:
            ydataSize = ysizes[0]

        # Check that there are the correct amount of inputs
        if len(yData) != net.inSize:
            raise ValueError(f"Given # of unique validations {len(yData)} does not match the net output size {net.inSize}!")
        
        # If sizes are correct, transform list into a matrix of vectors
        yData = np.zeros((ysizes[0], len(yData)))
        for i, data in enumerate(yData):
            yData[:, i] = data
    # Check validation data size if given as array
    elif type(yData) == np.ndarray:
        # Check for the correct amount of inputs
        try:
            yinSize = np.size(yData, 1)
        except:
            yinSize = 1
        ydataSize = np.size(yData, 0)
        if yinSize != net.outSize:
            raise ValueError(f"Given # of unique validations {yinSize} does not match the net output size {net.outSize}!")
        else:
            yData = yData
    else:
        raise TypeError("Unrecognized validation data type!")
    
    # Check that the len of the validation data matches the input len
    if dataSize != ydataSize:
        raise ValueError(f"Length of input data {dataSize} does not match the validation data length {ydataSize}!")
    
    # Z-score normalization
    if normalizeData:
        # Normalize input data
        for x in range(np.size(xData, 1)):
            mean = np.mean(xData[:, x])
            stdv = np.std(xData[:, x])
            xData[:, x] = (xData[:, x] - mean) / stdv

        # Get validation de-normalization data
        mu = []; st = []
        for i in range(yinSize):
            mu.append(np.mean(yData[:, i]))
            st.append(np.std(yData[:, i]))
        
    # Training loop
    improvments = []
    tweakAmp = startingTweakAmp
    try:    ## Plot validation values
        validationVals = yData[:, 0]
    except:
        validationVals = yData
    # print("\n-- Starting Training Loop --")
    for iterations in range(maxIterations):
        # Tweak net
        testNet = net.CopyNet()
        if iterations != 0:
            testNet.TweakWeights(tweakAmp, Selection=trainWeights)

        # Get current predictions
        predictions = []
        SSE = 0
        for i in range(dataSize):
            # Make inVector
            if inSize == 1:
                inVec = np.array([xData[i]]).reshape(1,1)
            else:
                inVec = np.array(xData[i, :]).reshape(1, inSize)

            # Get prediction
            val = testNet.Calculate(inVec, hiddenFunction=hiddenFunc)

            # Reverse normalization from the mu and st of the validation data
            if normalizeData and type(val) not in [int, float, np.float64]:
                for i in range(len(val)):
                    val[i] = (val[i]*st[i]) + mu[i]
                predictions.append(val[0])
            elif normalizeData:
                predictions.append((val*st[0]) / mu[0])
            elif not normalizeData:
                predictions.append(val)

            # Sum up current error
            if inSize == 1:
                correct = yData[i]
            else:
                correct = yData[i, :]
            SSE += np.sum((correct-val)**2)

        # Handel Starting case
        if iterations == 0:
            bestSSE = SSE
            plot = predictions

        # Reactions if there was an improvement
        avgError = (bestSSE/dataSize)**0.5
        if SSE < bestSSE:
            bestSSE = SSE
            net = testNet.CopyNet()
            plot = predictions
            improvments.append(1)
            # net.SaveNN(netName)
            # print(f"Net saved at iteration: {iterations}")
            # sleep(0.5)
        else:
            improvments.append(0)

        # Improvement history handling
        if len(improvments) > round(blockSize/1.2):
            improvments.pop(0)
        if sum(improvments) == 0 and iterations%blockSize == 0 and iterations > blockSize:
            tweakAmp = tweakAmp/2

        # if tweakAmp < 2e-4:
        #     break
        # Removed as when I lower this, the learning keeps getting 'deeper'
        # Perhaps no surprise there
        # Besides, maxIterations handles breaking the loop

        if plotLive:
            # Plot current best predictions vs the validation data
            plt.cla()
            plt.plot([*range(len(plot))], plot)
            plt.plot([*range(len(validationVals))], validationVals, 'g--')
            plt.title(f"Avg. Error: {avgError:.2e}, Iteration: {iterations}")
            plt.pause(0.0001)
        # else:
        if not Silent:
            print(f"Avg. Error: {avgError:.2e}, Iteration: {iterations}", end='\r')

    # Save resulting net after training
    # sleep(0.5)
    # net.SaveNN(netName)
    if not Silent:
        print(f"Final Avg. Error: {avgError:.2e}, Iteration: {iterations}")
    # print("-- Training Loop Finished --")

    if plotResults:
        # Final plot
        plt.cla()
        plt.plot([*range(len(plot))], plot)
        plt.plot([*range(len(validationVals))], validationVals)
        plt.title(f"<Done> -- Final Avg. Error: {avgError:.2e}")
        plt.show()
    
    return net, avgError

def CycleTrain(Net, inputData, yData, startingTweakAmp=0.8, 
          plotLive=False, plotResults=False, normalizeData=False, 
          hiddenFnc="ELU", maxIterations=1000, maxCycles=5,
          blockSize=30, Silent=False):
    """
    **** Depreceated since version 0.2.0 ****

    Train a specified neural net to the given validation data, using the 
    input data as the input vector to the net. Returns the trained net
    and the final average error after completion. Note the error is 
    found from the sum-square error divided amoung the number of data points.

    Inputs:

    1 - Net
    - Type == String or AdvNet
    - Looks for size and weights files with the given net name to load. If a 
    neural net is provided instead, this is simply the net used. If loading a 
    net, the size and weight files created from SaveNN(name) are gathered from
    their specific name made when generated. So pls don't change their names, thx.

    2 - inputData
    - Type == Numpy Array or List of (Numpy) Vectors
    - Size == # of unique data lists = Net input size
    - If an array is given, the columns should be the data points of that inputs

    3 - yData
    - Type == Numpy Array or List

    4 - startingTweakAmp
    - Type == Float
    - The starting amplitude used for tweaking the weight arrays, explained more in
    the tweakWeights() information.

    5 - plotLive
    - Type == Bool
    - Decides if pyplot is used to give a live visualization of the training.
    Currently just the first set of data/validation data are plotted against eachother

    6 - plotReults
    - Type == Bool
    - For the sake of speed, if a visualization is prefered but only needed after training
    this is the option to use. A plot of the net's predictions are plotted only after the
    trianing has finished.

    7 - normalizeData
    - Type == Bool
    - Decides of data is to be normalized according to its mean and standard
    deviation. This is particularly useful when there are multiple unique input
    data sets, with very different values.
    - If True, the mean and std. dev. of the validation data is used as the values
    to de-normalize the net's outputs

    8 - hiddenFnc
    - Type == String
    - Decides what hidden-layer processing should be used. 'NONE' simply passes
    through each summed up vector to the next layer with no processing. The default,
    'ELU', uses the function max(val, 0) before passing on to the next hidden layer.

    9 - trainWeights
    - Type == List (or 'gates' or 'middle')
    - Used to select specific weight set(s) to train instead of training all weights.
    Useful for making sure the first and last sets of weights are trained correctly to
    avoid 'gate keeping'.
    - The string 'gates' can be used to select the first and last weight arrays to train
    while 'middle' can be used to select all weight arrays (if any) in between the 'gates'.

    10 - maxIterations
    - Type == Int
    - Used to set an upper limit of training iterations. If no useful training is completed
    within a few hundred iterations (depending on tweak amplitude, 0.4 means 300 iterations),
    the training will end, assuming the maxIterations is greater than this limit.

    11 - maxCycles
    - Type == Int
    - Sets an upper limit to how many training cycles can be done.

    12 - blockSize
    - Type == Int
    - Used for determining how quickly the tweak amplitude is decreased from its starting value.
    Every blockSize # of iterations, if no improvements are made, the tweak amplitude is 
    divided by a factor of two. This progresses training and is what makes the automatic
    detection of how fine the net should be trained, possible. Training is ended once max 
    iterations are reached or the tweak amplitude gets to be very small (< 5e-4).

    13 - Silent
    - Type = Bool
    - Decides if the current error and iteration shouldn't be printed to the console.
    """

    # Get initial Error
    Net, bestError = Train(Net, inputData, yData, 
                        startingTweakAmp=startingTweakAmp, normalizeData=normalizeData, 
                        hiddenFunc=hiddenFnc, maxIterations=1, 
                        blockSize=blockSize, Silent=True)
    
    # Start cycle training (limit to 1000 just in case)
    for cycle in range(maxCycles):
        if Silent == False:
            print(f"Starting Cycle #{cycle+1}")
        
        Net, error = Train(Net, inputData, yData, 
                            startingTweakAmp=startingTweakAmp, normalizeData=normalizeData, 
                            hiddenFunc=hiddenFnc, maxIterations=maxIterations, 
                            blockSize=blockSize, Silent=Silent, trainWeights=[0],
                            plotLive=plotLive, plotResults=False)
        Net, error1 = Train(Net, inputData, yData, 
                            startingTweakAmp=startingTweakAmp, normalizeData=normalizeData, 
                            hiddenFunc=hiddenFnc, maxIterations=maxIterations, 
                            blockSize=blockSize, Silent=Silent, trainWeights='middle',
                            plotLive=plotLive, plotResults=False)
        Net, error2 = Train(Net, inputData, yData, 
                            startingTweakAmp=startingTweakAmp, normalizeData=normalizeData, 
                            hiddenFunc=hiddenFnc, maxIterations=maxIterations, 
                            blockSize=blockSize, Silent=Silent, trainWeights=[-1],
                            plotLive=plotLive, plotResults=False)
        
        # Check for enough improvement
        if (error2)/bestError > 0.997:
            if Silent == False:
                print("Improvement of less than 0.3% -- Training concluded")
            break
        
        # Set next best error
        bestError = error2

    # If max cycles reached
    if cycle + 1 == maxCycles:
        if Silent == False:
            print(f"Maximum cycles of {maxCycles} completed")
    
    # Plot results if desired
    if plotResults:
        plt.cla() # Clear anything from ealier training plots if applicable
        Forecast(Net, inputData, yData, plotResults=plotResults)

    # Finish by giving back the improved net and its final mean error
    return Net, bestError
