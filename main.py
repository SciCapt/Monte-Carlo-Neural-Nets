from matplotlib import pyplot as plt
import numpy as np
import textwrap as txt
import os
from time import perf_counter as pc
import random as rn

class AdvNet:
    ## Default Functions ## 
    def __init__(self, inputCount: int, hiddenCounts: list, outputCount: int, activations: any = 'LIN'):
        """
        Create a deep neural network from the given counts of neurons requested per layer. By default,
        uses a linear activation in all layers.

        Inputs:

        1 - inputCount
        - Number of input features
        - Type == int
        - Range == (0, inf]

        2 - hiddenCounts
        - Heights of the hidden layers desired
        - Type == List of int
        - Range == (0, inf]

        3 - outputCount
        - Number of output features
        - Type == int
        - Range == (0, inf]

        4 - activations
        - Possible Values: "lin", "elu", "atan", "relu", "rnd", "sqr", "root" "sig", "resu", "resu2"
            - LIN:   No applied function (linear)
            - ELU:   Exponential up to x=0, then uses y=x
            - ATAN:  Uses arctan(x)
            - RELU:  max(x, 0)
            - RND:   A linear layer that rounds outputs to the nearest integer
            - SQR:   Forces all negative numbers to be -1, and all positive to be 1
            - ROOT:  Essentially a 'leaky' atan function (no horizontal asymptotes)
            - SIG:   Sigmoid function
            - RESU:  Experimental, applies |sin(x)|*x
            - RESU2: Experimental, applies max(|sin(x)|*x, 0)
        - A list of different functions per layer can be given, otherwise, all calculation
          steps will use the single given activation function.
        - Examples:
            - AdvNet(1, [10, 20, 10], 2, "RELU") 
                - 5-layer net / 4 in-between spaces
                - Uses the "RELU" activation function for all (4) in-between spaces.
            - AdvNet(2, [15, 15], 5, ["RELU", "elu", "Atan"]) 
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

        For each step of forward propogation (n1 total steps / spaces) an activation function
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

        # # Setup Biases (Unused currently)
        # self.biases = [2*np.random.rand(self.sizes[b+1], 1) - 1 for b in range(len(self.weights)-1)]

        # Calculate the number of Parameters in the Weights/net
        weightSizes = [M.size for M in self.weights]
        self.parameters = sum(weightSizes)

        # Construct activation functions list (if only given a single function type)
        if type(activations) == str:
            self.activationFunction = [activations]*len(self.sizes)

        # Activation list if given a list of functions
        elif type(activations) == list:
            # Check that the correct amount of activation functions were provided
            LA = len(activations)
            LW = len(self.sizes)
            if LA == LW:
                self.activationFunction = activations
            elif LA < LW:
                raise ValueError(f"Not enough activation functions provided! ({LA} given, {LW} required)")
            elif LA > LW:
                raise ValueError(f"Too many activation functions provided! ({LA} given, {LW} required)")
            
        # Force upper case for activation functions
        # (allows giving them as lower case/whatever when constructed)
        try:
            self.activationFunction = [i.upper() for i in self.activationFunction]
        except:
            raise ValueError("Confirm all activation functions given in the list are strings!")
            
        # Test Calculation Time (for __str__)
        I = 2 # Do just 2 iterations
        t1 = pc()
        for i in range(I):
            self.Calculate(np.ones((self.inSize, 1)))
        t2 = pc()
        self.speed = format((t2-t1)/I, ".2e")

    def __str__(self):
        # Resulting string dimensions/title
        strlen = 64
        l1 = f"="
        l2 = f"Neural Net Characteristics:"

        # Show the layer heights of the net
        l3 = f"1. Layer Sizes = {self.sizes}"
        l3 = txt.fill(l3, strlen-2)

        # Get and show the medians weight values within the weight arrays
        weightMedians = []
        for weights in self.weights:
            weightMedians.append(round(np.median(weights), 2))
        l4 = f"2. Weight Medians = {weightMedians}"
        l4 = txt.fill(l4, strlen-2)

        # Show the number of paramaters that make up the net
        l6 = f"3. Number of Parameters: {self.parameters}"

        # Show activation functions and order
        l7a = f"4. Activation Functions: {self.activationFunction}"
        l7a = txt.fill(l7a, strlen-2)
        
        # Show the calculation time
        l8 = f"5. Calculation Time: ~{self.speed}"
        l8 = txt.fill(l8, strlen-2)

        # Resulting String
        full = (l1*strlen + '\n' + l2.center(strlen, ' ') + '\n' + l3 + 
                '\n' + l4 + '\n' + l6 + '\n' + 
                l7a + '\n' + l8 + '\n' + l1*strlen)
        return full

    ## Internal Functions ##
    def TweakWeights(self, Amplitude: float, Selection='all', returnChange:bool = False):
        """
        Adjust the weights of a neural net by a given amplitude. The net change in the 
        weight arrays can be optionally returned for gradient decent calculation later.
        
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

        The net weights are clipped to the range [-1,1], therefore, an amplitude magnitude 
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
            Selection = [*range(1, len(self.weights)-1)]
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

        # List of weight array changes if requested
        if returnChange:
            Wo = [wi.copy() for wi in self.weights]

        # Adjust all the weights randomly, with a maximum change magnitude of Amplitude
        for i in Selection:
            # Negative Indicies
            if i < 0:
                # raise ValueError("Negative Indicies to select weights is not supported!")
                i += len(self.weights)

            # Add tweaks to existing weights
            self.weights[i] += (np.random.rand(self.sizes[i], self.sizes[i+1]) - 0.5)*(2*Amplitude)

            # Make sure values are in the range [-1, 1]
            self.weights[i] = np.clip(self.weights[i], -1, 1)

        # Note change in weight array after clipping
        if returnChange:
            dW = [(self.weights[i] - Wo[i]) for i in range(len(self.weights))]
            return dW

        # # Adjust the bias values
        # for i in range(len(self.biases)):
        #     self.biases[i] = np.clip(self.biases[i] + 2*Amplitude*(np.random.rand(self.biases[i].size, 1) - 0.5), -5, 5)

    def Calculate(self, inVector, useFast:bool = False):
        """
        Returns the neural net's calculation for a given input vector. If the net used has an input size
        of one, a single float value can be given (array type not required).

        Inputs:

        1 - inVector
        - Type == NumPy Array
        - Size == (netInputCount, 1) or (1, netInputCount)
        - Note: 
            - If the net input size is 1, then a float can be given and a vector form is not required.

        2 - useFast
        - Type == Bool
        - Usage:
            - Determines whether or not fastCalc is used over Calculate. This can bring the
              calculation speed down to around 40-70% of the otherwise default calculation
              time, but minimal checks of the given data are done.
            - If you are sure the given inVector data is correct, then you can use this
              fast-calc setting, otherwise use caution as you will not receive useful 
              information about what might go wrong during the calculation.
        """  
        # Apply fast calculation mode if requested
        if useFast:
            return fastCalc(self, inVector)

        # Handling if a single number/float was given not in an array form
        if type(inVector) != np.ndarray:
            if self.inSize == 1:
                inVector = np.array(inVector).reshape(1, 1)
            else:
                raise ValueError(f"Net input size of 1 expected, got {self.inSize} instead")
            
        # Check for correct shape (x, 1) or (1, x)
        m1 = np.size(inVector, axis=0)
        try:
            m2 = np.size(inVector, axis=1)
        except:
            m2 = 0
        if m1*m2 == m1 or m1*m2 == m2:
            # One of the dimensions is one, so the input is indeed a vector
            pass
        else:
            # The input is not a vector but instead an array/matrix
            raise ValueError(f"Expected inVector of size {(self.inSize, 1)} or {(1, self.inSize)}, got {(np.size(inVector, axis=0), np.size(inVector, axis=1))})")

        # Handling for row vectors (convert to column vectors)
        passedRowTest = False
        if inVector.size == np.size(inVector, axis=0): # Check if already correct form
            passedRowTest = True
        elif inVector.size == np.size(inVector, axis=1): # If all entries are oriented along the horizontal axis
            # Convert row vectors to column vectors
            if self.inSize != 1:
                if inVector.size == np.size(inVector, 1):
                    inVector = inVector.T
                    passedRowTest = True
            else:
                # Just a single value so continue
                passedRowTest = True
        
        # Calculate if the input vector is indeed a column vector
        if inVector.size == np.size(inVector, axis=0) and passedRowTest:
            # Vector has the right column vector shape -- now check for correct size
            if inVector.size == self.inSize:
                # Vector is of the right shape and size so continue

                # Go through all of weights calculating the next vector
                calcVec = inVector # Start the calcVector with the given input vector
                
                # Triple-checking size throughout forward prop.
                calcVec = calcVec.reshape(calcVec.size, 1)

                # Apply first activation layer on input vector
                calcVec = applyActi(calcVec, self.activationFunction[0])

                for i, weights in enumerate(self.weights):
                    # Forward propogation
                    calcVec = sum(calcVec*weights)

                    # Use the net's activation function for the current space
                    hiddenFunction = self.activationFunction[i+1]

                    # Apply the activation function
                    calcVec = applyActi(calcVec, hiddenFunction)

                    # Triple-checking size throughout forward prop.
                    calcVec = calcVec.reshape(calcVec.size, 1)

                # Handling for single-number outputs (free it from the array at the end)
                if self.outSize == 1:
                    calcVec = calcVec[0][0]

                return calcVec
            
            # Vector is not of the correct shape for the used neural net
            else:
                raise ValueError(f"inVector size ({inVector.size}) does not match the net input size ({self.inSize})")
        else:
            raise RuntimeError("Failed to enter Calculation loop! (This shouldn't have happened)")

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

        # Check that the directory exists
        DIR = "MCNetData"
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        # Save the various Characteristics and weights of the NN to load later
        # name = str(name)
        np.savetxt(f"{DIR}/NN_{name}_Size", self.sizes)
        for i in range(len(self.sizes) - 1):
            # Redundant save cycles in case of 'onedrive' situations
            ## Limit save tries to 10 times, if thats not working
            ## something else is likely wrong that this won't fix
            for tries in range(10):
                try:
                    np.savetxt(f"{DIR}/NN_{name}_Weights{i+1}", self.weights[i])
                    # if made this far, it successfully saved so break the loop
                    break
                except:
                    # Something didn't work, try again
                    # BTW a 'onedrive issue' is when the local net files
                    # are saved on some cloud-like thing and if saved or
                    # accessed too quickly, will throw a 'permission denied'
                    # error that can be avoided if tried again with a short
                    # time delay.
                    continue
        
        # Convert activation data from list to numerical array
        data = []
        for function in self.activationFunction:
            # Conversions
            if function == "LIN":
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
                np.savetxt(f"{DIR}/NN_{name}_Activations", np.array(data))
                break
            except:
                continue

    def ApplyTweak(self, dW):
        for i, dWi in enumerate(dW):
            self.weights[i] = self.weights[i] + dWi

    def ExpTrain(self, XData, YData, Ieta:int = 5, Gamma:float = 1, Beta:int = 20,
              subIeta:int = 3, subGamma:int = 1, subBeta:int = 30,
              verbose:bool = True, useFast:bool = True):
        """
        **** Experimental Training Method ****

        Returns a trained version of the net used. Takes a slice of data to
        use for fitting a test net using genTrain then tests/confirms its
        predictive power using the remaining data (from XData & YData).

        1 - Ieta
            - Number of iterations to use for training the net
        
        2 - Gamma
            - Weight tweak amplitude decay factor; every Gamma # of
              iterations, the tweak ampltiude will be halved and the
              training "precision" will be doubled (but will look over
              a smaller possible range of values for the net's weights!)

        3 - Beta
            - Number of test nets to make for each batch, every iteration

        4 - subIeta
            - Number of iterations used per test net to train said net

        5 - subGamma
            - Same factor as Gamma, but instead used for the quick training
              of a test net (along with subIeta)

        6 - subBeta
            - Batch size used when doing the quick training of a test net
              (along with subIeta and subGamma)

        7 - Verbose
            - Decides if the current iteration number and best R2 score
              are printed throughout the training process

        8 - useFast
            - Decides if the 'fast' method is used for .Calculate() -- 
              The only major effect of this is the lack of error reporting
              when the net is making a calculation. If you suspect the 
              data arrays aren't correctly oriented then turn this to false
              to assist in error checking            
        """
        
        # Get initial score
        bestAverageScore = netMetrics(self, XData, YData, useFast=useFast)

        # Make train/test split
        xTrain, yTrain, xTest, yTest = TTSplit(XData, YData, percentTrain=70)
        bestNet = self
        for Itr in range(Ieta):
            # Get current tweak amplitude
            Tau = 2 ** (-Itr / Gamma)

            # Make test batch
            testNets = []
            for B in range(Beta):
                net = self.CopyNet()
                net.TweakWeights(Tau)
                testNets.append(net)

            # Test Test Batch
            for batchNet in testNets:
                # Do small fit to train data
                batchNet, R2_f = genTrain(batchNet, xTrain, yTrain, iterations=subIeta, 
                                          batchSize=subBeta, gamma=subGamma, Silent=True, 
                                          useFast=useFast)

                # Test net on test data
                R2_p = netMetrics(batchNet, xTest, yTest, useFast=useFast)
                
                # Get average score
                avgScore = (R2_f + R2_p) / 2

                # Check for improvement
                if avgScore > bestAverageScore:
                    bestNet = batchNet.CopyNet()
                    bestAverageScore = avgScore
                    if verbose:
                        print(f"Iteration #{Itr+1} | Best Score = {bestAverageScore}          ", end='\r')

            # Update status
            if verbose:
                print(f"Iteration #{Itr+1} | Best Score = {bestAverageScore}          ", end='\r')

        # Final printout
        if verbose:
            print(f"Iteration #{Itr+1} | Best Score = {bestAverageScore}          ")

        # Return the trained net
        return bestNet

    def Fit(self, xArr:np.ndarray, yArr:np.ndarray, Ieta:int = 9, 
                  Beta:int = 50, Gamma:int = 3, ScoreFunction = None,
                  Verbose:bool = True, useFast:bool = True):
        """
        Returns a trained version of the net object to the relative to the given X & Y data
        
        Instead of always using the first net that has any improvement from the previous best net,
        a 'batch' of nets are generated from the parent net, then tested to find the best of those.

        Net:
            - The deep neural net to be trained.

        xArr:
            - Input data for the net.

        yArr:
            - Validation data used to train the net to.

        Ieta:
            - Amount of iteration attempts at improving the net's fit to the data.

        Beta:
            - Amount of nets the be constructed and tested for improvements per iteration.
            - The default value (0) instead calculates a batch size based on the current iteration.
            The minimum batch size used in this case is 10.

        Gamma:
            - Sets the learning factor exponential decay rate. That is, every 'gamma' number 
            of iterations, the learning rate will have decreased by 1/2. 

        ScoreFunction:
            - The fuction used to "score" nets alonge training, the default method (None) uses
              an R^2 calculation.
            - Other functions are accepted given:
                - A net that does better should give a higher "score"
                - Function input should be (net, Xarray, Yarray)

        Verbose:
            - Turns off printing the current progress and best R^2 value during training.

        useFast:
            - Decides if the faster net calculation method is used; note that this has worse
              error handling but greatly increases net calculation speed for nets with few
              hidden layers.
        """
        # Scoop net
        net = self

        # Verify data types
        if type(net) == AdvNet and type(xArr) == np.ndarray and type(yArr) == np.ndarray:
            # print("All data is good")
            pass
        else:
            print("An input data type is not correct")

        # Get inital accuracy
        if ScoreFunction == None:
            currentScore = netMetrics(net, xArr, yArr, useFast=useFast)
        else:
            currentScore = ScoreFunction(net, xArr, yArr)

        # Generational Training method
        for I in range(Ieta):
            # Get current tweak amplitude
            twk = 2 ** (-I / Gamma)

            # Get new mutated test nets
            testNets = []
            for n in range(Beta):
                newNet = net.CopyNet()
                newNet.TweakWeights(twk)
                testNets.append(newNet)

            # Get the offspring's scores
            newNetScores = []
            for mutNet in testNets:
                if ScoreFunction == None:
                    newScore = netMetrics(mutNet, xArr, yArr, useFast=useFast)
                else:
                    newScore = ScoreFunction(net, xArr, yArr)
                newNetScores.append(newScore)

            # See if the best score is an improvement
            batchBestScore = max(newNetScores)
            if batchBestScore > currentScore:
                # Actions for improvement
                bestIndex = newNetScores.index(batchBestScore)
                net = testNets[bestIndex].CopyNet()
                currentScore = batchBestScore

            # Update fancy progress bar stuff
            if Verbose:
                pix1 = '='
                pix2 = '-'
                donePercent = (I + 1) / (Ieta)
                barLength = 40
                pix1Len = round(barLength * donePercent)
                pix2Len = barLength - pix1Len
                print(f"Score: {currentScore:.6f} | Training: {pix1*pix1Len}{pix2*pix2Len}", end='\r')
        
        # Finish with returning best net and its R2 value
        if Verbose:
            print(f"Score: {currentScore:.6f} | Training: {'=='*20}                   ")
        return net

    def Predict(self, X, useFast:bool = True):
        """
        Calculates and returns the net outputs for the given X input data
        """
        # Check that input X is array and correct shape
        if type(X) == np.ndarray:
            # Check for the correct amount of inputs
            if len(X.shape) > 1:
                inSize = np.size(X, 1)
            else:
                inSize = 1

            dataSize = np.size(X, 0)

            if inSize != self.inSize:
                raise ValueError(f"Given # of unique inputs {inSize} does not match the net input size {self.inSize}!")
            
        else:
            raise TypeError("Unrecognized X input data type (Should be np.ndarray)!")

        # Calculate predictions for given inputs
        predictions = []
        for i in range(dataSize):
            # Generate net input vector
            # if inSize > 1:
            #     invec = np.array(X[i]).reshape(1,inSize)
            # else:
            invec = np.array(X[i]).reshape(1,inSize)

            # Get net prediciton
            val = self.Calculate(invec, useFast=useFast)
            predictions.append(val)
        
        return np.array(predictions)


## External Net Functions ##
def LoadNet(name:str):
    """
    Returns a nerual net object with the loaded characteristics from the
    given nickname. 
    
    Inputs:

    1 - Name
    - Type == String
    """

    # Check that the mcnet data directory exsits
    # If not, make the folder and raise error
    DIR = "MCNetData"
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        raise NotADirectoryError(
            "The net data folder (MCNetData) was not found! It is now made; resave or move existing nets into this folder."
            )

    # Load the various Characteristics and weights of the NN
    # name = str(name)
    for _ in range(10): ## Redundant loading loop, reference SaveNN on 'onedrive issue'
        try:
            sizes = list(np.loadtxt(f"{DIR}/NN_{name}_Size"))
            break
        except:
            continue
    sizes = [int(i) for i in sizes]
    inSize = int(sizes[0])
    hiddenSize = sizes[1:len(sizes)-1]
    outSize = int(sizes[-1])

    # Load in the activation function data
    for _ in range(10): ## Redundant loading loop, reference SaveNN on 'onedrive issue'
        try:
            activations = list(np.loadtxt(f"{DIR}/NN_{name}_Activations", ndmin=1))
            break
        except:
            continue

    # Convert to str activation data
    strActivations = []
    for function in activations:
        # Conversions
        if function == 0:
            strActivations.append("LIN")
        elif function == 1:
            strActivations.append("RELU")
        elif function == 2:
            strActivations.append("ELU")
        elif function == 3:
            strActivations.append("ATAN")
        else:
            strActivations.append("LIN")

    # From the size, construct the net frame
    net = AdvNet(inSize, hiddenSize, outSize, strActivations)

    # Load in the saved net weights
    for i in range(len(net.sizes) - 1):
        for _ in range(10): ## Redundant loading loop, reference SaveNN on 'onedrive issue'
            try:
                weights = np.loadtxt(f"{DIR}/NN_{name}_Weights{i+1}")
                break
            except:
                continue
        weights = weights.reshape(net.sizes[i], net.sizes[i+1])
        net.weights[i] = weights

    # Return the generated neural net
    return net

def Forecast(Net, inputData, comparisonData=[], plotResults=False, useFast:bool=True):
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

    2 - inputData / XData
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
        val = net.Calculate(invec, useFast=useFast)
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

        plt.legend(["Model Values", "Comparison Values"])
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

def netMetrics(net:AdvNet, xArr:np.ndarray, yArr:np.ndarray, 
               method:str = 'r2', useFast:bool = True):
    """
    Returns the nets R^2 value for the given x input data and y validation data.

    method:
        - Options: 'r2' (more to be added)
            - 'r2' returns the R^2 fit of the model's predictions to the data.

    TODO:
        - Add Mean Average Error option
        - Add Sum Squared Error option
        - Add Root Mean Sqaure Error option
    """

    # Get predictions
    yHatList = Forecast(net, xArr, plotResults=False, useFast=useFast)

    ## R^2 Method ##
    if method == 'r2':
        return np.clip(r2d2(yHatList, yArr), -1, 1)
    
    ## Average Absolute Error ##
    elif method == 'aae':
        # Get AAE
        aae = sum(abs(np.array(yHatList).T - yArr))

        # Zero handling
        if aae == 0:
            return np.inf
        else:
            return 1/aae

    ## Sum Squared Error ##
    elif method == 'sse':
        # Get SSE
        sse = sum((np.array(yHatList).T - yArr)**2)

        # Zero handling
        if sse == 0:
            return np.inf
        else:
            return 1/sse

    ## Mean Average Error ##
    elif method == 'mae':
        mae = (1/len(yHatList)) * np.sum(np.abs(np.array(yHatList) - yArr))
        if mae == 0:
            return np.inf
        else:
            return 1/mae

def TTSplit(Xdata, Ydata, percentTrain:float = 50):
    """
    Universal Train-Test data split via random selection.

    Returns in the order xTrain, yTrain, xTest, yTest
    
    1. percentTrain
        - sets the amount of data given back for training
    data, while the rest is sent into the testing data set
    """

    # Defaults
    sortSets = True # No need to make other method rn, sorting isn't that slow
    xTest = []
    yTest = []
    xTrain = []
    yTrain = []

    # Generate the indicies used for collecting train data
    dataLen = len(Xdata)
    numTrainSamples = round(dataLen * percentTrain/100)
    numTestSamples = dataLen - numTrainSamples
    trainIndicies = rn.sample(range(dataLen), numTrainSamples)
    if sortSets:
        trainIndicies.sort()

    # All-in-one train/test collector (sorted method)
    if sortSets:
        crntIndex = 0
        while crntIndex < dataLen:
            # Check if its a training index
            if len(trainIndicies) >= 1:
                if crntIndex == trainIndicies[0]:
                    xTrain.append(Xdata[crntIndex])
                    yTrain.append(Ydata[crntIndex])
                    
                    trainIndicies.pop(0)
                    crntIndex += 1
                # Else add to test data
                else:
                    xTest.append(Xdata[crntIndex])
                    yTest.append(Ydata[crntIndex])

                    crntIndex += 1

            # Else add to test data
            else:
                xTest.append(Xdata[crntIndex])
                yTest.append(Ydata[crntIndex])

                crntIndex += 1

    # Shape data lists (accoutning for 1D lists)
    try:
        xTrain = np.array(xTrain).reshape(numTrainSamples, np.size(Xdata, 1))
        xTest = np.array(xTest).reshape(numTestSamples, np.size(Xdata, 1))
    except:
        xTrain = np.array(xTrain).reshape(numTrainSamples)
        xTest = np.array(xTest).reshape(numTestSamples)
    try:
        yTrain = np.array(yTrain).reshape(numTrainSamples, np.size(Ydata, 1))
        yTest = np.array(yTest).reshape(numTestSamples, np.size(Ydata, 1))
    except:
        yTrain = np.array(yTrain).reshape(numTrainSamples)
        yTest = np.array(yTest).reshape(numTestSamples)

    return xTrain, yTrain, xTest, yTest

def dataSelect(X, Y, count:int):
    """
    **** Experimental training data selection method ****

    **** Only useable when X and Y are 1D ****

    Returns shortened x, y data arrays with (count) # of data points/rows

    Automatically thins a database by selecting "important" data points more often than
    "non-important data points". This probability comes from the local 2nd derivative
    magnitude that are around a data point. If there are multiple Y columns (y1, y2, ...)
    then the columns the derivatives that are calculated can be specified with cols.

    X:
        - Input x array

    Y:
        - Input y array

    count:
        - Number of data points to select from the main X/Y arrays
    """

    # Input error catching
    if np.size(X, 0) < 2:
        raise ValueError("There should be at least 2 data points/rows in the X array!")
    if np.size(X, 0) != np.size(Y, 0):
        raise ValueError(f"The # of entries in X ({np.size(X, 0)}) shpuld match Y ({np.size(Y, 0)})")

    # Make numerical derivative data
    Xlength = np.size(X, 0)
    d2X = []
    
    # Get 2nd Derivative
    for row in range(1, Xlength-1):
        dydx1 = (Y[row] - Y[row-1]) / X[row] - X[row-1]
        dydx2 = (Y[row+1] - Y[row]) / X[row+1] - X[row]
        D2 = (dydx2 - dydx1) / (X[row+1] - X[row-1])
        d2X.append(D2)
    d2X = [d2X[0]] + d2X + [d2X[-1]]

    # Shape into array
    d2X = np.array(d2X).reshape(Xlength)

    # Make relative weights of each data point from 2nd derivative magnitudes
    d2X = np.abs(d2X)
    d2X /= np.sum(d2X)

    # delet this
    # if givepp:
    #     return d2X

    # Randomly select indicies taking into account their weight
    indicies = [*range(Xlength)]
    selection = np.random.choice(indicies, size=count, replace=False, p=d2X)
    selection = np.sort(selection)

    # Get the data for the small arrays
    x = []
    y = []
    for idx in selection:
        x.append(X[idx])
        y.append(Y[idx])

    # Shape the lists into array data
    if len(X.shape) == 2:
        x = np.array(x).reshape(count, np.size(X, 1))
    else:
        x = np.array(x).reshape(count)
    if len(Y.shape) == 2:
        y = np.array(y).reshape(count, np.size(Y, 1))
    else:
        y = np.array(y).reshape(count)

    return x, y


## Helper/Smol Functions ##
def applyActi(calcVec:np.ndarray, activationFunction:str):
    """Applies an activation function to the given vector/array of data in calcVec."""

    # Some idiot proofing (i should put this in more places)
    activationFunction = activationFunction.upper()

    # Activation function definitions tree
    ## TODO: turn this cute if tree into a case/switch thing: (Needs >= Python 3.10)
    # match activationFunction:
    #     case 'LIN':
    #         pass

    #     case 'RELU':
    #          calcVec[calcVec < 0] = 0

    #     case _:
    #         raise ValueError("Given hidden function is not of the avalible types!")

    if activationFunction == 'LIN':
        pass

    elif activationFunction == 'RELU':
        calcVec[calcVec < 0] = 0

    elif activationFunction == 'ELU':
        calcVec = 0.4 * (2.7182818284**calcVec - 1)

    elif activationFunction == 'ATAN':
        calcVec = np.arctan(calcVec)

    elif activationFunction == "RND":
        calcVec = np.round(calcVec)
    
    elif activationFunction == "SQR":
        calcVec[calcVec < 0] = -1
        calcVec[calcVec >= 0] = 1

    elif activationFunction == "ROOT":
        calcVec = (np.arctan(calcVec) * np.abs(calcVec)**0.5) / 1.5

    elif activationFunction == 'RESU':
        # calcVec[calcVec < 0] = 0
        calcVec = abs(np.sin(calcVec))*calcVec

    elif activationFunction == 'RESU2':
        calcVec[calcVec < 0] = 0
        calcVec = abs(np.sin(calcVec))*calcVec

    elif activationFunction == 'SIG':
        e_p = np.exp(calcVec)
        e_n = np.exp(-calcVec)
        calcVec = ((e_p - e_n) / (e_p + e_n)) + 1

    elif activationFunction == 'EXP':
        calcVec[calcVec < -0.4] = 0

    else:
        raise ValueError("Given hidden function is not of the avalible types!")

    # Finish
    return calcVec

def fastCalc(net:AdvNet, inVector:np.ndarray):
    """
    Mostly benefits smaller nets, especially with less than a few layers. Updated to 
    allow for using activation function on the input vector before any network calculations
    are done.
    
    ### NOTICE ###
    By voiding data shape checks, this is the equivalent of .Calculate() for a net, but
    takes less time. If you're using this, that means you are confident the inVector 
    data you are using is correct, and/or you do not need the many error checks that 
    .Calculate has.

    This function still supports single float inputs/outputs, just as .Calculate does.
    """
    # Basic check - Shape vector to what it should be
    # If this doesn't work, sounds like a you problem :P
    inVector = inVector.reshape((net.inSize, 1))

    # Possible fast calculations
    if net.sizes[0] == 1 and len(net.sizes) == 2: # ~45% of regular calc time
        inVector = applyActi(inVector, net.activationFunction[0])
        inVector = applyActi(inVector*net.weights[0], net.activationFunction[1])

    else:   # ~73% of regular calc time
        inVector = applyActi(inVector, net.activationFunction[0])
        for i in range(1, len(net.sizes)):
            if i != 0:
                inVector = inVector.reshape((inVector.size, 1))
            inVector = applyActi(sum(inVector*net.weights[i-1]), net.activationFunction[i])

    # Return the vector shaped as a column vector
    if inVector.size == 1:     # if a single number
        return inVector[0]
    else:                      # all other vectors
        inVector = inVector.reshape(inVector.size, 1)
        return inVector

def r2d2(yModel:np.ndarray, yTrue:np.ndarray):
    """
    Returns the R^2 value of the model values (yModel) against the true
    y data (yTrue).
    """

    # List checking for idoits like me
    if type(yModel) == list:
        yModel = np.array(yModel)
    if type(yTrue) == list:
        yTrue = np.array(yTrue)

    # R2 Calc
    yMean = np.mean(yTrue)
    RES = np.sum((yTrue - yModel) ** 2)
    TOT = np.sum((yTrue - yMean) ** 2)
    if TOT == 0:
        TOT = np.inf

    return 1 - (RES / TOT)
    

## Old / Deprecated ##
def oldFastCalc(net:AdvNet, inVector:np.ndarray):
    """
    **** Deprecated since version 1.2.2 ****
    - Used old calculation method that didn't allow for using activation functions on
      the input vector before any matrix calculations

    Mostly benefits smaller nets, especially with less than a few layers.
    
    ### NOTICE ###
    By voiding data shape checks, this is the equivalent of .Calculate() for a net, but
    takes less time. If you're using this, that means you are confident the inVector 
    data you are using is correct, and/or you do not need the many error checks that 
    .Calculate has.

    This function still supports single float inputs/outputs, just as .Calculate does.
    """
    # Basic check - Shape vector to what it should be
    # If this doesn't work, sounds like a you problem :P
    inVector = inVector.reshape((net.inSize, 1))

    # Possible fast calculations
    if net.sizes[0] == 1 and len(net.sizes) == 2: # ~45% of regular calc time
        inVector = applyActi(inVector*net.weights[0], net.activationFunction[0])

    else:   # ~73% of regular calc time
        for i in range(len(net.sizes)-1):
            if i != 0:
                inVector = inVector.reshape((inVector.size, 1))
            inVector = applyActi(sum(inVector*net.weights[i]), net.activationFunction[i])

    # Return the vector shaped as a column vector
    if inVector.size == 1:     # if a single number
        return inVector[0]
    else:                      # all other vectors
        inVector = inVector.reshape(inVector.size, 1)
        return inVector

def thinData(xData, yData, numPoints:int):
    """
    **** Deprecated since V1.1.0, use dataSelect() instead ****

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

def genTrain(net:AdvNet, xArr:np.ndarray, yArr:np.ndarray, iterations:int = 1000, 
                    batchSize:int = 0, gamma:int = 50, weightSelection:str = None, 
                    R2Goal = 0.999, Silent:bool = False, useFast:bool = False):
    """
    **** Replaced by .FastTrain since version 1.0.0 ****

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
    currentR2 = netMetrics(net, xArr, yArr, useFast=useFast)

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
            newR2 = netMetrics(mutNet, xArr, yArr, useFast=useFast)
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

def OldTrain(Net, inputData, yData, startingTweakAmp=0.8, 
          plotLive=False, plotResults=False, normalizeData=False, 
          hiddenFunc="ELU", trainWeights='all', maxIterations=1000, 
          blockSize=30, Silent=False):
    """
    **** Deprecated since version 0.2.0 ****

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
    - Decides what hidden-layer processing should be used. 'LIN' simply passes
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
    **** Deprecated since version 0.2.0 ****

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
    - Decides what hidden-layer processing should be used. 'LIN' simply passes
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
    Net, bestError = OldTrain(Net, inputData, yData, 
                        startingTweakAmp=startingTweakAmp, normalizeData=normalizeData, 
                        hiddenFunc=hiddenFnc, maxIterations=1, 
                        blockSize=blockSize, Silent=True)
    
    # Start cycle training (limit to 1000 just in case)
    for cycle in range(maxCycles):
        if Silent == False:
            print(f"Starting Cycle #{cycle+1}")
        
        Net, error = OldTrain(Net, inputData, yData, 
                            startingTweakAmp=startingTweakAmp, normalizeData=normalizeData, 
                            hiddenFunc=hiddenFnc, maxIterations=maxIterations, 
                            blockSize=blockSize, Silent=Silent, trainWeights=[0],
                            plotLive=plotLive, plotResults=False)
        Net, error1 = OldTrain(Net, inputData, yData, 
                            startingTweakAmp=startingTweakAmp, normalizeData=normalizeData, 
                            hiddenFunc=hiddenFnc, maxIterations=maxIterations, 
                            blockSize=blockSize, Silent=Silent, trainWeights='middle',
                            plotLive=plotLive, plotResults=False)
        Net, error2 = OldTrain(Net, inputData, yData, 
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

