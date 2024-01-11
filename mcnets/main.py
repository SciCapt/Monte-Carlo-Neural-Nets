# from matplotlib import pyplot as 
plt = None # Final dependancy (Forecast) Deprecated in V1.4.0
import numpy as np
import textwrap as txt
import os
from time import perf_counter as pc
from random import sample
from joblib import load, dump
from multiprocessing import Pool
import copy

# Neural Network Model
class MCNeuralNetwork:
    ## Default Functions ## 
    def __init__(self, hidden_counts:list = [100], activations: any = 'DEFAULT'):
        """
        Create a deep neural network from the given counts of neurons requested per layer. By default,
        uses a SILU activation function on all hidden layers and a Linear/Identity on the input and
        output layers.

        ## Inputs:

        - hidden_counts
            - Heights of the hidden layers desired

        - activations
            - LIN:   Identity (x)
            - RELU:  max(x, 0)
            - SILU:  x*sigmoid(x)
            - SIG:   sigmoid(x) (or just sig(x))
            - dSILU: sig(x)*(1 + x*(1 - sig(x)))
            - TANH:  tanh(x)
            - ELU:   0.4 * (2.7182818284**x - 1)
            - ROOT:  Essentially a 'leaky' atan function (no horizontal asymptotes)
            - SQR:   x = 0 for x <= 0, x = 1 for x > 0
            - RND:   round(x)
            - NOTES:
                - A list of different functions per layer can be given OR
                - A single string/Activation Function
                - Values are NOT case sensitive!

        ## Model Generation

        The model's number of input and output features are generated when it is
        fitted to some data set for the first time. Therefore, until .fit() is properly
        called on a model, the printout will look like the first section. After .fit()
        is used properly, it will show some stats about the model like in the second
        printout.

        ```
        import mcnets as mc

        model = mc.MCNeuralNetwork(hiddenCounts=[10], activations=['atan', 'relu', 'lin'])
        print(model)

        >>> ========================================================================
        >>>                      Neural Net Characteristics:                       
        >>> 1. Layer Sizes = [-1, 10, -1]
        >>> 2. Weight Medians = Not yet generated; call .fit first
        >>> 3. Number of Parameters: Not yet generated; call .fit first
        >>> 4. Activation Functions: ['ATAN', 'RELU', 'LIN']
        >>> 5. Avg. Calculation Time (s): Not yet generated; call .fit first
        >>> ========================================================================

        import numpy as np

        X = np.random.rand(100, 2)
        Y = 2*X[:, 0] - 4*X[:, 1]

        model = mc.MCNeuralNetwork(hiddenCounts=[10], activations=['atan', 'relu', 'lin'])

        model.fit(X, Y)

        print(model)

        >>> ========================================================================
        >>>                      Neural Net Characteristics:                       
        >>> 1. Layer Sizes = [2, 10, 1]
        >>> 2. Weight Medians = [0.01, -0.42]
        >>> 3. Number of Parameters: 30
        >>> 4. Activation Functions: ['ATAN', 'RELU', 'LIN']
        >>> 5. Avg. Calculation Time (s): 9.52e-05
        >>> ========================================================================
        ```
        """

        # Sizes (inSize and outSize not yet initiated)
        self.inSize = -1
        self.outSize = -1
        self.hiddenSize = hidden_counts
        self.sizes = [self.inSize] + hidden_counts + [self.outSize]

        # Biases
            ## use_biases allows for turning off a model from changing its biases
            ## as they are all zeros at first, if this is changed before the model is fitted
            ## this allows for using a model with now biases
            ## 
            ## bias_bounds allow for sneaky changing of the lower and upper bound of
            ## the model's biases by giving it a different tuple
        self._biases = []
        self.use_biases = True
        self.bias_bounds = (-2, 2)

        # Other params to be generated during the first .fit call
        self.parameters = "Not yet generated; call .fit first"
        self.speed = "Not yet generated; call .fit first"
        self.fitted = False
        self._weights = None

        # Sneaky thing
        self.weight_bounds = (-1, 1)

        # Construct activation functions list (if only given a single function type)
        if type(activations) == str:
            if activations == "DEFAULT":
                self.activationFunction = ['silu']*len(self.sizes)
                self.activationFunction[0] = 'lin'
                self.activationFunction[-1] = 'lin'
            else:
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

        # If given a single external activation function
        elif callable(activations):
            self.activationFunction = [activations]*len(self.sizes)
        else:
            raise ValueError(f"Type {type(activations)} not supported for activation functions")
            
        # Force upper case for activation functions
        # (allows giving them as lower case/whatever when constructed)
        # Checks that not given a single external AF
        # Checks that when applying .upper() it isnt trying it on an external AF
        if not callable(activations):
            try:
                for i, afi in enumerate(self.activationFunction):
                    if not callable(afi):
                        # Should be a string AF form
                        self.activationFunction[i] = self.activationFunction[i].upper()
                    else:
                        # Is a custom external AF
                        # self.activationFunction[i] = self.activationFunction[i]
                        pass
            except:
                raise ValueError("Confirm all activation functions given in the list are strings and/or functions!")        

    def __str__(self):
        # Resulting string dimensions/title
        strlen = 72
        l1 = f"="
        l2 = f"Neural Net Characteristics:"

        # Show the layer heights of the net
        l3 = f"1. Layer Sizes = {self.sizes}"
        l3 = txt.fill(l3, strlen-2)

        # Get and show the medians weight values within the weight arrays
        if self.weights != None:
            weightMedians = []
            for weights in self.weights:
                weightMedians.append(round(np.median(weights), 2))
            l4 = f"2. Weight Medians = {weightMedians}"
            l4 = txt.fill(l4, strlen-2)
        else:
            l4 = f"2. Weight Medians = Not yet generated; call .fit first"
            l4 = txt.fill(l4, strlen-2)

        # Show the number of paramaters that make up the net
        l6 = f"3. Number of Parameters: {self.parameters}"

        # Show activation functions and order
        l7a = f"4. Activation Functions: {self.activationFunction}"
        l7a = txt.fill(l7a, strlen-2)
        
        # Show the calculation time
        if type(self.speed) == str:
            l8 = f"5. Avg. Calculation Time (s): {self.speed}"
            l8 = txt.fill(l8, strlen-2)
        else:
            l8 = f"5. Avg. Calculation Time (s): {self.speed} per sample"
            l8 = txt.fill(l8, strlen-2)

        # Resulting String
        full = (l1*strlen + '\n' + l2.center(strlen, ' ') + '\n' + l3 + 
                '\n' + l4 + '\n' + l6 + '\n' + 
                l7a + '\n' + l8 + '\n' + l1*strlen)
        return full

    def __len__(self):
        return len(self.sizes)

    ## Weights Handling ##
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, newWeightList):
        # Apply to main weights attribute
        self._weights = [wi.copy() for wi in newWeightList]

    ## Biases Handling
    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, new_biases):
        self._biases = [bi.copy() for bi in new_biases]

    ## Internal Functions ##
    def TweakWeights(self, Amplitude: float, Selection='all', returnChange:bool = False):
        """
        Adjust the weights of a neural net by a given amplitude. The net change in the 
        weight arrays can be optionally returned for gradient decent calculation later.
        
        ## Inputs:

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

        ## Operation:

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

            # Make sure values are in the range [-1, 1] (default)
            self.weights[i] = np.clip(self.weights[i], self.weight_bounds[0], self.weight_bounds[1])

            # Adjust biases
            if self.use_biases:
                self.biases[i] += (np.random.rand(self.sizes[i+1]) - 0.5)*(2*Amplitude)
                self.biases[i] = np.clip(self.biases[i], self.bias_bounds[0], self.bias_bounds[1])

        # Note change in weight array after clipping
        if returnChange:
            dW = [(self.weights[i] - Wo[i]) for i in range(len(self.weights))]
            return dW

    def Calculate(self, inVector, useFast=True):
        """
        Returns the model's prediction to a given input vector

        1 - inVector
            - Sample vector of X data
            - If the model input size is 1, then a float can be given to calculate from

        2 - useFast (**Deprecated**)
            - Determines if speed is favored over error handling
            - Using this typically speeds up calculations by ~5x
        """

        # Simple Error Check (num features)
        if self.inSize > 1 and inVector.size != self.inSize:
            raise ValueError(f"Input X Sample Size ({inVector.size}) != Model Input Size ({self.inSize})!")

        # First Layer
        inVector = applyActi(inVector, self.activationFunction[0])

        # All other layers (passing through weights)
        for i in range(1, len(self.sizes)):
            inVector = applyActi(np.dot(inVector, self.weights[i-1]) + self.biases[i-1], self.activationFunction[i])

        # Finish
        if inVector.size == 1:   # Return if passing only a single value
            return inVector[0]
        else:   # Return if passing a resulting vector
            # if len(inVector.shape) >= 3:
            #     raise ValueError("Calculation Vector size >= 3")
            inVector = inVector.reshape(inVector.size, 1)
            return inVector

    def CopyNet(self):
        """
        Copy a neural net object to a new variable. Used in similar
        situations as NumPy's copy method to make an isolated net
        with the same values and properties as the parent net.
        """
        # Check that it is fitted
        if self.fitted == False:
            raise ValueError("Model is not yet fitted and therefore cannot be copied")

        return copy.deepcopy(self)

    def ApplyTweak(self, dW):
        for i, dWi in enumerate(dW):
            self.weights[i] = self.weights[i] + dWi

    def fit(self, xArr:np.ndarray, yArr:np.ndarray, Ieta:int = 9, Beta:int = 50, Gamma:int = 3, 
            ScoreFunction = None, verbose:bool = True, useFast:bool = True, zeroIsBetter:bool = True, 
            scoreType:str = 'r2', useMP:bool = False):
        """
        ## Overview

        Returns the score history from the model training.
        
        Instead of always using the first net that has any improvement from the previous best net,
        a 'batch' of nets are generated from the parent net, then tested to find the best of those.
        That is, this essentially does a search with depth 'Beta' to find the next best net.

        ## Inputs

        xArr:
            - Input data for the net.

        yArr:
            - Validation data used to train the net to.

        Ieta:
            - Amount of iteration attempts at improving the net's fit to the data.

        Beta:
            - Amount of nets the be constructed and tested for improvements per iteration.

        Gamma:
            - Sets the learning factor exponential decay rate. That is, every 'gamma' number 
            of iterations, the learning rate will decrease by a factor of 1/2. 

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
              error handling but increases net calculation speed for nets with few layers.

        zeroIsBetter:
            - Changes objective to getting a lower value score. Changes to False automatically 
              when using a scoring function like R^2 or RAE where higher is better.
            - NOTE: if using an external scoring function, you will have to change this
              value manually according to how that function works.

        scoreType:
            - Possible score methods in netMetrics: ['r2', 'sse', 'mre', 'mae', 'rae']

        useMP:
            - If the multiprocessing form is allowed to be used
            - Improvements differ vastly based on the machine being used
            - When Beta*len(xArr) >= ~1e5, multiprocessing is better
                - Automatically uses normal method otherwise (is faster below ~1e5)
                - The larger the net, the more efficient it is to use multiprocessing
                - Improvements of up to 4x are common with large nets/datasets
        """

        # Complete model setup during first fit
        if self.fitted == False:

            if self.inSize == -1:
                # Handling for 1D and 2D y data
                if len(xArr.shape) > 1:
                    self.inSize = np.size(xArr, axis=1)
                else:
                    self.inSize = 1
            
            if self.outSize == -1:
                # Handling for 1D and 2D y data
                if len(yArr.shape) > 1:
                    self.outSize = np.size(yArr, axis=1)
                    if self.outSize != 1:
                        print(f"# of Y features found to be {self.outSize} and not 1 per sample. No action required if this is correct.")
                else:
                    self.outSize = 1

            # Regenerate the counts parameter with data size from x/y
            if self.sizes[0] == -1:
                self.sizes[0] = self.inSize
            if self.sizes[-1] == -1:
                self.sizes[-1] = self.outSize

            # Setup Biases
            for si in self.sizes[1:]:
                self.biases.append(np.zeros(si))

            # Setup weights
            if self.weights == None:
                self.weights = []
                for i in range(len(self.sizes) - 1):
                    # Initialize weights with some random weights in range [-0.5, 0.5]
                    newWeightsArr = (np.random.rand(self.sizes[i], self.sizes[i+1]) - 0.5)
                    self.weights.append(newWeightsArr)

                # Calculate the number of Parameters in the Weights/net
                weightSizes = [M.size for M in self.weights]
                self.parameters = sum(weightSizes) + sum([b.size for b in self.biases])

            # Test Calculation Time (for __str__)
            t1 = pc()
            for _ in range(3):  # Do just 3 iterations
                self.Calculate(np.ones((self.inSize)))
            t2 = pc()
            self.speed = format((t2-t1)/3, ".2e")

            # Mark fitting as done
            self.fitted = True

        # Pandas DataFrame and Series handling
        Tx = str(type(xArr))
        if Tx in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            xArr = xArr.to_numpy()

        Ty = str(type(yArr))
        if Ty in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            yArr = yArr.to_numpy()

        # Verify data types
        if type(self) == MCNeuralNetwork and type(xArr) == np.ndarray and type(yArr) == np.ndarray:
            # print("All data is good")
            pass
        else:
            raise ValueError("An input data type is not correct")

        # Note if zeroIsBetter needs to be flipped for 'r2' or 'rae'
        if scoreType in ['r2', 'rae']:
            zeroIsBetter = False

        # Scoring constant (flips equality)
        if zeroIsBetter:
            SK = 1
        else:
            SK = -1

        # Get inital accuracy
        if ScoreFunction == None:
            currentScore = self.score(xArr, yArr, useFast=useFast, method=scoreType)
        else:
            currentScore = ScoreFunction(self, xArr, yArr)

        # Start score history
        history = [currentScore]

        # Dev feature: plotting all scores
        # scores = ['r2', 'sse', 'mre', 'mae', 'rae']
        # startingScores = [[self.score(xArr, yArr, useFast=True, method=scr)] for scr in scores]
        # historyALL = dict(zip(scores, startingScores))

        # Generational Training method
        for I in range(Ieta):
            # Get current tweak amplitude
            twk = 2 ** (-I / Gamma)

            # Get new mutated test nets
            testNets = []
            for n in range(Beta):
                newNet = self.CopyNet()
                newNet.TweakWeights(twk)
                testNets.append(newNet)

            # Get the offspring's scores
            if len(xArr)*Beta >= 1e5 and useMP: # Multiprocessing method
                newNetScores = multiNetScore(testNets, xArr=xArr, yTrue=yArr, scoreType=scoreType, useFast=useFast)
                
            else: # Normal method
                newNetScores = []
                for mutNet in testNets:
                    if ScoreFunction == None:
                        newScore = mutNet.score(xArr, yArr, useFast=useFast, method=scoreType)
                    else:
                        newScore = ScoreFunction(mutNet, xArr, yArr)
                    newNetScores.append(newScore)

            # See if the best score is an improvement
            if SK == 1:
                batchBestScore = min(newNetScores)
            else:
                batchBestScore = max(newNetScores)
            if SK*batchBestScore < SK*currentScore:
                # Actions for improvement
                bestIndex = newNetScores.index(batchBestScore)
                self.weights = testNets[bestIndex].weights
                self.biases = testNets[bestIndex].biases
                currentScore = batchBestScore

            # Update score history
            history.append(currentScore)

            # Dev thingy:
            # for scr in scores:
            #     historyALL[scr].append(self.score(xArr, yArr, useFast=True, method=scr))

            # Update fancy progress bar stuff
            if verbose:
                pix1 = '='
                pix2 = '-'
                donePercent = (I + 1) / (Ieta)
                barLength = 40
                pix1Len = round(barLength * donePercent)
                pix2Len = barLength - pix1Len
                print(f"{scoreType.upper()} Score: {currentScore:.6f} | Training: {pix1*pix1Len}{pix2*pix2Len}", end='\r')
        
        # Finish: print final score/progress and return the score history
        if verbose:
            print(f"{scoreType.upper()} Score: {currentScore:.6f} | Training: {'=='*20}       ")
        
        return np.array(history)
    
    def predict(self, xArr, useFast:bool = True):
        """
        Calculates and returns the net outputs for all the given X input data
        """
        # Pandas DataFrame and Series handling
        T = str(type(xArr))
        if T in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            xArr = xArr.to_numpy()

        # Check that input X is array and correct shape
        if type(xArr) == np.ndarray:
            # Check for the correct amount of inputs
            if len(xArr.shape) > 1:
                inSize = np.size(xArr, 1)
            else:
                inSize = 1

            dataSize = np.size(xArr, 0)

            if inSize != self.inSize:
                if self.fitted == False:
                    raise AttributeError("Model is not yet fitted, call .fit first")
                else:
                    raise ValueError(f"Given # of unique inputs {inSize} does not match the net input size {self.inSize}!")
        else:
            raise TypeError(f"Unrecognized X input data type (Is {type(xArr)})!")

        # Calculate predictions for given inputs
        predictions = []
        for i in range(dataSize):
            # Generate net input vector
            invec = np.array(xArr[i]).reshape(1,inSize)

            # Get net prediciton
            predictions.append(self.Calculate(invec, useFast=useFast))

        # Format predictions as numpy array
        predictions = np.array(predictions)
        predictions.reshape((len(xArr), self.outSize))  # For some reason, doing this in the line above screws up the predictions of the models???? idk just keep it here
        return predictions

    def score(self, xArr, yTrue, useFast:bool = True, method:str = 'r2'):
        """
        Returns the nets error/score from one of a few methods

        method:
            - 'r2' returns the R^2 fit of the model
            - 'sse' returns the Total Sum Squared Error
            - 'mre' returns the Mean Root Error
                - MRE = SQRT( SUM( (ytrue - ymodel)**2 ) / N )
            - 'mae' returns the Mean Absolute Error
            - 'rae' returns a custom value similar to r2 from raeScore()
        """

        # Get model predictions
        yHat = self.predict(xArr, useFast=useFast)
        yHat.shape = yHat.shape[:2]

        ## R^2 Method ##
        if method == 'r2':
            return r2d2(yHat, yTrue)

        ## Sum Squared Error ##
        elif method == 'sse':
            return np.sum((yHat - yTrue)**2)
        
        ## Mean Sum Squared Error ##
        elif method == 'mre':
            return (np.sum((yHat - yTrue)**2) / len(yHat))**0.5

        ## Mean Absolute Error ##
        elif method == 'mae':
            return (1/len(yHat)) * np.sum(np.abs(np.array(yHat) - yTrue))

        elif method == 'rae':
            return raeScore(yTrue, yHat)

        # Raise error if one of the possible methods was not given
        else:
            raise ValueError(f"Given score type '{method.upper()}' is not one of the avalible types.")

    def save(self, path_or_name:str):
        """
        Uses joblib function 'dump' to save the model object with the given
        file path or name.
        """
        dump(self, path_or_name)

    def optimize_fit(self, xArr, yArr, activationTests:list = ['silu', 'dsilu', 'tanh', 'root', 'lin'],
                    trainDepthFactor:int = 1, inputActivation:str = 'lin', outputActivation:str = 'lin',
                    testInput:bool = False, testOutput:bool = False, scoreType:str = 'sse',
                    useBest:bool = True, useFast:bool = True, Verbose:bool = True):
        """
        ## Overview

        A method that attempts to find a the most optimal combination of activation functions
        for an MCNeuralNetwork model, for a given X/Y set and possible activations in activationTests. 
        Returns in-place the best model that fits to the given data.

        Note that there is a test-train split done before testing activation sets.

        ## Inputs

        - xArr:
            - The input X samples
        - yArr:
            - The input Y target samples
        - activationTests:
            - The list that test activations are pulled from 
        - trainDepthFactor:
            - An integer [1, inf]; higher means longer training cycles for more a more accurate optimization
        - inputActivation:
            - If testInput is False (Default), this will be the input's activation function used in all tests
        - outputActivation:
            - If testOutput is False (Default), this will be the outputs's activation function used in all tests
        - testInput:
            - If True, the input activations will be tested as well (with activations pulled from activationTests)
            - inputActivation is ignored if True
        - testOutput:
            - If True, the output activations will be tested as well (with activations pulled from activationTests)
            - outputActivation is ignored if True
        - scoreType:
            - One of the scoring options in the .score method ('r2', 'sse', 'mae', or 'rae)
        - useBest:
            - If True (Default) the best-scoring model will replace the base model (returned in-place)
        - useFast:
            - If True (Default) fastCalculate will be used (less error reporting for faster calculation speeds)
        - Verbose:
            - If True (Default) the top 3 (or less) combinations and their validation scores are printed when completed
        """

        # Results lists
        scores = []
        afsets = []

        # Generate info for making generalized comboniation tests
        inoutNonTestLen = 2 - (testInput + testOutput)
        setLength = len(self) - inoutNonTestLen
        comboID = [0] * setLength
        baseNum = len(activationTests)

        # Fit the model to data shape if not already fitted
        if self.fitted == False:
            self.fit(xArr, yArr, Ieta=0, verbose=0, scoreType=scoreType)

        # Set top model as self to start
        topModel = self.CopyNet()

        def marchCombo(comboID, baseNum):
            """
            Progresses the test combo list as if it were a binary string,
            but with a base of baseNum, not 2.
            """
            carry = False
            comboID[0] += 1

            for i in range(len(comboID)):
                if carry:
                    comboID[i] += 1

                if comboID[i] >= baseNum:
                    comboID[i] = 0
                    carry = True
                else:
                    carry = False
                    break

            return comboID
        
        # Check if any optimization to be done
        numCombos = baseNum**setLength
        if setLength == 0:
            numCombos = 0
            print("No optimization done! No hidden layers & no input/output layers set to test mode")
            return None
        
        # Run through all possible combos of activations
        xt, xv, yt, yv = TTSplit(xArr, yArr)
        for i in range(numCombos):
            print(f"On Test #{i+1} / {numCombos}    ", end='\r')

            # Generate a test model with the new test activations list
            testModel = self.CopyNet()

            newAFset = [activationTests[ID] for ID in comboID]
            if inoutNonTestLen == 2:
                newAFset = [inputActivation] + newAFset + [outputActivation]
            elif inoutNonTestLen == 0:
                pass
            elif testInput:
                newAFset = newAFset + [outputActivation]
            elif testOutput:
                newAFset = [inputActivation] + newAFset
            
            testModel.activationFunction = newAFset

            # Test/validate model scores; add to scores and set lists
            testModel.fit(xt, yt, Ieta=9*trainDepthFactor, Gamma=3*trainDepthFactor, 
                          verbose=False, useFast=useFast, scoreType=scoreType)
            scores.append(testModel.score(xv, yv, method=scoreType, useFast=useFast))
            afsets.append(newAFset)

            # Check it last was a top model tested
            if scoreType in ['sse', 'mae', 'mre'] and scores[-1] == min(scores):
                topModel = testModel.CopyNet()
            elif scoreType in ['r2', 'rae'] and scores[-1] == max(scores):
                topModel = testModel.CopyNet()

            # Generate the next combo to try
            comboID = marchCombo(comboID, baseNum)

        # Get top 3 sets and corresponding scores
        topSetsScores = []
        leaderboardLen = min(3, len(scores))
        for _ in range(leaderboardLen):
            # Get best score according to score type used
            if scoreType in ['sse', 'mae', 'mre']:
                bestInd = scores.index(min(scores))
            elif scoreType in ['r2', 'rae']:
                bestInd = scores.index(max(scores))

            topSetsScores.append([scores[bestInd], afsets[bestInd]])

            # Remove current top
            scores.pop(bestInd)
            afsets.pop(bestInd)

        # Replace the model's activations and weights with the best set
        if useBest:
            self.activationFunction = topModel.activationFunction
            self.weights = topModel.weights

        # Return the top results for comparison
        if Verbose:
            print(f"Top {leaderboardLen} Combinations:")
            for i in range(leaderboardLen):
                print(f"{scoreType.upper()} Score: {topSetsScores[i][0]} | Set: {topSetsScores[i][1]}")

    def load(path_or_name:str):
        return load_model(path_or_name=path_or_name)


# Neural Network Ensemble Model
class MCSimpleEnsemble:
    def __init__(self, n_estimators = 10, hidden_counts = [100,], activations = "DEFAULT"):
        """
        A simple ensemble model using n models dervied from a cross validation.
        The model's weights are based off of their relative scores from the cross
        validation initially done
        """

        # Attributes
        self.n_estimators = n_estimators
        self.models = []
        self.weights = []
        self.fitted = False
        self.base_model = MCNeuralNetwork(hidden_counts=hidden_counts, activations=activations)

    def fit(self, X, Y, depth=1, verbose=True, scoreType='r2', **kwargs):
        self.models, self.weights = cross_val(self.base_model, X, Y, N=self.n_estimators, 
                                              return_models=True, train_depth=depth, verbose=verbose,
                                              scoreType=scoreType)
        
        # Fit check
        self.base_model.fitted = True
        self.fitted = True

    def predict(self, X, **kwargs):
        yhat = 0
        for wi, mi in zip(self.weights, self.models):
            yhat += wi*mi.predict(X)
        return yhat

    def score(self, X, y_true, method='r2'):
        # yhat = self.predict(X)
        # return r2d2(yhat, y_true)
        return MCNeuralNetwork.score(self, X, y_true, method=method)
    
    def CopyNet(self):
        """
        Copy a neural net object to a new variable. Used in similar
        situations as NumPy's copy method to make an isolated net
        with the same values and properties as the parent net.
        """
        # Check that it is fitted
        if self.fitted == False:
            raise ValueError("Model is not yet fitted and therefore cannot be copied")

        return copy.deepcopy(self)


# SUNN (Super Unique Neural Network)
class SUNN:
    def __init__(self, hidden_counts: list = [100,], activations: list = ['lin', 'silu', 'dsilu', 'elu', 'tanh', 'root'],
                 inputActivation:str = 'lin', outputActivation:str = 'lin', structured:bool = True):
        """
        A Neural Network regressor that takes using many activation functions (AFs) to
        the extreme. By applying a seperate AF at *every* node, some very unique behvaiors
        can be generated. 

        - hidden_counts:
            - Hidden layer sizes

        - activations:
            - Nodes will randomly pull an activation function from this list
                - LIN:   Identity (x)
                - RELU:  max(x, 0)
                - SILU:  x*sigmoid(x)
                - SIG:   sigmoid(x) (or just sig(x))
                - dSILU: sig(x)*(1 + x*(1 - sig(x)))
                - TANH:  tanh(x)
                - ELU:   0.4 * (2.7182818284**x - 1)
                - ROOT:  Essentially a 'leaky' atan function (no horizontal asymptotes)
                - SQR:   x = -1 for x < 0, x = 1 for x >= 0
                - RND:   round(x)
                - NOTES:
                    - A list of different functions per layer can be given OR
                    - A single string/Activation Function
                    - Values are NOT case sensitive!

        - inputActivation:
            - Force the input nodes to have this activation function
            - Set to None to disable

        - outputActivation:
            - Force the output nodes to have this activation function
            - Set to None to disable

        - structured:
            - Determines if activations are equally distributed amoung the nodes per layer instead of being random
        """

        # Sizes (inSize and outSize not yet initiated)
        self.inSize = -1
        self.outSize = -1
        
        self.hiddenSize = hidden_counts
        self.sizes = [self.inSize] + hidden_counts + [self.outSize]

        # Activation Function (AF) Stuff
        self.activationFunction = activations
        self.inputActivation = inputActivation
        self.outputActivation = outputActivation
        self.structured = structured

        if self.inputActivation not in self.activationFunction and self.inputActivation != None:
            self.activationFunction.append(self.inputActivation)
        if self.outputActivation not in self.activationFunction and self.outputActivation != None:
            self.activationFunction.append(self.outputActivation)

        # Biases (Not fully implemented)
        self.use_biases = False
        self.biases = None

        # Not yet generatable stuff
        self.parameters = "Not yet generated; call .fit first"
        self.speed = "Not yet generated; call .fit first"
        self._weights = None
        self._SUA = None
        self.fitted = False

        # Sneaky setting
        self.weight_bounds = (-1, 1)

        # Setup dictionary to make a map for the given activations
        self.SUmap = {}
        keyNum = 1
        for af in self.activationFunction:
            if af not in self.SUmap.values():
                self.SUmap[keyNum] = af
                keyNum += 1

    def __str__(self):
        print("=========================================")
        print("SUNN (Super Unique Nerual Network) Stats:")
        print("    **** NOTE: SUNN Is in Beta! ****")
        print()
        print(f"Avg. Calc. Speed (s): {self.speed}")
        print(f"Number of Parameters: {self.parameters}")
        print(f"SUMap (key -> AF Mapping): {self.SUmap}")
        print(f"SUA (Super Unique Arrays): {self.SUA}")
        print("=========================================")
        return ""

    ## Weights Handling ##
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, newWeightList):
        # Apply to main weights attribute
        self._weights = [wi.copy() for wi in newWeightList]

    ## SUA Handling ##
    @property
    def SUA(self):
        return self._SUA

    @SUA.setter
    def SUA(self, newSUAList):
        # Apply to main weights attribute
        self._SUA = [si.copy() for si in newSUAList]

    ## Methods Carried from elsewhere ##
    def Calculate(self, invec, useFast=True):
        return SUNN.calculate_su(self, invec)

    def predict(self, xarray, useFast=True):
        out = MCNeuralNetwork.predict(self, xarray, useFast=useFast)
        out = out.reshape((len(xarray), self.outSize))
        return out

    def TweakWeights(*args):
        return super(type(SUNN.TweakWeights), MCNeuralNetwork.TweakWeights(*args))

    def score(net, xArr, yTrue, method='r2', useFast=True):
        return MCNeuralNetwork.score(net, xArr, yTrue, useFast=useFast, method=method)
    
    def CopyNet(self):
        # # New Net Matrix/Stuff Sizes (These aren't actually outside of SUNN's init)
        # newInputCount = self.inSize
        # newOutputCount = self.outSize
        # newHiddenCounts = self.hiddenSize
        # if type(self.activationFunction) == list:
        #     newActivations = self.activationFunction.copy()
        # else:
        #     newActivations = self.activationFunction
        # newSizes = [newInputCount] + newHiddenCounts + [newOutputCount]

        # # Make net net shell
        # newNet = SUNN(newHiddenCounts, newActivations)
        # newNet.inSize = newInputCount
        # newNet.outSize = newOutputCount
        # # newNet.hiddenSize = newHiddenCounts
        # newNet.sizes = newSizes
        # newNet.activationFunction = newActivations
        # newNet.inputActivation = self.inputActivation
        # newNet.outputActivation = self.outputActivation
        # newNet.structured = self.structured
        
        # # Setup weights / SUAs (handlers copy this information correctly)
        # newNet.weights = self.weights
        # newNet.SUA = self.SUA
            
        # # Return the copied net
        # return newNet
        
        # Use new deepcopy method
        return MCNeuralNetwork.CopyNet(self)

    ## Custom SUNN Methods
    def fit(self, xArr:np.ndarray, yArr:np.ndarray, Ieta:int = 9, Beta:int = 50, 
                  Gamma:int = 3, ScoreFunction = None, verbose:bool = True, 
                  zeroIsBetter:bool = True, scoreType:str = 'r2', useMP = False, 
                  useFast=True):
        """
        Returns in-place a trained version of the model relative to the given X & Y data
        
        Instead of always using the first net that has any improvement from the previous best net,
        a 'batch' of nets are generated from the parent net, then tested to find the best of those.
        That is, this essentially does a search with depth 'Beta' to find the next best net
        iteration, and is not a best-first training method.

        ## Inputs

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

        Gamma:
            - Sets the learning factor exponential decay rate. That is, every 'gamma' number 
            of iterations, the learning rate will decrease by a factor of 1/2. 

        ScoreFunction:
            - The fuction used to "score" nets alonge training, the default method (None) uses
              an R^2 calculation.
            - Other functions are accepted given:
                - A net that does better should give a higher "score"
                - Function input should be (net, Xarray, Yarray)

        Verbose:
            - Turns off printing the current progress and best R^2 value during training.

        zeroIsBetter:
            - Changes objective to getting a lower value score. Changes to False when using
              a scoring function like R^2 or RAE where higher is better.
            - NOTE: if using an external scoring function, you will have to change this
              value manually according to how that function works.

        scoreType:
            - Possible score methods in netMetrics: ['r2', 'sse', 'mae', 'mre', 'rae']

        useMP:
            - Allows the use of multiprocessing when testing net batches
            - This may harm preformance for a lower number of samples
        """

        # Update to models # inputs/outputs to the given data
        if self.fitted == False:
            if self.inSize == -1:
                # Handling for 1D and 2D y data
                if len(xArr.shape) > 1:
                    self.inSize = np.size(xArr, axis=1)
                else:
                    self.inSize = 1
            
            if self.outSize == -1:
                # Handling for 1D and 2D y data
                if len(yArr.shape) > 1:
                    self.outSize = np.size(yArr, axis=1)
                    if self.outSize != 1:
                        print(f"# of Y features found to be {self.outSize} and not 1 per sample. No action required if this is correct.")
                else:
                    self.outSize = 1

            # Regenerate the counts parameter with data size from x/y
            if self.sizes[0] == -1:
                self.sizes[0] = self.inSize
            if self.sizes[-1] == -1:
                self.sizes[-1] = self.outSize

            # Setup weights if not yet done
            if self.weights == None:
                self.weights = []
                for i in range(len(self.sizes) - 1):
                    # Initialize weights with some random weights in range [-0.5, 0.5]
                    newWeightsArr = (np.random.rand(self.sizes[i], self.sizes[i+1]) - 0.5)
                    self.weights.append(newWeightsArr)

                # Calculate the number of Parameters in the Weights/net
                weightSizes = [M.size for M in self.weights]
                self.parameters = sum(weightSizes)

            # Setup SUAs
            if self.SUA == None:
                self.setSUAs()

            # Test Calculation Time (for __str__)
            if type(self.speed) == str:
                t1 = pc()
                for _ in range(3):  # Do just 3 iterations
                    self.calculate_su(np.ones((self.inSize)))
                t2 = pc()
                self.speed = format((t2-t1)/3, ".2e")

            # Complete fit setup
            self.fitted = True

        # Pandas DataFrame and Series handling
        Tx = str(type(xArr))
        if Tx in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            xArr = xArr.to_numpy()

        Ty = str(type(yArr))
        if Ty in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            yArr = yArr.to_numpy()

        # Verify data types
        if type(self) == SUNN and type(xArr) == np.ndarray and type(yArr) == np.ndarray:
            # print("All data is good")
            pass
        else:
            raise ValueError("An input data type is not correct")

        # Note if zeroIsBetter needs to be flipped for 'r2' or 'rae'
        if scoreType in ['r2', 'rae']:
            zeroIsBetter = False

        # Scoring constant (flips equality)
        if zeroIsBetter:
            SK = 1
        else:
            SK = -1

        # Get inital accuracy
        if ScoreFunction == None:
            currentScore = self.score(xArr, yArr, method=scoreType)
        else:
            currentScore = ScoreFunction(self, xArr, yArr)

        # Generational Training method
        for I in range(Ieta):
            # Get current tweak amplitude
            twk = 2 ** (-I / Gamma)

            # Get new mutated test nets
            testNets = []
            for n in range(Beta):
                newNet = self.CopyNet()
                newNet.TweakWeights(twk)
                testNets.append(newNet)

            # Get the offspring's scores
            if len(xArr)*Beta >= 110000 and useMP: # Multiprocessing method
                newNetScores = multiNetScore(testNets, xArr=xArr, yTrue=yArr, scoreType=scoreType, useFast=True)
                
            else: # Normal method
                newNetScores = []
                for mutNet in testNets:
                    if ScoreFunction == None:
                        newScore = mutNet.score(xArr, yArr, method=scoreType)
                    else:
                        newScore = ScoreFunction(mutNet, xArr, yArr)
                    newNetScores.append(newScore)

            # See if the best score is an improvement
            if SK == 1:
                batchBestScore = min(newNetScores)
            else:
                batchBestScore = max(newNetScores)
            if SK*batchBestScore < SK*currentScore:
                # Actions for improvement
                bestIndex = newNetScores.index(batchBestScore)
                self.weights = testNets[bestIndex].weights
                self.SUA = testNets[bestIndex].SUA
                currentScore = batchBestScore

            # Update fancy progress bar stuff
            if verbose:
                pix1 = '='
                pix2 = '-'
                donePercent = (I + 1) / (Ieta)
                barLength = 40
                pix1Len = round(barLength * donePercent)
                pix2Len = barLength - pix1Len
                print(f"{scoreType.upper()} Score: {currentScore:.6f} | Training: {pix1*pix1Len}{pix2*pix2Len}", end='\r')
        
        # Finish with returning best net and its R2 value
        if verbose:
            print(f"{scoreType.upper()} Score: {currentScore:.6f} | Training: {'=='*20}                   ")

    def setSUAs(self):
        # Set up first SU (Super Unique) AF arrays
        self.SUA = []
        # Using structured form (for easier activation function usefulness decoding)
        if self.structured:
            for i, si in enumerate(self.sizes):
                # Force input arrays to have input activation
                if self.inputActivation != None and i == 0:
                    self.SUA.append(np.ones(shape=(si)) * list(self.SUmap.keys())[list(self.SUmap.values()).index(self.inputActivation)])

                # Force output arrays to have output activation
                elif self.outputActivation != None and i == len(self.sizes)-1:
                    self.SUA.append(np.ones(shape=(si)) * list(self.SUmap.keys())[list(self.SUmap.values()).index(self.outputActivation)])

                # Hidden layers
                else:
                    layer = []
                    i = 0
                    while len(layer) < si:
                        if i >= len(self.SUmap.keys()):
                            i = 0
                        key = list(self.SUmap.keys())[i]
                        layer.append(key)
                        i += 1
                    layer.sort()
                    layer = np.array(layer)
                    layer.reshape(si)
                    self.SUA.append(layer)


        # Randomized SUA setup
        else:
            for i, si in enumerate(self.sizes):
                # Force input arrays to have input activation
                if self.inputActivation != None and i == 0:
                    self.SUA.append(np.ones(shape=(si)) * list(self.SUmap.keys())[list(self.SUmap.values()).index(self.inputActivation)])

                # Force output arrays to have output activation
                elif self.outputActivation != None and i == len(self.sizes)-1:
                    self.SUA.append(np.ones(shape=(si)) * list(self.SUmap.keys())[list(self.SUmap.values()).index(self.outputActivation)])

                # Hidden layers
                else:
                    self.SUA.append(np.random.randint(1, len(list(self.SUmap.keys()))+1, size=(si)))

    def calculate_su(self, xi):
        """
        Returns the model's prediction for a given X sample
        """
        # Basic check - Shape vector to what it should be
        # This should break if bad input size/shape/type is given
        # but thats fine lol >.<
        xi = xi.reshape((self.inSize))

        # Calculation for all layers (first moved into here)
        library = zip(list(self.SUmap.keys()), list(self.SUmap.values()))
        for i in range(len(self.sizes)):
            # # Non-First layer things
            if i > 0:
                xi = np.dot(xi, self.weights[i-1])

            # New SU AF handling
            for key, AFtype in library:
                # Get indicies to affect / first SUA
                afInds = np.where(self.SUA[i] == key)

                # Apply AF type where indicies say to
                xi[afInds] = applyActi(xi[afInds], AFtype)

        # Return the vector shaped as a column vector
        if xi.size == 1:     # if a single number
            return xi[0]
        else:                      # all other vectors
            xi = xi.reshape(xi.size, 1)
            return xi

    def TweakSUA(self, amplitude):
        """
        Version of the tweak function to modify the single-node AF
        values applied. Ampltiude [1, 0] is (approximately) the percent of
        nodes that will change AF type.
        """

        # Iterate over all of the SU arrays
        for i in range(len(self.SUA)):
            if (self.inputActivation != None and i == 0) or (self.outputActivation != None and i == len(self.SUA)-1):
                continue
            elif i not in [0, len(self.SUA)-1] and self.structured:
                continue
            else:
                # Make a map of true/false from random function in array that matches
                # a given SUA as self.SUA is being iterated over and find the indicies
                # that are randomly >= amplitude to select for changing
                tfmap = np.random.random_sample(self.SUA[i].shape)
                changeIndicies = np.where(tfmap < amplitude)
                del tfmap

                # Make a new random SUA to choose from
                newSUA = np.random.randint(1, len(list(self.SUmap.keys()))+1, size=self.SUA[i].shape)

                # Use the above indicies to apply random.choice(list(self.SUAmap.values()))
                # on those locations, applying a AF type change to there
                self.SUA[i][changeIndicies] = newSUA[changeIndicies]


# SOUP Regressor (Sub-Ordinary Universal Polynomial)
class MCSoupRegressor:
    def __init__(self, coef_bounds = (-1, 1), use_tan=False, round_threshold=1e-5):
        # Desc
        """
        ## Sub-Ordinary Universal Polynomial

        Creates a large fittable funtion/""polynomial"" for every X feature given in .fit

        - coef_bounds
            - These are the min and max bounds that the coefficients (k_i) in f(x) can take.
              Feel free to experiment with various ranges, though (-1, 1) tends to work just fine.
        - use_tan
            - Three TAN(x) terms are included in f(x), but due to the asymptotic nature of TAN, they
              can actually hurt model preformance. So, this is disabled by default, but left as a
              setting to try anyways.
        - round_threshold
            - When adjusting the coefficients of the model, if a single coefficient's magnitude falls
              below this threshold, it is rounded to 0. This makes it easier for the model to completely
              remove terms from its various f(x) equations if it finds that is better.


        ## Technical Breakdown

        For each column of (Normalized!) data, generates a function of best fit of the form:

        f(x) = k0 + k1*(|x|**0.5) + k2*(x) + k3*(x**2) + k4*sin(x/3) + k5*sin(x) + k6*sin(3x) + 

               k7*cos(x/3) + k8*cos(x) + k9*cos(3x) + k10*tan(x/3) + k11*tan(x) + k12*tan(3x) + 

               k13*e**(x/3) + k14*e**(x) + k15*e**(3x) + k16*e**(-x/3) + k17*e**(-x) + k18*e**(-3x)

        There is an f(x) for every x feature. This means the net model is:

        F(x) = SUM[f_i(x)] for i=[0, 1, ..., (# of features - 1)]

        And no, I will not write it out more than that. You can see how large one f(x) alone is!

        TODO:
        - Make classifier (just make another class that wraps sig(x) around f(x))
        - Add more function parts!
        - Function customization?
            - Add filter on which parts to ignore if any
        """

        # Unchanging attributes
        self.FUNCTION_LENGTH = 19
        self.USE_TAN = use_tan
        self.ROUND = round_threshold

        # Changable attributes
        self._coefs = 0
        self.coef_bounds = coef_bounds
        self.num_features = 0
        self.parameters = 0
        self.fitted = False

    ## coefs Handling ##
    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, new_coefs):      
        self._coefs = new_coefs.copy()
        self._coefs[np.abs(self._coefs) < self.ROUND] = 0

    ## Model Functions ##
    def predict(self, X:np.ndarray, run_debug=False):
        """
        Calculates each ungodly f(x) described in the __init__ for each row in X.

        (Actually iterates over columns/features to speed things up)
        """

        # Verify the shape of X (and num_features)
        if run_debug:
            if len(X.shape) == 1 and self.num_features > 1:
                raise ValueError(f"Expected X array shape of ({len(X)}, {self.num_features}), got {X.shape}")
            elif len(X.shape) > 1 and X.shape[1] != self.num_features:
                raise ValueError(f"Expected X array shape of ({len(X)}, {self.num_features}), got {X.shape}")
            
        # Main function, per feature
        def f(x, col_index):
            """Yes this is f(x) from above. Rip readability *shrug*"""

            # Get function coefficients for this feature
            k = self.coefs[col_index].flatten()

            # Good lord
            return (k[0] + k[1]*(np.abs(x)**0.5) + k[2]*x + k[3]*(x**2) + k[4]*np.sin(x/3) + k[5]*np.sin(x) + k[6]*np.sin(3*x) + 
                    k[7]*np.cos(x/3) + k[8]*np.cos(x) + k[9]*np.cos(3*x) + self.USE_TAN*k[10]*np.tan(x/3) + self.USE_TAN*k[11]*np.tan(x) + self.USE_TAN*k[12]*np.tan(3*x) + 
                    k[13]*np.exp(x/3) + k[14]*np.exp(x) + k[15]*np.exp(3*x) + k[16]*np.exp(-x/3) + k[17]*np.exp(-x) + k[18]*np.exp(-3*x))
        
        # Calculate the sum described in INIT
        result = 0
        for col_index in range(self.num_features):
            result += f(X[:, col_index], col_index=col_index)

        return result

    def fit(self, X, Y, Ieta=100, Beta=25, Gamma=50, dropout=0.9, init_adj_max=2, verbose=True):
        """
        ## Function
        Adjusts the model's coefficients for N iterations. Returns the fitted model in-place.

        ## Inputs
        - X
            - The input data to make predictions with
        - Y
            - The data to test model outputs too
        - N
            - The number of iterations to run to attempt to improve the model
        - beta
            - Number of adjustments tested to the current best model, per iteration
        - gamma
            - Every gamma # of iterations, the scale of the adjustments made to the model
              coefficients are reduced by 1/2
        - dropout
            - (Approximately) The % of coefficients that are NOT adjusted per beta test. These are picked randomly.
            - Stay in school!
        - init_adj_max
            - The initial maximum amplitude that adjustments can make to an individual model coefficient.
              Having this much larger than the coefficient bounds makes finding improvements slower. Having
              this value be too small will cause not many meaningful adjustments to be made.
        - verbose
            - Whether or not an update of iteration # and current model score is printed (in one line) every
              10 iterations.
        """

        # Check if model initial fit complete
        if not self.fitted:
            # Generate the coefficients for each feature
            if len(X.shape) == 2:
                self.num_features = X.shape[1]
                self.coefs = np.random.rand(self.num_features, self.FUNCTION_LENGTH)
            elif len(X.shape) == 1:
                # Assume a singular feature
                self.num_features = 1
                self.coefs = np.random.rand(self.num_features, self.FUNCTION_LENGTH)
            else:
                raise ValueError(f"X Array Must be 1 or 2 Dimensional! Not {len(X.shape)}-Dimensional")
            
            # Confirm initial fit
            self.parameters = self.coefs.size
            self.fitted = True

        # Tweak params for N iterations
        score = r2d2(self.predict(X), Y)
        for itr in range(Ieta):
            add_mode = itr%2

            # Get dropout filter and apply it (discards changing drapout % num of coefficients)
            filt = np.random.rand(self.num_features, self.FUNCTION_LENGTH) > dropout

            # Get Tweak amplitude adjustment
            adjustments = []
            for _ in range(Beta):
                adjustments.append(filt * init_adj_max*(2**(-(itr) / Gamma)) * 2*(np.random.rand(self.num_features, self.FUNCTION_LENGTH) - 0.5))

            # Apply adjustment to test coeff array
            og_coefs = self.coefs.copy()
            test_scores = []
            for adj in adjustments:
                if add_mode:
                    self.coefs = np.clip(og_coefs+adj, self.coef_bounds[0], self.coef_bounds[1])
                else:
                    self.coefs = np.clip(og_coefs*adj, self.coef_bounds[0], self.coef_bounds[1])
                test_scores.append(self.score(X, Y))

            # Test new coef array score
            best_score = max(test_scores)
            if best_score > score:
                score = best_score

                best_adj = adjustments[test_scores.index(best_score)]

                if add_mode:
                    self.coefs = np.clip(og_coefs+best_adj, self.coef_bounds[0], self.coef_bounds[1])
                else:
                    self.coefs = np.clip(og_coefs*best_adj, self.coef_bounds[0], self.coef_bounds[1])

            else:
                self.coefs = og_coefs

            # Print status
            if verbose and (itr+1)%10 == 0:
                print(f"Iteration #{itr+1} | Score = {format(score, '.6f')}        ", end='\r')
        
        if verbose:
            print(f"Iteration #{itr+1} | Score = {format(score, '.6f')}        ")

    def score(self, X, Y, **kwargs):
        """
        Return the models R2 score to the given X and Y Data
        """
        return r2d2(self.predict(X), Y)

    def CopyNet(self):
        """
        Copy a model object to a new variable. Used in similar
        situations as NumPy's copy method to make an isolated net
        with the same values and properties as the parent net.
        """
        # Check that it is fitted
        if self.fitted == False:
            raise ValueError("Model is not yet fitted and therefore cannot be copied")

        return copy.deepcopy(self)


# SOUP Classifier (Sub-Ordinary Universal Polynomial)
class MCSoupClassifier:
    def __init__(self, coef_bounds = (-1, 1), use_tan=False, round_threshold=1e-5):
        # Desc
        """
        ## Sub-Ordinary Universal Polynomial

        Creates a large fittable funtion/""polynomial"" for every X feature given in .fit

        - coef_bounds
            - These are the min and max bounds that the coefficients (k_i) in f(x) can take.
              Feel free to experiment with various ranges, though (-1, 1) tends to work just fine.
        - use_tan
            - Three TAN(x) terms are included in f(x), but due to the asymptotic nature of TAN, they
              can actually hurt model preformance. So, this is disabled by default, but left as a
              setting to try anyways.
        - round_threshold
            - When adjusting the coefficients of the model, if a single coefficient's magnitude falls
              below this threshold, it is rounded to 0. This makes it easier for the model to completely
              remove terms from its various f(x) equations if it finds that is better.


        ## Technical Breakdown

        For each column of (Normalized!) data, generates a function of best fit of the form:

        f(x) = k0 + k1*(|x|**0.5) + k2*(x) + k3*(x**2) + k4*sin(x/3) + k5*sin(x) + k6*sin(3x) + 

               k7*cos(x/3) + k8*cos(x) + k9*cos(3x) + k10*tan(x/3) + k11*tan(x) + k12*tan(3x) + 

               k13*e**(x/3) + k14*e**(x) + k15*e**(3x) + k16*e**(-x/3) + k17*e**(-x) + k18*e**(-3x)

        There is an f(x) for every x feature. This means the net model is:

        F(x) = SUM[f_i(x)] for i=[0, 1, ..., (# of features - 1)]

        And no, I will not write it out more than that. You can see how large one f(x) alone is!

        TODO:
        - Make classifier (just make another class that wraps sig(x) around f(x))
        - Add more function parts!
        - Function customization?
            - Add filter on which parts to ignore if any
        """

        # Unchanging attributes
        self.FUNCTION_LENGTH = 19
        self.USE_TAN = use_tan
        self.ROUND = round_threshold

        # Changable attributes
        self._coefs = 0
        self.coef_bounds = coef_bounds
        self.num_features = 0
        self.parameters = 0
        self.fitted = False

    ## coefs Handling ##
    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, new_coefs):      
        self._coefs = new_coefs.copy()
        self._coefs[np.abs(self._coefs) < self.ROUND] = 0

    ## Model Functions ##
    def predict(self, X:np.ndarray, run_debug=False):
        """
        Calculates each ungodly f(x) described in the __init__ for each row in X.

        (Actually iterates over columns/features to speed things up)
        """

        # Verify the shape of X (and num_features)
        if run_debug:
            if len(X.shape) == 1 and self.num_features > 1:
                raise ValueError(f"Expected X array shape of ({len(X)}, {self.num_features}), got {X.shape}")
            elif len(X.shape) > 1 and X.shape[1] != self.num_features:
                raise ValueError(f"Expected X array shape of ({len(X)}, {self.num_features}), got {X.shape}")
            
        # Main function, per feature
        def f(x, col_index):
            """Yes this is f(x) from above. Rip readability *shrug*"""

            # Get function coefficients for this feature
            k = self.coefs[col_index].flatten()

            # Good lord
            return (k[0] + k[1]*(np.abs(x)**0.5) + k[2]*x + k[3]*(x**2) + k[4]*np.sin(x/3) + k[5]*np.sin(x) + k[6]*np.sin(3*x) + 
                    k[7]*np.cos(x/3) + k[8]*np.cos(x) + k[9]*np.cos(3*x) + self.USE_TAN*k[10]*np.tan(x/3) + self.USE_TAN*k[11]*np.tan(x) + self.USE_TAN*k[12]*np.tan(3*x) + 
                    k[13]*np.exp(x/3) + k[14]*np.exp(x) + k[15]*np.exp(3*x) + k[16]*np.exp(-x/3) + k[17]*np.exp(-x) + k[18]*np.exp(-3*x))
        
        # Calculate the sum described in INIT
        result = 0
        for col_index in range(self.num_features):
            result += f(X[:, col_index], col_index=col_index)

        # Classifier Addition
        result = sig(result)

        return result

    def fit(self, X, Y, Ieta=100, Beta=25, Gamma=50, dropout=0.9, init_adj_max=2, verbose=True):
        """
        ## Function
        Adjusts the model's coefficients for N iterations. Returns the fitted model in-place.

        ## Inputs
        - X
            - The input data to make predictions with
        - Y
            - The data to test model outputs too
        - N
            - The number of iterations to run to attempt to improve the model
        - beta
            - Number of adjustments tested to the current best model, per iteration
        - gamma
            - Every gamma # of iterations, the scale of the adjustments made to the model
              coefficients are reduced by 1/2
        - dropout
            - (Approximately) The % of coefficients that are NOT adjusted per beta test. These are picked randomly.
            - Stay in school!
        - init_adj_max
            - The initial maximum amplitude that adjustments can make to an individual model coefficient.
              Having this much larger than the coefficient bounds makes finding improvements slower. Having
              this value be too small will cause not many meaningful adjustments to be made.
        - verbose
            - Whether or not an update of iteration # and current model score is printed (in one line) every
              10 iterations.
        """

        # Check if model initial fit complete
        if not self.fitted:
            # Generate the coefficients for each feature
            if len(X.shape) == 2:
                self.num_features = X.shape[1]
                self.coefs = np.random.rand(self.num_features, self.FUNCTION_LENGTH)
            elif len(X.shape) == 1:
                # Assume a singular feature
                self.num_features = 1
                self.coefs = np.random.rand(self.num_features, self.FUNCTION_LENGTH)
            else:
                raise ValueError(f"X Array Must be 1 or 2 Dimensional! Not {len(X.shape)}-Dimensional")
            
            # Confirm initial fit
            self.parameters = self.coefs.size
            self.fitted = True

        # Tweak params for N iterations
        score = r2d2(self.predict(X), Y)
        for itr in range(Ieta):
            add_mode = itr%2

            # Get dropout filter and apply it (discards changing drapout % num of coefficients)
            filt = np.random.rand(self.num_features, self.FUNCTION_LENGTH) > dropout

            # Get Tweak amplitude adjustment
            adjustments = []
            for _ in range(Beta):
                adjustments.append(filt * init_adj_max*(2**(-(itr) / Gamma)) * 2*(np.random.rand(self.num_features, self.FUNCTION_LENGTH) - 0.5))

            # Apply adjustment to test coeff array
            og_coefs = self.coefs.copy()
            test_scores = []
            for adj in adjustments:
                if add_mode:
                    self.coefs = np.clip(og_coefs+adj, self.coef_bounds[0], self.coef_bounds[1])
                else:
                    self.coefs = np.clip(og_coefs*adj, self.coef_bounds[0], self.coef_bounds[1])
                test_scores.append(self.score(X, Y))

            # Test new coef array score
            best_score = max(test_scores)
            if best_score > score:
                score = best_score

                best_adj = adjustments[test_scores.index(best_score)]

                if add_mode:
                    self.coefs = np.clip(og_coefs+best_adj, self.coef_bounds[0], self.coef_bounds[1])
                else:
                    self.coefs = np.clip(og_coefs*best_adj, self.coef_bounds[0], self.coef_bounds[1])

            else:
                self.coefs = og_coefs

            # Print status
            if verbose and (itr+1)%10 == 0:
                print(f"Iteration #{itr+1} | Score = {format(score, '.6f')}        ", end='\r')
        
        if verbose:
            print(f"Iteration #{itr+1} | Score = {format(score, '.6f')}        ")

    def score(self, X, Y, **kwargs):
        """
        Return the models R2 score to the given X and Y Data
        """
        return r2d2(self.predict(X), Y)

    def CopyNet(self):
        """
        Copy a model object to a new variable. Used in similar
        situations as NumPy's copy method to make an isolated net
        with the same values and properties as the parent net.
        """
        # Check that it is fitted
        if self.fitted == False:
            raise ValueError("Model is not yet fitted and therefore cannot be copied")

        return copy.deepcopy(self)


## External Functions ##
def save_model(model, name:str):
    """
    Load the model object saved under the given name
    """
    dump(model, filename=name)

def load_model(path_or_name:str) -> MCNeuralNetwork:
    """
    Uses joblib function 'load' to recall a model object previous saved
    """
    try:
        return load(path_or_name)
    except:
        raise ValueError("Could not load a model with the given path/name!")

def Extend(baseNet:MCNeuralNetwork, d_height:int, imputeType:str = "zeros"):
    """
    Returns an MCNeuralNetwork that has its hidden layers height increased by d_height,
    with the base weight values given from baseNet. 
    The input and output layer sizes are not changed.

    ## Inputs

    1. baseNet
        - MCNeuralNetwork to use as a starting point for the parameters

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

def TTSplit(Xdata, Ydata, percentTrain:float = 70):
    """
    Universal Train-Test data split via random selection.

    Returns in the order xTrain, xTest, yTrain, yTest
    
    - percentTrain
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
    trainIndicies = sample(range(dataLen), numTrainSamples)
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

    return xTrain, xTest, yTrain, yTest

def dataSelect(X, Y, count:int):
    """
    **** Experimental training data selection method ****

    **** Only useable when X and Y are 1D ****

    Returns shortened x, y data arrays with (count) # of data points/rows

    Automatically thins a database by selecting "important" data points more often than
    "non-important data points". This probability comes from the local 2nd derivative
    magnitude that are around a data point.

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

def cross_val(model:MCNeuralNetwork, X, Y, N=5, scoreType='r2', return_models=False, 
              train_depth=2, verbose=True):
    """
    Cross validation of model using X and Y with N number of splits

    ## Params

    - N
        - Number of train/validation split tests to complete

    - scoreType
        - One of the score/error functions avalible in .score
        - ['r2', 'sse', 'mae', 'mse', 'rae']

    - return_models
        - If True, returns (models, weights) from the models tested and their
          weights derived from their relative overall score to the others.

    - train_depth
        - 5*train_depth = number of iterations each model is trained to

    - verbose
        - 0 = No output printing
        - 1 = Outputs current step/progress
        - 2 = Prints the current step and each models .fit output
    """

    if len(X) < N:
        raise ValueError(f"Not enough data for {N} cycles")

    # Verbose settings
    fit_verbose = False
    step_verbose = False
    if verbose == 1:
        fit_verbose = False
        step_verbose = True
    elif verbose == 2:
        fit_verbose = True
        step_verbose = True

    # Main loop
    if return_models:
        models = []

    model.fit(X, Y, Ieta=0, verbose=False)
    base_model = model.CopyNet()

    scores = []
    step = round(len(X) / N)
    for n in range(N):
        # Data Split
        start = n*step
        if n == N-1:
            stop = len(X)
        else:
            stop  = (n+1)*step

        X_val = X[start:stop+1]
        Y_val = Y[start:stop+1]

        if len(X.shape) > 1:
            X_train = np.row_stack([X[:start], X[stop+1:]])
        else:
            X_train = np.array(X[:start].tolist() + X[stop+1:].tolist())

        if len(Y.shape) > 1:
            Y_train = np.row_stack([Y[:start], Y[stop+1:]])
        else:
            Y_train = np.array(Y[:start].tolist() + Y[stop+1:].tolist())

        # Train model
        model = base_model.CopyNet()
        model.fit(X_train, Y_train, Ieta=round(5*train_depth), verbose=fit_verbose)

        if return_models:
            models.append(model)

        # Record model score
        scores.append(model.score(X_val, Y_val, method=scoreType))

        # Print step results
        if step_verbose:
            print(f"Cross-Validation: Step {n+1}/{N} Complete     ", end='\r')

    # Generate model weights if needed
    if return_models:
        weights = [m.score(X, Y) for m in models]
        weights = np.array(weights)
        weights /= np.sum(weights)
        weights = weights.tolist()

    # Finish
    if step_verbose:
        print(f"Cross-Validation: All Steps Completed")
        print(f"Mean Model {scoreType} Validation Score/Error = {np.mean(scores)}")

    if return_models:
        return models, weights
    else:
        return scores

def generate_model(X, Y, hidden_layers = [100], 
             test_activations=['lin', 'relu', 'sig', 'tanh', 'silu', 'dsilu', 'elu'],
             initial_activations='lin', train_split_percent=5, greedy=True, 
             target_score=0.99, iterations=5, Ieta=10, Beta=10, Gamma=2, 
             skip_current=True, verbose=True):

        """
        Automatically builds and optimizes an MCNeuralNetwork model to the given X/Y data via a round-robin style testing method. 

        Returns the best trained model according to R^2 scoring.

        ### Parameters
        - X
            - Input features
        - Y
            - Output features
        - hidden_layers
            - Number and height of each (if any) of the test models' hidden layers
        - test_activations
            - The list of activations to test during the round-robin testing
        - initial_activations
            - IF STR: uses this activation for all layers and finds improvements from there
            - IF LIST: uses the list as the initial model activations 
                - This allows further testing of a model later on if desired
        - train_split_percent
            - Percent of given X/Y data that goes into model training
            - The rest of data goes into validation data used for comparing model scores
            - The default is a very low amount for speed and to help the model find true patterns
        - greedy
            - Skips to the next layer if any improvement is made testing a current layer's activation
            - Might require a higher number of iterations if True
        - target_score
            - During testing, if the best model's score is at or above this value, training concludes
        - iterations
            - The number of cycles preformed during the round-robin testing
        - Ieta
            - Number of iterations to use for each model's .fit call
        - Beta
            - Batch size to use for each model's .fit call
        - Gamma
            - 'Learning rate' decay for each model's .fit call
        - skip_current
            - If the current test AF for a layer during training is the model's current AF for that layer,
              skip that test and move on the the next test AF for that layer
            - This helps prevent the model form getting stuck on an initial combination like all 'lin' layers.
        - verbose
            - If 0: no printed output
            - If 1: Single-line progress update
            - If 2: Prints a new line for every next step
        """

        # Initialize model
        num_layers = len(hidden_layers) + 2
        
        if type(initial_activations) == str:
            af_start = [initial_activations]*num_layers
        elif type(initial_activations) == list and len(initial_activations) == num_layers:
            af_start = initial_activations
        else:
            raise ValueError(f"initial_activations input type incorrect (check num of entries if a list)")

        model = MCNeuralNetwork(hidden_counts=hidden_layers, activations=af_start)

        # Make data split
        X, X_val, Y, Y_val = TTSplit(X, Y, percentTrain=train_split_percent)
        
        # Test initial model
        model.fit(X, Y, Ieta=Ieta, Beta=Beta, Gamma=Gamma, verbose=False)
        best_score = model.score(X_val, Y_val)
        best_model = model.CopyNet()

        # Do round-robin improvments
        num_test_models = iterations*num_layers*len(test_activations)
        i = 0
        for I in range(iterations):

            # Go layer-by-layer
            for li in range(num_layers):

                # Test other activations for this layer
                for afi, AF in enumerate(test_activations):
                    # Get current iteration number
                    i = (afi+1) + li*len(test_activations) + I*num_layers*len(test_activations)

                    # Check for str AF, not a function type
                    if type(AF) != str:
                        raise ValueError(f"A test activation is not of type str! (Note external function types not supported)")

                    # Apply AF
                    initial_af = model.activationFunction[li]
                    model.activationFunction[li] = AF.upper()

                    # Skip test AF if its the current one
                    if skip_current and initial_af.upper() == AF.upper():
                        continue

                    # Test model score
                    model.fit(X, Y, Ieta=Ieta, Beta=Beta, Gamma=Gamma, verbose=False)
                    test_score = model.score(X_val, Y_val)

                    if test_score > best_score:
                        # Keep model as best
                        best_score = test_score
                        best_model = model.CopyNet()
                        improved = True

                        # Check if score is at or beyond target threshold
                        if best_score >= target_score:
                            if verbose >= 1:
                                print(f"Iteration {num_test_models}/{num_test_models} | Best Score = {round(best_score, 4)} | Best Activations = {model.activationFunction}")
                                # print("== Model score >= target_score ==")
                                # print("== Generation Completed! ==\n")
                            return best_model
                    else:
                        model.activationFunction[li] = initial_af
                        improved = False

                    # Progress Update
                    if verbose == 1:
                        print(f"Iteration {i}/{num_test_models} | Best Score = {round(best_score, 4)} | Best Activations = {model.activationFunction}       ", end='\r')
                    elif verbose >= 2:
                        print(f"Iteration {i}/{num_test_models} | Best Score = {round(best_score, 4)} | Best Activations = {model.activationFunction}")

                    # Finish with layer if using greedy algorithm
                    if improved and greedy:
                        break

        # Final Progress Update
        if verbose >= 1:
            print(f"Iteration {i}/{num_test_models} | Best Score = {round(best_score, 4)} | Best Activations = {model.activationFunction}")
            # print("== Generation Completed! ==\n")

        return best_model

def run_module_tests(print_sub_steps=True) -> None:
    """
    Run the standardized module tests to verify base functionality. Returns nothing,
    all info is given in print statements.
    """
    
    # Step 1 (Data)
    ## Verify TTSplit and normalize
    X = np.random.rand(100, 3)
    X[:, 0] = 2*(X[:, 0] - 0.5)
    X[:, 1] = X[:, 1] + 0.67
    X[:, 2] = X[:, 2]**2 - 1.5*X[:, 2]
    Y = np.sum(X, axis=1)

    X, xms = normalize(X)
    Y, yms = normalize(Y)

    xt, xv, yt, yv = TTSplit(X, Y, percentTrain=75)

    print("Step 1 Done - TTSplit() and normalize() verified\n")

    # Step 2 (MCNeuralNetwork)
    model = MCNeuralNetwork([10, 10], activations=['lin', 'relu', 'silu', 'lin'])
    if print_sub_steps:
        print("Step 2 Part 1/4 Done (make MCNeuralNetwork)")

    model.fit(xt, yt, Ieta=8, Beta=25, Gamma=3, verbose=False, scoreType='mae')
    if print_sub_steps:
        print("Step 2 Part 2/4 Done (fit MCNeuralNetwork)")

    model.score(xv, yv)
    if print_sub_steps:
        print("Step 2 Part 3/4 Done (score MCNeuralNetwork)")

        print("Cross Validation Scores:", cross_val(model, X, Y, N=3, train_depth=1, verbose=False))
        print("Step 2 Part 4/4 Done (cross_val on MCNeuralNetwork)")
    print("Step 2 Done - MCNeuralNetwork verified\n")

    # Step 3 (MCSimpleEnsemble)
    model = MCSimpleEnsemble(n_estimators=4, hidden_counts=[10, 10], activations=['lin', 'relu', 'silu', 'lin'])
    if print_sub_steps:
        print("Step 3 Part 1/4 Done (make MCSimpleEnsemble)")

    model.fit(xt, yt, Ieta=8, Beta=25, Gamma=3, verbose=False, scoreType='mae')
    if print_sub_steps:
        print("Step 3 Part 2/4 Done (fit MCSimpleEnsemble)")

    model.score(xv, yv)
    if print_sub_steps:
        print("Step 3 Part 3/4 Done (score MCSimpleEnsemble)")

        print("Cross Validation Scores:", cross_val(model, X, Y, N=3, train_depth=1, verbose=False))
        print("Step 3 Part 4/4 Done (cross_val on MCSimpleEnsemble)")
    print("Step 3 Done - MCSimpleEnsemble verified\n")

    # Step 4 (SUNN)
    model = SUNN(hidden_counts=[10, 10], activations=['lin', 'silu', 'dsilu'])
    if print_sub_steps:
        print("Step 4 Part 1/4 Done (make SUNN)")

    model.fit(xt, yt, Ieta=8, Beta=25, Gamma=3, verbose=False, scoreType='mae')
    if print_sub_steps:
        print("Step 4 Part 2/4 Done (fit SUNN)")

    model.score(xv, yv)
    if print_sub_steps:
        print("Step 4 Part 3/4 Done (score SUNN)")

        print("Cross Validation Scores:", cross_val(model, X, Y, N=3, train_depth=1, verbose=False))
        print("Step 4 Part 4/4 Done (cross_val on SUNN)")
    print("Step 4 Done - SUNN verified\n")

    # Step 5 (MCSoupRegressor)
    model = MCSoupRegressor()
    if print_sub_steps:
        print("Step 5 Part 1/4 Done (make MCSoupRegressor)")

    model.fit(xt, yt, Ieta=100, Beta=25, Gamma=50, verbose=False)
    if print_sub_steps:
        print("Step 5 Part 2/4 Done (fit MCSoupRegressor)")

    model.score(xv, yv)
    if print_sub_steps:
        print("Step 5 Part 3/4 Done (score MCSoupRegressor)")

        print("Cross Validation Scores:", cross_val(model, X, Y, N=3, train_depth=1, verbose=False))
        print("Step 5 Part 4/4 Done (cross_val on MCSoupRegressor)")
    print("Step 5 Done - MCSoupRegressor verified\n")

    # Step 6 (MCSoupClassifier)
    model = MCSoupClassifier()
    if print_sub_steps:
        print("Step 6 Part 1/4 Done (make MCSoupClassifier)")

    model.fit(xt, yt, Ieta=100, Beta=25, Gamma=50, verbose=False)
    if print_sub_steps:
        print("Step 6 Part 2/4 Done (fit MCSoupClassifier)")

    soup_score = model.score(xv, yv)
    if print_sub_steps:
        print("Step 6 Part 3/4 Done (score MCSoupClassifier)")

        print("Cross Validation Scores:", cross_val(model, X, Y, N=3, train_depth=1, verbose=False))
        print("Step 6 Part 4/4 Done (cross_val on MCSoupClassifier)")
    print("Step 6 Done - MCSoupClassifier verified\n")

    # Step 7 (Save/Load)
    save_model(model, "soup")
    if print_sub_steps:
        print("Step 7 Part 1/4 Done (save_model on MCSoupClassifier)")

    soup = load_model("soup")
    if print_sub_steps:
        print("Step 7 Part 2/4 Done (load_model on MCSoupClassifier)")

    if abs(soup_score-soup.score(xv, yv)) < 1e-6:
        if print_sub_steps:
            print("Step 7 Part 3/4 Done (score check on loaded MCSoupClassifier)")
    else:
        print("Step 7 Part 3/4 Done Incorrectly! (loaded model score != true model score)")

    os.remove('soup')
    if print_sub_steps:
        print("Step 7 Part 4/4 Done (removed test save model 'soup')")
    print("Step 7 Done - save_model and load_model verified\n")



## Activation Functions ##
# Identity
def lin(calcVec): 
    return calcVec
# RELU-Like
def relu(calcVec): 
    calcVec[calcVec < 0] = 0
    return calcVec
def silu(calcVec):
    return calcVec / (1 + np.exp(-np.clip(calcVec, -7e2, None)))
# SIG-Like
def sig(calcVec):
    return 1 / (1 + np.exp(-calcVec))
def dsilu(calcVec):
    return sig(calcVec) * (1 + calcVec * (1 - sig(calcVec)))
def tanh(calcVec):
    return np.tanh(calcVec)
# Exponential-Like
def elu(calcVec):
    return 0.4 * (np.expm1(np.clip(calcVec, None, 7e2)))
# Logirithmic-like
def root(calcVec):
    return np.arctan(calcVec) * np.abs(calcVec)**0.5 / 1.5
# dRELU-Like
def sqr(calcVec):
    calcVec[calcVec <= 0] = 0
    calcVec[calcVec > 0] = 1
    return calcVec
def rnd(calcVec):
    calcVec = np.round(calcVec)
    return calcVec
# Chaotic / Experimental
def resu(calcVec):
    return np.abs(np.sin(calcVec))*calcVec
def resu2(calcVec):
    calcVec[calcVec < 0] = 0
    return abs(np.sin(calcVec))*calcVec
def exp(calcVec):
    return calcVec**2


## Helper/Smol Functions ##
def applyActi(calcVec:np.ndarray, activationFunction):
    """Applies an activation function to the given vector/array of data in calcVec."""

    # Use a given function (if it is an external function)
    if callable(activationFunction):
        return activationFunction(calcVec)

    # Doing this again to be safe
    else:
        activationFunction = activationFunction.upper()

    # AF Dictionary (avoids using match/case stuff and needing Python >= 3.10)
    AFDict = {
        "LIN": lin,
        "RELU": relu,
        "SILU": silu,
        "SIG": sig,
        "DSILU": dsilu,
        "TANH": tanh,
        "ELU": elu,
        "ROOT": root,
        "SQR": sqr,
        "RND": rnd,
        "RESU": resu,
        "RESU2": resu2,
        "EXP": exp
    }

    # Look for ones in dictionary
    # Using try/except for speed I thinks
    try:
        return AFDict[activationFunction](calcVec)
    except:
        raise ValueError(f"Given hidden function ({activationFunction}) is not of the avalible types!")

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

    # Check if needs to be flattened
    # if len(yTrue.shape) == 1 and len(yModel.shape) > 1:
    #     yModel = yModel.flatten()
    
    # if len(yTrue.shape) > 1:
    #     if yTrue.shape[1] == 1:
    #         yTrue = yTrue.flatten()
    #         yModel = yModel.flatten()
    yTrue = yTrue.reshape(yModel.shape)

    # R2 Calc
    yMean = np.mean(yTrue)
    RES = np.sum(np.clip(yTrue - yModel, -1e154, 1e154) ** 2)
    TOT = np.sum((yTrue - yMean) ** 2)
    if TOT == 0:
        TOT = np.inf

    return 1 - (RES / TOT)
    
def raeScore(yTrue:np.ndarray, yPred:np.ndarray):
    """
    ## Relative Average Error Score

    ### Preforms the following calculation sequence:

    #### 1 Element-wise Absolute Relative Error (ARE)

        - score_ij = |yTrue_ij - yPred_ij| / |yTrue_ij| -> [0, inf]

    or

        - score_ij = |yPred_ij| -> [0, inf] if yTrue_ij == 0

    #### 2 Get Average ARE

        - <score> = MEAN(score_array)

    #### 3 Convert to return in range [0, 1]

        - RAE = e^(-<score>)
    """

    # Get locations of zeros and all else
    zeroLocs = np.where(yTrue == 0)
    nonZeroLocs = np.where(yTrue != 0)

    # Get scores form zero locations and non-zero locations
    if len(zeroLocs[0]) > 0:
        score_zeros = np.mean(np.abs(yPred[zeroLocs]))
    else:
        score_zeros = 0

    if len(nonZeroLocs[0]) > 0:
        score_nonzeros = np.mean(np.abs(yTrue[nonZeroLocs] - yPred[nonZeroLocs]) / np.abs(yTrue[nonZeroLocs]))
    else:
        score_nonzeros = 0

    # Get score average
    avgScore = (score_zeros + score_nonzeros) / 2

    # Get/return RAE
    return np.e ** (-avgScore)

def normalize(array):
    """
    Returns the normalized array and the columns (mean, st. dev.)

    Normalizes a given array (per col.) such that each point is the z-score for the given columns mean and standard deviation.

    Columns with only two unique values are automatically converted to be just 0's and 1's
    """

    # Normalize depending on if many cols or one
    ms_data = []
    if len(array.shape) > 1:
        # Normalize col-by-col
        for ci in range(array.shape[1]):
            # Check if is a bool col
            if len(set(array[:, ci])) == 2:
                vals = set(array[:, ci])
                for i, val in enumerate(vals):
                    array[array == val] = i

                # Dont make M/S adjustments
                ms_data.append((0, 1))
                continue

            # Get Mean and STD
            M = np.mean(array[:, ci])
            S = np.std(array[:, ci])
            ms_data.append((M, S))

            # Adjust col in array
            array[:, ci] -= M
            array[:, ci] /= S
            
    else:
        # Normalize single-col array
        M = np.mean(array)
        S = np.std(array)
        ms_data.append((M, S))

        array -= M
        array /= S

    return array, ms_data


# Multiprocessing for getting many model's calculation to the same array
def multiNetScore(netList, xArr, yTrue, scoreType='r2', useFast=True, processes=10):
    # Get/return the model's scores
    pool = Pool(processes=processes)
    return pool.starmap(func=score_, iterable=[(net, xArr, yTrue, scoreType, useFast) for net in netList])

def score_(net, xArr, yTrue, scoreType, useFast=True):
    # Alt. predict form for MP variable handling
    return net.score(xArr=xArr, yTrue=yTrue, useFast=useFast, method=scoreType)


## Neat Paper/Other References ##
notes = """

1. SILU and dSILU activation functions
Stefan Elfwing, Eiji Uchibe, Kenji Doya,
Sigmoid-weighted linear units for neural network function approximation in reinforcement learning,
Neural Networks,
Volume 107,
2018,
Pages 3-11,
ISSN 0893-6080,
https://doi.org/10.1016/j.neunet.2017.12.012.
(https://www.sciencedirect.com/science/article/pii/S0893608017302976)

"""


## Old / Deprecated ##
def debug_calculate(self, inVector):
    """
    Slower form of model calculation used for more error handling / debugging

    Returns the neural net's calculation for a given input vector. If the net used has an input size
    of one, a single float value can be given (array type not required).

    ## Inputs:

    1 - inVector
    - Type == NumPy Array
    - Size == (netInputCount, 1) or (1, netInputCount)
    - Note: 
        - If the net input size is 1, then a float can be given and a vector form is not required.
    """  

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

            for i, wi in enumerate(self.weights):
                # Forward propogation
                calcVec = sum(calcVec*wi)

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

def build_model(X, Y, max_hidden_layers=2, base_height=50, 
                activation_tests=['lin', 'silu', 'dsilu', 'tanh', 'elu', 'root'],
                validation_size=0.3, training_depth=1,
                test_ends=False, verbose=True):

    ## Training settings ##
    model_layer_sizes = []
    model_activations = []
    input_activation = 'lin'
    output_activation = 'lin'

    I = round(10 * training_depth)
    B = round(10 * training_depth)
    G = round(1.5 * training_depth)

    x_train, x_val, y_train, y_val = TTSplit(X, Y, percentTrain=100*(1 - validation_size))


    ## Main loop (For First Layer Activation) ##
    if test_ends:
        initial_scores = []
        for i, l1 in enumerate(activation_tests):
            # Make test model
            model = MCNeuralNetwork(hidden_counts=[], activations=[l1, 'lin'])

            # Test model
            model.fit(x_train, y_train, Ieta=I, Beta=B, Gamma=G, verbose=False)
            initial_scores.append(model.score(x_val, y_val))

            # Progress
            if verbose:
                print(f"Step 0a Progress: {i+1}/{len(activation_tests)}     ", end='\r')
        
        if verbose:
            print(f"Step 0a Progress: {len(activation_tests)}/{len(activation_tests)} - Done!")
            print(f"Current Model Score = {max(initial_scores)}\n")

        # Get best input activation, add to overall model
        input_activation = (activation_tests[initial_scores.index(max(initial_scores))])


    ## Main loop (For Last Layer Activation) ##
    if test_ends:
        initial_scores = [max(initial_scores)]
        for i, l1 in enumerate(activation_tests[1:]):
            # Make test model
            model = MCNeuralNetwork(hidden_counts=[], activations=[input_activation, l1])

            # Test model
            model.fit(x_train, y_train, Ieta=I, Beta=B, Gamma=G, verbose=False)
            initial_scores.append(model.score(x_val, y_val))

            # Progress
            if verbose:
                print(f"Step 0b Progress: {i+1}/{len(activation_tests)}     ", end='\r')

        if verbose:
            print(f"Step 0b Progress: {len(activation_tests)}/{len(activation_tests)} - Done!")
            print(f"Current Model Score = {max(initial_scores)}\n")

        # Get best input activation, add to overall model
        output_activation = (activation_tests[initial_scores.index(max(initial_scores))])


    ## Main loop (For Hidden Layers) ##
    for layer in range(max_hidden_layers):
        scores = []

        # Add another hidden layer
        model_layer_sizes.append(base_height)

        # Test current layer for its best activation
        for i, l1 in enumerate(activation_tests):
            # Make test model
            model = MCNeuralNetwork(hidden_counts=model_layer_sizes, activations=[input_activation] + model_activations + [l1] + [output_activation])

            # Test model
            model.fit(x_train, y_train, Ieta=I, Beta=B, Gamma=G, verbose=False)
            scores.append(model.score(x_val, y_val))

            # Progress
            if verbose:
                print(f"Step {layer+1}/{max_hidden_layers} Progress: {i+1}/{len(activation_tests)}     ", end='\r')

        # Get best input activation, add to overall model
        model_activations.append(activation_tests[scores.index(max(scores))])
        print(f"Step {layer+1}/{max_hidden_layers} Progress: {len(activation_tests)}/{len(activation_tests)} - Done!")
        print(f"Current Model Score = {max(scores)}\n")


    ## Test Hidden Layer Sizes for Improvements


    ## Make, Train, and Return the Best Model ##
    if verbose:
        print(f"==== Fitting the Best Model Architecture ====")

    model = MCNeuralNetwork(hidden_counts=model_layer_sizes, activations=[input_activation] + model_activations + [output_activation])
    model.fit(x_train, y_train, Ieta=12, Gamma=2, verbose=verbose)
    val_score = model.score(x_val, y_val)

    if verbose:
        print(f"Validation Score = {val_score}")

    return model

def fastCalc(net:MCNeuralNetwork, inVector:np.ndarray):
    """
    **** Deprecated since version 1.4.3 ****
    
    **** Replaced by improved default calculation method ****

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
    ### Old Method
    # # Basic check - Shape vector to what it should be
    # # If this doesn't work, sounds like a you problem :P
    # inVector = inVector.reshape((net.inSize, 1))

    # # Possible fast calculations
    # if net.sizes[0] == 1 and len(net.sizes) == 2: # ~45% of regular calc time
    #     inVector = applyActi(inVector, net.activationFunction[0])
    #     inVector = applyActi(inVector*net.weights[0], net.activationFunction[1])

    # else:   # ~73% of regular calc time
    #     inVector = applyActi(inVector, net.activationFunction[0])
    #     for i in range(1, len(net.sizes)):
    #         if i != 0:
    #             inVector = inVector.reshape((inVector.size, 1))
    #         inVector = applyActi(sum(inVector*net.weights[i-1]), net.activationFunction[i])

    ### New Method
    inVector = inVector.reshape((net.inSize))
    inVector = applyActi(inVector, net.activationFunction[0])
    for i in range(1, len(net.sizes)):
        inVector = applyActi(np.dot(inVector, net.weights[i-1]), net.activationFunction[i])

    # Return the vector shaped as a column vector
    if inVector.size == 1:     # if a single number
        return inVector[0]
    else:                      # all other vectors
        inVector = inVector.reshape(inVector.size, 1)
        if len(inVector.shape) >= 3:
            raise ValueError("Calculation Vector size >= 3")
        return inVector

def netMetrics(net, xArr:np.ndarray, yArr:np.ndarray, 
               method:str = 'sse', useFast:bool = True):
    """
    **** Deprecated since version 1.4.0 ****
    
    **** All options moved to the .score method ****

    Returns the nets error/score from one of a few methods

    method:
        - 'r2' returns the R^2 fit of the model
        - 'sse' returns the Sum Squared Error
        - 'mae' returns the Mean Absolute Error
        - 'rae' returns a custom value similar to r2 from raeScore()
    """

    # Get predictions
    if type(net) in [MCNeuralNetwork, SUNN]:
        yHat = net.predict(xArr, useFast=useFast)
    else:
        raise ValueError(f"Unrecognized model type: {type(net)}")

    ## R^2 Method ##
    if method == 'r2':
        return np.clip(r2d2(yHat, yArr), -1, 1)

    ## Sum Squared Error ##
    elif method == 'sse':
        # Get SSE
        return np.sum((yHat - yArr)**2)

    ## Mean Absolute Error ##
    elif method == 'mae':
        mae = (1/len(yHat)) * np.sum(np.abs(np.array(yHat) - yArr))
        return mae

    elif method == 'rae':
        return raeScore(yArr, yHat)

    # Raise error if one of the possible methods was not given
    else:
        raise ValueError(f"Given score type {method} is not one of the avalible types.")

def Forecast(Net:MCNeuralNetwork, inputData, comparisonData=[], plotResults=False, useFast:bool=True):
    """
    **** Deprecated since version 1.4.0 ****

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
    if type(Net) not in [str, MCNeuralNetwork]:
        raise TypeError(f"netName should be a string or MCNeuralNetwork! Not {type(Net)} ")
    if type(Net) == str:
        net = load_model(Net)
    elif type(Net) == MCNeuralNetwork:
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
    if inSize > 1: # V1.4.0 I think this helps, might just yeet the function though
        xPlot = xData[:, 0]

    if plotResults:
        plt.cla() # Clear anything from ealier training plots if applicable

        # plt.plot([*range(len(forPlot))], forPlot)
        plt.plot(xPlot, forPlot)

        if len(forPlot) == np.size(comparisonData):

            # plt.plot([*range(np.size(comparisonData))], comparisonData, "--")
            plt.plot(xPlot, comparisonData, "--")

        plt.legend(["Model Values", "Comparison Values"])
        plt.title("Net Predictions")
        plt.show()

    # Return forecast values
    return predictions

def ExpTrain(self, XData, YData, Ieta:int = 5, Gamma:float = 1, Beta:int = 20,
              subIeta:int = 3, subGamma:int = 1, subBeta:int = 30,
              verbose:bool = True, useFast:bool = True):
        """
        **** Deprecated since version 1.4.0 ****

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

def oldLoadNet(name:str):
    """
    **** Deprecated since version 1.4.0 ****

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
    net = MCNeuralNetwork(inSize, hiddenSize, outSize, strActivations)

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

def oldSaveNN(Net:MCNeuralNetwork, name: str):
        """
        **** Deprecated since version 1.4.0 ****

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
        np.savetxt(f"{DIR}/NN_{name}_Size", Net.sizes)
        for i in range(len(Net.sizes) - 1):
            # Redundant save cycles in case of 'onedrive' situations
            ## Limit save tries to 10 times, if thats not working
            ## something else is likely wrong that this won't fix
            for tries in range(10):
                try:
                    np.savetxt(f"{DIR}/NN_{name}_Weights{i+1}", Net.weights[i])
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
        for function in Net.activationFunction:
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

def oldFastCalc(net:MCNeuralNetwork, inVector:np.ndarray):
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

def genTrain(net:MCNeuralNetwork, xArr:np.ndarray, yArr:np.ndarray, iterations:int = 1000, 
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
    if type(net) == MCNeuralNetwork and type(xArr) == np.ndarray and type(yArr) == np.ndarray:
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

def OldTrain(Net:MCNeuralNetwork, inputData, yData, startingTweakAmp=0.8, 
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
    if type(Net) not in [str, MCNeuralNetwork]:
        raise TypeError(f"netName should be a string or AdvNet! Not {type(Net)} ")
    if type(Net) == str:
        net = load_model(Net)
    elif type(Net) == MCNeuralNetwork:
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

def CycleTrain(Net:MCNeuralNetwork, inputData, yData, startingTweakAmp=0.8, 
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
