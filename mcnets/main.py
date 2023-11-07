# from matplotlib import pyplot as 
plt = None # Final dependancy (Forecast) Deprecated in V1.4.0
import numpy as np
import textwrap as txt
import os
from time import perf_counter as pc
import random as rn
from joblib import load, dump


# Primary Regressor
class MCRegressor:
    ## Default Functions ## 
    def __init__(self, hiddenCounts:list = [25,], activations: any = 'DEFAULT'):
        """
        Create a deep neural network from the given counts of neurons requested per layer. By default,
        uses a SILU activation function on all hidden layers and a Linear/Identity on the input and
        output layers.

        ## Inputs:

        - hiddenCounts
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

        model = mc.MCRegressor(hiddenCounts=[10], activations=['atan', 'relu', 'lin'])
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

        model = mc.MCRegressor(hiddenCounts=[10], activations=['atan', 'relu', 'lin'])

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
        self.hiddenSize = hiddenCounts
        self.sizes = [self.inSize] + hiddenCounts + [self.outSize]

        self.parameters = "Not yet generated; call .fit first"
        self.speed = "Not yet generated; call .fit first"
        self._weights = None

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

    ## Weights Handling ##
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, newWeightList):
        # Apply to main weights attribute
        self._weights = [wi.copy() for wi in newWeightList]

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

            # Make sure values are in the range [-1, 1]
            self.weights[i] = np.clip(self.weights[i], -1, 1)

        # Note change in weight array after clipping
        if returnChange:
            dW = [(self.weights[i] - Wo[i]) for i in range(len(self.weights))]
            return dW

    def Calculate(self, inVector, useFast:bool = True):
        """
        Returns the neural net's calculation for a given input vector. If the net used has an input size
        of one, a single float value can be given (array type not required).

        ## Inputs:

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
        if type(self.activationFunction) == list:
            newActivations = self.activationFunction.copy()
        else:
            newActivations = self.activationFunction
        newSizes = [newInputCount] + newHiddenCounts + [newOutputCount]

        # Make net net shell
        newNet = MCRegressor(newHiddenCounts, newActivations)
        newNet.inSize = newInputCount
        newNet.outSize = newOutputCount
        # newNet.hiddenSize = newHiddenCounts
        newNet.sizes = newSizes
        newNet.activationFunction = newActivations
        
        # Setup weights
        newNet.weights = self.weights
            
        # Return the copied net
        return newNet

    def ApplyTweak(self, dW):
        for i, dWi in enumerate(dW):
            self.weights[i] = self.weights[i] + dWi

    def fit(self, xArr:np.ndarray, yArr:np.ndarray, Ieta:int = 9, 
                  Beta:int = 50, Gamma:int = 3, ScoreFunction = None,
                  Verbose:bool = True, useFast:bool = True, 
                  zeroIsBetter:bool = True, scoreType:str = 'sse'):
        """
        Returns a trained version of the net object to the relative to the given X & Y data
        
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

        zeroIsBetter:
            - Changes objective to getting a lower value score. Changes to False when using
              a scoring function like R^2 or RAE where higher is better.
            - NOTE: if using an external scoring function, you will have to change this
              value manually according to how that function works.

        scoreType:
            - Possible score methods in netMetrics: ['r2', 'sse', 'mae', 'rae']
        """

        # Update to models # inputs/outputs to the given data
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
                print(f"# of Y features found to be {np.size(yArr, axis=1)} and not 1 per sample. No action required if this is correct.")
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

        # Test Calculation Time (for __str__)
        if type(self.speed) == str:
            t1 = pc()
            for _ in range(3):  # Do just 3 iterations
                self.Calculate(np.ones((self.inSize, 1)))
            t2 = pc()
            self.speed = format((t2-t1)/3, ".2e")

        # Pandas DataFrame and Series handling
        Tx = str(type(xArr))
        if Tx in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            xArr = xArr.to_numpy()

        Ty = str(type(yArr))
        if Ty in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            yArr = yArr.to_numpy()

        # Verify data types
        if type(self) in [MCRegressor, SUNN] and type(xArr) == np.ndarray and type(yArr) == np.ndarray:
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
                currentScore = batchBestScore

            # Update fancy progress bar stuff
            if Verbose:
                pix1 = '='
                pix2 = '-'
                donePercent = (I + 1) / (Ieta)
                barLength = 40
                pix1Len = round(barLength * donePercent)
                pix2Len = barLength - pix1Len
                print(f"{scoreType.upper()} Score: {currentScore:.6f} | Training: {pix1*pix1Len}{pix2*pix2Len}", end='\r')
        
        # Finish with returning best net and its R2 value
        if Verbose:
            print(f"{scoreType.upper()} Score: {currentScore:.6f} | Training: {'=='*20}                   ")

    def predict(self, X, useFast:bool = True):
        """
        Calculates and returns the net outputs for all the given X input data
        """
        # Pandas DataFrame and Series handling
        T = str(type(X))
        if T in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            X = X.to_numpy()

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
            raise TypeError(f"Unrecognized X input data type (Is {type(X)})!")

        # Calculate predictions for given inputs
        predictions = []
        for i in range(dataSize):
            # Generate net input vector
            invec = np.array(X[i]).reshape(1,inSize)

            # Get net prediciton
            val = self.Calculate(invec, useFast=useFast)
            predictions.append(val)
        
        return np.array(predictions)

    def score(self, xArr, yTrue, useFast:bool = True, method:str = 'r2'):
        """
        Returns the nets error/score from one of a few methods

        method:
            - 'r2' returns the R^2 fit of the model
            - 'sse' returns the Sum Squared Error
            - 'mae' returns the Mean Absolute Error
            - 'rae' returns a custom value similar to r2 from raeScore()
        """

        # Get predictions
        if type(self) == MCRegressor:
            yHat = self.predict(xArr, useFast=useFast)
        elif type(self) == SUNN:
            yHat = self.predict_su(xArr)
        else:
            raise ValueError(f"Unrecognized model type: {type(self)}")

        ## R^2 Method ##
        if method == 'r2':
            return np.clip(r2d2(yHat, yTrue), -1, 1)

        ## Sum Squared Error ##
        elif method == 'sse':
            return np.sum((yHat - yTrue)**2)

        ## Mean Absolute Error ##
        elif method == 'mae':
            return (1/len(yHat)) * np.sum(np.abs(np.array(yHat) - yTrue))

        elif method == 'rae':
            return raeScore(yTrue, yHat)

        # Raise error if one of the possible methods was not given
        else:
            raise ValueError(f"Given score type {method} is not one of the avalible types.")

    def save(self, pathOrName:str):
        """
        Uses joblib function 'dump' to save the model object with the given
        file path or name.
        """
        dump(self, pathOrName)


# Primary Classifier


# SUNN (Super Unique Neural Network) *BETA*
class SUNN():
    def __init__(self, hiddenCounts: list = [25,], activations: list = ['lin', 'relu', 'sig', 'silu', 'dsilu', 'elu'],
                 inputActivation:str = 'lin', outputActivation:str = 'lin'):
        """
        A Neural Network regressor that takes using many activation functions (AFs) to
        the extreme. By applying a seperate AF at *every* node, some very unique behvaiors
        can be generated. 

        - hiddenCounts:
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

        - inputActivation
            - Force the input nodes to have this activation function
            - Set to None to disable

        - outputActivation
            - Force the output nodes to have this activation function
            - Set to None to disable
        """

        # Sizes (inSize and outSize not yet initiated)
        self.inSize = -1
        self.outSize = -1
        self.hiddenSize = hiddenCounts
        self.sizes = [self.inSize] + hiddenCounts + [self.outSize]

        # Activation Function (AF) Stuff
        self.activationFunction = activations
        self.inputActivation = inputActivation
        self.outputActivation = outputActivation

        if self.inputActivation not in self.activationFunction and self.inputActivation != None:
            self.activationFunction.append(self.inputActivation)
        if self.outputActivation not in self.activationFunction and self.outputActivation != None:
            self.activationFunction.append(self.outputActivation)

        # Not yet generatable stuff
        self.parameters = "Not yet generated; call .fit first"
        self.speed = "Not yet generated; call .fit first"
        self._weights = None
        self._SUA = None

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

    def predict(self, xarray):
        return MCRegressor.predict(self, xarray)

    def TweakWeights(*args):
        return super(type(SUNN.TweakWeights), MCRegressor.TweakWeights(*args))

    def score(self, xarray, yarray, method:str = 'r2', useFast=True):
        return MCRegressor.score(self, xarray, yarray, useFast=True, method=method)

    ## Custom SUNN Methods
    def fit(self, xArr:np.ndarray, yArr:np.ndarray, Ieta:int = 9, Beta:int = 25, 
                  Gamma:int = 3, ScoreFunction = None, Verbose:bool = True, 
                  zeroIsBetter:bool = True, scoreType:str = 'sse'):
        """
        Returns a trained version of the net object to the relative to the given X & Y data

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

        zeroIsBetter:
            - Changes objective to getting a lower value score. Changes to False when using
              a scoring function like R^2 or RAE where higher is better.
            - NOTE: if using an external scoring function, you will have to change this
              value manually according to how that function works.

        scoreType:
            - Possible score methods in netMetrics: ['r2', 'sse', 'mae', 'rae']
        """

        # Update to models # inputs/outputs to the given data
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
                print(f"# of Y features found to be {np.size(yArr, axis=1)} and not 1 per sample. No action required if this is correct.")
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
                self.calculate_su(np.ones((self.inSize, 1)))
            t2 = pc()
            self.speed = format((t2-t1)/3, ".2e")

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
                newNet = self.copy_su()
                newNet.TweakWeights(twk)
                testNets.append(newNet)

            # Get the offspring's scores
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
            if Verbose:
                pix1 = '='
                pix2 = '-'
                donePercent = (I + 1) / (Ieta)
                barLength = 40
                pix1Len = round(barLength * donePercent)
                pix2Len = barLength - pix1Len
                print(f"{scoreType.upper()} Score: {currentScore:.6f} | Training: {pix1*pix1Len}{pix2*pix2Len}", end='\r')
        
        # Finish with returning best net and its R2 value
        if Verbose:
            print(f"{scoreType.upper()} Score: {currentScore:.6f} | Training: {'=='*20}                   ")

    def setSUAs(self):
        # Set up first SU (Super Unique) AF arrays
        self.SUA = []
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
        Calculate function for the SUNN model
        """
        # Basic check - Shape vector to what it should be
        # This should break if bad input size/shape/type is given
        # but thats fine lol >.<
        xi = xi.reshape((self.inSize, 1))

        # Calculation for all layers (first moved into here)
        library = zip(list(self.SUmap.keys()), list(self.SUmap.values()))
        for i in range(len(self.sizes)):
            # Non-First layer things
            if i != 0:
                # Keep column vector shape
                xi = xi.reshape((xi.size, 1))

                # Do matrix multiplication as per usual
                xi = sum(xi*self.weights[i-1])

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

    def predict_su(self, xarray):
        """
        Version of the predict function for the SUNN model
        """
        # Pandas DataFrame and Series handling
        T = str(type(xarray))
        if T in ["<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>"]:
            xarray = xarray.to_numpy()

        # Check that input X is array and correct shape
        if type(xarray) == np.ndarray:
            # Check for the correct amount of inputs
            if len(xarray.shape) > 1:
                inSize = np.size(xarray, 1)
            else:
                inSize = 1

            dataSize = np.size(xarray, 0)

            if inSize != self.inSize:
                raise ValueError(f"Given # of unique inputs {inSize} does not match the net input size {self.inSize}!")
            
        else:
            raise TypeError(f"Unrecognized xarray input data type (Is {type(xarray)})!")

        # Calculate predictions for given inputs
        predictions = []
        for i in range(dataSize):
            # Generate net input vector
            invec = np.array(xarray[i]).reshape(1,inSize)

            # Get net prediciton
            val = self.calculate_su(invec)
            predictions.append(val)
        
        return np.array(predictions)

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

    def copy_su(self):
        # New Net Matrix/Stuff Sizes (These aren't actually outside of SUNN's init)
        newInputCount = self.inSize
        newOutputCount = self.outSize
        newHiddenCounts = self.hiddenSize
        if type(self.activationFunction) == list:
            newActivations = self.activationFunction.copy()
        else:
            newActivations = self.activationFunction
        newSizes = [newInputCount] + newHiddenCounts + [newOutputCount]

        # Make net net shell
        newNet = SUNN(newHiddenCounts, newActivations)
        newNet.inSize = newInputCount
        newNet.outSize = newOutputCount
        # newNet.hiddenSize = newHiddenCounts
        newNet.sizes = newSizes
        newNet.activationFunction = newActivations
        newNet.inputActivation = self.inputActivation
        newNet.outputActivation = self.outputActivation
        
        # Setup weights / SUAs (handlers copy this information correctly)
        newNet.weights = self.weights
        newNet.SUA = self.SUA
            
        # Return the copied net
        return newNet


## External Functions ##
def loadMC(pathOrName:str) -> MCRegressor:
    """
    Uses joblib function 'load' to recall a model object previous saved
    """
    try:
        return load(pathOrName)
    except:
        raise ValueError("Could not load a model with the given path/name!")

def Extend(baseNet:MCRegressor, d_height:int, imputeType:str = "zeros"):
    """
    Returns an MCRegressor that has its hidden layers height increased by d_height,
    with the base weight values given from baseNet. 
    The input and output layer sizes are not changed.

    ## Inputs

    1. baseNet
        - MCRegressor to use as a starting point for the parameters

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


## Activation Functions ##
# Identity
def lin(calcVec): 
    return calcVec
# RELU-Like
def relu(calcVec): 
    calcVec[calcVec < 0] = 0
    return calcVec
def silu(calcVec):
    return calcVec / (1 + 2.7182818284**(-calcVec))
# SIG-Like
def sig(calcVec):
    return 1 / (1 + 2.7182818284**(-calcVec))
def dsilu(calcVec):
    return sig(calcVec)*(1 + calcVec*(1 - sig(calcVec)))
def tanh(calcVec):
    return np.tanh(calcVec)
# Exponential-Like
def elu(calcVec):
    calcVec = 0.4 * (2.7182818284**calcVec - 1)
    return calcVec
# Logirithmic-like
def root(calcVec):
    calcVec = (np.arctan(calcVec) * np.abs(calcVec)**0.5) / 1.5
    return calcVec
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
    calcVec[calcVec < -0.4] = 0
    return calcVec


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

def fastCalc(net:MCRegressor, inVector:np.ndarray):
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
    if type(net) == MCRegressor:
        yHat = net.predict(xArr, useFast=useFast)
    elif type(net) == SUNN:
        yHat = net.predict_su(xArr)
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

def Forecast(Net:MCRegressor, inputData, comparisonData=[], plotResults=False, useFast:bool=True):
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
    if type(Net) not in [str, MCRegressor]:
        raise TypeError(f"netName should be a string or MCRegressor! Not {type(Net)} ")
    if type(Net) == str:
        net = loadMC(Net)
    elif type(Net) == MCRegressor:
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
    net = MCRegressor(inSize, hiddenSize, outSize, strActivations)

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

def oldSaveNN(Net:MCRegressor, name: str):
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

def oldFastCalc(net:MCRegressor, inVector:np.ndarray):
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

def genTrain(net:MCRegressor, xArr:np.ndarray, yArr:np.ndarray, iterations:int = 1000, 
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
    if type(net) == MCRegressor and type(xArr) == np.ndarray and type(yArr) == np.ndarray:
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

def OldTrain(Net:MCRegressor, inputData, yData, startingTweakAmp=0.8, 
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
    if type(Net) not in [str, MCRegressor]:
        raise TypeError(f"netName should be a string or AdvNet! Not {type(Net)} ")
    if type(Net) == str:
        net = loadMC(Net)
    elif type(Net) == MCRegressor:
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

def CycleTrain(Net:MCRegressor, inputData, yData, startingTweakAmp=0.8, 
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

