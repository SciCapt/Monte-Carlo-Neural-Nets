# Python Dependancies
from random import sample
import copy

# External Dependancies
import numpy as np
from joblib import load, dump

# MCNet Dependancies
from mcnets.activations import *

## Models ##
# Primary Monte-Carlo Neural Network Model
class NeuralNetwork:
    def __init__(self, hidden_counts:'tuple[int]'=(100), activations:'tuple[str, function]'=('relu'), input_acti='identity', 
                 output_acti='identity', max_iter=1000, learning_rate_init=1, learning_rate_mode=('adaptive', 'dynamic', 'constant'), 
                 gamma=0.0025, n_iter_no_change=100, l2_reg=0, l1_reg=0, dropout=0.0, validation_fraction:float=0, early_stopping=True, 
                 quad_tol:float=-0.01, tri_tol:float=-0.01, verbose=False):
        """
        Neural Network that uses Monte-Carlo training. Can be either a regressor or classifier depending on the
        output_activation used (i.e. use sigmoid/sig for a classifier).

        - `hidden_counts`
            - List of count of units to have per hidden layer
            - Length of this list == number of hidden layers
            - Can pass an empty list to generate a model with no hidden layers
        
        - `activations`
            - List of activations (strings and/or functions) for the hidden layers with the same length as hidden_counts
            - Strings:
                - Should be one of the accepted strings below (NOT case sensitive):
                    - `'LIN'/'LINEAR'/'IDENTITY', 'RELU', 'LRELU', 'SILU', 'SIG', 'DSILU', 'TANH', 'ELU', 'ROOT', 'SQR', 'RND'`
            - Functions:
                - Should take an array, and output an array of the same size
                - Example: A ReLU function could be: `lambda x: np.maximum(x, 0)`

        - `input_acti`
            - The activation used in the input layer (before any matrix calculations). Same requirements as `activations`
        
        - `output_acti`
            - The activation used in the output layer (after all matrix calculations). Same requirements as `activations`

        - `max_iter`
            - The maxmimum iterations to use when fitting the model

        - `learning_rate_init`
            - The initial learning rate magnitude to used when fitting the model

        - `learning_rate_mode`
            - 'adaptive': Keeps learning rate constant until no improvment for `n_iter_no_change` # of iterations, then halves the learning rate
            - 'dynamic': Continually decreases the learning rate via exponential decay, with gamma as the decay factor (`rate * 2**(-iter*gamma)`)
            - 'constant': Doesn't change the learning rate throughout the fitting process

        - `gamma`
            - Exponential decay factor when using 'dynamic' learning rate mode

        - `n_iter_no_change`
            - Amount of iterations waited when using the 'adaptive' learning mode before halving the learning rate

        - `l2_reg`
            - Magnitude of the L2 regularization penalty applied

        - `l1_reg`
            - Magnitude of the L1 regularization penalty applied

        - `dropout`
            - Average (decimal) percentage of weights that aren't altered with every change

        - `validation_fraction`
            - Decimal percent of fit data put aside to use as a validation set
            - If above 0, the early stopping tolerances `quad_tol` and `tri_tol` use the validation
            score history over the training score history

        - `early_stopping`
            - Determines if the `quad_tol` and `tri_tol` early stopping methods are used

        - `quad_tol`
            - Early stopping method
            - Fits a polynomial (degree 2) to the current training scores during fitting
            - If the polynomial has a tangent/slope value less than this tolerance, training stops
            - If None:
                - Doesn't consider this for early stopping, even if early_stopping is True
                - Helpful for only considering one of the tols (quad or tri)

        - `tri_tol`
            - Early stopping method
            - Fits a polynomial (degree 3) to the current training scores during fitting
            - If the polynomial has a tangent/slope value less than this tolerance, training stops
            - If None:
                - Doesn't consider this for early stopping, even if early_stopping is True
                - Helpful for only considering one of the tols (quad or tri)
        """

        # Assemble hidden counts to a list
        if isinstance(hidden_counts, list):
            self.hidden_counts = hidden_counts
        elif isinstance(hidden_counts, tuple):
            self.hidden_counts = list(hidden_counts)
        elif isinstance(hidden_counts, (str, int)):
            self.hidden_counts = [hidden_counts]

        # Assemble activations to a list
        if isinstance(activations, list):
            self.activations = activations
        elif isinstance(activations, tuple):
            self.activations = list(activations)
        elif isinstance(activations, str):
            self.activations = [activations]

        # Main params
        self.input_acti = input_acti
        self.output_acti = output_acti
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.learning_rate_mode = 'adaptive' if isinstance(learning_rate_mode, tuple) else learning_rate_mode
        self.gamma = gamma
        self.n_iter_no_change = n_iter_no_change
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.dropout = dropout
        self.val_frac = validation_fraction
        self.stop_early = early_stopping
        self.quad_tol = quad_tol
        self.tri_tol = tri_tol
        self.verbose = verbose

        # Fit-generated params
        self._use_score_history = 'train' if self.val_frac <= 0 else 'val'
        self._is_fitted = False
        self.input_size = None
        self.output_size = None
        self.weights = []
        self.biases = []

        # Initial checks
        if len(self.hidden_counts) != len(self.activations):
            raise ValueError(f"Length of hidden_counts ({self.hidden_counts}) != Length of activations ({self.activations})")
        
        # Compile activations
        for i, af in enumerate(self.activations):
            if type(af) == str:
                try:
                    self.activations[i] = AFDict[af.upper()]
                except:
                    raise ValueError(f"Activation name '{af}' is not in the accepted activations")
        
        if type(self.input_acti) == str:
            try:
                self.input_acti = AFDict[self.input_acti.upper()]
            except:
                if not callable(input_acti):
                    raise ValueError(f"Input activation name '{self.input_acti}' is not in the accepted activations")
                else:
                    self.input_acti = input_acti
            
        if type(self.output_acti) == str:
            try:
                self.output_acti = AFDict[self.output_acti.upper()]
            except:
                if not callable(self.output_acti):
                    raise ValueError(f"Output activation name '{self.output_acti}' is not in the accepted activations")
                else:
                    self.output_acti = self.output_acti

    def set_weights(self, new_weights:'list[np.ndarray]'):
        self.weights = [wi.copy() for wi in new_weights]

    def set_biases(self, new_biases:'list[np.ndarray]'):
        self.biases = [bi.copy() for bi in new_biases]

    def get_weights(self):
        return [wi.copy() for wi in self.weights]
    
    def get_biases(self):
        return [bi.copy() for bi in self.biases]

    def predict(self, X:np.ndarray):
        # Apply initial activation
        X = self.input_acti(X)

        # Hidden Layers yipeeeeeeeeeee
        for afunc, W, B in zip(self.activations+[self.output_acti], self.weights, self.biases):
            X = afunc(np.dot(X, W) + B)

        # Shape accordingly
        if self.output_size == 1:
            X = X.flatten()
        return X
    
    def _current_l2_penalty(self):
        return self.l2_reg*np.sum([np.sum(wi**2) for wi in self.get_weights()])
    
    def _current_l1_penalty(self):
        return self.l1_reg*np.sum([np.sum(np.abs(wi)) for wi in self.get_weights()])
    
    def score(self, X, Y, score_type='r2'):
        return score_model(self, X, Y, method=score_type) - self._current_l2_penalty() - self._current_l1_penalty()
    
    def _initialize_model(self, input_size, output_size, force_generate=False):
        """Generates the models weights and biases with the given input and ouput size.
        Can be forced to regen (if already previously fitted) by setting force_generate to True."""
        if not self._is_fitted or force_generate:
            # Get feature sizes
            self.input_size = input_size
            self.output_size = output_size

            # Make Weights
            self.weights = [2*np.random.rand(s1, s2)-1 for s1, s2 in zip([self.input_size]+self.hidden_counts, self.hidden_counts+[self.output_size])]
            self.biases  = [2*np.random.rand(s1)-1 for s1 in self.hidden_counts+[self.output_size]]

            # Complete
            self._is_fitted = True
    
    def fit(self, X:np.ndarray, Y:np.ndarray, score_type='r2'):
        """Fit the model to the given data. 
        
        Returns a dict with the training score/error history under 'train'
        """

        # History dict
        history = {'train': [], 'val': []}

        # Train/Validation split
        if self.val_frac > 0:
            xt, xv, yt, yv = TTSplit(X, Y, percentTrain=(1-self.val_frac))
        else:
            xv = None; yv = None

        # Make val score function for easy handling of both cases (val_frac == 0 and val_frac > 0)
        def val_score(self:NeuralNetwork, xv, yv, score_type=score_type):
            if self.val_frac <= 0:
                return 0
            else:
                return self.score(xv, yv, score_type=score_type)

        # Generate model
        self._initialize_model(input_size=1 if len(X.shape)==1 else X.shape[1],
                               output_size=1 if len(Y.shape)==1 else Y.shape[1],
                               force_generate=False)

        # Single Column inputs - check for correct size (len(X), 1)
        if len(X.shape) == 1:
            print(f"MCNet WARN: X seems to be 1 column, but has shape {X.shape} not {(len(X), 1)} (this will be corrected but reshape X to avoid this warning).")
            X = X.reshape((len(X), 1))

        # Initial stats
        score = self.score(X, Y, score_type=score_type) if self.val_frac <= 0 else self.score(xt, yt, score_type=score_type)
        current_rate = self.learning_rate_init
        i_last_improved = 0

        # Update history
        history['train'].append(score)
        history['val'].append(val_score(self, xv, yv, score_type=score_type))

        # Main iteration loop
        for i in range(self.max_iter):
            # Get learning rate
            if self.learning_rate_mode == 'dynamic':
                current_rate = self.learning_rate_init * 2**(-i * self.gamma)
            elif self.learning_rate_mode == 'adaptive':
                if i-i_last_improved >= self.n_iter_no_change:
                    current_rate /= 2
                    i_last_improved = i

            # Test tweaking weights
            init_weights = self.get_weights()
            for ind in range(len(init_weights)):
                # Get dropout filter/mask with dropout
                if self.dropout > 0:
                    mask = np.random.random(self.weights[ind].shape)
                    mask[mask < self.dropout] = 0
                    mask[mask > self.dropout] = 1
                    adjustment = mask * np.random.random(self.weights[ind].shape)

                # Get adjustment with no dropout
                else:
                    adjustment = np.random.random(self.weights[ind].shape)

                # Modify weights using above adjustment (scaled to [-learning_rate, +learning_rate])
                self.weights[ind] += current_rate*(2*adjustment - 1)

                # Get models score with modified weights
                test_score = self.score(X, Y, score_type=score_type) if self.val_frac <= 0 else self.score(xt, yt, score_type=score_type)

                # Keep weights and set new best score if better
                if test_score > score:
                    score = test_score
                    i_last_improved = i

                    # Update history
                    history['train'].append(score)
                    history['val'].append(val_score(self, xv, yv, score_type=score_type))

                    # Do early stopping calculations (only after initial bit of training for better polynomials)
                    if self.stop_early and i >= 0.1*self.max_iter:
                        ## Build and check polynomial fits for early stopping conditions ##
                        poly_x = [*range(len(history[self._use_score_history]))]

                        ## Quad Tol check #
                        if self.quad_tol != None:
                            # Coefficients
                            p2_coefs = np.polyfit(poly_x, history[self._use_score_history], deg=2)

                            # Y (score) predictions
                            p2_y = [sum([ci*x**(2-i) for i, ci in enumerate(p2_coefs)]) for x in poly_x]

                            # Numerical tangents
                            dp2_y = [y1-y2 for y1, y2 in zip(p2_y[1:], p2_y[:-1])]

                            # Check min quad tangent slope
                            if min(dp2_y) < self.quad_tol:
                                if self.verbose:
                                    print(f"Itr: {i+1}/{self.max_iter} | {score_type.upper()}: {score:.6f} | Learning Rate: {current_rate}")
                                    print("Training Stopped: quad tolerance has been surpassed")
                                return history
                            
                        ## Tri Tol check #
                        if self.tri_tol != None:
                            # Coefficients
                            p3_coefs = np.polyfit(poly_x, history[self._use_score_history], deg=3)

                            # Y (score) predictions
                            p3_y = [sum([ci*x**(3-i) for i, ci in enumerate(p3_coefs)]) for x in poly_x]

                            # Numerical tangents
                            dp3_y = [y1-y2 for y1, y2 in zip(p3_y[1:], p3_y[:-1])]

                            # Check min quad tangent slope
                            if min(dp3_y) < self.tri_tol:
                                if self.verbose:
                                    print(f"Itr: {i+1}/{self.max_iter} | {score_type.upper()}: {score:.6f} | Learning Rate: {current_rate}")
                                    print("Training Stopped: tri tolerance has been surpassed")
                                return history

                # Reset weights to original values if not better
                else:
                    self.weights[ind] = init_weights[ind].copy()

            # Status
            if self.verbose:
                print(f"Itr: {i+1}/{self.max_iter} | {score_type.upper()}: {score:.6f} | Learning Rate: {current_rate}      ", end='\r')

        # Final printout
        if self.verbose:
            print(f"Itr: {i+1}/{self.max_iter} | {score_type.upper()}: {score:.6f} | Learning Rate: {current_rate}")
        return history

    def save(self, name:str):
        save_model(self, name=name)

    def get_param_ranges_for_optuna(self):
        """Returns a dictionary of params and their possible values either via tuples of (min_val, max_val)
        or lists for discrete categorical params/options.
        
        Params not included in the dictionary:
        - `hidden_counts`
        - `activations`
        - `verbose`
        
        These params should be directly set as desired in an optimizer."""

        return {
            'input_acti': ['LIN', 'RELU', 'LRELU', 'SILU', 'SIG', 'DSILU', 'TANH', 'ELU', 'ROOT', 'SQR', 'RND'],
            'output_acti': ['LIN', 'RELU', 'LRELU', 'SILU', 'SIG', 'DSILU', 'TANH', 'ELU', 'ROOT', 'SQR', 'RND'],
            'max_iter': (1, 10000),
            'learning_rate_init': (1e-3, 5),
            'learning_rate_mode': ['adaptive', 'dynamic', 'constant'],
            'gamma': (1e-6, 10),
            'n_iter_no_change': (1, 5000),
            'l2_reg': (1e-10, 1),
            'l1_reg': (1e-10, 1),
            'dropout': (0, 0.9), 
            'validation_fraction': (0.1, 0.75),
            'early_stopping': [True, False],
            'quad_tol': (-1, 1),
            'tri_tol': (-1, 1),
        }

    def copy(self):
        """Returns a deep copy of the model"""
        return copy.deepcopy(self)
    
    def make_mutation(self, learning_rate=1):
        """Returns a copy of the model with weights randomly via values
        in the range [-learning_rate, +learning_rate]."""
        # Get a model copy, adjust its weights
        new = self.copy()
        for wi in new.weights:
            wi += learning_rate*(2*np.random.random(wi.shape)-1)
        return new

# Large flexible "polynomial" 
class SoupRegressor:
    def __init__(self, use_tan=False, round_threshold=1e-5, max_iter=100, dropout=0., learning_rate_init=20., 
                 learning_rate_mode:str=('dynamic', 'adaptive', 'constant'), gamma=50, n_iter_no_change=10, 
                 use_biases=True, trainable_biases=False, l1_reg=0., l2_reg=0., verbose=False):
        """
        Creates and combines outputs of large ""polynomials"" made for every X feature given in .fit(). Only
        fits to one target.

        - `use_tan`
            - Three TAN(x) terms are included in f(x), but due to the asymptotic nature of TAN, they
              can actually hurt model preformance. So, this is disabled by default, but left as a
              setting to try anyways.

        - `round_threshold`
            - When adjusting the coefficients of the model, if a single coefficient's magnitude falls
              below this threshold, it is rounded to 0. This makes it easier for the model to completely
              remove terms from its various f(x) equations if it finds that is better.

        - `max_iter`
            - Maximum iterations used to train the model
            - As of V2.0.3 there is no early stopping for this model, so this is simply the number of 
            training iterations completed

        - `dropout`
            - Average (decimal) percent of coefficients ignored (not trained) per sub iteration

        - `learning_rate_init`
            - Initial learning rate
            - Note that this model type typically requires a much higher learning rate than others so
            the typical value of 1 is too slow more often than not

        - `learning_rate_mode`
            - 'adaptive': Keeps learning rate constant until no improvment for `n_iter_no_change` # of iterations, then halves the learning rate
            - 'dynamic': Continually decreases the learning rate via exponential decay, with gamma as the decay factor (`rate * 2**(-iter*gamma)`)
            - 'constant': Doesn't change the learning rate throughout the fitting process

        - `gamma`
            - Exponential decay factor when using 'dynamic' learning rate mode

        - `n_iter_no_change`
            - Amount of iterations waited when using the 'adaptive' learning mode before halving the learning rate

        - `use_biases`
            - Decides if a constant [-1, 1] is included at the end of each feature's f(x)

        - `trainable_biases`
            - Decides of the above constants are trainable

        - `l2_reg`
            - Magnitude of the L2 regularization penalty applied

        - `l1_reg`
            - Magnitude of the L1 regularization penalty applied

        ## Technical Breakdown

        For each column of data, generates a function of best fit of the form:

        f_i(x) = k0 + k1*(|x|**0.5) + k2*(x) + k3*(x**2) + k4*sin(x/3) + k5*sin(x) + k6*sin(3x) + 

               k7*cos(x/3) + k8*cos(x) + k9*cos(3x) + k10*tan(x/3) + k11*tan(x) + k12*tan(3x) + 

               k13*e**(x/3) + k14*e**(x) + k15*e**(3x) + k16*e**(-x/3) + k17*e**(-x) + k18*e**(-3x)

        There is an f(x) for every x feature. This means for N features the net model is:

        F(x) = SUM[f_i(x)] for i=[0, 1, ..., N-1]
        """

        # Unchanging attributes
        self.FUNCTION_LENGTH = 19 if not use_biases else 20
        self.USE_TAN = use_tan
        self.ROUND = round_threshold

        # Fit params
        self.ieta = max_iter
        self.dropout = dropout
        self.init_rate = learning_rate_init
        self.learning_mode = 'adaptive' if isinstance(learning_rate_mode, tuple) else learning_rate_mode
        self.gamma = gamma
        self.n_iter_no_change = n_iter_no_change
        self.use_biases = use_biases
        self.train_biases = trainable_biases
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.verbose = verbose

        # Attributes generated in fit
        self.coefs = 0
        self.num_features = 0
        self.parameters = 0
        self.fitted = False

    ## Model Functions ##
    def predict(self, X:np.ndarray, run_debug=False):
        """
        Calculates each f(x) described, for each row in X.
        """

        # Verify the shape of X (and num_features)
        if run_debug:
            given_size = 1 if len(X.shape) == 1 else X.shape[1]
            if self.num_features != given_size:
                raise ValueError(f"Expected X array shape of ({len(X)}, {self.num_features}), got {X.shape}")
            
        # Main function, per feature
        def f(x, col_index):
            # Get function coefficients for this feature
            k = self.coefs[:, col_index].flatten()

            # Complete f(x) prediction, add k[19] (bias) if included in model
            SUM = k[0] + k[1]*(np.abs(x)**0.5) + k[2]*x + k[3]*(x**2) + k[4]*np.sin(x/3) + k[5]*np.sin(x) + k[6]*np.sin(3*x) + \
                  k[7]*np.cos(x/3) + k[8]*np.cos(x) + k[9]*np.cos(3*x) + self.USE_TAN*k[10]*np.tan(x/3) + self.USE_TAN*k[11]*np.tan(x) + self.USE_TAN*k[12]*np.tan(3*x) + \
                  k[13]*np.exp(x/3) + k[14]*np.exp(x) + k[15]*np.exp(3*x) + k[16]*np.exp(-x/3) + k[17]*np.exp(-x) + k[18]*np.exp(-3*x)
            
            if self.use_biases:
                SUM += k[19]
            
            return SUM
        
        # Calculate the sum described in INIT
        result = 0
        for col_index in range(self.num_features):
            result += f(X[:, col_index], col_index=col_index)
        return result
    
    def _initialize_model(self, input_size, force_generate=False):
        if not self.fitted or force_generate:
            # Setup coefficients
            self.num_features = input_size
            self.coefs = 2*np.random.rand(self.FUNCTION_LENGTH, self.num_features) - 1
            
            # Confirm initial fit
            self.parameters = self.coefs.size
            self.fitted = True

    def fit(self, X, Y, score_type='r2'):
        # Check if model initial fit complete
        self._initialize_model(input_size=1 if len(X.shape)==1 else X.shape[1], force_generate=False)

        # Get coefs to adjust (include or exclude bias, the last one)
        if self.use_biases and not self.train_biases:
            end_len = self.FUNCTION_LENGTH - 1
        else:
            end_len = self.FUNCTION_LENGTH

        # Tweak params for N iterations
        i_last_improved = 0
        current_rate = self.init_rate
        score = self.score(X, Y, method=score_type)
        for itr in range(self.ieta):
            # Get dropout filter and apply it (discards changing drapout % num of coefficients)
            filt = np.random.rand(end_len) > self.dropout

            # Get learning rate
            if self.learning_mode == 'dynamic':
                current_rate = self.learning_rate_init * 2**(-itr * self.gamma)
            elif self.learning_mode == 'adaptive':
                if itr-i_last_improved >= self.n_iter_no_change:
                    current_rate /= 2
                    i_last_improved = itr

            # Loop over each f(x) coefs (self.coefs cols per feature)
            for c in range(self.coefs.shape[1]):
                # Get original values to reset to if no improvement
                init_coefs = self.coefs[:end_len, c].copy()

                # Make adjustment
                self.coefs[:end_len, c] += current_rate*(2*filt*np.random.rand(end_len) - 1)

                # Test adjustment
                new_score = self.score(X, Y, method=score_type)
                if new_score > score:
                    # Keep new coefs, set new best score
                    score = new_score
                    i_last_improved = itr

                    # Round coefs below threshold to 0
                    self.coefs[np.abs(self.coefs) < self.ROUND] = 0
                else:
                    # Worse score, reset coefs changed
                    self.coefs[:end_len, c] = init_coefs

            # Print status
            if self.verbose:
                print(f"Itr. #{itr+1} | Score = {format(score, '.6f')} | Learning Rate = {current_rate:.6f}        ", end='\r')
        
        if self.verbose:
            print(f"Itr. #{itr+1} | Score = {format(score, '.6f')} | Learning Rate = {current_rate:.6f}        ")

    def _current_l2_penalty(self):
        return self.l2_reg*np.sum(self.coefs**2)
    
    def _current_l1_penalty(self):
        return self.l1_reg*np.sum(np.abs(self.coefs))

    def score(self, X, Y, method='r2'):
        return score_model(self, X, Y, method=method) - self._current_l2_penalty() - self._current_l1_penalty()
    
    def copy(self):
        """Returns a copy of the model"""
        return copy.deepcopy(self)


## External Functions ##
def save_model(model, name:str):
    """
    Load the model object saved under the given name
    """
    dump(model, filename=name)

def load_model(name:str):
    """
    Uses joblib function 'load' to recall a model object previous saved
    """
    try:
        return load(name)
    except:
        raise ValueError("Could not find a model with the given name")

def TTSplit(Xdata, Ydata, percentTrain:float = 70):
    """
    Universal Train-Test data split via random selection.

    Returns in the order xTrain, xTest, yTrain, yTest
    
    - percentTrain
        - sets the amount of data given back for training data, 
          while the rest is sent into the testing data set
        - Values less than 1 are interpreted as decimal percent
    """

    # Scale percent
    if percentTrain < 1:
        percentTrain *= 100

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

def cross_val(model, X, Y, cv=5, score_type='r2', return_models=False, verbose=0, **kwargs) -> np.ndarray:
    """
    Cross validation of model using X and Y with N number of splits

    ## Params

    - `cv`
        - Number of train/validation split tests to complete

    - `scoreType`
        - One of the score/error functions avalible in score()
        - ['r2', 'sse', 'mae', 'mse', 'rae', 'acc']
        - If a function is given, should be the form:
            - scorer_func(predictor, X, Y)

    - `return_models`
        - If True, returns (models, weights) from the models tested and their
          weights derived from their relative overall score to the others.

    - `verbose`
        - 0 = No output printing
        - 1 = Outputs current step/progress
    """

    # Old 'N' kwarg for cv
    if 'N' in kwargs.keys():
        cv = kwargs['N'] 

    # Old for previous gen MCNN
    if 'train_depth' in kwargs.keys():
        train_depth = kwargs['train_depth']
    else:
        train_depth = 2

    if 'scoreType' in kwargs.keys():
        score_type = kwargs['scoreType']

    # Check for enough data 
    if len(X) < cv:
        raise ValueError(f"Not enough data for {cv} cycles")

    # Verbose settings
    fit_verbose = False     # Old for previous gen MCNN
    step_verbose = False    # Verbosity for each CV step
    if verbose == 1:
        fit_verbose = False
        step_verbose = True
    elif verbose == 2:
        fit_verbose = True
        step_verbose = True

    # Main loop
    if return_models:
        models = []

    base_model = copy.deepcopy(model)

    scores = []
    step = round(len(X) / cv)
    for n in range(cv):
        # Data Split
        start = n*step
        if n == cv-1:
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
        model = copy.deepcopy(base_model)
        try:
            # Old for previous generation MCNN
            model.fit(X_train, Y_train, Ieta=round(5*train_depth), verbose=fit_verbose)
        except:
            model.fit(X_train, Y_train)

        if return_models:
            models.append(model)

        # Record model score
        scores.append(score_model(model, X_val, Y_val, method=score_type))

        # Print step results
        if step_verbose:
            print(f"Cross-Validation: Step {n+1}/{cv} Complete     ", end='\r')

    # Generate model weights if needed
    if return_models:
        weights = [m.score(X, Y) for m in models]
        weights = np.array(weights)
        weights /= np.sum(weights)
        weights = weights.tolist()

    # Finish
    if step_verbose:
        # print(f"Cross-Validation: All Steps Completed")
        print(f"Mean Model {score_type} Validation Score/Error = {np.mean(scores):.4e} +- {np.std(scores):.4e}")

    if return_models:
        return models, weights
    else:
        return np.array(scores)

def score(ytrue, ypred, method='r2') -> float:
    """
    Main scorer function given a model's output and true values.

    - `method`
        - 'r2': R^2 Score
        - 'sse': -Sum Squared Error
        - 'mre': -Root Mean Squared Error
        - 'mae': -Mean Absolute Error
        - 'rae': Custom R^2-like Score
        - 'acc'/'accuracy': Accuracy Score
        - Function: form of `scorer(ytrue, ypred)`
    """

    # Force correct case
    method = method.lower()

    ## R^2 Method ##
    if method == 'r2':
        return r2d2(ypred, ytrue)

    ## Sum Squared Error ##
    elif method == 'sse':
        return -np.sum((ypred - ytrue)**2)
    
    ## Mean Root Error ##
    elif method == 'mre':
        return -(np.sum((ypred - ytrue)**2) / len(ypred))**0.5

    ## Mean Absolute Error ##
    elif method == 'mae':
        return -(1/len(ypred)) * np.sum(np.abs(np.array(ypred) - ytrue))

    ## RAE Score ##
    elif method == 'rae':
        return raeScore(ytrue, ypred)
    
    ## Accuracy Score ##
    elif method in ['acc', 'accuracy']:
        return np.sum(ypred == ytrue) / ypred.size
    
    ## Custom Function ##
    elif isinstance(method, callable):
        return method(ytrue, ypred)

    # Raise error if one of the possible methods was not given
    else:
        raise ValueError(f"Given score type '{method.upper()}' is not one of the avalible types.")

def score_model(model, X:np.ndarray, ytrue:np.ndarray, method='r2') -> float:
    """
    Main scorer function given a model, input data, and true values.

    - `method`
        - 'r2': R^2 Score
        - 'sse': -Sum Squared Error
        - 'mre': -Root Mean Squared Error
        - 'mae': -Mean Absolute Error
        - 'rae': Custom R^2-like Score
        - 'acc'/'accuracy': Accuracy Score
        - Function: form of `scorer(model, X, ytrue)`
    """

    # Force lowercase
    method = method.lower()
    
    # Get model output
    ypred = model.predict(X)

    # For callable scorers given
    if callable(method):
        return method(model, X, ytrue)
    
    # Otherwise use main score function
    else:
        return score(ytrue, ypred, method=method)


## Helper/Smol Functions ##
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

def normalize(array:np.ndarray):
    """
    ## Function

    Normalizes a given array (per col.) such that each point is the z-score for the given 
    columns mean and standard deviation. The returned array is a copy.

    Columns with only two unique values are automatically converted to be just 0's and 1's

    ## Returns

    normalized_array, mean_std_data

    ## Usage

    The following should generate a normalized array where each column of the array has a
    mean of (essentially*) 0, and a standard deviation of (essentially*) 1.

    *Due to floating point errors, the values might be off by a very minimal amount

    ```
    import numpy as np
    import mcnets as mc

    # Column index to look at
    col_index = 2

    # Generate an array with 5 columns and 100 samples
    array = np.random.rand(100, 5)
    mean = np.mean(array[:, col_index])
    std  = np.std(array[:, col_index])
    print("Original array:")
    print(f"{mean = }")
    print(f"{std = }")

    # Normalize the array on a per-column basis
    norm_array, MS_data = mc.normalize(array)

    # Use a columns to prove it is normalized (mean of 0, std of 1)
    mean = np.mean(norm_array[:, col_index])
    std  = np.std(norm_array[:, col_index])
    print()
    print("Normalized array:")
    print(f"{mean = }")
    print(f"{std = }")
    ```
    """

    # dtype Check
    array = array.astype(np.float64).copy()

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
            array[:, ci] /= S if S != 0 else 1
            
    else:
        # Normalize single-col array
        M = np.mean(array)
        S = np.std(array)
        ms_data.append((M, S))

        array -= M
        array /= S

    return array, ms_data

def denormalize(normalized_array:np.ndarray, mean_std_data):
    """
    ## Function

    Denormalizes an array that has been normalized by the normalize() function, and
    given the mean_std_data from the array's normalization process. The returned array
    is a copy.

    ## Returns

    denormalized_array

    ## Usage

    The following should generate a normalized array where each column of the array has a
    mean of (essentially*) 0, and a standard deviation of (essentially*) 1. After this, it 
    will denormalize it using this function and bring the array back to its original mean/std.

    *Due to floating point errors, the values might be off by a very minimal amount

    ```
    import numpy as np
    import mcnets as mc

    # Column index to look at
    col_index = 2

    # Generate an array with 5 columns and 100 samples
    array = np.random.rand(100, 5)
    mean = np.mean(array[:, col_index])
    std  = np.std(array[:, col_index])
    print("Original array:")
    print(f"{mean = }")
    print(f"{std = }")

    # Normalize the array on a per-column basis
    norm_array, MS_data = mc.normalize(array)

    # Use a columns to prove it is normalized (mean of 0, std of 1)
    mean = np.mean(norm_array[:, col_index])
    std  = np.std(norm_array[:, col_index])
    print()
    print("Normalized array:")
    print(f"{mean = }")
    print(f"{std = }")

    # Denormalize the array, per-column
    denorm_array = mc.denormalize(norm_array, MS_data)

    # Use a columns to prove it is no longer normalized
    mean = np.mean(denorm_array[:, col_index])
    std  = np.std(denorm_array[:, col_index])
    print()
    print("Denormalized array (Should be same as original):")
    print(f"{mean = }")
    print(f"{std = }")
    ```
    """

    # Make array copy
    denorm_arr = normalized_array.copy()

    # Get number of columns
    if len(denorm_arr.shape) > 1:
        # Do denormalization
        for col in range(denorm_arr.shape[1]):
            denorm_arr[:, col] = (denorm_arr[:, col] * mean_std_data[col][1]) + mean_std_data[col][0]
    else:
        # Do denormalization (single col)
        denorm_arr[:] = (denorm_arr[:] * mean_std_data[0][1]) + mean_std_data[0][0]

    return denorm_arr


## Libraries ##
AFDict = {
        "LIN": lin,
        "LINEAR": lin,
        "IDENTITY": lin,
        "RELU": relu,
        "LRELU": lrelu,
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