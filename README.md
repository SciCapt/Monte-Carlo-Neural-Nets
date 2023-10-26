# Monte-Carlo-Neural-Nets

## Overview / Lore

A package made to see how well a basic architecture of neural nets could be. The nets can be created with custom input and output heights, as well as full customization of the hidden layers' sizes (height and count of layers) and activation functions that are applied at each space between the layers.

The basic operation of these nets is that they can be trained to some data set (or some score in an unsupervised case) by randomly 'tweaking' the different parameter weight values within the net. These weight values are clipped to be restrained to the range [-1, 1].

This is on PyPI, view the latest release at:
https://pypi.org/project/mcnets/

## Quickstart
### Example Code (Curve Fitting to f(x) = x^0.5)
Following the training algorithm enchancement of V1.0.0, there has been a few renamings, functions fixes, scoring function metrics improved, and more. Below is the quick start code that shows the syntax for creating a network, a few ways to write in the activation functions to be used, how to write the sizing (automatic input and output sizes to come soon), the included train-test split function, fitting the models, getting their predictions, and the plots of the resulting predictions, etc. This is most of what is needed to be able to use these networks, but there is more to show in other niche cases (examples to come later).
```
import matplotlib.pyplot as plt
import numpy as np
import mcnets as mc

# Data to fit to (f(x) = x^0.5)
# You can increase the number of samples by changing the X variable
X = np.random.rand(20)*2 # Gives domain of [0, 2)
Y = X**0.5

# Assemble a few models to test
net_atan  = mc.AdvNet(1, [25], 1, 'atan')
net_lin   = mc.AdvNet(1, [25], 1)
net_combo = mc.AdvNet(1, [25], 1, ['relu', 'elu', 'lin'])

# An equal alternative definition for the ATAN model:
# net_atan = AdvNet(1, [25], 1, ['atan', 'atan', 'atan'])

# An equal alternative definition for the linear model (lin is default):
# net_atan = AdvNet(1, [25], 1)  

# Train-Test Split (Taking only the train group)
x, y, _, _ = mc.TTSplit(X, Y, percentTrain=50)

# Fit the models to the training data group
print("ATAN Model Training:")
net_atan  = net_atan.Fit(x, y, useFast=False)

print("\nLinear Model Training:")
net_lin   = net_lin.Fit(x, y)

print("\nCombo Model Training")
net_combo = net_combo.Fit(x, y)

# Get the models predictions to the full data set
ym_atan  = net_atan.Predict(X, useFast=False)
ym_lin   = net_lin.Predict(X)
ym_combo = net_combo.Predict(X)

# Plot and compare the results of the models
print("\nPlotted Results:")
plt.plot(Y, 'o')
plt.plot(ym_atan, 'r--')
plt.plot(ym_lin, 'g--')
plt.plot(ym_combo, 'b--')
plt.title(f"y=x^0.5 vs Networks of Various Activation Functions")
plt.legend(["True Data", "ATAN Model", "Linear Model", "Combo Model"])
plt.show()
```

It is worth noting that before V0.2.1, directly fitting a netork to the values for curve fitting was either a nightmare or not really possible (First Method below). Now with many changes to the activations functions, net customizability, training algorithm, etc. it is quite straight forward. 

Note in the example above that it is also quite easy to be able to test variations of activation functions used within a network. This allows for being able to find a model for nearly every dataset, though it can be hard to find the right/best combination sometimes (yes, this is foreshadowing to an automatic activation function combination finder I am working on presently).

## Old Curve Fitting Examples
While the quickstart above shows the current ability of nets to converge onto a small set of data points, over 1000s of points have been fit to in under 10 seconds. There were a few versions before this, however. Below are some examples of the limited training possible before, shown purely for comparison to what is now possible.

### First (choppy) Method
The original way I used these nets for curve fitting was by taking advantage of the non-linear behavior of the RELU calculation. At the time, I also had an ELU option implemented, but it didn't give me any useful fit back. This method worked and did allow the net to have some "intelligence" to fit to the given data, but it was very choppy at best and required massive nets to do an okay-ish job. Below is a "short" training session example of what it gives.

Training Details:
- Net Size = 1176 parameters
- Training Time = 213.4 s

Result:

![](Examples/ghFit1b.png)

### Second Method (fitting slopes using ELU)
The new method still makes use of the AdvNet, but instead of directly using the output as some sort of "y" value, the output is used as a sort of rough derivative estimator (ie. a slope) that gets added to the previous point. By finding the points this way, using the ELU hidden function, much nicer results can be generated. Also importantly, note the much smaller net and training time as compared to above.

Training Details:
- Net Size = 312 params.
- Training Time = 18.1 s

Result:

![](Examples/ghFit2b.png)


## Speed Tests
### Convention
A speed test was completed on what I like to refer to as 'rectangular' neural nets. This is, excluding the input and output layer sizes, each hidden layer has some size/height H and there are W layers (the net has a hidden 'width' of W). So, again excluding in/out sizes as they will be the same, the nets can be referenced as their hidden size of HxW. For example, a net with sizes 1x12x12x12x1 (in and out size of 1) as used in the above quickstart example can be thought of as a 12x3 net.

### Results
Using this convention, a test was conducted on 50^2 nets, that is, with H and W ranging from 1 to 50. The test simply consisted of performing the same numerical calculation (using the .Calculate() method) 100 times with the same net. The time recorded was then the total time divided by the 100 repeats to give the average calculation time. This test was done on all of the 2500 different nets which took a few minutes as the median calculation time was about 1.2 ms. Here is the pretty heat map that sums up the results better than the entirety of this paragraph if I kept rambling:

![](Examples/ghSpeedTest1a.png)

Note that the cooler colors are faster, and the warmer colors are slower times.

The most important takeaway is that 'tall' and 'wide' nets of equal parameters take significantly different times. That is, having only a few hidden layers that are tall (ie. like a 12x3 net) is faster than the reverse with a similar number of parameters (ie. like a 5x13 net). It is quite worse, in fact:

Net 1:
- Shape = 12x3
- Param.s = 312
- Avg. calculation time = 49.2 microseconds

Net 2:
- Shape = 5x13
- Param.s = 310
- Avg. calculation time = 124 microseconds
