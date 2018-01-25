
11th January 2018
-----------------

Rand the PNASNet1 structure on CIFAR-10 using this code, and it was only
able to converge to less than 90% accuracy, which isn't really good enough.
It's going to be near impossible to debug all the potential issues that
could be causing it to fail, because the Tensorflow network is so
complicated.

The only thing I can try is to switch the optimizer to RMSProp and see if
that happens to magically fix anything.

Tried adding RMSProp. Fails to converge at all. Loss is extremely large to
start with. Suspect initialization problem.

25th January 2018
-----------------

Tried training with the cosine schedule described in the paper, and a new
definition based on Cadene's work. Model underfit:

* Train: 91.87%
* Test: 92.25%

Separately trained with a full anytime SGDR schedule and was able to get
the model to overfit the way we'd expect a neural network to:

* Train: 99%
* Test: 93.34%

However, both of these fall short of the 97.6% accuracy reported on
CIFAR-10 for NASNet-A in the paper.
