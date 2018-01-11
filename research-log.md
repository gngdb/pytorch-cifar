
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
