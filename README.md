# ABORTED ATTEMPT TO PORT PNASNET TO PYTORCH

I tried to port the [tensorflow definition of nasnet][tf] and couldn't get
it to work. Was trying to do this just by inspection and trying to respect
pytorch idioms. Unfortunately, nasnet is too complicated to do this well,
and I've probably made some mistake.

Network fails to train past 90% accuracy, and completely fails to train
with regularisation enabled. Suspect problem with intiailisation or
optimizer, but could easily be some problem with the network definition.

[tf]: https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet
