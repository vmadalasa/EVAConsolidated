# EVA4S6
L1 L2 regulatization

Team
ABK Syed
Jahnavi
Sachin CV
Madalasa

Observation:

We have used the code in 8 versions: 2 batch norm versions (batch norm and ghost batch norm); L1/L2/Both/None

No (L1 or L2) with batch Norm

No (L1 or L2) with Ghost batch Norm

with BN

L1 loss value +BN

L2 Loss value +BN

L1+L2 with BN

with Ghost batch Norm

L1 + Ghost Batch Norm

L2  + Ghost Batch Norm

L1+L2 with GBN

Observations • Without any loss functions the code ran for 40 epochs and the 

best value of loss obtained was ~0.01623 with  L1+L2 with Ghost BN 

Best value of accuracy was L2+GBN with 99.56

So confidence in the higher in the model with L1+L2+GhostBN whereas L2 +GBN had better sensitivity maybe because l1 threw away some features that might be important bringing it down to zero



The max acc for NoL1_NoL2 with BN is:  99.4

The max acc for WithL1_NoL2 with BN is:  99.46

The max acc for NoL1_WithL2 with BN is:  99.46

The max acc for WithL1_WithL2 with BN is:  99.47

The max acc for NoL1_NoL2 with GhostBN is:  99.48

The max acc for WithL1_NoL2 with GhostBN is:  99.54

The max acc for NoL1_WithL2 with GhostBN is:  99.56

The max acc for WithL1_WithL2 with GhostBN is:  99.47


The min loss for NoL1_NoL2 with BN is:  0.017138663053512575

The min loss for WithL1_NoL2 with BN is:  0.01711288378238678

The min loss for NoL1_WithL2 with BN is:  0.01704927146434784

The min loss for WithL1_WithL2 with BN is:  0.017386202383041383

The min loss for NoL1_NoL2 with GhostBN is:  0.017153030371665955

The min loss for WithL1_NoL2 with GhostBN is:  0.016235638618469238

The min loss for NoL1_WithL2 with GhostBN is:  0.016875307750701906

The min loss for WithL1_WithL2 with GhostBN is:  0.01724003961086273
