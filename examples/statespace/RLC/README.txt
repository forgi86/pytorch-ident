Main scripts:
-------------

- RLC_generate_train.py: generate the training dataset
- RLC_generate_test.py: generate the test dataset
- RLC_train_1step.py: train with a 1-step-ahead method, as discussed in [1]
- RLC_train_ae.py: train with a multi-step-ahead method, using an LSTM for state estimation.
  Similar to [3] and [4], but with a recurrent encoder network
- RLC_test.py: plot model simulation on a test dataset and compute metrics

Bibliography
------------
[1] M. Forgione and D. Piga. Model structures and fitting criteria for system identification with neural networks. In Proceedings of the 14th IEEE International Conference Application of Information and Communication Technologies, 2020.
[2] M. Forgione and D. Piga. Continuous-time system identification with neural networks: model structures and fitting criteria. European Journal of Control, 59:68-81, 2021.
[3] D. Masti and A. Bemborad. Learning nonlinear state–space models using autoencoders
[4] G. Beintema, R. Toth, M. Schoukens. Nonlinear State-Space Identification using Deep Encoder Networks; Submitted to l4dc 2021a