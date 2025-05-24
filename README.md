# Low Density Parity Check Code

 - This project implements the generation, encoding, and decoding of LDPC codes. This is included in the file
ldpc.py. Helper functions used are included in utils/ldfc_utils.py.

 - Furthermore, the method from the paper "A heuristic search for good low-density parity-check codes at short block lengths"[1]
is implemented and tested in some simulations.

 - The simulation of the transmission channels Binary Symmetric Channel (BSC) and Additive White Gaussian Noise Channel (AWGN)
are included in channels.

[1]: Yongyi Mao and A. H. Banihashemi, "A heuristic search for good low-density parity-check codes at short block lengths," ICC 2001. IEEE International Conference on Communications. Conference Record (Cat. No.01CH37240), Helsinki, Finland, 2001, pp. 41-44 vol.1, doi: 10.1109/ICC.2001.936269.
keywords: {Parity check codes;Iterative decoding;Educational institutions;Systems engineering and theory;Broadband communication;Delay effects;Signal to noise ratio;Distributed computing;Computational modeling;Graph theory},