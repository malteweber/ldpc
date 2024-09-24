# Programmierprojekt Informationstheorie
## Sommersemester 2024

 - In diesem Projekt wird die Erzeugung, Kodierung und Dekodierung von LDPC-Codes implementiert. Diese ist in der Dateil
`ldpc.py` enthalten. Hilfsfunktionen die genutzt werden sind in `utils/ldfc_utils.py` enthalten.

 - Weiterhin wird die Methode aus dem Paper "A heuristic search for good low-density parity-check codes at short block lengths"[1]
implementiert und in einigen Simulationen getestet.

 - Die Simulation der Übertragungskanäle Binary Symmetric Channel (BSC) und Additive White Gaussian Noise Channel (AWGN)
sind in `channels` enthalten.

 - Die Datei `utils/simulation_utils.py` enthält Funktionen für die Simulationsexperimente.
 - Die genutzte Python-Version in Python 3.12.0
 - Die genutzten Bibliotheken sind in `requirements.txt` enthalten.
   - numpy: Datenstrukturen für Matrizen und Vektoren, trigonometrische Funktionen
   - scipy: Datenstrukturen für spärlich besetzte Matrizen
   - matplotlib: Visualisierung von Daten
   - networkx: Graphenstruktur für Tannergraphen


[1]: Yongyi Mao and A. H. Banihashemi, "A heuristic search for good low-density parity-check codes at short block lengths," ICC 2001. IEEE International Conference on Communications. Conference Record (Cat. No.01CH37240), Helsinki, Finland, 2001, pp. 41-44 vol.1, doi: 10.1109/ICC.2001.936269.
keywords: {Parity check codes;Iterative decoding;Educational institutions;Systems engineering and theory;Broadband communication;Delay effects;Signal to noise ratio;Distributed computing;Computational modeling;Graph theory},