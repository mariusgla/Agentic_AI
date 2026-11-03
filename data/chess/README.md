UCI (Universal Chess Interface) beschreibt einen Zug rein über Start- und Zielfeld, z. B. e2e4 oder g1f3. Eine Umwandlung hängt den Figurenbuchstaben an, z. B. e7e8q. Das ist eindeutig, brauchst keinen Brettzustand zum Lesen und hat nur rund 1.900 mögliche Tokens. Für ein Move-Level-Vokabular ist es ideal.


Besonderheit bei Lichess-Dumps mit Uhrzeit-Annotationen

Viele Partien (vor allem neuere Monate) enthalten Kommentare mit der Restzeit nach jedem Halbzug, und nach einem Kommentar wird die Zugnummer für Schwarz als 1... wiederholt:

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Nf3 { [%clk 0:02:58] } 2... Nf6 { [%clk 0:02:57] } 3. Nxe5 ...

Bei Juni 2014 sind diese Annotationen meist noch nicht dabei, aber bei neueren Dateien schon. Deshalb entfernt clean_moves() aus meiner früheren Antwort Kommentare und 1...-Marker.

Nach der Bereinigung (Character-Level-Format für NanoGPT) sieht die Partie so aus:

;1.e4 e5 2.Nf3 Nf6 3.Nxe5 d6 4.Nf3 Nxe4 5.d4 d5 6.Bd3 Nc6 7.O-O Be7 8.c4 Nb4 9.Be2 O-O 10.Nc3 Nxc3 11.bxc3 Nc6 12.Qd3 Bf5 13.Qxf5

Und als UCI (Move-Level):

e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5 f1d3 b8c6 e1g1 f8e7 c2c4 c6b4 d3e2 e8g8 b1c3 e4c3 b2c3 b4c6 d1d3 c8f5 d3f5


UCI-Moves (full moves as tokens) are more efficient to train and with a short context, you can make every training sequence start at the beginning of a game, so the model always knows whose turn it is and where in the game it is.


What it does:

For each game it goes through these steps:

Read the PGN (.pgn or .pgn.zst) line by line.
Drop the game if either player’s Elo is below MIN_ELO (default 1400).
Extract the SAN moves and drop the game if it has fewer than 10 or more than BLOCK_SIZE - 1 half-moves. With the default cap of 128, the limit is 127 half-moves.
Replay the game on a board, translate each move to UCI, and drop the game if a move is illegal.
Write <start> moves... <end> padded with <pad> to BLOCK_SIZE + 1 tokens.

Defaults configs:

Minimum game length: 10 half-moves (MIN_MOVES).
Validation split: 0.5% of games, picked at random per game (VAL_FRAC).
Termination: no filter, so time-forfeit games are kept. Their moves are normal, only the ending differs.
Vocabulary: fixed at 1,971 tokens (<pad>=0, <start>=1, <end>=2, plus all 1,968 possible UCI moves), independent of your data.
Speed: a single process, no multiprocessing, to keep it simple. Replaying games with python-chess is the slow part, so expect it to run for a while on very large files. If it’s too slow, splitting the work per monthly file and running several instances in parallel is the easy fix.