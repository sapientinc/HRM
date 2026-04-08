#!/usr/bin/env python
#coding: utf-8
from __future__ import print_function
import numpy as np
import random

'''
sticker indices:
       ┌──┬──┐
       │ 0│ 1│
       ├──┼──┤
       │ 2│ 3│
 ┌──┬──┼──┼──┼──┬──┬──┬──┐
 │16│17│ 8│ 9│ 4│ 5│20│21│
 ├──┼──┼──┼──┼──┼──┼──┼──┤
 │18│19│10│11│ 6│ 7│22│23│
 └──┴──┼──┼──┼──┴──┴──┴──┘
       │12│13│
       ├──┼──┤
       │14│15│
       └──┴──┘

face colors:
    ┌──┐
    │ 0│
 ┌──┼──┼──┬──┐
 │ 4│ 2│ 1│ 5│
 └──┼──┼──┴──┘
    │ 3│
    └──┘

moves:
[ U , U', U2, R , R', R2, F , F', F2, D , D', D2, L , L', L2, B , B', B2, x , x', x2, y , y', y2, z , z', z2]

'''

# move indices
moveInds = { \
  "U": 0, "U'": 1, "U2": 2, "R": 3, "R'": 4, "R2": 5, "F": 6, "F'": 7, "F2": 8, \
  "D": 9, "D'": 10, "D2": 11, "L": 12, "L'": 13, "L2": 14, "B": 15, "B'": 16, "B2": 17, \
  "x": 18, "x'": 19, "x2": 20, "y": 21, "y'": 22, "y2": 23, "z": 24, "z'": 25, "z2": 26 \
}

# move definitions
moveDefs = np.array([ \
  [  2,  0,  3,  1, 20, 21,  6,  7,  4,  5, 10, 11, 12, 13, 14, 15,  8,  9, 18, 19, 16, 17, 22, 23], \
  [  1,  3,  0,  2,  8,  9,  6,  7, 16, 17, 10, 11, 12, 13, 14, 15, 20, 21, 18, 19,  4,  5, 22, 23], \
  [  3,  2,  1,  0, 16, 17,  6,  7, 20, 21, 10, 11, 12, 13, 14, 15,  4,  5, 18, 19,  8,  9, 22, 23], \
  [  0,  9,  2, 11,  6,  4,  7,  5,  8, 13, 10, 15, 12, 22, 14, 20, 16, 17, 18, 19,  3, 21,  1, 23], \
  [  0, 22,  2, 20,  5,  7,  4,  6,  8,  1, 10,  3, 12,  9, 14, 11, 16, 17, 18, 19, 15, 21, 13, 23], \
  [  0, 13,  2, 15,  7,  6,  5,  4,  8, 22, 10, 20, 12,  1, 14,  3, 16, 17, 18, 19, 11, 21,  9, 23], \
  [  0,  1, 19, 17,  2,  5,  3,  7, 10,  8, 11,  9,  6,  4, 14, 15, 16, 12, 18, 13, 20, 21, 22, 23], \
  [  0,  1,  4,  6, 13,  5, 12,  7,  9, 11,  8, 10, 17, 19, 14, 15, 16,  3, 18,  2, 20, 21, 22, 23], \
  [  0,  1, 13, 12, 19,  5, 17,  7, 11, 10,  9,  8,  3,  2, 14, 15, 16,  6, 18,  4, 20, 21, 22, 23], \
  [  0,  1,  2,  3,  4,  5, 10, 11,  8,  9, 18, 19, 14, 12, 15, 13, 16, 17, 22, 23, 20, 21,  6,  7], \
  [  0,  1,  2,  3,  4,  5, 22, 23,  8,  9,  6,  7, 13, 15, 12, 14, 16, 17, 10, 11, 20, 21, 18, 19], \
  [  0,  1,  2,  3,  4,  5, 18, 19,  8,  9, 22, 23, 15, 14, 13, 12, 16, 17,  6,  7, 20, 21, 10, 11], \
  [ 23,  1, 21,  3,  4,  5,  6,  7,  0,  9,  2, 11,  8, 13, 10, 15, 18, 16, 19, 17, 20, 14, 22, 12], \
  [  8,  1, 10,  3,  4,  5,  6,  7, 12,  9, 14, 11, 23, 13, 21, 15, 17, 19, 16, 18, 20,  2, 22,  0], \
  [ 12,  1, 14,  3,  4,  5,  6,  7, 23,  9, 21, 11,  0, 13,  2, 15, 19, 18, 17, 16, 20, 10, 22,  8], \
  [  5,  7,  2,  3,  4, 15,  6, 14,  8,  9, 10, 11, 12, 13, 16, 18,  1, 17,  0, 19, 22, 20, 23, 21], \
  [ 18, 16,  2,  3,  4,  0,  6,  1,  8,  9, 10, 11, 12, 13,  7,  5, 14, 17, 15, 19, 21, 23, 20, 22], \
  [ 15, 14,  2,  3,  4, 18,  6, 16,  8,  9, 10, 11, 12, 13,  1,  0,  7, 17,  5, 19, 23, 22, 21, 20], \
  [  8,  9, 10, 11,  6,  4,  7,  5, 12, 13, 14, 15, 23, 22, 21, 20, 17, 19, 16, 18,  3,  2,  1,  0], \
  [ 23, 22, 21, 20,  5,  7,  4,  6,  0,  1,  2,  3,  8,  9, 10, 11, 18, 16, 19, 17, 15, 14, 13, 12], \
  [ 12, 13, 14, 15,  7,  6,  5,  4, 23, 22, 21, 20,  0,  1,  2,  3, 19, 18, 17, 16, 11, 10,  9,  8], \
  [  2,  0,  3,  1, 20, 21, 22, 23,  4,  5,  6,  7, 13, 15, 12, 14,  8,  9, 10, 11, 16, 17, 18, 19], \
  [  1,  3,  0,  2,  8,  9, 10, 11, 16, 17, 18, 19, 14, 12, 15, 13, 20, 21, 22, 23,  4,  5,  6,  7], \
  [  3,  2,  1,  0, 16, 17, 18, 19, 20, 21, 22, 23, 15, 14, 13, 12,  4,  5,  6,  7,  8,  9, 10, 11], \
  [ 18, 16, 19, 17,  2,  0,  3,  1, 10,  8, 11,  9,  6,  4,  7,  5, 14, 12, 15, 13, 21, 23, 20, 22], \
  [  5,  7,  4,  6, 13, 15, 12, 14,  9, 11,  8, 10, 17, 19, 16, 18,  1,  3,  0,  2, 22, 20, 23, 21], \
  [ 15, 14, 13, 12, 19, 18, 17, 16, 11, 10,  9,  8,  3,  2,  1,  0,  7,  6,  5,  4, 23, 22, 21, 20]  \
])

# piece definitions
pieceDefs = np.array([ \
  [  0, 21, 16], \
  [  2, 17,  8], \
  [  3,  9,  4], \
  [  1,  5, 20], \
  [ 12, 10, 19], \
  [ 13,  6, 11], \
  [ 15, 22,  7], \
])

# OP representation from (hashed) piece stickers
pieceInds = np.zeros([58, 2], dtype=np.int_)
pieceInds[50] = [0, 0]; pieceInds[54] = [0, 1]; pieceInds[13] = [0, 2]
pieceInds[28] = [1, 0]; pieceInds[42] = [1, 1]; pieceInds[ 8] = [1, 2]
pieceInds[14] = [2, 0]; pieceInds[21] = [2, 1]; pieceInds[ 4] = [2, 2]
pieceInds[52] = [3, 0]; pieceInds[15] = [3, 1]; pieceInds[11] = [3, 2]
pieceInds[47] = [4, 0]; pieceInds[30] = [4, 1]; pieceInds[40] = [4, 2]
pieceInds[25] = [5, 0]; pieceInds[18] = [5, 1]; pieceInds[35] = [5, 2]
pieceInds[23] = [6, 0]; pieceInds[57] = [6, 1]; pieceInds[37] = [6, 2]

# piece stickers from OP representation
pieceCols = np.zeros([7, 3, 3], dtype=np.int_)
pieceCols[0, 0, :] = [0, 5, 4]; pieceCols[0, 1, :] = [4, 0, 5]; pieceCols[0, 2, :] = [5, 4, 0]
pieceCols[1, 0, :] = [0, 4, 2]; pieceCols[1, 1, :] = [2, 0, 4]; pieceCols[1, 2, :] = [4, 2, 0]
pieceCols[2, 0, :] = [0, 2, 1]; pieceCols[2, 1, :] = [1, 0, 2]; pieceCols[2, 2, :] = [2, 1, 0]
pieceCols[3, 0, :] = [0, 1, 5]; pieceCols[3, 1, :] = [5, 0, 1]; pieceCols[3, 2, :] = [1, 5, 0]
pieceCols[4, 0, :] = [3, 2, 4]; pieceCols[4, 1, :] = [4, 3, 2]; pieceCols[4, 2, :] = [2, 4, 3]
pieceCols[5, 0, :] = [3, 1, 2]; pieceCols[5, 1, :] = [2, 3, 1]; pieceCols[5, 2, :] = [1, 2, 3]
pieceCols[6, 0, :] = [3, 5, 1]; pieceCols[6, 1, :] = [1, 3, 5]; pieceCols[6, 2, :] = [5, 1, 3]

# useful arrays for hashing
hashOP = np.array([1, 2, 10])
pow3 = np.array([1, 3, 9, 27, 81, 243])
pow7 = np.array([1, 7, 49, 343, 2401, 16807])
fact6 = np.array([720, 120, 24, 6, 2, 1])

# get FC-normalized solved state
def initState():
  return np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])

# apply a move to a state
def doMove(s, move):
  return s[moveDefs[move]]

# apply multiple moves based on move indices
def doMoves(s, moves):
  for move in moves:
    s = doMove(s, move)
  return s

# apply a string sequence of moves to a state
def doAlgStr(s, alg):
  moves = alg.split(" ")
  for m in moves:
    if m in moveInds:
      s = doMove(s, moveInds[m])
  return s

# generate a random scramble sequence
def scramble(length=20):
  s = initState()
  move_names = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
  moves = []
  
  for _ in range(length):
    move = random.choice(move_names)
    moves.append(move)
    s = doMove(s, moveInds[move])
  
  return s

# check if state is solved
def isSolved(s):
  for i in range(6):
    if not (s[4 * i:4 * i + 4] == s[4 * i]).all():
      return False
  return True

# normalize stickers relative to a fixed DLB corner
def normFC(s):
  normCols = np.zeros(6, dtype=np.int_)
  normCols[s[18] - 3] = 1
  normCols[s[23] - 3] = 2
  normCols[s[14]] = 3
  normCols[s[18]] = 4
  normCols[s[23]] = 5
  return normCols[s]

# get OP representation given FC-normalized sticker representation
def getOP(s):
  return pieceInds[np.dot(s[pieceDefs], hashOP)]

# get sticker representation from OP representation
def getStickers(sOP):
  s = np.zeros(24, dtype=np.int_)
  s[[14, 18, 23]] = [3, 4, 5]
  for i in range(7):
    s[pieceDefs[i]] = pieceCols[sOP[i, 0], sOP[i, 1], :]
  return s

# get a unique index for the piece orientation state (0-728)
def indexO(sOP):
  return np.dot(sOP[:-1, 1], pow3)

# get a unique index for the piece permutation state (0-117648)
def indexP(sOP):
  return np.dot(sOP[:-1, 0], pow7)

# get a (gap-free) unique index for the piece permutation state (0-5039)
def indexP2(sOP):
  return np.dot([sOP[i, 0] - np.count_nonzero(sOP[:i, 0] < sOP[i, 0]) for i in range(6)], fact6)
  '''
  ps = np.arange(7)
  P = 0
  for i, p in enumerate(sOP[:, 0]):
    P += fact6[i] * np.where(ps == p)[0][0]
    ps = ps[ps != p]
  return P
  '''
  

# get a unique index for the piece orientation and permutation state (0-3674159)
def indexOP(sOP):
  return indexO(sOP) * 5040 + indexP2(sOP)

# print state of the cube
def printCube(s):
  print("      ┌──┬──┐")
  print("      │ {}│ {}│".format(s[0], s[1]))
  print("      ├──┼──┤")
  print("      │ {}│ {}│".format(s[2], s[3]))
  print("┌──┬──┼──┼──┼──┬──┬──┬──┐")
  print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(s[16], s[17], s[8], s[9], s[4], s[5], s[20], s[21]))
  print("├──┼──┼──┼──┼──┼──┼──┼──┤")
  print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(s[18], s[19], s[10], s[11], s[6], s[7], s[22], s[23]))
  print("└──┴──┼──┼──┼──┴──┴──┴──┘")
  print("      │ {}│ {}│".format(s[12], s[13]))
  print("      ├──┼──┤")
  print("      │ {}│ {}│".format(s[14], s[15]))
  print("      └──┴──┘")

if __name__ == "__main__":
  # get solved state
  s = initState()
  printCube(s)
  # do some moves
  s = doAlgStr(s, "x y R U' R' U' F2 U' R U R' U F2")
  printCube(s)
  # normalize stickers relative to DLB
  s = normFC(s)
  printCube(s)

 