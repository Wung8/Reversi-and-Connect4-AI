import sys; args = sys.argv[1:]
import math, time, random
import numpy as np
import torch

def shift(x,i):
  if i > 0: x<<=i
  else: x>>=-i
  return x

def remove(a,b):
  return a ^ b & a

  
def convBoard(board,player):
  if player == True: board = (board[1],board[0],board[2])
  x = [[1 & (board[i]>>j) for j in range(72)] for i in range(3)]
  x = np.array(x)
  #m = 72
  # converts board into a np array of 0s and 1s
  #x = (((1 & (x[:,None] >> np.arange(m).astype('int32')))) > 0).astype(int)
  x = np.reshape(x,(3,8,9))
  # gets rid of the 0 buffer on the right
  x = x[:,:,:-1]
  x = torch.from_numpy(x).float()
  return x


def setGlobals():
  global H,W,DIRS,DIRLOOKUP,OUTOFBOUNDS,CORNERMASK
  global COUNT

  COUNT = 0

  H,W = 8,9
  DIRS = [1,-1,8,-8,9,-9,10,-10]
  
  OUTOFBOUNDS = 0
  for i in range(H):
    OUTOFBOUNDS |= 2**(9*i+8)
  for i in range(W+1):
    OUTOFBOUNDS |= 2**(72+i)

  CORNERMASK = 1 + 2**7 + 2**63 + 2**70

def beginGame():
  return (1100585369600,551903297536,1652488667136)

def displayBit(x):
  print()
  for row in range(H+1):
    print((('0'*90+bin(x)[2:])[::-1][row*W:row*W+W]).replace('0','. ').replace('1','* '))

def displayBoard(board,player,indexed=False,return_only=False):
  moves = getMoves(board,player)
  players = {('0','0','0'):'.',('1','0','0'):'x',('0','1','0'):'o',('0','0','1'):'*',('1','1','0'):'!'}
  n = 65
  disp = '--------'
  for row in range(H):
    if indexed: players[('0','0','1')] = chr(n)
    zipped = zip(('0'*90+bin(board[0])[2:])[::-1][row*W:row*W+W-1],('0'*90+bin(board[1])[2:])[::-1][row*W:row*W+W-1],('0'*90+bin(moves)[2:])[::-1][row*W:row*W+W-1])
    disp += '\n'
    for z in zipped:
      disp += players[z] + ' '
      if indexed and z == ('0','0','1'):
        n += 1
        players[('0','0','1')] = chr(n)
  disp += '\n--------'
  if not return_only:
    print()
    print(disp)
    print(f'{board}, {player}')
    getCounts(board,p=True,player=player)
  return disp.strip()


def checkWin(board,player):
  pscore = sum([1 & board[player]>>i for i in range(72)])
  escore = sum([1 & board[not player]>>i for i in range(72)])
  if pscore == escore: return 0
  return [-1,1][pscore > escore]*20

def getCounts(board,p=False,player=None):
  oscore = sum([1 & board[0]>>i for i in range(72)])
  xscore = sum([1 & board[1]>>i for i in range(72)])
  if p:
    if str(player)=="None": print(f"x: {oscore}   o: {xscore}")
    elif player: print(f"x: {oscore}   o*: {xscore}")
    else: print(f"x*: {oscore}   o: {xscore}")
  return (oscore,xscore)

def getMoves(board,player,as_list=False):
  moves = 0
  enemy = board[not player]
  for d in DIRS:
    candidates = shift(board[player],d) & enemy
    while candidates:
      moves |= remove(shift(candidates,d),board[2])
      candidates = shift(candidates,d) & enemy
      
  moves = remove(moves,OUTOFBOUNDS)

  if as_list: return [2**i for i in range(72) if 2**i & moves]
  return moves

def makeMove(board,player,move):
  captures = 0
  enemy = board[not player]
  for d in DIRS:
    capt = 0
    mark = shift(move,d)
    while mark & enemy:
      capt |= mark
      mark = shift(mark,d)

    if mark & board[player]: captures |= capt

  if player: board = (remove(enemy,captures), board[player]|captures|move, board[2]|move)
  else: board = (board[player]|captures|move, remove(enemy,captures), board[2]|move)

  return board

def getNeighbors(board,player):
  toreturn = []
  for move in getMoves(board,player,as_list=True):
    toreturn.append(makeMove(board,player,move))

  return toreturn

setGlobals()

if __name__ == '__main__':
  board = beginGame()
  player = True
  while 1:
    displayBoard(board,player,indexed=True)
    usr = input()
    usr = ord(usr)-65
    board = getNeighbors(board,player)[usr]
    player = not player
    






