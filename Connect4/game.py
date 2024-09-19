import sys; args = sys.argv[1:]
import math, time, random
import numpy as np
import torch


'''
setGlobals
beginGame
convBoard
displayBoard
getNeighbors
randomMove

'''


def ffs(i):
  return i&-i

def setGlobals():
  global H,W,DIRS,DIRLOOKUP,COLLOOKUP,ROWMASK,COLMASK

  H,W = 6,8
  DIRLOOKUP,DIRS = [],{1,-1,7,-7,8,-8,9,-9}

  # top left is 0th index
  # 6x8 board, rightmost is going to be all 0's to make the math simpler
  for i in range(48):
    DIRLOOKUP.append(DIRS.copy())
    if i//8 == 0: DIRLOOKUP[i] ^= {-7,-8,-9}
    if i//8 == 5: DIRLOOKUP[i] ^= {7,8,9}
  DIRLOOKUP[0] ^= {-1}
  DIRLOOKUP[-1] ^= {1}

  ROWMASK = [int('1111111',2)<<(i*8) for i in range(7)]
  COLMASK = [int('01'*7,16)<<i for i in range(7)]

  global WINMASK
  WINMASK = []
  horz = int('1111',2)<<24
  for i in range(4): WINMASK.append(horz<<i)
  vert = int('01010101',16)<<3
  for i in range(4): WINMASK.append(vert<<(i*8))
  dia1 = int('000000001'*4,2)
  for i in range(4): WINMASK.append(dia1<<(i*9))
  dia2 = int('0000001'*4+'000000',2)
  for i in range(4): WINMASK.append(dia2<<(i*7))


def beginGame():
  return (0,0,0)

def getRows(board):
  return [(board & ROWMASK[i])>>(i*8) for i in range(7)]

def getCols(board):
  return [(board & COLMASK[i]) for i in range(7)]

# converts 3x48 to 3x6x7
def convBoard(board,player=True):
  if player == False: board = (board[1],board[0],board[2])
  x = np.array(board)
  m = 48
  # converts board into a np array of 0s and 1s
  x = (((x[:,None] & (1 << np.arange(m).astype('int64')))) > 0).astype(int)
  x = np.reshape(x,(3,6,8))
  # gets rid of the 0 buffer on the right
  x = x[:,:,:-1]
  x = torch.from_numpy(x).float()
  return x
  

def displayBoard(board,player=None,return_only=False):
  players = {('0','0'):'.',('1','0'):'x',('0','1'):'o',('1','1'):'!'}
  disp = '-------'
  for row in range(6):
    zipped = zip(('0'*90+bin(board[0])[2:])[::-1][row*8:row*8+7],('0'*90+bin(board[1])[2:])[::-1][row*8:row*8+7])
    disp += '\n'+' '.join([players[z] for z in zipped])
  disp += '\n-------'
  if not return_only:
    print()
    print(disp)
    print('1 2 3 4 5 6 7')
    print(board,',',player)
  return disp.strip()

def displayBit(board):
  for row in range(7):
    print(bin(board)[2:][::-1][row*8:row*8+7])
  print()
  
# if indexed==True will return list of possible boards with invalid moves as -1
def getNeighbors(board,player,indexed=False,return_move=False):
  toreturn = []
  for n,col in enumerate(getCols(board[2])):
    if col & ROWMASK[0]:
      if indexed: toreturn.append(-1)
      continue
    i = ffs(col)>>8
    if not i: i = 1<<(40+n)
    newboard = (board[0]|(i*(not player)), board[1]|(i*player), board[2]|i)
    if return_move: toreturn.append((newboard,i))
    else: toreturn.append(newboard)

  return toreturn
    
def checkWin(board,player,move):
  check = board[player]
  shift = 27-int(math.log(move,2))
  if shift < 0: check>>=abs(shift)
  if shift > 0: check<<=abs(shift)
    
  for mask in WINMASK:
    if check & mask == mask:
      return True

  return False



def randomMove(board,player):
  moves = getNeighbors
  if moves:
    return random.choice(moves)
  return -1

# player == True if o player == False if x
# board = [x tokens, o tokens, x+o tokens] in bits
if __name__ == '__main__':
  setGlobals()
  board = beginGame()
  player = True
  while 1:
    displayBoard(board)
    if 0 and player:
      nbrs = getNeighbors(board,player,indexed=True)
      usr = int(input())-1
    else:
      nbrs = getNeighbors(board,player,indexed=False)
      usr = random.randint(0,len(nbrs)-1)
    move = board[2]^nbrs[usr][2]
    board = nbrs[usr]
    if checkWin(board,player,move): break
    player = not player

  print(['x','o'][player],'wins')
  displayBoard(board)






    
