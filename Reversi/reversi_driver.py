import sys; args = sys.argv[1:]
import math, time, random
import numpy as np
import cv2

import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pickle

from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import game
game.setGlobals()

from network import Network
import network

def e1(board,player):
  return 1/math.log(board[player]+2)


NEARCS = [1+2**2,1+2**18,
          2**7+2**5,2**7+2**25,
          2**63+2**45,2**63+2**65,
          2**70+2**52,2**70+2**68]
CORNERS = [1,2**7,2**63,2**70]
use_model = True

def evaluate(board,player):
  if not use_model: return (random.random()-.5)/100
  i_board = game.convBoard(board,player).to(device)
  with torch.no_grad():
    val = model(i_board).item()

  for c in CORNERS:
    if board[player] & c: val += 4
    if board[not player] & c: val -= 4
    
  if val < -20: val=-20
  if val > 20: val=20
  return val


##def evaluate(board,player):
##  if not use_model: return (random.random()-.5)/100
##    #return len(game.getMoves(board,player,as_list=True))/100
##  
##  global model
##
##  r = 0
##  for c in CORNERS:
##    if board[player] & c: r += 8
##  r = min(r,10)
##  #for i,c in enumerate(NEARCS):
##  #  if board[player] & c == c-CORNERS[i//2]: r += .3
##
##  board = game.convBoard(board,player).to(device)
##  with torch.no_grad():
##    val = model(board)
##  val = val.item()
##  val += r/(max(1,val))
##  if val > 15.0: val = 15.0
##  if val < -15.0: val = -15.0
##
##  if val == 0: print(f'{board},{player}')
##  
##  return val


def displayBoardPlt(board):
    H,W = 8,9
    colors = {('0','0'):(50,100,50),('1','0'):(255,255,255),('0','1'):(0,0,0)}
    scale = int(200 / len(board))

    img = []
    for row in range(H):
        img_row = []
        zipped = zip(('0'*90+bin(board[0])[2:])[::-1][row*W:row*W+W-1],('0'*90+bin(board[1])[2:])[::-1][row*W:row*W+W-1])
        for z in zipped:
            img_row.append(colors[z])
        img.append(img_row)
    
  
    img = np.array(img,dtype=np.uint8)
    img = img.repeat(scale,axis=0).repeat(scale,axis=1)
    cv2.imshow("img", img)
    cv2.waitKey(20)


def play(board=game.beginGame(),p1=True,player=True,ptype=1):
  while 1:
    game.displayBoard(board,player,indexed=player==p1)
    displayBoardPlt(board)
    if player==p1:
      if ptype==2: board = random.choice(game.getNeighbors(board,player))
      if ptype==1:
        usr = input().upper()
        usr = ord(usr)-65
        board = game.getNeighbors(board,player)[usr]
    else:
      COUNT = 0
      TIME = time.time()
      iters = 0
      while iters<4000 or time.time()-TIME<1:
        iters += 1
        mcts(board,player)
      val,newboard = mcts(board,player,return_move=True)
      print(f'\n\nP: {evaluate(board,player)}   Q: {Q[board][1]}   evaluation: {val}')
      move = int(math.log(board[2]^newboard[2],2))+1
      x,y = move%9,move//9
      board = newboard
      print(f'searched {COUNT} moves {iters} iterations in {str(time.time()-TIME)[:5]}s')
      print(f'played to {x},{y}')
    player = not player
    if not game.getMoves(board,player): break
  game.displayBoard(board,player)
  displayBoardPlt(board)

def norm(val):
  return (val+20)/40

def getU(p,q,n,sp):
  c = .4
  c_punct = .7
  p,q = -p,-q
  u = (norm(q)*n+1.5)/(n+1) + c_punct*norm(p)/((n+1)**.5) + c*((math.log(sp)/n)**.5)
  return u

def mcts(board,player,return_move=False):
  # Q = {B(board):[P(initial evaluation),Q(average reward),N(number visits)]}
  global Q
  global model

  if not game.getMoves(board,player):
    val = game.checkWin(board,player)
    Q[board] = [val,val,1]
    return val
    
  if board not in Q:
    val = evaluate(board,player)
    Q[board] = [val,0,1]
    return val

  sp = Q[board][2]
  max_u, best_a = -999, -1
  for nbr in game.getNeighbors(board,player):
    # u is how a move is ranked based on exploitation + exploration 
    if nbr not in Q: u = 999
    else: u = getU(*Q[nbr],sp)

    if u > max_u:
      max_u = u
      best_a = nbr

  # if no moves possible, return draw
  if max_u == -999: print('ERROR')

  nbr = best_a
  val = -mcts(nbr,not player)

  q,n = Q[board][1:]
  Q[board][1] = (q*n+val)/(n+1)
  Q[board][2] += 1

  if return_move:
    boards = []
    for nbr in game.getNeighbors(board,player):
      boards.append((-Q[nbr][1],nbr,Q[nbr][2]))
    return max(boards)[:2]

  return val

alpha = .1
e = .1

def clip(board,player,val):
  qval = evaluate(board,player)
  a,b = (1-e)*qval-1,(1+e)*qval+1
  if val < a: val = a
  if val > b: val = b
  return val

def selfPlay(mc_iter=3000,evals=False,e_greedy=0.1):
  global COUNT
  global Q
  print('.',end='',flush=True)
  train_vals = []
  board = game.beginGame()
  player = True
  qvals = []
  c = 0
  while game.getMoves(board,player):
    c += 1
    if evals:
      game.displayBoard(board,player)
      displayBoardPlt(board)
    times_seen = 0
    if board in Q:
      times_seen = Q[board][2]
    for i in range(int(max(500,mc_iter-times_seen/3))):
      mcts(board,player)
    val,newboard = mcts(board,player,return_move=True)

    if not evals and random.random() > 1-e_greedy:
    #if not evals and c<40 and random.random() > 1-e_greedy-[0,.1][c<=10]:
      newboard = random.choice(game.getNeighbors(board,player))
      
    if evals: print(f'\n\nP: {Q[board][0]}   Q: {Q[board][1]}   evaluation: {val}')
    if train_vals:
      train_vals[-1][1] += alpha*(-val-train_vals[-1][1])
      #train_vals[-1][1] = clip(*train_vals[-1][2],train_vals[-1][1])
    train_vals.append([game.convBoard(board,player),min(max(val,-20),20),(board,player)])

    #if (board[player] ^ newboard[player]) & game.CORNERMASK:
    #  dv = 8
    #  for i in range(len(train_vals)):
    #    vv = train_vals[-i-1][1]
    #    train_vals[-i-1][1] += dv/max(1,vv)
    #    dv *= -decay
         
    board = newboard
    player = not player
  val = game.checkWin(board,player)
  train_vals.append((game.convBoard(board,player),val,player))
  if evals:
      game.displayBoard(board,player)
      displayBoardPlt(board)
  return train_vals



EEE = 25
if __name__ == '__main__':
  model = Network()
  model.load_state_dict(torch.load(f'reversi-model_{EEE}.pt',map_location=torch.device(device)))
  model.to(device)

  Q = {}
  usr = input("play [0], train [1] ")=="1"
  
  if not usr:
    p1 = input("player 1 or 2? ")=="1"
    ptype = int(input("player [0], minimax [1], self play [2] "))
    play(p1=p1,ptype=ptype)
    exit()
  
  usr = input("default [0], custom [1] ")=="1"
  save,episodes,repeat,lr = 1,1500,2,.0005
  if usr:
    save = int(input("save every: "))
    episodes = int(input("train every: "))
    repeat = int(input("repeat training: "))
    lr = float(input("learning rate: "))

  usr = 'y' in input("train base? [y/N] ")
    
  batch_size = 1
  optimizer = optim.Adam(model.parameters(), lr=lr)
  loss_fn = torch.nn.HuberLoss()
  
  #selfPlay(evals=True)
  
  for epoch in range(EEE+1,5000):
    device = 'cpu'
    model.to(device)

    selfPlay(evals=True)
    print(f"epoch {epoch}")
    
    Q,QVALS,newvals = {},[],[]
    if epoch == 0 and usr: use_model=False
    else: use_model=True

    newvals = []
    #newvals = sum(Parallel(n_jobs=4)(delayed(selfPlay)(mc_iter=[5000,3600][use_model],e_greedy=.05) for ep in range([500,episodes][use_model])),[])
    newvals = sum(Parallel(n_jobs=12)(delayed(selfPlay)() for ep in range(episodes)),[])


    #for ep in range([500,episodes][use_model]):
    #  print('.',end='',flush=True)
    #  newvals += selfPlay(mc_iter=[5000,3600][use_model],e_greedy=.05)
    #print()

    device = 'cuda'
    model.to(device)
    train_data = network.DataLoader(newvals)
    QVALS += newvals

    train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=0,
     )

    print()
    for j in range(repeat):
      total_loss = 0
      for i, (boards,vals) in enumerate(train_dataloader):
        boards,vals = boards.squeeze().to(device),vals.to(device)

        outputs = model(boards)
        loss = loss_fn(outputs, vals)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      print(total_loss/len(newvals))

    if epoch%save == 0:
      Q = {}
      torch.save(model.state_dict(), f'reversi-model_{epoch}.pt')
      with open(f'qvals_{epoch}.pkl', 'wb') as f:
        pickle.dump(QVALS, f)
        
    exit()

  print('Finished Training')

  torch.save(model.state_dict(), 'othello-model_final.pt')







