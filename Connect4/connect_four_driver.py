import sys; args = sys.argv[1:]
import math, time, random
import numpy as np

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

import game
game.setGlobals()

from network import Network
import network

COUNT = 0

def e1(board,player):
  return 1/math.log(board[player]+2)

use_model = True
def evaluate(board,player):
  if not use_model: return 0
  
  global model
  board = game.convBoard(board,player).to(device)
  with torch.no_grad():
    val = model(board)
  val = val.item()
  if val > 15: val = 15.0
  if val < -15: val = -15.0
  return val

def play(p1=True,ptype=0):
  global COUNT
  global TRANSTABLE
  global TIME

  global Q
  Q = {}
  
  board = game.beginGame()

  player = True
  qvals = []
  while 1:
    game.displayBoard(board,player)    
    if player==p1:
      # MCTS
      if ptype == 2:
        COUNT = 0
        TIME = time.time()
        depth = 10
        iters = 0
        log = True
        while iters<3600 or time.time()-TIME<1:
          iters += 1
          mcts(board,player)
        val,newboard,move = mcts(board,player,return_move=True)
        print(f'\n\nP: {evaluate(board,player)}   evaluation: {val}')
        board = newboard
        print(f'searched {COUNT} moves {iters} iterations in {str(time.time()-TIME)[:5]}s')
      
      # MINIMAX
      if ptype == 1:
        TIME = time.time()
        depth = 10
        while time.time()-TIME < .1:
          COUNT = 0
          TRANSTABLE = {}
          val,newboard,move = minimax(board,player,depth=depth,return_move=True)
          depth += 5
          if depth==40: break
        print(f'depth: {depth}')
        board = newboard

      # PLAYER
      if ptype == 0:
        nbrs = game.getNeighbors(board,player,indexed=True)
        usr = -1
        while 1:
          try:
            usr = int(input())-1
            if 0<=usr<=6:
              if nbrs[usr]!=-1: break
          except KeyboardInterrupt:
            exit()
          except:
            xxx = 0
          print('invalid input')
        move = board[2]^nbrs[usr][2]
        board = nbrs[usr]
    else:
      COUNT = 0
      TIME = time.time()
      depth = 10
      iters = 0
      log = True
      while iters<4000 or time.time()-TIME<1:
      #for i in range(3600):
        iters += 1
        mcts(board,player)
      val,newboard,move = mcts(board,player,return_move=True)
      print(f'\n\nP: {evaluate(board,player)}   Q: {Q[board][1]}   evaluation: {val}')
      board = newboard
      print(f'searched {COUNT} moves {iters} iterations in {str(time.time()-TIME)[:5]}s')
    if game.checkWin(board,player,move): break
    if not game.getNeighbors(board,player): break
    player = not player
  game.displayBoard(board)
  return board


def selfPlay(mc_iter=2400,evals=False,e_greedy=0):
  global COUNT
  global Q
  train_vals = []
  board = game.beginGame()
  player = True
  qvals = []
  c = 0
  while 1:
    c += 1
    if evals:
      game.displayBoard(board,player)
    times_seen = 0
    if board in Q:
      times_seen = Q[board][2]
    for i in range(int(max(500,mc_iter-times_seen/3))):
      mcts(board,player)
    val,newboard,move = mcts(board,player,return_move=True)
    
    if not evals and c<20 and random.random() > 1-e_greedy-[0,.2][c<=6]:
      newboard,move = random.choice(game.getNeighbors(board,player,return_move=True))

    if c <= 6:
      boards = []
      for nbr,move in game.getNeighbors(board,player,return_move=True):
        boards.append((-Q[nbr][1]+random.gauss(sigma=1.25),nbr,move,Q[nbr][2]))
      val,newboard,move = max(boards)[:3]
      
    if evals: print(f'\n\nP: {Q[board][0]}   Q: {Q[board][1]}   evaluation: {val}')
    train_vals.append((game.convBoard(board,player),val,player))
    board = newboard
    if game.checkWin(board,player,move): break
    if not game.getNeighbors(board,player): break
    player = not player
  player = not player
  val = -val
  train_vals.append((game.convBoard(board,player),val,player))
  if evals: game.displayBoard(board,player)
  return train_vals


# squishes evaluation of a board (-20 to 20) to be between 0 and 1 for ucb
def norm(val):
  return (val+20)/40  

def mcts(board,player,move=1,sp=1,return_move=False):
  global COUNT
  # Q = {B(board):[P(initial evaluation),Q(average reward),N(number visits)]}
  global Q
  global model

  # exploration importance over exploitation
  c = .4
  c_punct = .7

  COUNT += 1
  if game.checkWin(board,not player,move):
    Q[board] = [-20,-20,1]
    return -20

  if board not in Q:
    val = evaluate(board,player)
    Q[board] = [val,0,1]
    return val

  sp = Q[board][2]
  max_u, best_a = -999, -1
  for nbr,move in game.getNeighbors(board,player,return_move=True):
    # u is how a move is ranked based on exploitation + exploration 
    if nbr not in Q: u = 999
    else:
      p,q,n = Q[nbr]
      p,q = -p,-q
      u = norm((q*n+30)/(n+1)) + c_punct*(norm(p)/((n+1)**.5)) + c*((math.log(sp)/n)**.5)

    if u > max_u:
      max_u = u
      best_a = (nbr,move)

  # if no moves possible, return draw
  if max_u == -999: return 0

  nbr,move = best_a
  val = -mcts(nbr,not player,move=move,sp=Q[board][2])

  q,n = Q[board][1:]
  Q[board][1] = (q*n+val)/(n+1)
  Q[board][2] += 1

  if return_move:
    boards = []
    for nbr,move in game.getNeighbors(board,player,return_move=True):
      boards.append((-Q[nbr][1],nbr,move,Q[nbr][2]))
      #print(-Q[nbr][1],-mcts(nbr,not player,move=move,sp=Q[board][2]),Q[nbr][2])
    return max(boards)[:3]

  return val


def minimax(board,player,depth,alpha=-999,beta=999,move=1,return_move=False):
  global COUNT
  global TRANSTABLE
  global TIME

  if game.checkWin(board,not player,move): return -20

  if board in TRANSTABLE: return TRANSTABLE[board]
  
  COUNT += 1

  best = 0
  falloff = .98

  # if max depth reached, evaluate position and return
  if depth == 0: best = e1(board,player)
  
  else:
    boards = []
    for nbr,move in game.getNeighbors(board,player,return_move=True):
      val = -minimax(nbr,not player,depth-1,-beta,-alpha,move)*falloff
      if return_move: boards.append((val,nbr,move))
      else: boards.append(val)

      if val > beta: break
      if val > alpha: alpha = val

    if return_move:
      for b in boards: print(int(math.log(b[2],2)%8+1),b[0])

    # if no moves, return draw. else return value of the best move
    if boards: best = max(boards)

  TRANSTABLE[board] = best
  return best


if __name__ == '__main__':
  model = Network()
  model.load_state_dict(torch.load('c4-model_30.pt',map_location=torch.device(device)))
  #model.load_state_dict(torch.load('c4-model_225.pt',map_location=torch.device(device)))
  model.to(device)

  Q = {}
  usr = input("play [0], train [1] ")=="1"
  
  if not usr:
    p1 = input("player 1 or 2? ")=="1"
    ptype = int(input("player [0], minimax [1], self play [2] "))
    play(p1=p1,ptype=ptype)
    exit()
  
  usr = input("default [0], custom [1] ")=="1"
  save,episodes,repeat,lr = 1,250,2,.0005
  if usr:
    save = int(input("save every: "))
    episodes = int(input("train every: "))
    repeat = int(input("repeat training: "))
    lr = float(input("learning rate: "))

  usr = input("train base? [y/N] ")=='y'
    
  batch_size = 1
  optimizer = optim.Adam(model.parameters(), lr=lr)
  loss_fn = torch.nn.MSELoss()
  
  play(ptype=2)
  
  for epoch in range(31,5000):
    print(f"epoch {epoch}")
    Q,QVALS,newvals = {},[],[]
    if epoch == 0 and usr: use_model=False
    else: use_model=True
      
    for ep in range([500,episodes][use_model]):
      print('.',end='',flush=True)
      newvals += selfPlay(mc_iter=[5000,3600][use_model],e_greedy=.05)
    print()

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
      torch.save(model.state_dict(), f'c4-model_{epoch}.pt')
      with open(f'qvals_{epoch}.pkl', 'wb') as f:
        pickle.dump(QVALS, f) 
      #selfPlay(evals=True)
      play(ptype=2)


  print('Finished Training')

  torch.save(model.state_dict(), 'c4-model_final.pt')










