import pygame
import threading

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

def getMove(board, player):
  global EVAL
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
  EVAL = round(val,1)
  print(f'searched {COUNT} moves {iters} iterations in {str(time.time()-TIME)[:5]}s')
  print(f'played to {x},{y}')
  return move, newboard

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

def scale(tp, s):
  return tuple(int(x*s) for x in tp)

def combine(tp1, tp2):
  return tuple(int(tp1[i]+tp2[i]) for i in range(len(tp1)))

T_PRESSED = False
EVAL_TOGGLE = False
def win_mainloop(stop_event):
  global USR
  global T_PRESSED, EVAL_TOGGLE
  pygame.init()
  winsize = (900, 900)
  display = pygame.display.set_mode(winsize, pygame.RESIZABLE)
  clock = pygame.time.Clock()
  pygame.font.init()
  font = pygame.font.SysFont('Pokemon GB', 50)

  while not stop_event.is_set():
    pygame.event.pump()
    board, player = BOARD, PLAYER
    display.fill((35,65,30))  # fill screen with black
    todisplay = game.displayBoard(board, player=PLAYER, return_only=True).replace(' ','').split('\n')
    colors = {'.':(20,50,20),
              '*':(60,80,20),
              'x':(0,0,0),
              'o':(255,255,255)}

    valid_moves = set()
    for y in range(1,9):
      row = todisplay[y]
      for x in range(8):
        color = colors[row[x]]
        pygame.draw.circle(display, color, (x*100+100,y*100), 40)
        if row[x]=='*': valid_moves.add((x,y-1))

    x,y = pygame.mouse.get_pos()
    x = round((x-100)/100)
    x = max(min(x,7),0)
    y = round((y-100)/100)
    y = max(min(y,7),0)
    valid_move = (x,y) in valid_moves
    if valid_move: pygame.draw.circle(display, (90,110,40), (x*100+100,y*100+100), 40)

    if EVAL_TOGGLE:
      text = font.render(f'{EVAL}', False, (255,255,255))
      text_rect = (800, 850)
      display.blit(text,text_rect)
    
    if USR == -1:
      usr = -1
      for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
          if event.button == 1:  # Left mouse button
            if valid_move:
              usr = (x,y)
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_r: # r=restart
            usr = -2
          if event.key == pygame.K_t and not T_PRESSED: 
            T_PRESSED = True
            EVAL_TOGGLE = not EVAL_TOGGLE
        if event.type == pygame.KEYUP:
          if event.key == pygame.K_t:
            T_PRESSED = False
      USR = usr

            
    clock.tick(30)   # runs game at fps
    pygame.display.update()

  pygame.quit()


def startscreen_mainloop():
  global USR
  pygame.init()
  winsize = (900, 900)
  display = pygame.display.set_mode(winsize, pygame.RESIZABLE)
  clock = pygame.time.Clock()
  pygame.font.init()
  font = pygame.font.SysFont('Pokemon GB', 150)
  p1 = None
  
  while p1 == None:
    pygame.event.pump()
    display.fill((20,50,20))  # fill screen with black

    x,y = pygame.mouse.get_pos()
    if x<400: pygame.draw.rect(display, scale((255,255,255),0.7), pygame.Rect(0, 0, 450, 900))
    else: pygame.draw.rect(display, scale((0,0,0),0.7), pygame.Rect(450, 0, 450, 900))
    
    text = font.render('P1', False, (255,255,255))
    text_rect = text.get_rect(center=(225,450))
    text_rect = (text_rect.left, text_rect.top)
    display.blit(text,text_rect)
    text = font.render('P2', False, (255,255,255))
    text_rect = text.get_rect(center=(675,450))
    text_rect = (text_rect.left, text_rect.top)
    display.blit(text,text_rect)

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
          if event.button == 1:  # Left mouse button
            p1 = x<400
            break
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    clock.tick(30)   # runs game at fps
    pygame.display.update()
    
  pygame.quit()
  return p1

def endscreen_mainloop():
  pygame.init()
  winsize = (900, 900)
  display = pygame.display.set_mode(winsize, pygame.RESIZABLE)
  pygame.font.init()
  font = pygame.font.SysFont('Pokemon GB', 150)
  font_sub = pygame.font.SysFont('Pokemon GB', 30)
  font_head = pygame.font.SysFont('Pokemon GB', 100)

  while True:
    board, player = BOARD, PLAYER
    display.fill((60,60,60))  # fill screen with black
    todisplay = game.displayBoard(board, player=PLAYER, return_only=True).replace(' ','').replace('*','.').split('\n')
    colors = {'.':(20,50,20),
              'x':(0,0,0),
              'o':(255,255,255)}

    for y in range(1,9):
      row = todisplay[y]
      for x in range(8):
        color = colors[row[x]]
        pygame.draw.circle(display, color, (x*100+100,y*100), 40)

    # darken background
    dark_surface = pygame.Surface(display.get_size(), pygame.SRCALPHA)
    dark_surface.fill((0,0,0,100))
    display.blit(dark_surface, (0, 0))

    score = game.getCounts(board)
    player = score[0]<score[1]

    winning_player = ['black','white'][player]
    winning_color = [(255,255,255),(255,255,255)][player]
    if score[0]!=score[1]: msg = f'{winning_player} wins!'
    else: # draw
      msg = 'draw'
    text = font.render(msg, False, winning_color)
    text_rect = text.get_rect(center=(450,450))
    text_rect = (text_rect.left, text_rect.top)
    display.blit(text,text_rect)
    text = font_sub.render('click to play again', False, winning_color)
    text_rect = text.get_rect(center=(450,500))
    text_rect = (text_rect.left, text_rect.top)
    display.blit(text,text_rect)
    text = font_head.render(f'{score[1]} - {score[0]}', False, winning_color)
    text_rect = text.get_rect(center=(450,350))
    text_rect = (text_rect.left, text_rect.top)
    display.blit(text,text_rect)

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
          if event.button == 1:  # Left mouse button
            return
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
  

def play(board=game.beginGame(),p1=True,player=True):
  global BOARD, PLAYER, USR
  while 1:
    game.displayBoard(board,player,indexed=player==p1)
    BOARD, PLAYER = board, player
    if player==p1:
      USR, usr = -1, -1
      while usr==-1:
        usr = USR
      if usr == -2: return board
      disp = game.displayBoard(board,player,indexed=player==p1,return_only=True)
      disp = disp.replace(' ','').split('\n')
      usr = disp[usr[1]+1][usr[0]]
      
      #usr = input().upper()
      usr = ord(usr)-65
      board = game.getNeighbors(board,player)[usr]
      USR, usr = -1, -1
    else:
      # MCTS
      move, newboard = getMove(board, player)
      board = newboard
    player = not player
    if not game.getMoves(board,player): break
  game.displayBoard(board,player)
  BOARD, PLAYER = board, player
  return board



EEE = 25
if __name__ == '__main__':
  model = Network()
  model.load_state_dict(torch.load(f'reversi-model_{EEE}.pt',map_location=torch.device(device)))
  model.to(device)

  while True:
    Q = {}

    p1 = startscreen_mainloop()

    # begin display thread
    USR = -2
    PLAYER = True
    EVAL = 0
    BOARD = game.beginGame()
    stop_event = threading.Event()
    thread = threading.Thread(target=win_mainloop, args=(stop_event,))
    thread.start()

    play(p1=p1)
    stop_event.set()
    thread.join()

    if USR != -2: endscreen_mainloop()



   
