import pygame
import threading

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

use_model = True
def evaluate(board, player):
  if not use_model: return 0
  
  global model
  board = game.convBoard(board,player).to(device)
  with torch.no_grad():
    val = model(board)
  val = val.item()
  if val > 15: val = 15.0
  if val < -15: val = -15.0
  return val

def getMove(board, player):
  global EVAL
  COUNT = 0
  TIME = time.time()
  iters = 0
  log = True
  while iters<3600 or time.time()-TIME<1:
    iters += 1
    mcts(board,player)
  val,newboard,move = mcts(board,player,return_move=True)
  EVAL = round(val,1)
  if log:
    print(f'\n\nP: {evaluate(board,player)}   evaluation: {val}')
    print(f'searched {COUNT} moves {iters} iterations in {str(time.time()-TIME)[:5]}s')
  return move, newboard

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
    return max(boards)[:3]

  return val


def play(p1=True):
  global BOARD, PLAYER, USR

  global Q
  Q = {}
  
  board = game.beginGame()

  player = True
  qvals = []
  exit_case = -1
  while 1:
    game.displayBoard(board,player)
    BOARD, PLAYER = board, player
    if player==p1:
      # PLAYER
      nbrs = game.getNeighbors(board,player,indexed=True)
      USR, usr = -1, -1
      while usr==-1:
        usr = USR
      if usr == -2: return board, exit_case
      move = board[2]^nbrs[usr][2]
      board = nbrs[usr]
      USR, usr = -1, -1
    else:
      # MCTS
      move, newboard = getMove(board, player)
      board = newboard
    if game.checkWin(board,player,move):
      exit_case = 0
      break
    if not game.getNeighbors(board,player):
      exit_case = 1
      break
    player = not player
  game.displayBoard(board)
  BOARD, PLAYER = board, player
  return board, exit_case

def scale(tp, s):
  return tuple(int(x*s) for x in tp)

T_PRESSED = False
EVAL_TOGGLE = False
def win_mainloop(stop_event):
  global USR
  global T_PRESSED, EVAL_TOGGLE
  pygame.init()
  winsize = (800, 700)
  display = pygame.display.set_mode(winsize, pygame.RESIZABLE)
  clock = pygame.time.Clock()
  pygame.font.init()
  font = pygame.font.SysFont('Pokemon GB', 50)

  while not stop_event.is_set():
    pygame.event.pump()
    board, player = BOARD, PLAYER
    display.fill((30,30,100))  # fill screen with black
    todisplay = game.displayBoard(board, return_only=True).replace(' ','').split('\n')
    colors = {'.':(20,20,40),
              'x':(255,0,0),
              'o':(255,255,0)}

    for y in range(1,7):
      row = todisplay[y]
      for x in range(7):
        color = colors[row[x]]
        pygame.draw.circle(display, color, (x*100+100,y*100), 40)

    invalid_row = False
    x,y = pygame.mouse.get_pos()
    x = round((x-100)/100)
    x = max(min(x,6),0)
    y = round((y-100)/100)
    y = max(min(y,7),0)
    if todisplay[y+1][x]=='.':
      # fall
      while todisplay[y+1][x]=='.':
        y += 1
      y -= 1
      pygame.draw.circle(display, scale(colors['xo'[player]],0.5), (x*100+100,y*100+100), 40)
    else:
      # rise
      while todisplay[y+1][x] in 'ox':
        y -= 1
      if todisplay[y+1][x]=='.':
        pygame.draw.circle(display, scale(colors['xo'[player]],0.5), (x*100+100,y*100+100), 40)
      else:
        invalid_row = True

    if EVAL_TOGGLE:
      text = font.render(f'{EVAL}', False, (255,255,255))
      text_rect = (700, 650)
      display.blit(text,text_rect)
    
    if USR == -1 and not invalid_row:
      usr = -1
      for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
          if event.button == 1:  # Left mouse button
            usr = x
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
  winsize = (800, 700)
  display = pygame.display.set_mode(winsize, pygame.RESIZABLE)
  clock = pygame.time.Clock()
  pygame.font.init()
  font = pygame.font.SysFont('Pokemon GB', 150)
  p1 = None
  
  while p1 == None:
    pygame.event.pump()
    display.fill((30,30,100))  # fill screen with black

    x,y = pygame.mouse.get_pos()
    opacities = [0., 0.]
    opacities[x>400] = 0.7
    pygame.draw.rect(display, scale((255,255,0),opacities[0]), pygame.Rect(0, 0, 400, 700))
    pygame.draw.rect(display, scale((255,0,0),opacities[1]), pygame.Rect(400, 0, 400, 700))
    
    text = font.render('P1', False, (255,255,255))
    text_rect = text.get_rect(center=(200,350))
    text_rect = (text_rect.left, text_rect.top)
    display.blit(text,text_rect)
    text = font.render('P2', False, (255,255,255))
    text_rect = text.get_rect(center=(600,350))
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


def endscreen_mainloop(result):
  pygame.init()
  winsize = (800, 700)
  display = pygame.display.set_mode(winsize, pygame.RESIZABLE)
  pygame.font.init()
  font = pygame.font.SysFont('Pokemon GB', 150)
  font_sub = pygame.font.SysFont('Pokemon GB', 30)

  while True:
    board, player = BOARD, PLAYER
    display.fill((30,30,100))  # fill screen with black
    todisplay = game.displayBoard(board, return_only=True).replace(' ','').split('\n')
    colors = {'.':(20,20,40),
              'x':(255,0,0),
              'o':(255,255,0)}

    for y in range(1,7):
      row = todisplay[y]
      for x in range(7):
        color = colors[row[x]]
        pygame.draw.circle(display, color, (x*100+100,y*100), 40)

    # darken background
    dark_surface = pygame.Surface(display.get_size(), pygame.SRCALPHA)
    dark_surface.fill((0,0,0,100))
    display.blit(dark_surface, (0, 0))

    if result==0:
      winning_player = ['red','yellow'][player]
      winning_color = [(255,0,0),(255,255,0)][player]
      text = font.render(f'{winning_player} wins!', False, winning_color)
    else:
      winning_color = (255,255,255)
      text = font.render(f'draw', False, winning_color)
    text_rect = text.get_rect(center=(400,350))
    text_rect = (text_rect.left, text_rect.top)
    display.blit(text,text_rect)
    text = font_sub.render('click to play again', False, winning_color)
    text_rect = text.get_rect(center=(400,400))
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

PLAYER = True
BOARD = game.beginGame()
if __name__ == '__main__':
  model = Network()
  model.load_state_dict(torch.load('c4-model_30.pt',map_location=torch.device(device)))
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
    
    _, result = play(p1=p1)
    stop_event.set()
    thread.join()

    if USR != -2: endscreen_mainloop(result)
 
