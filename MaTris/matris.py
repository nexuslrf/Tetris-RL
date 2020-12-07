#!/usr/bin/env python
import pygame
from pygame import Rect, Surface
import random
import numpy as np
import os
import kezmenu
from tetrominoes import list_of_tetrominoes, list_of_start_tetrominoes
from tetrominoes import rotate, tetrominoes

from scores import load_score, write_score, Score

class GameOver(Exception):
    """Exception used for its control flow properties"""

class DummySound(object):
    def play(self):
        return

def get_sound(filename, fake=True):
    if fake:
        return DummySound()
    else:
        return pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "resources", filename))

BGCOLOR = (15, 15, 20)
BORDERCOLOR = (140, 140, 140)

BLOCKSIZE = 30
BORDERWIDTH = 10

MATRIS_OFFSET = 20

MATRIX_WIDTH = 10
MATRIX_HEIGHT = 22

LEFT_MARGIN = 340

WIDTH = MATRIX_WIDTH*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2 + LEFT_MARGIN
HEIGHT = (MATRIX_HEIGHT-2)*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2

TRICKY_CENTERX = WIDTH-(WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2))/2

VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2

DROP_TRIALS = 4

MAX_LEVEL = 20

DROP_RATIO = 1 - 0.2

INIT_GRID = [
    'oooxxxxxoo',
    'ooooxxxooo',
    'xxxoxxxoxx',
    'xxooxxxoox',
    'xxxoxxxoxx'
]

class Matris(object):
    def __init__(self, screen):
        self.surface = None
        if screen:
            self.surface = screen.subsurface(Rect((MATRIS_OFFSET+BORDERWIDTH, MATRIS_OFFSET+BORDERWIDTH),
                                              (MATRIX_WIDTH * BLOCKSIZE, (MATRIX_HEIGHT-2) * BLOCKSIZE)))

        self.color_map = {
            "blue": 1,"yellow": 2,"pink": 3,"green": 4,
            "red": 5,"cyan": 6,"orange": 7,"grey": 8
            }
        self.colors = ["none","blue","yellow","pink","green","red","cyan","orange","grey"]

        self.matrix = np.zeros([3, MATRIX_HEIGHT, MATRIX_WIDTH], dtype=int)
        """
        `self.matrix` is the current state of the tetris board, that is, it records which squares are
        currently occupied. It does not include the falling tetromino. The information relating to the
        falling tetromino is managed by `self.set_tetrominoes` instead. When the falling tetromino "dies",
        it will be placed in `self.matrix`.
        """

        # Use 7-Packed random piece
        next_shape = random.choice(list_of_start_tetrominoes)
        self.next_tetromino = tetrominoes[next_shape]
        # self.next_tetromino = tetrominoes[list_of_tetrominoes[2]]
        self.next_tetromino_bag = list(np.random.permutation(list_of_tetrominoes))
        repeat_idx = self.next_tetromino_bag.index(next_shape)
        self.next_tetromino_bag = [next_shape] + self.next_tetromino_bag
        self.next_tetromino_idx = 1

        self.held_tetromino = None
        self.recently_swapped = True
        self.drop_trials = DROP_TRIALS
        self.garbage_block = self.block('grey')
        self.t_spin = 0 # 0: no t-spin, 1: t-spin mini, 2: t-spin 
        self.difficult_clear = False
        self.done = False
        self.locked = False
        self.needs_redraw = True

        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.4 # Move down every 400 ms

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.level = 1
        self.score = 0
        self.lines = 0

        self.combo = 1 # Combo will increase when you clear lines with several tetrominos in a row
        
        self.paused = False

        self.highscore = load_score()
        self.played_highscorebeaten_sound = False

        self.levelup_sound  = get_sound("levelup.wav")
        self.gameover_sound = get_sound("gameover.wav")
        self.linescleared_sound = get_sound("linecleared.wav")
        self.highscorebeaten_sound = get_sound("highscorebeaten.wav")


    def set_tetrominoes(self):
        """
        Sets information for the current and next tetrominos
        """
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = tetrominoes[self.next_tetromino_bag[self.next_tetromino_idx]]
        self.next_tetromino_idx = (self.next_tetromino_idx + 1) % 7
        if self.next_tetromino_idx == 0:
            self.next_tetromino_bag = list(np.random.permutation(list_of_tetrominoes))
        # self.next_tetromino = tetrominoes[list_of_tetrominoes[2]]
        self.locked = False
        self.t_spin = 0
        self.recently_swapped = False
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.surface_of_held_tetromino = self.construct_surface_of_held_tetromino()
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)

    def predef_tetrominoes(self, init_grid):
        """
        Add some garbage blocks as initial pre-defined grid
        """
        n_row = len(init_grid)
        for i in range(n_row):
            y = i + MATRIX_HEIGHT - n_row
            for x in range(MATRIX_WIDTH):
                if init_grid[i][x] == 'x':
                    self.matrix[0, y,x] = self.color_map['grey']

    def swap_held(self):
        if self.recently_swapped:
            return False
        if self.held_tetromino is None:
            # Hold current piece, promote next piece
            self.held_tetromino = self.current_tetromino
            self.set_tetrominoes()
            self.recently_swapped = True
        else:
            tmp = self.held_tetromino
            self.held_tetromino = self.current_tetromino
            self.current_tetromino = tmp

            self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
            self.tetromino_rotation = 0
            self.tetromino_block = self.block(self.current_tetromino.color)
            self.shadow_block = self.block(self.current_tetromino.color, shadow=True)
            self.surface_of_held_tetromino = self.construct_surface_of_held_tetromino()
            self.recently_swapped = True
        return True
    
    def hard_drop(self):
        """
        Instantly places tetrominos in the cells below
        """
        amount = 0
        while self.request_movement('down'):
            amount += 1

        self.score += 2 * amount
        self.lock_tetromino()


    def update(self, timepassed):
        """
        Main game loop
        """
        self.needs_redraw = False
        
        pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
        unpressed = lambda key: event.type == pygame.KEYUP and event.key == key

        events = pygame.event.get()
        #Controls pausing and quitting the game.
        for event in events:
            if pressed(pygame.K_p):
                if self.surface:
                    self.surface.fill((0,0,0))
                self.needs_redraw = True
                self.paused = not self.paused
            elif event.type == pygame.QUIT:
                self.gameover(full_exit=True)
            elif pressed(pygame.K_ESCAPE):
                self.gameover(full_exit=True)

        if self.paused:
            return self.needs_redraw

        for event in events:
            #Controls movement of the tetromino
            if pressed(pygame.K_SPACE):
                self.hard_drop()
            elif pressed(pygame.K_UP) or pressed(pygame.K_w):
                self.request_rotation()
            elif pressed(pygame.K_x):
                self.request_forward_rotation()
            elif pressed(pygame.K_z):
                self.request_reverse_rotation()
            elif pressed(pygame.K_LEFT) or pressed(pygame.K_a):
                self.request_movement('left')
                self.movement_keys['left'] = 1
            elif pressed(pygame.K_RIGHT) or pressed(pygame.K_d):
                self.request_movement('right')
                self.movement_keys['right'] = 1
            elif pressed(pygame.K_LSHIFT) or pressed(pygame.K_RSHIFT):
                self.swap_held()

            elif unpressed(pygame.K_LEFT) or unpressed(pygame.K_a):
                self.movement_keys['left'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2
            elif unpressed(pygame.K_RIGHT) or unpressed(pygame.K_d):
                self.movement_keys['right'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2


        self.downwards_speed = self.base_downwards_speed * (1 - DROP_RATIO * (self.level -1) /(MAX_LEVEL - 1))

        self.downwards_timer += timepassed
        sonic_drop = False
        downwards_speed = self.downwards_speed 
        if any([pygame.key.get_pressed()[pygame.K_DOWN], pygame.key.get_pressed()[pygame.K_s]]):
            downwards_speed = self.downwards_speed*0.2
            sonic_drop = True

        if self.downwards_timer > downwards_speed:
            if not self.request_movement('down'): #Places tetromino if it cannot move further down
                self.drop_trials -= 1
                if self.drop_trials == 0:
                    self.lock_tetromino()
                    self.drop_trials = DROP_TRIALS
            # else:
            #     if sonic_drop:
            #         self.score += 1

            self.downwards_timer %= downwards_speed


        if any(self.movement_keys.values()):
            self.movement_keys_timer += timepassed
        if self.movement_keys_timer > self.movement_keys_speed:
            self.request_movement('right' if self.movement_keys['right'] else 'left')
            self.movement_keys_timer %= self.movement_keys_speed
        
        return self.needs_redraw

    def step_update(self, actions, timepassed):
        """
        One step update
        """
        pre_score = self.score
        self.needs_redraw = False
        sonic_drop = False
        for action in actions:
            #Controls movement of the tetromino
            if action == 'hard drop':
                self.hard_drop()
            elif action == 'sonic drop':
                sonic_drop = True
            elif action == 'forward':
                self.request_forward_rotation()
            elif action == 'reverse':
                self.request_reverse_rotation()
            elif action == 'left':
                self.request_movement('left')
                self.movement_keys['left'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2
            elif action == 'right':
                self.request_movement('right')
                self.movement_keys['right'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2
            elif action == 'left-press':
                self.request_movement('left')
                self.movement_keys['left'] = 1
            elif action == 'right-press':
                self.request_movement('right')
                self.movement_keys['right'] = 1
            elif action == 'hold':
                self.swap_held()

        self.downwards_speed = self.base_downwards_speed * (1 - 0.9 * (self.level -1) /(MAX_LEVEL - 1))

        self.downwards_timer += timepassed
        downwards_speed = self.downwards_speed 
        if sonic_drop:
            downwards_speed = self.downwards_speed*0.2

        if self.downwards_timer > downwards_speed:
            if not self.request_movement('down'): #Places tetromino if it cannot move further down
                self.drop_trials -= 1
                if self.drop_trials == 0:
                    self.lock_tetromino()
                    self.drop_trials = DROP_TRIALS
            # else:
            #     if sonic_drop:
            #         # self.score += 1
            #         pass

            self.downwards_timer %= downwards_speed


        if any(self.movement_keys.values()):
            self.movement_keys_timer += timepassed
        if self.movement_keys_timer > self.movement_keys_speed:
            self.request_movement('right' if self.movement_keys['right'] else 'left')
            self.movement_keys_timer %= self.movement_keys_speed
        
        return self.score - pre_score

    def draw_surface(self):
        """
        Draws the image of the current tetromino
        """
        self.matrix[1:,:,:] = 0
        self.matrix = self.blend(matrix=self.place_shadow())
        if self.surface:
            grid = self.matrix[0] + self.matrix[1]
            complete_overlap = np.abs(self.matrix[1] - self.matrix[2]).sum() == 0
            for y in range(MATRIX_HEIGHT):
                for x in range(MATRIX_WIDTH):
                    #                                       I hide the 2 first rows by drawing them outside of the surface
                    block_location = Rect(x*BLOCKSIZE, (y*BLOCKSIZE - 2*BLOCKSIZE), BLOCKSIZE, BLOCKSIZE)

                    self.surface.fill(BGCOLOR, block_location)
                    if self.matrix[2,y,x]:
                        self.surface.blit(self.shadow_block, block_location)
                        if complete_overlap: continue

                    if grid[y,x]:
                        self.surface.blit(self.block(self.colors[grid[y,x]]), block_location)
                    
    def gameover(self, full_exit=False):
        """
        Gameover occurs when a new tetromino does not fit after the old one has died, either
        after a "natural" drop or a hard drop by the player. That is why `self.lock_tetromino`
        is responsible for checking if it's game over.
        """

        write_score(self.score)
        
        if full_exit:
            exit()
        else:
            self.paused = True
        #     raise GameOver("Sucker!")

    def place_shadow(self):
        """
        Draws shadow of tetromino so player can see where it will be placed
        """
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)) is not None:
            posY += 1

        position = (posY-1, posX)

        return self.blend(position=position, shadow=True)

    def valid_pos(self, posY, posX):
        return posX >=0 and posX < MATRIX_WIDTH and posY >= 0 and posY < MATRIX_HEIGHT

    def fits_in_matrix(self, shape, position):
        """
        Checks if tetromino fits on the board
        """
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                good_pos = self.valid_pos(y,x)
                if (not good_pos and shape[y-posY][x-posX] # outside matrix
                    or # coordinate is occupied by something else which isn't a shadow
                    good_pos and self.matrix[0,y,x] and shape[y-posY][x-posX]):
                    return False

        return position

    def t_spin_test(self, shape, position, rotation):
        """
        Checks T-Spin
        """
        posY, posX = position
        if rotation == 2:
            cnt = 0
            for x in range(posX, posX+len(shape), 2):
                for y in range(posY, posY+len(shape), 2):
                    if  self.matrix[0,y,x] and shape[y-posY][x-posX] is None:
                        cnt += 1
            return 2 if cnt >= 3 else 0
        elif rotation != 0:
            cnt = 0
            for x in range(posX, posX+len(shape), 2):
                for y in range(posY, posY+len(shape), 2):
                    good_pos = self.valid_pos(y,x)
                    if  (not good_pos and shape[y-posY][x-posX] is None# outside matrix
                        or # coordinate is occupied by something else which isn't a shadow
                        good_pos and self.matrix[0,y,x] and shape[y-posY][x-posX] is None):
                        cnt += 1
            return 1 if cnt >= 3 else 0
        return 0

    def request_rotation(self, direction=1):
        """
        Checks if tetromino can rotate
        Returns the tetromino's rotation position if possible
        """
        rotation = (self.tetromino_rotation + direction) % 4
        shape = self.rotated(rotation)

        y, x = self.tetromino_position

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2))
                    )
        
        if position and self.current_tetromino[0] == 'pink':
            t_spin = self.t_spin_test(shape, position, rotation)
            self.t_spin = t_spin if t_spin > self.t_spin else self.t_spin

        if not position:
            position = (self.fits_in_matrix(shape, (y+1, x+1)) or
                        self.fits_in_matrix(shape, (y+1, x-1)))
            if position and self.current_tetromino[0] == 'pink':
                t_spin = 2 if rotation == 2 else 1
                self.t_spin = t_spin if t_spin > self.t_spin else self.t_spin
        
        if not position:
            position = (self.fits_in_matrix(shape, (y+2, x+1)) or
                        self.fits_in_matrix(shape, (y+2, x-1)))
            if position and self.current_tetromino[0] == 'pink':
                self.t_spin = 2
        
        # ^ That's how wall-kick is implemented

        if position and self.blend(shape, position) is not None:
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            
            self.needs_redraw = True
            return self.tetromino_rotation
        else:
            return False

    def request_forward_rotation(self):
        self.request_rotation(1)

    def request_reverse_rotation(self):
        self.request_rotation(-1)
            
    def request_movement(self, direction):
        """
        Checks if teteromino can move in the given direction and returns its new position if movement is possible
        """
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX-1)) is not None:
            self.tetromino_position = (posY, posX-1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'right' and self.blend(position=(posY, posX+1)) is not None:
            self.tetromino_position = (posY, posX+1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'up' and self.blend(position=(posY-1, posX)) is not None:
            self.needs_redraw = True
            self.tetromino_position = (posY-1, posX)
            return self.tetromino_position
        elif direction == 'down' and self.blend(position=(posY+1, posX)) is not None:
            self.needs_redraw = True
            self.tetromino_position = (posY+1, posX)
            return self.tetromino_position
        else:
            return False

    def rotated(self, rotation=None):
        """
        Rotates tetromino
        """
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def block(self, color, shadow=False):
        """
        Sets visual information for tetromino
        """
        colors = {'blue':   (105, 105, 255),
                  'yellow': (225, 242, 41),
                  'pink':   (242, 41, 195),
                  'green':  (22, 181, 64),
                  'red':    (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan':   (10, 255, 226),
                  'grey':   (200, 200, 200)}


        if shadow:
            end = [90] # end is the alpha value
        else:
            end = [] # Adding this to the end will not change the array, thus no alpha value

        border = Surface((BLOCKSIZE, BLOCKSIZE), pygame.SRCALPHA, 32)
        border.fill(list(map(lambda c: c*0.5, colors[color])) + end)

        borderwidth = 2

        box = Surface((BLOCKSIZE-borderwidth*2, BLOCKSIZE-borderwidth*2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(list(map(lambda c: min(255, int(c*random.uniform(0.8, 1.2))), colors[color])) + end) 

        del boxarr # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))


        return border

    def lock_tetromino(self):
        """
        This method is called whenever the falling tetromino "dies". `self.matrix` is updated,
        the lines are counted and cleared, and a new tetromino is chosen.
        """
        self.locked = True
        self.matrix[1:,:,:] = 0
        self.matrix = self.blend()

        lines_cleared = self.remove_lines()
        self.lines += lines_cleared

        score_type = ""
        if self.t_spin > 0:
            score_type = "t-spin " if self.t_spin == 2 else "t-spin mini "
        if lines_cleared:
            score_type += Score.score_list[lines_cleared - 1]
        else:
            score_type += "no lines"

        b2b = 1.5 if self.difficult_clear and score_type in Score.difficult_list else 1
        self.difficult_clear = score_type in Score.difficult_list

        if self.difficult_clear:
            if lines_cleared >= 4:
                self.linescleared_sound.play()
            print("Wow: " + score_type + "!")

        self.score = self.score + (b2b * Score.score_table[score_type] + (self.combo - 1) * 50) * self.level

        if not self.played_highscorebeaten_sound and self.score > self.highscore:
            if self.highscore != 0:
                self.highscorebeaten_sound.play()
            self.played_highscorebeaten_sound = True

        if self.level < MAX_LEVEL and self.lines >= self.level*10:
            self.levelup_sound.play()
            self.level += 1

        self.combo = self.combo + 1 if score_type in Score.score_list else 1

        self.set_tetrominoes()

        if  self.blend() is None:
            self.gameover_sound.play()
            self.done = True
            self.gameover()
            
        self.needs_redraw = True

    def remove_lines(self):
        """
        Removes lines from the board
        """
        lines = []
        for y in range(MATRIX_HEIGHT):
            #Checks if row if full, for each row
            line = (y, [])
            for x in range(MATRIX_WIDTH):
                if self.matrix[0,y,x]:
                    line[1].append(x)
            if len(line[1]) == MATRIX_WIDTH:
                lines.append(y)

        for line in sorted(lines):
            #Moves lines down one row
            for x in range(MATRIX_WIDTH):
                self.matrix[0,line,x] = 0
            for y in range(0, line+1)[::-1]:
                for x in range(MATRIX_WIDTH):
                    self.matrix[0,y,x] = self.matrix[0,y-1,x] if self.valid_pos(y-1,x) else 0

        return len(lines)

    def blend(self, shape=None, position=None, matrix=None, shadow=False):
        """
        Does `shape` at `position` fit in `matrix`? If so, return a new copy of `matrix` where all
        the squares of `shape` have been placed in `matrix`. Otherwise, return False.
        
        This method is often used simply as a test, for example to see if an action by the player is valid.
        It is also used in `self.draw_surface` to paint the falling tetromino and its shadow on the screen.
        """
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        copy = self.matrix.copy() if matrix is None else matrix.copy()
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                good_pos = self.valid_pos(y,x)
                if (not good_pos and shape[y-posY][x-posX] # shape is outside the matrix
                    or # coordinate is occupied by something else which isn't a shadow
                    good_pos and copy[0,y,x] and shape[y-posY][x-posX]):

                    return None # Blend failed; `shape` at `position` breaks the matrix

                elif shape[y-posY][x-posX]:
                    color_id = self.color_map[self.current_tetromino[0]]
                    if self.locked:
                        copy[0,y,x] = color_id
                    elif shadow:
                        copy[2,y,x] = color_id
                    else:
                        copy[1,y,x] = color_id
        return copy

    def construct_surface_of_next_tetromino(self):
        """
        Draws the image of the next tetromino
        """
        shape = self.next_tetromino.shape
        surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
        return surf

    def construct_surface_of_held_tetromino(self):
        """
        Draws the image of the held tetromino
        """
        if self.held_tetromino is not None:
            shape = self.held_tetromino.shape
            surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)

            for y in range(len(shape)):
                for x in range(len(shape)):
                    if shape[y][x]:
                        surf.blit(self.block(self.held_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
            return surf
        return None

    def get_state(self):
        board = self.matrix[:,2:,:].copy()
        board[board>0] = 1
        # combine 3 channels into one
        state_info = (board[0] + board[1] * 2 + board[2] * 3)[np.newaxis,:,:]

        # state_info = {
        #     'board': board,
        #     'current': self.color_map[self.current_tetromino[0]] - 1,
        #     'next': self.color_map[self.next_tetromino[0]] - 1,
        #     'hold': self.color_map[self.held_tetromino[0]] - 1 if self.held_tetromino else 7,
        # }
        
        return state_info


class Game(object):
    def main(self, screen):
        """
        Main loop for game
        Redraws scores and next tetromino each time the loop is passed through
        """
        self.clock = pygame.time.Clock()

        self.matris = Matris(screen)

        # self.matris.predef_tetrominoes(INIT_GRID)
        self.screen = screen

        screen.blit(construct_nightmare(screen.get_size()), (0,0))
        
        matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
        matris_border.fill(BORDERCOLOR)
        screen.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
        
        self.redraw()

        while True:
            try:
                timepassed = self.clock.tick(50)
                if self.matris.update((timepassed / 1000.) if not self.matris.paused else 0):
                    self.redraw()
            except GameOver:
                return
      
    def gym_init(self, screen=None):

        self.clock = pygame.time.Clock()
        self.matris = Matris(screen)

        # self.matris.predef_tetrominoes(INIT_GRID)
        self.screen = screen

        if self.screen:
            screen.blit(construct_nightmare(screen.get_size()), (0,0))
            
            matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
            matris_border.fill(BORDERCOLOR)
            screen.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
            
            self.redraw()


    def redraw(self):
        """
        Redraws the information panel and next termoino panel
        """
        if not self.matris.paused:
            self.blit_next_tetromino(self.matris.surface_of_next_tetromino)
            self.blit_held_tetromino(self.matris.surface_of_held_tetromino)
            self.blit_info()

            self.matris.draw_surface()

        pygame.display.flip()


    def blit_info(self):
        """
        Draws information panel
        """
        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 30)
        width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

        def renderpair(text, val):
            text = font.render(text, True, textcolor)
            val = font.render(str(val), True, textcolor)

            surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)

            surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            surf.blit(val, val.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
            return surf
        
        #Resizes side panel to allow for all information to be display there.
        scoresurf = renderpair("Score", self.matris.score)
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

        height = 20 + (levelsurf.get_rect().height + 
                       scoresurf.get_rect().height +
                       linessurf.get_rect().height + 
                       combosurf.get_rect().height )
        
        #Colours side panel
        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))
        
        #Draws side panel
        area.blit(levelsurf, (0,0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))

        self.screen.blit(area, area.get_rect(bottom=HEIGHT-MATRIS_OFFSET, centerx=TRICKY_CENTERX))


    def blit_next_tetromino(self, tetromino_surf):
        """
        Draws the next tetromino in a box to the side of the board
        """
        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]
        # ^^ I'm assuming width and height are the same

        center = areasize/2 - tetromino_surf_size/2
        area.blit(tetromino_surf, (center, center))

        self.screen.blit(area, area.get_rect(top=MATRIS_OFFSET, centerx=TRICKY_CENTERX))

    
    def blit_held_tetromino(self, tetromino_surf):
        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        areasize = area.get_size()[0]
        if tetromino_surf is not None:
            tetromino_surf_size = tetromino_surf.get_size()[0]
            # ^^ I'm assuming width and height are the same

            center = areasize/2 - tetromino_surf_size/2
            area.blit(tetromino_surf, (center, center))

        self.screen.blit(area, area.get_rect(top=MATRIS_OFFSET * 2 + BLOCKSIZE * 5, centerx=TRICKY_CENTERX))

class Menu(object):
    """
    Creates main menu
    """
    running = True
    def main(self, screen):
        clock = pygame.time.Clock()
        menu = kezmenu.KezMenu(
            ['Play!', lambda: Game().main(screen)],
            ['Quit', lambda: setattr(self, 'running', False)],
        )
        menu.position = (50, 50)
        menu.enableEffect('enlarge-font-on-focus', font=None, size=60, enlarge_factor=1.2, enlarge_time=0.3)
        menu.color = (255,255,255)
        menu.focus_color = (40, 200, 40)
        
        nightmare = construct_nightmare(screen.get_size())
        highscoresurf = self.construct_highscoresurf() #Loads highscore onto menu

        timepassed = clock.tick(30) / 1000.

        while self.running:
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    exit()

            menu.update(events, timepassed)

            timepassed = clock.tick(30) / 1000.

            if timepassed > 1: # A game has most likely been played 
                highscoresurf = self.construct_highscoresurf()

            screen.blit(nightmare, (0,0))
            screen.blit(highscoresurf, highscoresurf.get_rect(right=WIDTH-50, bottom=HEIGHT-50))
            menu.draw(screen)
            pygame.display.flip()

    def construct_highscoresurf(self):
        """
        Loads high score from file
        """
        font = pygame.font.Font(None, 50)
        highscore = load_score()
        text = "Highscore: {}".format(highscore)
        return font.render(text, True, (255,255,255))

def construct_nightmare(size):
    """
    Constructs background image
    """
    surf = Surface(size)

    boxsize = 8
    bordersize = 1
    vals = '1235' # only the lower values, for darker colors and greater fear
    arr = pygame.PixelArray(surf)
    for x in range(0, len(arr), boxsize):
        for y in range(0, len(arr[x]), boxsize):

            color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)

            for LX in range(x, x+(boxsize - bordersize)):
                for LY in range(y, y+(boxsize - bordersize)):
                    if LX < len(arr) and LY < len(arr[x]):
                        arr[LX][LY] = color
    del arr
    return surf


if __name__ == '__main__':
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MaTris")
    # Menu().main(screen)
    Game().main(screen)
