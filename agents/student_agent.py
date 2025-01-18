# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


def uct(wins, n_visits, p_visits, c):  # upper confidence bound for approximating value of actions
    value = wins / n_visits + c * np.sqrt(np.log(p_visits) / n_visits)
    return value


class MCNode:  # node in MC search tree
    def __init__(self, player, opponent, state, parent=None, parent_move=None):
        self.player = player
        self.opponent = opponent
        self.state = np.copy(state)
        self.parent = parent
        self.parent_move = parent_move
        self.moves = get_valid_moves(state, player)
        self.children = []
        self.wins = 0
        self.visits = 0
        return

    def select_node(self):  # select next best node
        highest_val = -1
        next_node = self.children[0]
        for child in self.children:
            val = uct(child.wins, child.visits, child.parent.visits, 3)
            if val > highest_val:  # update to child node with highest uct value
                highest_val = val
                next_node = child
        return next_node

    def expand_tree(self):  # add node to tree
        move = self.moves.pop()
        cur_state = np.copy(self.state)
        execute_move(cur_state, move, self.player)
        child = MCNode(player=self.opponent, opponent=self.player, state=cur_state, parent=self, parent_move=move)
        self.children.append(child)
        return child  # return expansion node

    def swap(self):  # swap players in current node (used in rollout)
        p = self.player
        self.player = self.opponent
        self.opponent = p

    def bad_moves(self):  # get bad positions based on board size
        n = self.state.shape[0]
        bad_pos = [  # positions adjacent to corners
            (1, 1), (n-2, n-2), (n-2, 1), (1, n-2),
            (0, 1), (1, 0), (n-2, 0), (n-1, 1),
            (n-1, n-2), (n-2, n-1), (0, n-2), (1, n-1)
        ]
        return bad_pos

    def rollout(self, p, o, bad_moves, start_time):
        cur_state = np.copy(self.state)
        cur_p = self.player  # record node player & opponent
        cur_o = self.opponent
        end, p1, p2 = check_endgame(cur_state, p, o)

        while not end:  # (mostly) randomize moves for both players until game ends
            if time.time() - start_time > 1.9:  # end simulation early if nearing time limit
                return None

            move = random_move(cur_state, self.player)
            if move is None:  # swap players if no valid moves
                self.swap()
                move = random_move(cur_state, self.player)
            if move in bad_moves:  # heuristic: if move is 'bad', tries again up to 2x to get a 'non-bad' move
                move = random_move(cur_state, self.player)
                if move in bad_moves:
                    move = random_move(cur_state, self.player)

            execute_move(cur_state, move, self.player)
            self.swap()  # swap players at end of turn
            end, p1, p2 = check_endgame(cur_state, p, o)

        self.player = cur_p  # reset node's original players
        self.opponent = cur_o

        if p == 1:  # check scores for our agent vs opponent
            p_score = p1
            o_score = p2
        else:
            p_score = p2
            o_score = p1

        if p_score > o_score:
            val = 1
        elif p_score == o_score:
            val = 0.3
        else:
            val = 0
        return val  # return val depending on simulation outcome

    def backprop(self, win):  # update wins/visits along path
        self.wins += win
        self.visits += 1
        if self.parent:
            self.parent.backprop(win)

    def tree_policy(self, start_time):  # select path in tree and node to be expanded
        cur = self
        end, _, _ = check_endgame(cur.state, self.player, self.opponent)
        while not end:  # while not terminal node
            if time.time() - start_time > 1.9:  # end loop if nearing time limit
                return None
            if cur.moves:  # if there are untried moves
                return cur.expand_tree()  # add new node for some move
            elif cur.children:  # if cur.moves is empty then the node is fully expanded
                cur = cur.select_node()  # select best child node
            else:
                break  # break if fully expanded and no children
            end, _, _ = check_endgame(cur.state, self.player, self.opponent)
        return cur  # return expansion node

    def next_step(self):
        start = time.time()  # record start time
        bad = self.bad_moves()  # get list of bad positions

        while time.time() - start < 1.75:  # run simulations until time limit reached (set lower for safety)
            leaf = self.tree_policy(start)  # choose node to do rollout from
            if leaf is None:  # if node selection terminated early bc of time limit, end loop
                break
            win = leaf.rollout(self.player, self.opponent, bad, start)  # run simulation
            if win is None:  # if rollout terminated early, no backprop
                break
            leaf.backprop(win)  # update wins & visits along path

        best_step = max(self.children, key=lambda child: child.wins/child.visits)
        return best_step.parent_move  # return move from node with the highest win:visit ratio


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"

    def step(self, chess_board, player, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
          where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
          and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

        You should return a tuple (r,c), where (r,c) is the position where your agent
        wants to place the next disc. Use functions in helpers to determine valid moves
        and more helpful tools.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        root = MCNode(player, opponent, chess_board)
        move = root.next_step()
        return move
