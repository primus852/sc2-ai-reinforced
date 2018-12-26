import random
import os

import pandas as pd
import numpy as np
import math

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from src import QLearningTable as qlt
from src import Helper as helper
from src import Visualizer as vs

# python -m pysc2.bin.agent --map Simple64 --agent refined.DeepAgent --agent_race terran --norender

# Constants for ACTION Shortcuts
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

# Constants for FEATURE Shortcuts
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

# General Constants
_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5
_NEUTRAL_MINERAL_FIELD = 341
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

# Terran Constants
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

# Files/Foldes to save to
DATA_FILE = 'data/q_table'
STATS_FILE = 'data/stats'

# QLearning Actions
ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]

# ???
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))


class DeepAgent(base_agent.BaseAgent):
    def __init__(self):
        super(DeepAgent, self).__init__()

        # Init the QLearning
        self.qlearn = qlt.QLearningTable(actions=list(range(len(smart_actions))))

        # Init the Vars
        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0

        self.base_top_left = None

        # Read previous Learning
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

        # Init stats DataFrame
        if os.path.isfile(DATA_FILE + '.gz'):
            self.stats = pd.read_pickle(STATS_FILE + '.gz', compression='gzip')
        else:
            self.stats = pd.DataFrame(columns=['Episode', 'Outcome', 'WinPct', 'LossPct', 'DrawPct'])

        # Get total episodes
        self.stats_total = len(self.stats)

        # Get Wins so far
        self.wins = len(self.stats[self.stats['Outcome'] == 1]) if self.stats_total > 0 else 0

        # Get Losses so far
        self.losses = len(self.stats[self.stats['Outcome'] == -1]) if self.stats_total > 0 else 0

        # Get Draws so far
        self.draws = len(self.stats[self.stats['Outcome'] == 0]) if self.stats_total > 0 else 0

    @staticmethod
    def split_action(action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return smart_action, x, y

    def step(self, obs):
        super(DeepAgent, self).step(obs)

        if obs.last():
            reward = obs.reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.move_number = 0

            # Total Episodes
            self.stats_total += 1

            # Check Result
            if obs.reward == 1:
                self.wins += 1
            elif obs.reward == 0:
                self.draws += 1
            elif obs.reward == -1:
                self.losses += 1
            else:
                print('UNKNOWN RESULT, Episode %i, Result: %i' % (self.episodes, obs.reward))

            # Append to Stats DataFrame
            self.stats.loc[-1] = {
                'Episode': self.stats_total,
                'Outcome': obs.reward,
                'WinPct': round((self.wins * 100 / self.stats_total), 2),
                'LossPct': round((self.losses * 100 / self.stats_total), 2),
                'DrawPct': round((self.draws * 100 / self.stats_total), 2),
            }  # adding a row
            self.stats.index = self.stats.index + 1  # shifting index
            self.stats = self.stats.sort_index()  # sorting by index

            # Save Stats to Pickle
            self.stats.to_pickle(STATS_FILE + '.gz', 'gzip')

            # Save the Plot
            vis = vs.Visualizer()
            vis.save_plot()

            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation.feature_screen[_UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        supply_used = obs.observation.player.food_used
        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_army
        worker_supply = obs.observation.player.food_workers

        supply_free = supply_limit - supply_used

        if self.move_number == 0:
            self.move_number += 1

            current_state = np.zeros(12)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation.player[_ARMY_SUPPLY]

            hot_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            green_squares = np.zeros(4)
            friendly_y, friendly_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32))
                x = int(math.ceil((friendly_x[i] + 1) / 32))

                green_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                green_squares = green_squares[::-1]

            for i in range(0, 4):
                current_state[i + 8] = green_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            excluded_actions = []
            if supply_depot_count == 2 or worker_supply == 0:
                excluded_actions.append(1)

            if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
                excluded_actions.append(2)

            if supply_free == 0 or barracks_count == 0:
                excluded_actions.append(3)

            if army_supply == 0:
                excluded_actions.append(4)
                excluded_actions.append(5)
                excluded_actions.append(6)
                excluded_actions.append(7)

            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.split_action(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation.available_actions:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.split_action(self.previous_action)

            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation.available_actions:
                    if self.cc_y.any():
                        target = None
                        if supply_depot_count == 0:
                            target = helper.transform_distance(self, round(self.cc_x.mean()), -35,
                                                               round(self.cc_y.mean()), 0)
                        elif supply_depot_count == 1:
                            target = helper.transform_distance(self, round(self.cc_x.mean()), -25,
                                                               round(self.cc_y.mean()), -25)

                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_BARRACKS:
                if barracks_count < 2 and _BUILD_BARRACKS in obs.observation.available_actions:
                    if self.cc_y.any():
                        target = None
                        if barracks_count == 0:
                            target = helper.transform_distance(self, round(self.cc_x.mean()), 15,
                                                               round(self.cc_y.mean()), -9)
                        elif barracks_count == 1:
                            target = helper.transform_distance(self, round(self.cc_x.mean()), 15,
                                                               round(self.cc_y.mean()), 12)

                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation.available_actions:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_ATTACK:
                do_it = True

                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if do_it and _ATTACK_MINIMAP in obs.observation.available_actions:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    return actions.FunctionCall(_ATTACK_MINIMAP,
                                                [_NOT_QUEUED,
                                                 helper.transform_location(self,
                                                                           int(x) + (
                                                                                   x_offset * 8),
                                                                           int(y) + (
                                                                                   y_offset * 8))])

        elif self.move_number == 2:
            self.move_number = 0

            smart_action, x, y = self.split_action(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if _HARVEST_GATHER in obs.observation.available_actions:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

        return actions.FunctionCall(_NO_OP, [])
