import os
import pickle

from .. import utils
import blosc
from gym_minigrid.minigrid import WorldObj
from babyai.levels.verifier import ObjDesc
import time
from datetime import datetime
import numpy as np

def get_demos_path(demos=None, env=None, origin=None, length=None, valid=False, time_stamp=False, jobs=0):
    demos_path = (demos
                  if demos
                  else env + "_" + origin + str(length))
    
    if time_stamp:
        demos_path = demos_path + datetime.now().strftime('-%m-%d-%H:%M:%S.%f')[:-3] + "-job{}".format(jobs)
    
    if valid:
        demos_path = demos_path + "_valid"
        
    demos_path = demos_path + '.pkl'
    return os.path.join(utils.storage_dir(), 'demos', demos_path)


def load_demos(path, raise_not_found=True):
    print("start loading: ", path)
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))
        else:
            print("error")
            return []


def save_demos(demos, path):
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))


def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    if len(demos) > 0:
        print('Demo num frames: {}'.format(num_frames_per_episode))

def transform_demos_tc(demos):
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]
        subgoals = demo[4]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        
        prev_sg = ''

        for i in range(n_observations):
            is_new_sg = subgoals[i] != prev_sg

            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission,
                   'subgoal': subgoals[i],
                   'is_new_sg': is_new_sg
                   }
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))

            prev_sg = subgoals[i]

        new_demos.append(new_demo)
    return new_demos

def transform_demos(demos):
    '''
    takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
    i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
    returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
    '''
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        for i in range(n_observations):
            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission}
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos

def futureSubgoals(self):
    """
    type: subgoal class name

    object: (int, int) tuple or `ObjDesc` or object reference
    The position or the decription of the object or
    the object to which we are going.

    reason: reason string

    current pos: extra info current pos (int, int)

    """
    future_subgoals = []
    for subgoal in self.stack:
        subgoal_entry = {
            "type" : type(subgoal).__name__, 
            "object" : subgoal.datum, 
            "reason" : subgoal.reason, 
            "current pos": self.mission.agent_pos
            }
        future_subgoals.append(subgoal_entry)
    
    return future_subgoals

def translate(subgoal, stack):
    """
    Convert subgoal to sentence.
    """


    from babyai.bot import CloseSubgoal, OpenSubgoal, DropSubgoal, PickupSubgoal, GoNextToSubgoal, ExploreSubgoal

    sg_datum = subgoal.datum
    sg_reason = subgoal.reason
    fwd_cell = subgoal.fwd_cell

    subgoal_sentence = ''

    def CloseSubgoalTrans(subgoal, fwd_cell):
        """
        close an opened door.

        Note: if the close door action block agent itself, consider removing this.
        """
        assert fwd_cell is not None, 'Forward cell is empty'
        assert fwd_cell.type == 'door', 'Forward cell has to be a door'
        assert fwd_cell.is_open, 'Forward door must be open'

        color = fwd_cell.color
        return "close {} door".format(color)

    def OpenSubgoalTrans(subgoal, fwd_cell):
        """
        Situations:
            A. door is opened. (unreachable)
            B. door is closed and unlocked.
            C. door is closed and locked; agent has key.
            D. door is closed and locked; agent do not has key and carry nothing. (unreachable)
            E. door is closed and locked; agent do not has key and carry other thing. (unreachable)

            Note: After C, bot will find a place drop the the key. 
            If C is caused by E, bot will pickup the previous dropped item.
        
        Reasons:
            1. from insturctions
            2. unlock after found a key (no need to provide this info)
            3. explore
        """
        assert fwd_cell is not None, 'Forward cell is empty'
        assert fwd_cell.type == 'door', 'Forward cell has to be a door'

        # got_the_key = (subgoal.carrying and subgoal.carrying.type == 'key'
        #     and subgoal.carrying.color == subgoal.fwd_cell.color)

        

        if isinstance(fwd_cell, ObjDesc):
            subgoal_sentence = "open {} door".format(fwd_cell.color if fwd_cell.color else "a")
            if subgoal.note:
                subgoal_sentence = subgoal_sentence + " " + subgoal.note
            return subgoal_sentence
        
        color = fwd_cell.color if fwd_cell.color else "a"

        if fwd_cell.is_open:
            #A
            # raise NotImplementedError("reach A")
            subgoal_sentence = "open {} door".format(color)
        elif not fwd_cell.is_locked:
            #B
            subgoal_sentence = "open {} door".format(color)
        elif fwd_cell.is_locked:
            #CDE
            subgoal_sentence = "unlock {} door".format(color)
        
        if subgoal.note:
            subgoal_sentence = subgoal_sentence + " " + subgoal.note

        return subgoal_sentence

    def DropSubgoalTrans(subgoal):
        """
        Reasons:
            1. after PickUp mission
            2. to putNextTo mission
            3. temporary drop to find a key
            4. after open the door, drop the key
            5. temporary drop to remove blocking object
            6. drop blocking object

        """
        assert subgoal.bot.mission.carrying
        # assert not subgoal.fwd_cell

        carrying = subgoal.bot.mission.carrying
        subgoal_sentence = "drop {} {}".format(carrying.color, carrying.type)

        if subgoal.note:
            subgoal_sentence = subgoal_sentence + " " + subgoal.note
            if subgoal.note == 'to place blocking object':
                return "remove blocking object {} {}".format(carrying.color, carrying.type)
            elif subgoal.note ==  'after pickup mission':
                return "pickup {} {} to complete pickup mission".format(carrying.color, carrying.type)
        return subgoal_sentence
    
    def PickupSubgoalTrans(subgoal, fwd_cell):
        """
        Reasons:
            1. to complete PutNext mission
            2. to complete Pickup mission
            3. to remove blocking object
            4. to pick up previous carrying
            5. to open the door
        """
        assert not subgoal.bot.mission.carrying
    
        assert fwd_cell
        
        

        subgoal_sentence = "pickup {} {}".format(fwd_cell.color if fwd_cell.color else "a", fwd_cell.type)

        if subgoal.note:
            subgoal_sentence = subgoal_sentence + " " + subgoal.note
            if subgoal.note == 'to remove blocking object':
                return "remove blocking object {} {}".format(fwd_cell.color, fwd_cell.type)
        
        return subgoal_sentence

    if isinstance(subgoal, CloseSubgoal):
        subgoal_sentence = CloseSubgoalTrans(subgoal, fwd_cell)
    elif isinstance(subgoal, OpenSubgoal):
        subgoal_sentence = OpenSubgoalTrans(subgoal, fwd_cell)
    elif isinstance(subgoal, DropSubgoal):
        subgoal_sentence = DropSubgoalTrans(subgoal)
    elif isinstance(subgoal, PickupSubgoal):
        subgoal_sentence = PickupSubgoalTrans(subgoal, fwd_cell)
    elif isinstance(subgoal, GoNextToSubgoal):
        """
        Reasons:
            1. to pick up an object 
                , note='to pick it up'
            2. to drop an object
                , note='to drop it'
            3. to drop an object temporary
                , note='to drop it temporary'
            4. to open a door
                , note='to open it'
            5. to complete GoTo mission
                , note='to complete GoTo mission'
            6. to find a key
                , note='to find a key'
            7. to explore
                , note='explore'
        """

        if not subgoal.note:
            if isinstance(stack[-2], PickupSubgoal):
                subgoal.note = 'to pick it up'
            elif isinstance(stack[-2], OpenSubgoal):
                subgoal.note = 'to open it'
            elif isinstance(stack[-2], DropSubgoal):
                subgoal.note = 'to drop it'
            else:
                subgoal.note = 'to complete GoTo mission'
        if subgoal.note == 'explore':
            subgoal_sentence = "explore unseen area"
        elif subgoal.note == 'to find a key':
            # assert sg_datum.type == 'key'
            # subgoal_sentence = "find {} {}".format(sg_datum.color if sg_datum.color else "a",
            #                                         sg_datum.type)
            subgoal_sentence = PickupSubgoalTrans(stack[-2], sg_datum)
        elif subgoal.note == 'to complete GoTo mission':
            if isinstance(sg_datum, np.ndarray):
                sg_datum = subgoal.bot.mission.grid.get(*sg_datum)
            assert isinstance(sg_datum, WorldObj) or isinstance(sg_datum, ObjDesc)
            subgoal_sentence = "go to {} {} to complete GoTo mission".format(sg_datum.color if sg_datum.color else "a",
                                                                            sg_datum.type)
        elif subgoal.note == 'to open it':
            # if isinstance(sg_datum, WorldObj) or isinstance(sg_datum, ObjDesc):
            #     subgoal_sentence = "go to {} door to open".format(sg_datum.color if sg_datum.color else "a",
            #                                                                 sg_datum.type)
            # else:
            #     raise TypeError()
            if isinstance(sg_datum, np.ndarray):
                sg_datum = subgoal.bot.mission.grid.get(*sg_datum)
            subgoal_sentence = OpenSubgoalTrans(stack[-2], sg_datum)    
        elif subgoal.note == 'to drop it temporary' or subgoal.note == 'to drop it':
            # assert subgoal.bot.mission.carrying
            # carrying = subgoal.bot.mission.carrying
            # subgoal_sentence = "find a empty grid to drop {} {}".format(carrying.color if carrying.color else "a",
            #                                                             carrying.type)
            # if subgoal.note == 'to drop it temporary':
            #     subgoal_sentence = subgoal_sentence + " temporarily"
            subgoal_sentence = DropSubgoalTrans(stack[-2])
        elif subgoal.note == 'to pick it up':
            if isinstance(sg_datum, WorldObj) or isinstance(sg_datum, ObjDesc):
                # subgoal_sentence = "go to {} {} to pickup".format(sg_datum.color if sg_datum.color else "a",
                #                                                         sg_datum.type)
                subgoal_sentence = PickupSubgoalTrans(stack[-2], sg_datum)
            elif fwd_cell:
                # subgoal_sentence = "go to {} {} to pickup".format(fwd_cell.color if fwd_cell.color else "a",
                #                                                         fwd_cell.type)
                subgoal_sentence = PickupSubgoalTrans(stack[-2], fwd_cell)
            elif isinstance(sg_datum, np.ndarray):
                fwd_cell = subgoal.bot.mission.grid.get(*sg_datum)
                subgoal_sentence = PickupSubgoalTrans(stack[-2], fwd_cell)
            else:
                raise TypeError()
            # subgoal_sentence = PickupSubgoalTrans(stack[-2])
        elif isinstance(subgoal.note, WorldObj):
            # fwd_cell = subgoal.note
            # subgoal_sentence = "go to {} {} to pickup".format(fwd_cell.color if fwd_cell.color else "a",
            #                                                             fwd_cell.type)
            subgoal_sentence = PickupSubgoalTrans(stack[-2], subgoal.note)
        else:
            raise NotImplementedError("Unknown GoNextTo type")

    elif isinstance(subgoal, ExploreSubgoal):
        raise NotImplementedError("ExploreSubgoal should be unreachable")
    else:
        raise NotImplementedError("other subgoal type: ", type(subgoal).__name__)

    
    return subgoal_sentence

def get_subgoal_sentence(bot, suggested_action):
    """
    Current version only return current subgoal
    """

    stack = bot.stack
    current_pos = bot.mission.agent_pos
    sg = stack[-1]

    subgoal_sentence = translate(sg, stack)
    return subgoal_sentence