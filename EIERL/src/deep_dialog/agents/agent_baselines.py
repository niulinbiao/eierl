import copy, random
from deep_dialog import dialog_config
from agent import Agent


class InformAgent(Agent):
    """ A simple agent to test the system. This agent should simply inform all the slots and then issue: taskcomplete. """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """
        
        self.state['turn'] += 2
        if self.current_slot_id < len(self.slot_set.keys()):
            slot = self.slot_set.keys()[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {slot: "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}
            act_slot_response['turn'] = self.state['turn']
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}, 'turn': self.state['turn']}
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}



class RequestAllAgent(Agent):
    """ A simple agent to test the system. This agent should simply request all the slots and then issue: thanks(). """
    
    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """
        
        self.state['turn'] += 2
        if self.current_slot_id < len(dialog_config.sys_request_slots):
            slot = dialog_config.sys_request_slots[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "PLACEHOLDER"}
            act_slot_response['turn'] = self.state['turn']
        else:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}, 'turn': self.state['turn']}
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}



class RandomAgent(Agent):
    """ A simple agent to test the interface. This agent should choose actions randomly. """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1


    def state_to_action(self, state):
        """ Run current policy on state and produce an action """
        
        self.state['turn'] += 2
        act_slot_response = copy.deepcopy(random.choice(dialog_config.feasible_actions))
        act_slot_response['turn'] = self.state['turn']
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}



class EchoAgent(Agent):
    """ A simple agent that informs all requested slots, then issues inform(taskcomplete) when the user stops making requests. """

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = ''
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1


    def state_to_action(self, state):
        """ Run current policy on state and produce an action """
        user_action = state['user_action']
        
        self.state['turn'] += 2
        act_slot_response = {}
        act_slot_response['inform_slots'] = {}
        act_slot_response['request_slots'] = {}
        ########################################################################
        # find out if the user is requesting anything
        # if so, inform it
        ########################################################################
        if user_action['diaact'] == 'request':
            requested_slot = user_action['request_slots'].keys()[0]

            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'][requested_slot] = "PLACEHOLDER"
        else:
            act_slot_response['diaact'] = "thanks"

        act_slot_response['turn'] = self.state['turn']
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class RequestBasicsAgent(Agent):
    """ A simple agent to test the system. This agent should simply request all the basic slots and then issue: thanks(). """
    
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None, request_set=None):
        self.request_set = request_set
        
        
    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = 'UNK'
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_slot_id = 0
        #self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
        #self.request_set = ["restaurantname", "date", "numberofpeople", "starttime", "address"]
        self.phase = 0

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """
        
        self.state['turn'] += 2
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
            act_slot_response['turn'] = self.state['turn']
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}, 'turn':self.state['turn']}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}, 'turn': self.state['turn']}
        else:
            raise Exception("THIS SHOULD NOT BE POSSIBLE (AGENT CALLED IN UNANTICIPATED WAY)")
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}


class RequestInformSlotAgent(Agent):
    """ This agent should simply request and inform all the basic slots and then issue: thanks(). """
    
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None, request_set=None, inform_set=None):
        self.request_set = request_set
        self.inform_set = inform_set
        
    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = 'UNK'
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['turn'] = -1
        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0
        #self.request_set = ['or_city', 'dst_city', 'seat', 'depart_date_dep', 'depart_time_dep', 'return_date_dep', 'return_time_dep', 'numberofpeople','hotel_name', 'hotel_city', 'hotel_numberofpeople', 'hotel_date_checkin', 'hotel_date_checkout']
        #self.inform_set = ['or_city', 'dst_city', 'seat', 'depart_date_dep', 'depart_time_dep', 'return_date_dep', 'return_time_dep','price', 'hotel_name', 'hotel_city', 'hotel_date_checkin', 'hotel_date_checkout', 'hotel_price']
        self.phase = 0

    def state_to_action(self, state):
        """ Run current policy on state and produce an action """
        
        self.state['turn'] += 2
        if self.current_request_slot_id < len(self.request_set):
            slot = self.request_set[self.current_request_slot_id]
            self.current_request_slot_id += 1

            act_slot_response = {} 
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
            act_slot_response['turn'] = self.state['turn']
        elif self.current_inform_slot_id < len(self.inform_set):
            slot = self.inform_set[self.current_inform_slot_id]
            self.current_inform_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {slot: 'PLACEHOLDER'}
            act_slot_response['request_slots'] = {}
            act_slot_response['turn'] = self.state['turn']
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}, 'turn':self.state['turn']}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}, 'turn': self.state['turn']}
        else:
            raise Exception("THIS SHOULD NOT BE POSSIBLE (AGENT CALLED IN UNANTICIPATED WAY)")
        
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}