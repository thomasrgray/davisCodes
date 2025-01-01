from pyControl.utility import *
import hardware_definition as hw

#-------------------------------------------------------------------------
# States and events.
#-------------------------------------------------------------------------

states = ['wait_for_beam_break',
          'wait_for_poke',
          'reward_available',
          'reward']

events = ['odor_one',
          'odor_two',
          'session_timer',
          'beam_timer',
          'inter_trial_timer']

initial_state = 'wait_for_beam_break'

#-------------------------------------------------------------------------
# Variables.
#-------------------------------------------------------------------------

v.ratio = 5
v.trials_solenoid1 = 10
v.trials_solenoid2 = 10

#-------------------------------------------------------------------------        
# Define behavior.
#-------------------------------------------------------------------------

def wait_for_beam_break(event):
    if event == 'entry':
        set_timer('beam_timer', 2 * second)
    elif event == 'exit':
        clear_timer('beam_timer')
    elif event == 'beam_timer':
        goto_state('odor_delivery')

def odor_delivery(event):
    if event == 'entry':
        set_timer('inter_trial_timer', 15 * second)
    elif event == 'odor_one':
        if withprob(1/v.ratio):
            if v.trials_solenoid1 > 0:
                hw.odor_one.SOL.on()
                v.trials_solenoid1 -= 1
                goto_state('reward_available')
            elif v.trials_solenoid2 > 0:
                hw.odor_two.SOL.on()
                v.trials_solenoid2 -= 1
                goto_state('reward_available')
            else:
                stop_framework()

    elif event == 'inter_trial_timer':
        goto_state('wait_for_beam_break')

def reward_available(event):
    if event == 'odor_two':
        goto_state('reward')

def reward(event):
    if event == 'entry':
        timed_goto_state('odor_delivery', 100 * ms)  # Assuming v.reward_duration is 100 ms as in your original code.
        if v.trials_solenoid1 > 0:
            hw.odor_one.SOL.off()
        elif v.trials_solenoid2 > 0:
            hw.odor_two.SOL.off()
    elif event == 'exit':
        if v.trials_solenoid1 > 0:
            hw.odor_one.SOL.off()
        elif v.trials_solenoid2 > 0:
            hw.odor_two.SOL.off()

def all_states(event):
    if event == 'session_timer':
        stop_framework()
