## An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals

**This document describes how to run the simulation of EIERL Agent, please also check the `example.sh`.**

## Content
* [Requirement](#requirement)
* [Data](#data)
* [Parameter](#parameter)
* [Running Dialogue Agents](#running-dialogue-agents)
* [Evaluation](#evaluation)

## Requirement
main required packages:

* Python 2.7
* PyTorch 0.3.1
* seaborn
* matplotlib

If you are using conda as package/environment management tool, you can create a environment by the `spec-file.txt`.

`$ conda create --name eierl --file spec-file.txt`


## Data
all the data is under this folder: `./src/deep_dialog/data_{domain}`,the domain includes movie, testraurant, and taxi. 

The following is an example of the movie dataset:

* Movie Knowledge Bases<br/>
`movie_kb.1k.p` <br/>
`movie_kb.v2.p` 

* User Goals<br/>
`user_goals_first_turn_template.v2.p` --- user goals extracted from the first user turn<br/>
`user_goals_first_turn_template.part.movie.v1.p` --- a subset of user goals 

* NLG Rule Template<br/>
`dia_act_nl_pairs.v6.json` --- some predefined NLG rule templates for both User simulator and Agent.

* Dialog Act Intent<br/>
`dia_acts.txt`

* Dialog Act Slot<br/>
`slot_set.txt`

## Parameter

### Agent setting 

**(Note: these are the key difference between the models (DQN and EIERL)**<br/>
`--rollout_size`: number of individuals in the DQN population]<br/>`--pop_size`：number of individuals in evolved populations

### Basic setting

`--agt`: the agent id  [14:EIERL,15:ICM_DQN,16:NOISY_DQN,17:EA ]<br/>
`--usr`: the user (simulator) id<br/>
`--max_turn`: maximum turns<br/>
`--episodes`: how many dialogues to run<br/>
`--slot_err_prob`: slot level err probability<br/>
`--intent_err_prob`: intent level err probability


### Data setting
`--goal_file_path`: the user goal file path for user simulator side

### Model setting
`--dqn_hidden_size`: hidden size for RL agent<br/>
`--batch_size`: batch size for DDQ training<br/>
`--simulation_epoch_size`: how many dialogue to be simulated in one epoch<br/>
`--warm_start`: use rule policy to fill the experience replay buffer at the beginning<br/>
`--warm_start_epochs`: how many dialogues to run in the warm start

### Display setting
`--run_mode`: 0 for display mode (NL); 1 for debug mode (Dia_Act); 2 for debug mode (Dia_Act and NL); 3 for no display (i.e. training)<br/>
`--act_level`: 0 for user simulator is Dia_Act level; 1 for user simulator is NL level<br/>

### Others
`--write_model_dir`: the directory to write the models<br/>


## Running Dialogue Agents

**DQN  Agent**:

```
python run.py --agt 14 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data_movie/movie_kb.1k.p --goal_file_path ./deep_dialog/data_movie/user_goals_first_turn_template.part.movie.v1.p --slot_set ./deep_dialog/data_movie/slot_set.txt  --act_set ./deep_dialog/data_movie/dia_acts.txt  --dict_path ./deep_dialog/data_movie/dicts.v3.p   --nlg_model_path ./deep_dialog/models/nlg/movie/lstm_tanh_relu_[1468202263.38]_2_0.610.p  --nlu_model_path ./deep_dialog/models/nlu/movie/lstm_[1468447442.91]_39_80_0.921.p     --diaact_nl_pairs ./deep_dialog/data_movie/dia_act_nl_pairs.v6.json  --dqn_hidden_size 80 --experience_replay_pool_size 5000 --episodes 500 --simulation_epoch_size 100  --write_model_dir ./deep_dialog/checkpoints/movie/nl/eierl/  --run_mode 3  --act_level 0  --slot_err_prob 0.00   --intent_err_prob 0.00   --batch_size 16   --warm_start 1   --warm_start_epochs 100 --epsilon 0 --rollout_size 4 --pop_size 0 --torch_seed 9841
```

**EIERL Agent**:

```
python run.py --agt 14 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data_movie/movie_kb.1k.p --goal_file_path ./deep_dialog/data_movie/user_goals_first_turn_template.part.movie.v1.p --slot_set ./deep_dialog/data_movie/slot_set.txt  --act_set ./deep_dialog/data_movie/dia_acts.txt  --dict_path ./deep_dialog/data_movie/dicts.v3.p   --nlg_model_path ./deep_dialog/models/nlg/movie/lstm_tanh_relu_[1468202263.38]_2_0.610.p  --nlu_model_path ./deep_dialog/models/nlu/movie/lstm_[1468447442.91]_39_80_0.921.p     --diaact_nl_pairs ./deep_dialog/data_movie/dia_act_nl_pairs.v6.json  --dqn_hidden_size 80 --experience_replay_pool_size 5000 --episodes 500 --simulation_epoch_size 100  --write_model_dir ./deep_dialog/checkpoints/movie/nl/eierl/  --run_mode 3  --act_level 0  --slot_err_prob 0.00   --intent_err_prob 0.00   --batch_size 16   --warm_start 1   --warm_start_epochs 100 --epsilon 0 --rollout_size 1 --pop_size 3 --torch_seed 9841
```

**EA Agent**:

```
python run.py --agt 17 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data_movie/movie_kb.1k.p --goal_file_path ./deep_dialog/data_movie/user_goals_first_turn_template.part.movie.v1.p --slot_set ./deep_dialog/data_movie/slot_set.txt  --act_set ./deep_dialog/data_movie/dia_acts.txt  --dict_path ./deep_dialog/data_movie/dicts.v3.p   --nlg_model_path ./deep_dialog/models/nlg/movie/lstm_tanh_relu_[1468202263.38]_2_0.610.p  --nlu_model_path ./deep_dialog/models/nlu/movie/lstm_[1468447442.91]_39_80_0.921.p     --diaact_nl_pairs ./deep_dialog/data_movie/dia_act_nl_pairs.v6.json  --dqn_hidden_size 80 --experience_replay_pool_size 5000 --episodes 5000 --simulation_epoch_size 100  --write_model_dir ./deep_dialog/checkpoints/movie/nl/ea/  --run_mode 3  --act_level 0  --slot_err_prob 0.00   --intent_err_prob 0.00   --batch_size 16   --warm_start 1   --warm_start_epochs 100 --torch_seed 9841
```

**ICM_DQN Agent**:

```
python run.py --agt 15 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data_movie/movie_kb.1k.p --goal_file_path ./deep_dialog/data_movie/user_goals_first_turn_template.part.movie.v1.p --slot_set ./deep_dialog/data_movie/slot_set.txt  --act_set ./deep_dialog/data_movie/dia_acts.txt  --dict_path ./deep_dialog/data_movie/dicts.v3.p   --nlg_model_path ./deep_dialog/models/nlg/movie/lstm_tanh_relu_[1468202263.38]_2_0.610.p  --nlu_model_path ./deep_dialog/models/nlu/movie/lstm_[1468447442.91]_39_80_0.921.p     --diaact_nl_pairs ./deep_dialog/data_movie/dia_act_nl_pairs.v6.json  --dqn_hidden_size 80 --experience_replay_pool_size 5000 --episodes 500 --simulation_epoch_size 100  --write_model_dir ./deep_dialog/checkpoints/movie/nl/icm/  --run_mode 3  --act_level 0  --slot_err_prob 0.00   --intent_err_prob 0.00   --batch_size 16   --warm_start 1   --warm_start_epochs 100 --torch_seed 9841
```

**NOISY_DQN Agent**:

```
python run.py --agt 16 --usr 1 --max_turn 40 --kb_path ./deep_dialog/data_movie/movie_kb.1k.p --goal_file_path ./deep_dialog/data_movie/user_goals_first_turn_template.part.movie.v1.p --slot_set ./deep_dialog/data_movie/slot_set.txt  --act_set ./deep_dialog/data_movie/dia_acts.txt  --dict_path ./deep_dialog/data_movie/dicts.v3.p   --nlg_model_path ./deep_dialog/models/nlg/movie/lstm_tanh_relu_[1468202263.38]_2_0.610.p  --nlu_model_path ./deep_dialog/models/nlu/movie/lstm_[1468447442.91]_39_80_0.921.p     --diaact_nl_pairs ./deep_dialog/data_movie/dia_act_nl_pairs.v6.json  --dqn_hidden_size 80 --experience_replay_pool_size 5000 --episodes 500 --simulation_epoch_size 100  --write_model_dir ./deep_dialog/checkpoints/movie/nl/noisy/  --run_mode 3  --act_level 0  --slot_err_prob 0.00   --intent_err_prob 0.00   --batch_size 16   --warm_start 1   --warm_start_epochs 100 --torch_seed 9841
```


## Experiments
You can train the model by the example commands above or check the `example.sh`.

## Evaluation
This work focuses on training efficiency, therefore we evaluate the performance by learning curves. Please check the example code in the `deep_dialog/draw_figure/draw_figure.py`.

```
$ python draw_figure.py 
```

