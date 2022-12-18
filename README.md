# Introduction
Implementation of [Capture the Flag: the emergence of complex cooperative agents](https://deepmind.com/blog/article/capture-the-flag-science) of  DeepMind. I first describe how to set the DeepMind lab for running the Capture The Flag map. Next, I also add a way how to desing your own simple CTF map. Finally, I am going to train the agent for Capture the Flag game in 1 vs 1 case. The scale of network will be little small than original paper. However, you can know a basic knowleade how to build the agent for CTF game.

# Version
1. Python 3.8

# Demo video 
To make everything simple, I first make a Capture the Flag map for 1 vs 1 situation without any item and obstable.

[![1 vs 1 game demo](https://img.youtube.com/vi/88dNnX357eY/sddefault.jpg)](https://youtu.be/88dNnX357eY
 "Capture The Flag Implementation - Click to Watch!")
<strong>Click to Watch!</strong>

In this game, agent can grab the opponent flag if it reach to it closely. And then, agent can obtain 1 score if it brings opponent flag to home base.

# Setting
At first, we are going run the Capture The Flag map as human playing mode. Try to follow the below intructions for that.

1. First, you need to clone the official DeepMind Lab from https://github.com/deepmind/lab.
2. Second, check that you can run one of human play example of DMLab.
3. Third, place [ctf_simple.aas](https://github.com/kimbring2/dmlab_ctf/blob/main/ctf_simple.aas), [ctf_simple.bsp](https://github.com/kimbring2/dmlab_ctf/blob/main/ctf_simple.bsp) file into under [build](https://github.com/deepmind/lab/tree/master/assets/maps/built) folder of your DMLab folder.
4. Fourth, copy [ctf_simple_factory.lua](https://github.com/kimbring2/dmlab_ctf/blob/main/ctf_simple_factory.lua) file under [factories](https://github.com/deepmind/lab/tree/master/game_scripts/factories) folder and [ctf_simple.lua](https://github.com/kimbring2/dmlab_ctf/blob/main/ctf_simple.lua) file under [levels](https://github.com/deepmind/lab/tree/master/game_scripts/levels) folder.
5. Finally, run 'bazel run :game -- -l ctf_simple -s logToStdErr=true' command from your DMLab root.

# PIP install
Next, we will run same map as Python script. You

1. First, you need to install DMLab using PIP package of Python. Follow official intruction for that from https://github.com/deepmind/lab/blob/master/python/pip_package/README.md.
2. After installing the whl file that is generated from first step, you need to copy the ctf_simple.lua file under the 'deepmind_lab/baselab/game_scripts/levels' path of installed Python package. Next, copy the ctf_simple_factory.lua file under 'deepmind_lab/baselab/game_scripts/factories' folder. Finally, copy the ctf_simple.aas and ctf_simple.bsp file under 
'deepmind_lab/baselab/maps' folder.
3. Second, open [DMLab_Test.ipynb](https://github.com/kimbring2/dmlab_ctf/blob/main/DMLab_Test.ipynb) file.
4. Third, check that you can import DmLab using 'import deepmind_lab' code.
5. Finally, run entire Jupyter Notebook code.

# How to customize map
You can design your own map using program called GtkRadiant. I also make the ctf_simple map like a below image.

<img src="image/gtk_radiant_sample.png" width="1000">

For that, you need to install three program mentioned in https://github.com/deepmind/lab#upstream-sources. After that, open GtkRadiant. You just need to make a closed room and put essential component for Capture The Flag game sush as info_player_intermission, team_ctf_blueflag, team_ctf_redflag, team_ctf_blueplayer, team_ctf_redplayer, team_ctf_bluespawn and team_ctf_redspawn.

If you finish designing map and make a map format file, you should convert it to binary format called the bsp, aas. The DeepMind also provides [tool for that](https://github.com/deepmind/lab/blob/master/deepmind/level_generation/compile_map.sh].

In my case, I use a command 'sudo ./compile_map_test.sh -a -m /home/kimbring2/GtkRadiant/test.map' for conversion. Please beware there is no gap in your map. That will make error named the leaked.

# Lua script
You also need to prepare a Lua script for running map file with DmLab. Tutorial for that can be found at [minimal_level_tutorial](https://github.com/deepmind/lab/blob/master/docs/developers/minimal_level_tutorial.md). The only important thing is setting the game type as CAPTURE_THE_FLAG. 

# Training agent
You can train the agent using [Jupyter Notebook of PPO agent](https://github.com/kimbring2/dmlab_ctf/blob/main/DMLab_PPO_TF2.ipynb) at simple CTF game of above. 

# Reference
1. DeepMind Lab: https://github.com/deepmind/lab
2. CTF map design tip: https://www.quake3world.com/forum/viewtopic.php?f=10&t=51042
