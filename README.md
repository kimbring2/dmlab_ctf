Implementation of Capture the Flag: the emergence of complex cooperative agents of DeepMind

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
2. Second, open [dmlab_test.ipynb](https://github.com/kimbring2/dmlab_ctf/blob/main/dmlab_test.ipynb) file.
3. Third, check that you can import DmLab using 'import deepmind_lab' code.
4. Finally, run entire Jupyter Notebook code.
