Implementation of Capture the Flag: the emergence of complex cooperative agents of DeepMind

[![1 vs 1 game demo](https://img.youtube.com/vi/88dNnX357eY/sddefault.jpg)](https://youtu.be/88dNnX357eY
 "Capture The Flag Implementation - Click to Watch!")
<strong>Click to Watch!</strong>

# Usage
1. First, you need to clone the official DeepMind Lab from https://github.com/deepmind/lab.
2. Second, check that you can run one of human play example of DMLab.
3. Third, place [ctf_simple.aas](https://github.com/kimbring2/dmlab_ctf/blob/main/ctf_simple.aas), [ctf_simple.bsp](https://github.com/kimbring2/dmlab_ctf/blob/main/ctf_simple.bsp) file into under [build](https://github.com/deepmind/lab/tree/master/assets/maps/built) folder of your DMLab folder.
4. Fourth, copy [ctf_simple_factory.lua](https://github.com/kimbring2/dmlab_ctf/blob/main/ctf_simple_factory.lua) file under [factories](https://github.com/deepmind/lab/tree/master/game_scripts/factories) folder and [ctf_simple.lua](https://github.com/kimbring2/dmlab_ctf/blob/main/ctf_simple.lua) file under [levels](https://github.com/deepmind/lab/tree/master/game_scripts/levels) folder.
5. Finally, run 'bazel run :game -- -l ctf_simple -s logToStdErr=true' command from your DMLab root.
