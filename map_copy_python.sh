DMLAB_PATH=$1

sudo cp maps/ctf_simple.aas "${DMLAB_PATH}/baselab/maps"
sudo cp maps/ctf_simple.bsp "${DMLAB_PATH}/baselab/maps"
sudo cp maps/ctf_simple.lua "${DMLAB_PATH}/baselab/game_scripts/levels"
sudo cp maps/ctf_simple_factory.lua "${DMLAB_PATH}/baselab/game_scripts/factories"

sudo cp maps/ctf_middle.aas "${DMLAB_PATH}/baselab/maps"
sudo cp maps/ctf_middle.bsp "${DMLAB_PATH}/baselab/maps"
sudo cp maps/ctf_middle.lua "${DMLAB_PATH}/baselab/game_scripts/levels"
sudo cp maps/ctf_middle_factory.lua "${DMLAB_PATH}/baselab/game_scripts/factories"