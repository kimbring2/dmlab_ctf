DMLAB_PATH=$1

cp maps/ctf_simple.aas "${DMLAB_PATH}/assets/maps/built"
cp maps/ctf_simple.bsp "${DMLAB_PATH}/assets/maps/built"
cp maps/ctf_simple.lua "${DMLAB_PATH}/game_scripts/levels"
cp maps/ctf_simple_factory.lua "${DMLAB_PATH}/game_scripts/factories"

cp maps/ctf_middle.aas "${DMLAB_PATH}/assets/maps/built"
cp maps/ctf_middle.bsp "${DMLAB_PATH}/assets/maps/built"
cp maps/ctf_middle.lua "${DMLAB_PATH}/game_scripts/levels"
cp maps/ctf_middle_factory.lua "${DMLAB_PATH}/game_scripts/factories"