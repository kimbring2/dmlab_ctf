NUM_ACTORS=$1
GPU_USE=$2
BATCH_SIZE=$3
UNROLL_LENGTH=$4
EXP_NAME=$5

tmux new-session -d -t impala_ctf

tmux new-window -d -n learner
COMMAND_LEARNER='python3.8 learner.py --env_num '"${NUM_ACTORS}"' --gpu_use '"${GPU_USE}"' --batch_size '"${BATCH_SIZE}"' --unroll_length '"${UNROLL_LENGTH}"' --exp_name '"${EXP_NAME}"''
echo $COMMAND_LEARNER

tmux send-keys -t "learner" "$COMMAND_LEARNER" ENTER

sleep 8.0

for ((id=0; id < $NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND='CUDA_VISIBLE_DEVICES=-1 python3.8 actor.py --env_id '"${id}"' --exp_name '"${EXP_NAME}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER

    sleep 0.5
done

tmux attach -t impala_ctf
