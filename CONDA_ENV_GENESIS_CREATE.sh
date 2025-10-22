GENESIS_ENV=${1:-"./.conda_envs/genesis"}
if [ ! -d "$GENESIS_ENV" ]
then
	conda env create -f ./conda/env.yaml --prefix $GENESIS_ENV
	conda config --set env_prompt '({name})'
	conda info --envs
    # conda init # -> This will add auto conda init script to ~/.bashrc
	conda activate $GENESIS_ENV
else
	echo "$GENESIS_ENV already exists"
fi
