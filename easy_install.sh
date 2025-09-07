uv venv --python=3.13
source .venv/bin/activate

uv pip install -e .
uv pip install 'stable-baselines3[extra]'
uv pip install "gymnasium[box2d]"