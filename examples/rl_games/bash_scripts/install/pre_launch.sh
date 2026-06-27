cd /workspace/ 

git clone https://github.com/ZihanWang314/latency-sensitive-bench

git config -f .gitmodules submodule.flappy-bird-gymnasium.url https://github.com/mindorigin150/flappy-bird-gymnasium.git
git config -f .gitmodules submodule.sample-factory.url https://github.com/mindorigin150/sample-factory.git
git config -f .gitmodules submodule.starVLA.url https://github.com/talha1503/starVLA.git

git submodule sync --recursive
git submodule update --init --recursive

cd /workspace/latency-sensitive-bench

export PYTHONPATH="/workspace/latency-sensitive-bench:${PYTHONPATH:-}"