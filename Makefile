.PHONY: help clean check autoformat rl-games-bootstrap
.DEFAULT: help

# Generates a useful overview/help message for various make features - add to this as necessary!
help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run code style and linting (black, ruff) *without* changing files!"
	@echo "make autoformat"
	@echo "    Apply formatting in place with black and fixable Ruff edits without failing on existing lint backlog."
	@echo "make rl-games-bootstrap"
	@echo "    One-command conda bootstrap for RL-games dependencies."

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	black --check .
	ruff check --show-source .

autoformat:
	black .
	ruff check --fix-only --show-fixes .

rl-games-bootstrap:
	bash examples/rl_games/install/bootstrap.sh
