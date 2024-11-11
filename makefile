TIMESTAMP := $(shell date +%Y%m%d%H%M%S)

default:
	@echo "\nSee readme for further usage instructions."
	@echo "Options are: \n"
	@echo "- make init: for project initialization and dependency installation"
	@echo "- make data: for creating project dataset from a data source"
	@echo "- make train: for training the model \n"

init:
	@python3 -m pip install poetry
	@poetry init --no-interaction
	@poetry install
	@echo "Please authorize the Wandb API if it's your first time using it."
	@wandb login
	@python3 src/config/config.py

init_no_wandb:
	@python3 -m pip install poetry
	@poetry init --no-interaction
	@poetry install
	@python3 src/config/config.py

demo:
	@python3 main.py --step demo