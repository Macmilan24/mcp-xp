COMPOSE := docker compose

ifneq (,$(wildcard .env))
    include .env
    export
endif

PORT ?= 8000

# Local development commands
start:
	uvicorn app.main:app --timeout-keep-alive 1 --host 0.0.0.0 --port $(PORT)

up:
	uvicorn app.main:app --port ${PORT}

start-up:
	nohup uvicorn app.main:app --reload --host 0.0.0.0 --port $(PORT) > uvicorn.log 2>&1 &

log:
	tail -f uvicorn.log

down:
	@pid=$$(sudo lsof -ti :$(PORT)); \
	if [ -n "$$pid" ]; then \
		echo "Killing process on port $(PORT) with PID: $$pid"; \
		sudo kill -9 $$pid; \
	else \
		echo "No process found on port $(PORT)"; \
	fi

## Docker commands
# Build Docker images
docker-build:
	$(COMPOSE) build

# Start containers in detached mode
docker-up:
	$(COMPOSE) up -d

# Stop and remove containers
docker-down:
	$(COMPOSE) down

## Maintainance

# Stop containers and remove volumes
docker-clean:
	$(COMPOSE) down -v

# Complete cleanup (remove images, volumes, containers)
docker-destroy:
	$(COMPOSE) down -v --rmi all


## Shell access
# Get bash shell in app container
docker-shell:
	$(COMPOSE) exec app bash

# Access Redis CLI
docker-shell-redis:
	$(COMPOSE) exec redis redis-cli

##  Show container status
docker-status:
	$(COMPOSE) ps

## Force rebuild without cache

docker-rebuild:
	$(COMPOSE) build --no-cache

## Clean up unused Docker resources
docker-prune:
	docker system prune -f
	docker volume prune -f