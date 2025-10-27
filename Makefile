# Define Docker Compose command
COMPOSE := docker compose

# Load environment variables from .env file
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Local development commands
start:
	uvicorn app.main:app --timeout-keep-alive 1 --host 0.0.0.0 --port $(APP_PORT)

server:
	python -m app.bioblend_server --port $(MCP_PORT)

mcp-server:
	nohup python -m app.bioblend_server --port $(MCP_PORT) > logs/MCP_server.log 2>&1 &

gx-service:
	nohup uvicorn app.main:app --reload --host 0.0.0.0 --port $(APP_PORT) > logs/GX_integration.log 2>&1 &

gx-down:
	@pid=$$(lsof -ti :$(APP_PORT)); \
	if [ -n "$$pid" ]; then \
		echo "Killing process on port $(APP_PORT) with PID: $$pid"; \
		kill -9 $$pid; \
	else \
		echo "No process found on port $(APP_PORT)"; \
	fi

mcp-down:
	@pid=$$(lsof -ti :$(MCP_PORT)); \
	if [ -n "$$pid" ]; then \
		echo "Killing process on port $(MCP_PORT) with PID: $$pid"; \
		kill -9 $$pid; \
	else \
		echo "No process found on port $(MCP_PORT)"; \
	fi

# Docker commands
# Build Docker images
docker-build:
	$(COMPOSE) build

# Start containers in detached mode
docker-up:
	$(COMPOSE) up -d

# Stop and remove containers
docker-down:
	$(COMPOSE) down

# Maintenance
# Stop containers and remove volumes
docker-clean:
	$(COMPOSE) down -v

# Complete cleanup (remove images, volumes, containers)
docker-destroy:
	$(COMPOSE) down -v --rmi all

# Shell access
# Get bash shell in app container
docker-shell:
	$(COMPOSE) exec app bash

# Access Redis CLI
docker_redis:
	$(COMPOSE) exec -it redis redis-cli -p $(REDIS_PORT)

# Show container status
docker-status:
	$(COMPOSE) ps

# Force rebuild without cache
docker-rebuild:
	$(COMPOSE) build --no-cache

# Clean up unused Docker resources
docker-prune:
	docker system prune -f
	docker volume prune -f

# Inspect MCP (unchanged)
mcp-inspect:
	npx @modelcontextprotocol/inspector