PORT := 8888


start:
	uvicorn app.main:app --timeout-keep-alive 1 --host 0.0.0.0 --port $(PORT)
# nohup uvicorn app.main:app --reload --host 0.0.0.0 --port $(PORT) > uvicorn.log 2>&1 &

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