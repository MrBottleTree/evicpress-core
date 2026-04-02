SHELL     := /bin/bash
PYTHON    ?= python
VENV      := venv/bin/activate
PROTO     := proto/evicpress.proto
GEN_DIR   := generated
LOG       := /tmp/evicpress.log
PID_FILE  := /tmp/evicpress.pid

.PHONY: help setup proto run start stop restart logs status stats ping clean

help:       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*##"}{printf "  %-12s %s\n",$$1,$$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────

setup:      ## Install dependencies + generate proto stubs
	python3 -m venv $(dir $(VENV))
	source $(VENV) && pip install -r requirements.txt
	$(MAKE) proto

proto:      ## (Re)generate gRPC Python stubs from evicpress.proto
	source $(VENV) && python -m grpc_tools.protoc \
		-I proto \
		--python_out=$(GEN_DIR) \
		--grpc_python_out=$(GEN_DIR) \
		$(PROTO)
	@sed -i 's/^import evicpress_pb2 as/from generated import evicpress_pb2 as/' \
		$(GEN_DIR)/evicpress_pb2_grpc.py
	@echo "[proto] stubs regenerated in $(GEN_DIR)/"

# ── Run ────────────────────────────────────────────────────────────────────────

run:        ## Start Machine B in the foreground (Ctrl-C to stop)
	source $(VENV) && PYTHONUNBUFFERED=1 python -u main.py

start:      ## Start Machine B in the background; logs → $(LOG)
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "[evicpress] already running (PID=$$(cat $(PID_FILE)))"; \
	else \
		source $(VENV) && PYTHONUNBUFFERED=1 nohup python -u main.py > $(LOG) 2>&1 & \
		echo $$! > $(PID_FILE); \
		sleep 2 && echo "[evicpress] started (PID=$$(cat $(PID_FILE)))"; \
	fi

stop:       ## Stop the background EvicPress process
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		kill $$(cat $(PID_FILE)) && rm -f $(PID_FILE) && echo "[evicpress] stopped"; \
	else \
		pkill -f "python.*main.py" 2>/dev/null && echo "[evicpress] stopped (fallback)" || echo "[evicpress] not running"; \
		rm -f $(PID_FILE); \
	fi

restart:    ## Stop and restart in the background
	$(MAKE) stop
	sleep 1
	$(MAKE) start

# ── Observe ────────────────────────────────────────────────────────────────────

logs:       ## Tail the background log (Ctrl-C to exit)
	tail -f $(LOG)

status:     ## Show process status and port bindings
	@echo "=== Process ===" && \
	ps aux | grep "[p]ython.*main.py" || echo "  not running"
	@echo "=== Ports ===" && \
	sudo lsof -i :50051 -i :8080 2>/dev/null | grep LISTEN || echo "  no ports bound"

stats:      ## Print current cache stats from the dashboard API
	@curl -s http://localhost:8080/api/state | python3 -m json.tool

ping:       ## Quick gRPC connectivity self-test (GetStats)
	@source $(VENV) && python -c "\
import sys; sys.path.insert(0,'generated'); \
import grpc, evicpress_pb2 as pb2, evicpress_pb2_grpc as g; \
ch = grpc.insecure_channel('localhost:50051'); \
r = g.EvicPressServiceStub(ch).GetStats(pb2.StatsRequest()); \
print(f'[ping OK] tier2={r.tier2_blocks}blk tier3={r.tier3_blocks}blk hit_rate={r.hit_rate:.3f} t1_promotions={r.tier1_promotions}'); \
ch.close()"

# ── Clean ──────────────────────────────────────────────────────────────────────

clean:      ## Remove generated proto stubs
	rm -f $(GEN_DIR)/evicpress_pb2.py $(GEN_DIR)/evicpress_pb2_grpc.py
	@echo "[clean] run 'make proto' to regenerate"
