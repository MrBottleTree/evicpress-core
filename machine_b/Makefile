PYTHON   ?= python
PROTO    := proto/evicpress.proto
GEN_DIR  := generated

.PHONY: help setup proto run clean

help:         ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*##"}{printf "  %-10s %s\n",$$1,$$2}'

setup:        ## Install dependencies + generate proto code
	$(PYTHON) -m pip install -r requirements.txt
	$(MAKE) proto

proto:        ## (Re)generate gRPC Python stubs from the .proto file
	$(PYTHON) -m grpc_tools.protoc \
		-I proto \
		--python_out=$(GEN_DIR) \
		--grpc_python_out=$(GEN_DIR) \
		$(PROTO)
	$(PYTHON) -c "\
f = open('$(GEN_DIR)/evicpress_pb2_grpc.py','r'); \
txt = f.read(); f.close(); \
txt = txt.replace('import evicpress_pb2 as', 'from generated import evicpress_pb2 as'); \
f = open('$(GEN_DIR)/evicpress_pb2_grpc.py','w'); f.write(txt); f.close(); \
print('[proto] import fix applied')"
	@echo "[proto] Done. Generated files in $(GEN_DIR)/"

run:          ## Start Machine B (gRPC + dashboard)
	$(PYTHON) main.py

run-config:   ## Start with a custom config: make run-config CFG=path/to/config.yaml
	$(PYTHON) main.py --config $(CFG)

clean:        ## Remove generated proto stubs
	rm -f $(GEN_DIR)/evicpress_pb2.py $(GEN_DIR)/evicpress_pb2_grpc.py
	@echo "[clean] Proto stubs removed. Run 'make proto' to regenerate."
