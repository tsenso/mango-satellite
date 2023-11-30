# Mango harvesting date prediction data broker

## Prerequisites
Dependencies are resolved by Anaconda, so to run application locally you need Anaconda to be installed

## How to generate gRPC python interfaces
```bash
python -m grpc_tools.protoc -I. --python_out=generated --pyi_out=generated --grpc_python_out=generated data_broker.proto 
```

## Docker support

### How to build
```bash
docker build -t mhdp_data_broker:latest .
```

### How to run
```bash
docker run -it --rm --expose 8061 mhdp_data_broker:latest
```
