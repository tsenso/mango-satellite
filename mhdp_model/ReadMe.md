# Mango harvesting date prediction model

## Prerequisites
Dependencies are resolved by Anaconda, so to run application locally you need Anaconda to be installed

## How to generate gRPC python interfaces
```bash
python -m grpc_tools.protoc -I. --python_out=generated --pyi_out=generated --grpc_python_out=generated model.proto 
```

## Docker support

### How to build
```bash
docker build -t mhdp_model:latest .
```

### How to run
```bash
docker run -it --rm --expose 8061 mhdp_model:latest
```

## aiexp.ai4europe.eu
### Model name
`mango-harvesting-date-prediction-with-satellite-photos`
