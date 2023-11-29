from generated.model_pb2 import Prediction


class MangoHarvestingPrediction:
    predicted_harvesting_year: int
    success: bool
    predicted_harvesting_date_offset: int
    description: str

    def __init__(self, predicted_harvesting_year: int, success: bool = False, predicted_harvesting_date_offset: int = 0,
                 description: str = ""):
        self.predicted_harvesting_year = predicted_harvesting_year
        self.success = success
        self.predicted_harvesting_date_offset = predicted_harvesting_date_offset
        self.description = description

    def map_to_grpc_model(self) -> Prediction:
        grpc_prediction = Prediction(predicted_harvesting_year=self.predicted_harvesting_year, success=self.success,
                                     predicted_harvesting_date_offset=self.predicted_harvesting_date_offset)
        if self.description:
            grpc_prediction.description = self.description

        return grpc_prediction
