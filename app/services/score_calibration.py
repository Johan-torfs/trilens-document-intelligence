from dataclasses import dataclass


@dataclass(frozen=True)
class LinearScoreCalibrator:
    noise_floor: float
    ceiling: float

    def __post_init__(self) -> None:
        if self.ceiling <= self.noise_floor:
            raise ValueError(
                "Calibration ceiling must be greater "
                "than the noise floor."
            )

    def calibrate(
        self,
        raw_score: float,
    ) -> float:
        calibrated = (
            raw_score - self.noise_floor
        ) / (
            self.ceiling - self.noise_floor
        )

        return max(
            0.0,
            min(1.0, calibrated),
        )