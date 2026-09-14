from typing import Literal
from pydantic import Field, model_validator
from mm_sae.config import StrictModel


class RQ1Config(StrictModel):
    selection_split: str = "train2017"
    correlation_split: str = "train2017"
    validation_split: str | None = "val2017"
    label_correlation: Literal["image_image", "image_text"] = "image_image"
    bin_width: float = Field(default=0.2, gt=0, le=2)
    intervention_repeats: int = Field(default=5, ge=1)
    intervention_seed: int = Field(default=0, ge=0)
    area_strata: int = Field(default=5, ge=1)
    save_intervention_panels: bool = False

    @model_validator(mode="after")
    def bins(self):
        if abs(round(2 / self.bin_width) * self.bin_width - 2) > 1e-9:
            raise ValueError("bin_width must divide [-1, 1]")
        return self

    def validate_splits(self, config):
        for split in [self.selection_split, self.correlation_split, self.validation_split]:
            if split is not None and split not in config.data.splits:
                raise ValueError(f"RQ1 requests unavailable split {split}")
        if self.validation_split in {self.selection_split, config.training.split}:
            raise ValueError("Validation cannot reuse images used to select features or train SAEs")
