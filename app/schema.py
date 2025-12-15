from pydantic import BaseModel, Field

class SmartphoneInput(BaseModel):
    brand: str = Field(..., example="Samsung")
    ram: int = Field(..., ge=1, description="RAM in GB")
    storage: int = Field(..., ge=1, description="Storage in GB")
    battery: int = Field(..., ge=100, description="Battery in mAh")
    camera: float = Field(..., ge=0, description="Main Rear Camera MP")
    screen_size: float = Field(..., ge=1.0, description="Display size in inches")
    processor: str = Field(..., example="Snapdragon", description="Processor Brand")
