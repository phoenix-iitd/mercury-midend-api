from pydantic import BaseModel, Field, model_validator

# Pydantic Models
class Group(BaseModel):
    id: str = Field(..., pattern=r"^[0-9]+(-[0-9]+)?@[sg]\.(whatsapp\.net|us)$")
    name: str


class QueueRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    data: list[Group]
    filePath: str = ""
    fileName: str = ""
    device_id: str
    user_id: str

    @model_validator(mode="after")
    def check_file_fields(self):
        if bool(self.filePath) != bool(self.fileName):
            raise ValueError("Both filePath and fileName must be provided together")
        return self


class QueueResponse(BaseModel):
    success: bool
    executionTime: float
    summary: dict
    correlation_id: str


class QueueStatusRequest(BaseModel):
    user_id: str
    correlation_id: str


class QueueStatusResponse(BaseModel):
    success: bool
    status: dict


class RevokeRequest(BaseModel):
    user_id: str
    device_id: str
    message_text: str
    max_age_hours: int = Field(default=8, ge=1, le=24)


class CreditsResponse(BaseModel):
    success: bool
    credits: dict