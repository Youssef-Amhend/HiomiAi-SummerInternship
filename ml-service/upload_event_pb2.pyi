from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class UploadCreatedEvent(_message.Message):
    __slots__ = ("Image_Id", "user_id")
    IMAGE_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    Image_Id: str
    user_id: str
    def __init__(self, Image_Id: _Optional[str] = ..., user_id: _Optional[str] = ...) -> None: ...
