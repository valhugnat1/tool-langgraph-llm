from pydantic import BaseModel
from typing import List
from fastapi import APIRouter

router = APIRouter()

class DocRequest(BaseModel):
    keys: List[str]
