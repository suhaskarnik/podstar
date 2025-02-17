from pyannote.audio import Pipeline
from typing import Dict, List, Any

from pydantic import BaseModel
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook

class DiaryEntry(BaseModel):
    speaker: str
    start: float
    end: float

class Diary(BaseModel):
    entries: List[DiaryEntry]


def diarize_episode(input_file: str, output_file: str):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=True
    ) 

    pipeline.to(torch.device("cuda"))

    with ProgressHook() as hook:
        diarization = pipeline(input_file, hook=hook)
    
    diary = Diary(entries=[])
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        entry = DiaryEntry(
            start = turn.start,
            end = turn.end,
            speaker = speaker,
        )
        diary.entries.append(entry)

    json = diary.model_dump_json()
    with open(output_file, "w") as f:
        f.write(json)
