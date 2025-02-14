from pyannote.audio import Pipeline
from datasets import load_dataset
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook


def diarize(input_file, output_file):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=True
    ) 

    pipeline.to(torch.device("cuda"))

    with ProgressHook() as hook:
        diarization = pipeline(input_file, hook=hook)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    with open(output_file, 'w') as f:
        diarization.write_rttm(f)
