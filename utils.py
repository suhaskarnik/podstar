import json

from datetime import datetime, timedelta

from torch import result_type

import pysrt


def convert_time(data: str):
    seconds, ms = map(int, str(data).split("."))

    time_delta = timedelta(seconds=seconds, milliseconds=ms)
    base_time = datetime(2000,1,1)

    result_time = base_time + time_delta
    result_str = result_time.strftime("%H:%M:%S.%f")[:-3]

    return result_str

def json_to_srt(input_file: str, output_file: str):
    with open(input_file, 'r') as f:
        data = json.load(f)

    to_srt(data, output_file)
    

def hf_pipeline_to_srt(result, output_file: str):
    to_srt(result["chunks"], output_file)

def to_srt(data: list, output_file: str):    
    srt = pysrt.SubRipFile()
    for idx, chk in enumerate(data):
        text = chk["text"]
        start, end = map(convert_time, chk["timestamp"])

        sub = pysrt.SubRipItem(idx,
                               start=start, end=end, text=text.strip())
        srt.append(sub)

    srt.save(output_file)

