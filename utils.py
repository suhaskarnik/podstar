import json

from datetime import datetime, timedelta

from torch import result_type

import pysrt

# adapted from https://github.com/Lyken17/tiny-whisper/blob/main/utils.py
def convert_time(data: str):
    try:
        seconds, ms = str(data).split(".")
       
        seconds = int(seconds)
        if len(ms) > 2:
            ms = ms[:2]

        ms = int(ms)
        time_delta = timedelta(seconds=seconds, milliseconds=ms)
        base_time = datetime(2000,1,1)

        result_time = base_time + time_delta
        result_str = result_time.strftime("%H:%M:%S.%f")[:-3]
    except Exception as e:
        print(f"{e}\n{data}\nseconds: {seconds}\nms: {ms}\ntimedelta: {time_delta}")
        raise e
    return result_str

def json_to_srt(input_file: str, output_file: str):
    with open(input_file, 'r') as f:
        data = json.load(f)

    to_srt(data[0]["offsets"], output_file)
    

def hf_transcript_to_srt(transcript, output_file: str):
    to_srt(transcript, output_file)

def to_srt(data: list, output_file: str):    
    srt = pysrt.SubRipFile()
    for idx, chk in enumerate(data):
        text = chk["text"]
        start, end = map(convert_time, chk["timestamp"])

        sub = pysrt.SubRipItem(idx,
                               start=start, end=end, text=text.strip())
        srt.append(sub)

    srt.save(output_file)

