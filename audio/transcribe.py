import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

def transcribe_episode(input_file: str, output_file: str):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id ='openai/whisper-large-v3-turbo' 

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipe = pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype = torch_dtype,
        device=device,
        
    )

    result = asr_pipe(
        input_file, generate_kwargs={"max_new_tokens": 256, "language": "english"},
        return_timestamps=True,
    )
    
    with open(output_file, 'w') as f:
        f.write(json.dumps(result["chunks"]))
