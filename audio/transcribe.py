import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

from datasets import Audio, Dataset

def transcribe_episode(input_file: str, output_file: str = None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id ='openai/whisper-medium' 

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    # https://github.com/huggingface/transformers/issues/31942#issuecomment-2593004180
    processor = AutoProcessor.from_pretrained(model_id)

    dataset = Dataset.from_dict({
        "audio": [input_file]
    })
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    input_features = processor(
        dataset[0]["audio"]["array"], return_tensors="pt", truncation=False, sampling_rate=16000
    ).input_features

    

    input_features=input_features.to(device, torch_dtype)

    generated_ids = model.generate(input_features, return_timestamps=True, return_segments=True)
    
    transcript = processor.batch_decode(generated_ids["sequences"], skip_special_tokens=True, output_offsets=True)

    result = transcript[0]["offsets"]
    if output_file:  
        with open(output_file, 'w') as f:
            f.write(json.dumps(result))
    
    return result

