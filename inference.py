from nemo.collections.asr.models import ASRModel
import torch

model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
model.eval()

def inference(audio, lang, timestamps=False):

    # Transcribe
    with torch.inference_mode():
        output = model.transcribe(
            audio, 
            source_lang=lang, 
            target_lang=lang, 
            timestamps=timestamps, 
            batch_size=4
        )
    
    # Print results
    if timestamps:
        word_timestamps = output[0].timestamp['word']
        segment_timestamps = output[0].timestamp['segment']

        for stamp in segment_timestamps:
            print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")
    else:
        print(output[0].text)

# Example usage
audio = ['2086-149220-0033.wav']
inference(audio, "en", True)
