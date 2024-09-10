
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import datetime

model_name = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_name)
model = MusicgenForConditionalGeneration.from_pretrained(model_name)


prompt = ["chill hip hop with lofi mix"]
inputs = processor(text=prompt,
                   padding=True,
                   return_tensors="pt")


audio_values = model.generate(**inputs, max_new_tokens=1024)


current_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
scipy.io.wavfile.write(f"music/music_{current_timestamp}.wav",
                       rate=model.config.audio_encoder.sampling_rate,
                       data=audio_values[0, 0].numpy())
