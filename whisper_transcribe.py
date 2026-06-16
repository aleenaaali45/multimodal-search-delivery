import sys
import whisper
model = whisper.load_model('tiny')
result = model.transcribe(sys.argv[1], fp16=False, verbose=False)
print(result['text'].strip())
