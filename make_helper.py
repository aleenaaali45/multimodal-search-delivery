f=open('whisper_transcribe.py','w') 
f.write("import sys\nimport whisper\nmodel = whisper.load_model('tiny')\nresult = model.transcribe(sys.argv[1], fp16=False, verbose=False)\nprint(result['text'].strip())\n") 
f.close() 
