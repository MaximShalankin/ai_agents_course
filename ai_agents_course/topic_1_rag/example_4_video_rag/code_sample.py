from faster_whisper import WhisperModel

# small - если нужно совсем мало RAM (<1GB)
# large-v3-turbo - если есть 2-3GB RAM (рекомендую)
model_size = "large-v3-turbo" 

model = WhisperModel(
    model_size, 
    device="cpu",          # На Mac M-series CTranslate2 быстрее всего на CPU!
    compute_type="int8",   # Оптимально для ARM
    cpu_threads=4          # Подбери под кол-во p-ядер
)

segments, info = model.transcribe(
    "audio.mp3", 
    beam_size=5, 
    language="ru",
    vad_filter=True,       # Обязательно True, иначе будет "Спасибо за просмотр" в тишине
    vad_parameters=dict(min_silence_duration_ms=500)
)
