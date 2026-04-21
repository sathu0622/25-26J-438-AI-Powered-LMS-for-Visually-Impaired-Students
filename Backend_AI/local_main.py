import base64
import io
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List

import nltk
import numpy as np
import pandas as pd
import requests
import soundfile as sf
from pydub import AudioSegment


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SOUNDS_DIR = os.path.join(DATA_DIR, "sounds")
GRADE10_CSV = os.path.join(DATA_DIR, "grade10_dataset.csv")
GRADE11_CSV = os.path.join(DATA_DIR, "grade11_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "generated_output")
MODEL_BASE_NAME = "google/flan-t5-small"
MODEL_ADAPTER_DIR = os.path.join(PROJECT_DIR, "models", "lesson_multitask_lora")

# Mandatory external endpoint for this project.
API_URL = "https://25-26-j-438-ai-powered-lms-for-visu.vercel.app/api/tts"


class LessonMultitaskModel:
    def __init__(self, base_model: str = MODEL_BASE_NAME, adapter_dir: str = MODEL_ADAPTER_DIR):
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self._loaded = False
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._loaded:
            return

        if not os.path.isdir(self.adapter_dir):
            self._loaded = True
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from peft import PeftModel
            import torch
        except Exception:
            self._loaded = True
            return

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
        model = PeftModel.from_pretrained(base, self.adapter_dir)
        model.eval()

        if torch.cuda.is_available():
            model = model.to("cuda")

        self._tokenizer = tokenizer
        self._model = model
        self._loaded = True

    @property
    def is_ready(self) -> bool:
        self._load()
        return self._model is not None and self._tokenizer is not None

    def _infer(self, prompt: str, max_new_tokens: int = 160) -> str:
        if not self.is_ready:
            return ""

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.1,
            )

        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def generate_fields(self, chapter: str, topic: str, source_text: str) -> Dict[str, str]:
        source_text = (source_text or "").strip()
        prompt_context = (
            f"chapter: {chapter}\n"
            f"topic: {topic}\n"
            f"source: {source_text[:900]}"
        )

        narrative = self._infer(
            f"task: narrate\n{prompt_context}\n"
            "instruction: create a clear, friendly narration for visually impaired students."
        )
        emotion = self._infer(
            f"task: emotion\n{prompt_context}\n"
            "instruction: output one emotion label only."
            ,
            max_new_tokens=16,
        )
        sound_effects = self._infer(
            f"task: sound_effects\n{prompt_context}\n"
            "instruction: output comma-separated sound effect tags.",
            max_new_tokens=48,
        )

        return {
            "narrative_text": narrative,
            "emotion": emotion,
            "sound_effects": sound_effects,
        }


def ensure_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def preprocess_data(df_to_process: pd.DataFrame) -> pd.DataFrame:
    df_to_process = df_to_process.copy()

    if "sound_effects" in df_to_process.columns and "sound_effects_list" not in df_to_process.columns:
        df_to_process["sound_effects_list"] = (
            df_to_process["sound_effects"]
            .fillna("")
            .astype(str)
            .apply(lambda x: [item.strip() for item in x.split(",") if item.strip()])
        )

    if "chapter" in df_to_process.columns and "chapter_num" not in df_to_process.columns:
        df_to_process["chapter_num"] = df_to_process["chapter"].astype(str).str.extract(r"(\d+)\.")

    if "chapter" in df_to_process.columns and "chapter_title" not in df_to_process.columns:
        df_to_process["chapter_title"] = (
            df_to_process["chapter"]
            .astype(str)
            .apply(lambda x: x.split(".", 1)[1].strip() if "." in x else x.strip())
        )

    return df_to_process


def load_and_preprocess_dataset(grade_level: str) -> pd.DataFrame:
    file_path = GRADE10_CSV if grade_level == "Grade 10" else GRADE11_CSV
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, encoding="latin-1")
    return preprocess_data(df)


class SoundEffectsMixer:
    def __init__(self, sounds_folder: str):
        self.sounds_folder = sounds_folder
        self.sound_effects: Dict[str, AudioSegment] = {}
        self.load_sound_effects()

    def load_sound_effects(self) -> None:
        self.sound_effects = {}
        if not os.path.exists(self.sounds_folder):
            print(f"Warning: sounds folder not found: {self.sounds_folder}")
            return

        for sound_file in os.listdir(self.sounds_folder):
            if sound_file.lower().endswith((".mp3", ".wav", ".ogg")):
                name = os.path.splitext(sound_file)[0]
                path = os.path.join(self.sounds_folder, sound_file)
                try:
                    self.sound_effects[name] = AudioSegment.from_file(path)
                except Exception as exc:
                    print(f"Failed to load sound effect '{sound_file}': {exc}")


class ApiTTSWrapper:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self._sample_rate = 22050

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def tts(self, text: str, lang: str = "en-IN") -> np.ndarray:
        payload = {"text": text, "lang": lang}

        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        audio_base64 = data.get("audio_base64")
        if not audio_base64:
            raise ValueError("API response did not contain 'audio_base64'.")

        audio_bytes = base64.b64decode(audio_base64)
        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if sample_rate != self._sample_rate:
            self._sample_rate = sample_rate

        return np.asarray(waveform, dtype=np.float32)


def get_sample_rate(tts_model: ApiTTSWrapper) -> int:
    return tts_model.sample_rate


def generate_tts_wav(tts_model: ApiTTSWrapper, text: str) -> str:
    sample_rate = get_sample_rate(tts_model)
    waveform_np = tts_model.tts(text=text)

    waveform_np = np.asarray(waveform_np, dtype=np.float32)
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.squeeze()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        temp_path = tmp_file.name
        sf.write(temp_path, waveform_np, sample_rate)

    return temp_path


def parse_sound_annotations(annotation_text: str) -> Dict[str, str]:
    keyword_to_sound_map: Dict[str, str] = {}
    if pd.isna(annotation_text) or not str(annotation_text).strip():
        return keyword_to_sound_map

    for entry in re.split(r"[;\n]", str(annotation_text)):
        entry = entry.strip()
        if ":" not in entry:
            continue

        sound_name, keywords_str = entry.split(":", 1)
        keywords = [k.strip().lower() for k in keywords_str.split("|") if k.strip()]
        for keyword in keywords:
            keyword_to_sound_map[keyword] = sound_name.strip()

    return keyword_to_sound_map


def build_lesson_segments(text_to_speak: str, keyword_to_sound_map: Dict[str, str]) -> List[Dict[str, List[str]]]:
    sentences = nltk.sent_tokenize(text_to_speak)
    lesson_segments = []

    for sentence in sentences:
        associated_sound_effects = []
        for keyword, sound_name in keyword_to_sound_map.items():
            if keyword in sentence.lower():
                associated_sound_effects.append(sound_name)

        associated_sound_effects = list(dict.fromkeys(associated_sound_effects))
        lesson_segments.append(
            {
                "sentence": sentence,
                "associated_sound_effects": associated_sound_effects,
            }
        )

    return lesson_segments


def choose_index(prompt: str, options: List[str]) -> int:
    if not options:
        raise ValueError("No options available.")

    print("\n" + prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")

    while True:
        raw_value = input("Select number: ").strip()
        if not raw_value.isdigit():
            print("Enter a valid number.")
            continue

        selected = int(raw_value)
        if 1 <= selected <= len(options):
            return selected - 1

        print("Selection out of range.")


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "lesson"


def maybe_resample_effect(effect: AudioSegment, sample_rate: int) -> AudioSegment:
    if effect.frame_rate != sample_rate:
        return effect.set_frame_rate(sample_rate)
    return effect


def build_audio(
    lesson_data: pd.Series,
    tts_model: ApiTTSWrapper,
    sound_mixer: SoundEffectsMixer,
    sound_mode: str,
    emotion_intensity: float,
) -> str:
    sample_rate = get_sample_rate(tts_model)
    text_to_speak = str(lesson_data.get("simplified_text", "")).replace("e.g.", "example is")

    if not text_to_speak.strip():
        raise ValueError("simplified_text is empty for selected lesson.")

    keyword_to_sound_map = parse_sound_annotations(lesson_data.get("sound_annotations", ""))
    lesson_segments = build_lesson_segments(text_to_speak, keyword_to_sound_map)

    chapter_title_clean = str(lesson_data.get("chapter_title", "")).strip()
    topic_clean = str(lesson_data.get("Grade/Topic", "")).split(":")[-1].strip()

    intro_part1_text = f"Hi. You selected chapter {chapter_title_clean} and topic {topic_clean}. "
    intro_part2_text = f"Today, I am going to discuss about {topic_clean}. "
    closing_text = "That is all for this section. Thank you for using me."

    # Preserved from notebook semantics. This controls pace in a narrow range.
    length_scale = max(0.8, min(1.4, 1.2 - ((emotion_intensity - 1.0) * 0.2)))
    _ = length_scale  # API backend controls voice properties; kept for compatibility.

    current_effects_list = list(lesson_data.get("sound_effects_list", []))

    intro_part1_path = generate_tts_wav(tts_model, intro_part1_text)
    intro_part1_audio = AudioSegment.from_file(intro_part1_path)
    os.remove(intro_part1_path)

    chime_mid_intro_audio = AudioSegment.silent(duration=500, frame_rate=sample_rate)
    if sound_mode in ["With Effects", "Only Effects"] and "chime" in current_effects_list and "chime" in sound_mixer.sound_effects:
        chime_effect = maybe_resample_effect(sound_mixer.sound_effects["chime"], sample_rate) - 15
        chime_mid_intro_audio = chime_effect
        current_effects_list.remove("chime")

    intro_part2_path = generate_tts_wav(tts_model, intro_part2_text)
    intro_part2_audio = AudioSegment.from_file(intro_part2_path)
    os.remove(intro_part2_path)

    full_intro_narration_audio = intro_part1_audio + chime_mid_intro_audio + intro_part2_audio
    final_intro_audio = full_intro_narration_audio

    if sound_mode in ["With Effects", "Only Effects"] and "soft_background_music" in current_effects_list and "soft_background_music" in sound_mixer.sound_effects:
        bgm_effect = maybe_resample_effect(sound_mixer.sound_effects["soft_background_music"], sample_rate)
        while len(bgm_effect) < len(full_intro_narration_audio):
            bgm_effect += bgm_effect
        bgm_effect = bgm_effect[: len(full_intro_narration_audio)] - 20
        final_intro_audio = final_intro_audio.overlay(bgm_effect)
        current_effects_list.remove("soft_background_music")

    main_lesson_audio_segments = []
    used_segment_effects = set()

    for segment in lesson_segments:
        sentence_audio_path = generate_tts_wav(tts_model, segment["sentence"])
        sentence_audio_segment = AudioSegment.from_file(sentence_audio_path)
        os.remove(sentence_audio_path)

        processed_segment_audio = sentence_audio_segment

        if sound_mode == "With Effects" and segment["associated_sound_effects"]:
            for effect_name in segment["associated_sound_effects"]:
                if effect_name in sound_mixer.sound_effects:
                    effect = maybe_resample_effect(sound_mixer.sound_effects[effect_name], sample_rate) - 15
                    if len(effect) > len(processed_segment_audio):
                        effect = effect[: len(processed_segment_audio)]
                    processed_segment_audio = processed_segment_audio.overlay(effect, position=0)
                    used_segment_effects.add(effect_name)

        main_lesson_audio_segments.append(processed_segment_audio)

    current_effects_list = [effect for effect in current_effects_list if effect not in used_segment_effects]

    main_lesson_combined_audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)
    if main_lesson_audio_segments:
        main_lesson_combined_audio = sum(main_lesson_audio_segments)

    combined_narration_audio = final_intro_audio + main_lesson_combined_audio

    closing_path = generate_tts_wav(tts_model, closing_text)
    closing_audio = AudioSegment.from_file(closing_path)
    os.remove(closing_path)

    combined_narration_audio += closing_audio

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_base:
        base_audio_path = temp_base.name
        combined_narration_audio.export(base_audio_path, format="wav")

    final_audio_path = base_audio_path

    if sound_mode == "With Effects":
        mixed_audio = AudioSegment.from_file(base_audio_path)
        general_effects_to_add = [eff for eff in current_effects_list if eff in sound_mixer.sound_effects]

        if general_effects_to_add:
            interval_ms = len(mixed_audio) / (len(general_effects_to_add) + 1)
            current_position_ms = interval_ms
            for effect_name in general_effects_to_add:
                effect = maybe_resample_effect(sound_mixer.sound_effects[effect_name], sample_rate) - 20
                if current_position_ms + len(effect) < len(mixed_audio):
                    mixed_audio = mixed_audio.overlay(effect, position=current_position_ms)
                current_position_ms += interval_ms

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_mixed:
            final_audio_path = temp_mixed.name
            mixed_audio.export(final_audio_path, format="wav")

    elif sound_mode == "Only Effects":
        only_effects_audio = AudioSegment.silent(duration=len(combined_narration_audio), frame_rate=sample_rate)

        all_possible_effects = set(lesson_data.get("sound_effects_list", []))
        for segment in lesson_segments:
            all_possible_effects.update(segment["associated_sound_effects"])

        if "chime" in lesson_data.get("sound_effects_list", []) and "chime" in sound_mixer.sound_effects:
            chime_effect = maybe_resample_effect(sound_mixer.sound_effects["chime"], sample_rate)
            only_effects_audio = only_effects_audio.overlay(chime_effect - 15, position=len(intro_part1_audio))

        if "soft_background_music" in lesson_data.get("sound_effects_list", []) and "soft_background_music" in sound_mixer.sound_effects:
            bgm_effect = maybe_resample_effect(sound_mixer.sound_effects["soft_background_music"], sample_rate)
            while len(bgm_effect) < len(only_effects_audio):
                bgm_effect += bgm_effect
            bgm_effect = bgm_effect[: len(only_effects_audio)] - 20
            only_effects_audio = only_effects_audio.overlay(bgm_effect)

        effects_for_only_mode = [
            effect
            for effect in all_possible_effects
            if effect in sound_mixer.sound_effects and effect not in ["chime", "soft_background_music"]
        ]

        if effects_for_only_mode:
            interval_ms = len(only_effects_audio) / (len(effects_for_only_mode) + 1)
            current_position_ms = interval_ms
            for effect_name in effects_for_only_mode:
                effect = maybe_resample_effect(sound_mixer.sound_effects[effect_name], sample_rate) - 15
                if current_position_ms + len(effect) < len(only_effects_audio):
                    only_effects_audio = only_effects_audio.overlay(effect, position=current_position_ms)
                current_position_ms += interval_ms

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_only:
            final_audio_path = temp_only.name
            only_effects_audio.export(final_audio_path, format="wav")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chapter_part = safe_filename(str(lesson_data.get("chapter", "chapter")))
    topic_part = safe_filename(str(lesson_data.get("Grade/Topic", "topic")))
    mode_part = safe_filename(sound_mode.lower().replace(" ", "_"))

    output_filename = f"{timestamp}_{chapter_part}_{topic_part}_{mode_part}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    AudioSegment.from_file(final_audio_path).export(output_path, format="wav")

    if os.path.exists(base_audio_path):
        os.remove(base_audio_path)
    if final_audio_path != base_audio_path and os.path.exists(final_audio_path):
        os.remove(final_audio_path)

    return output_path


def print_lesson_preview(lesson_data: pd.Series) -> None:
    print("\nChapter:", lesson_data.get("chapter", "N/A"))
    print("Topic:", lesson_data.get("Grade/Topic", "N/A"))
    print("Emotion:", lesson_data.get("emotion", "N/A"))
    print("Sound effects:", lesson_data.get("sound_effects", ""))

    preview_text = str(lesson_data.get("simplified_text", "")).strip()
    if preview_text:
        print("\nText preview:")
        print((preview_text[:300] + "...") if len(preview_text) > 300 else preview_text)


def _resolve_grade_label(grade: int) -> str:
    if grade == 10:
        return "Grade 10"
    if grade == 11:
        return "Grade 11"
    raise ValueError("Grade must be 10 or 11.")


def get_lesson_data_by_indices(grade: int, chapter_idx: int, topic_idx: int) -> pd.Series:
    grade_label = _resolve_grade_label(grade)
    df = load_and_preprocess_dataset(grade_label)

    chapter_df = df[["chapter_num", "chapter_title"]].dropna().drop_duplicates().reset_index(drop=True)
    if chapter_idx < 0 or chapter_idx >= len(chapter_df):
        raise ValueError(
            f"Invalid chapter index {chapter_idx}. Available chapter indices: 0-{max(0, len(chapter_df) - 1)}"
        )

    chapter_num = str(chapter_df.iloc[chapter_idx]["chapter_num"]).strip()
    matched_rows = df[df["chapter_num"].astype(str) == chapter_num].reset_index(drop=True)

    if matched_rows.empty:
        raise ValueError("No lessons found for selected chapter.")

    if topic_idx < 0 or topic_idx >= len(matched_rows):
        raise ValueError(
            f"Invalid topic index {topic_idx}. Available topic indices: 0-{max(0, len(matched_rows) - 1)}"
        )

    return matched_rows.iloc[topic_idx]


def generate_audio_for_selection(
    grade: int,
    chapter_idx: int,
    topic_idx: int,
    sound_mode: str = "With Effects",
    emotion_intensity: float = 1.0,
    api_url: str = API_URL,
    use_model: bool = True,
    model_adapter_dir: str = MODEL_ADAPTER_DIR,
) -> str:
    ensure_nltk()

    valid_modes = {"With Effects", "Only Effects", "Without Effects"}
    if sound_mode not in valid_modes:
        raise ValueError(f"Invalid sound_mode '{sound_mode}'. Use one of: {sorted(valid_modes)}")

    emotion_intensity = max(0.5, min(2.0, float(emotion_intensity)))
    lesson_data = get_lesson_data_by_indices(grade, chapter_idx, topic_idx).copy()

    if use_model:
        topic_name = str(lesson_data.get("Grade/Topic", "")).strip()
        chapter_name = str(lesson_data.get("chapter", "")).strip()
        source_text = str(lesson_data.get("original_text") or lesson_data.get("simplified_text") or "")

        multitask_model = LessonMultitaskModel(adapter_dir=model_adapter_dir)
        if multitask_model.is_ready:
            predicted = multitask_model.generate_fields(
                chapter=chapter_name,
                topic=topic_name,
                source_text=source_text,
            )

            if predicted.get("narrative_text"):
                lesson_data["simplified_text"] = predicted["narrative_text"]
            if predicted.get("emotion"):
                lesson_data["emotion"] = predicted["emotion"]
            if predicted.get("sound_effects"):
                lesson_data["sound_effects"] = predicted["sound_effects"]
                lesson_data["sound_effects_list"] = [
                    item.strip()
                    for item in str(predicted["sound_effects"]).split(",")
                    if item.strip()
                ]

    tts_model = ApiTTSWrapper(api_url)
    sound_mixer = SoundEffectsMixer(SOUNDS_DIR)

    return build_audio(
        lesson_data=lesson_data,
        tts_model=tts_model,
        sound_mixer=sound_mixer,
        sound_mode=sound_mode,
        emotion_intensity=emotion_intensity,
    )


def main() -> None:
    ensure_nltk()

    print("Local lesson audio generator")
    print("API URL:", API_URL)

    grade_options = ["Grade 10", "Grade 11"]
    grade_idx = choose_index("Choose grade", grade_options)
    selected_grade = grade_options[grade_idx]

    df = load_and_preprocess_dataset(selected_grade)

    chapter_df = df[["chapter_num", "chapter_title"]].dropna().drop_duplicates()
    chapter_options = [f"{row['chapter_num']}. {row['chapter_title']}" for _, row in chapter_df.iterrows()]
    chapter_idx = choose_index("Choose chapter", chapter_options)

    chapter_num = str(chapter_options[chapter_idx]).split(".")[0].strip()
    lessons = (
        df[df["chapter_num"].astype(str) == chapter_num]["Grade/Topic"]
        .dropna()
        .astype(str)
        .tolist()
    )
    lesson_idx = choose_index("Choose lesson topic", lessons)
    selected_topic = lessons[lesson_idx]

    matched_rows = df[
        (df["chapter_num"].astype(str) == chapter_num)
        & (df["Grade/Topic"].astype(str) == str(selected_topic))
    ]
    if matched_rows.empty:
        raise ValueError("No lesson found for selected chapter/topic.")

    lesson_data = matched_rows.iloc[0]
    print_lesson_preview(lesson_data)

    sound_mode_options = ["With Effects", "Only Effects", "Without Effects"]
    sound_mode_idx = choose_index("Choose sound mode", sound_mode_options)
    sound_mode = sound_mode_options[sound_mode_idx]

    emotion_raw = input("Emotion intensity (0.5 to 2.0, default 1.0): ").strip()
    emotion_intensity = 1.0
    if emotion_raw:
        try:
            emotion_intensity = float(emotion_raw)
        except ValueError:
            print("Invalid value. Using default 1.0.")
            emotion_intensity = 1.0
    emotion_intensity = max(0.5, min(2.0, emotion_intensity))

    print("\nInitializing TTS model wrapper...")
    tts_model = ApiTTSWrapper(API_URL)
    sound_mixer = SoundEffectsMixer(SOUNDS_DIR)

    print("Generating lesson audio. This can take some time...")
    output_path = build_audio(
        lesson_data=lesson_data,
        tts_model=tts_model,
        sound_mixer=sound_mixer,
        sound_mode=sound_mode,
        emotion_intensity=emotion_intensity,
    )

    print("\nAudio generated successfully.")
    print("Output file:", output_path)


if __name__ == "__main__":
    main()
