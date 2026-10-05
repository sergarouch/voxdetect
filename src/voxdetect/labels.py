"""Label schema for anonymous community collection.

Four classes stay separate all the way to training:

- silence: no speech and no distinct sound
- noise: sound that is not speech, plus a noise kind
- english: speech, language fixed as en
- non_english: speech in any other language, with an ISO 639-1 code

Silence and noise never carry a language. English is never filed under
non_english. There is no free-text field, so a filename or a comment cannot
smuggle a name into the dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCHEMA_VERSION = 1
CLASS_ORDER = ("silence", "noise", "english", "non_english")
RECORD_SECONDS = 4
MIN_SECONDS = 0.3
MAX_SECONDS = 15.0
SAMPLE_RATE = 16000
MAX_BYTES = 8 * 1024 * 1024

NOISE_KINDS = (
    ("room", "Room", "Fan or vent. No voices."),
    ("traffic", "Traffic", "Street. No talk."),
    ("music", "Music", "Music only."),
    ("typing", "Typing", "Keys. No talk."),
    ("crowd", "Crowd", "Far voices. No words."),
    ("appliance", "Machine", "Machine running."),
    ("outdoors", "Outside", "Wind, birds, water."),
    ("other", "Other", "Not speech."),
)

# Featured languages for the picker. English is intentionally absent.
LANGUAGES = (
    ("am", "አማርኛ", "Amharic"),
    ("ar", "العربية", "Arabic"),
    ("bn", "বাংলা", "Bengali"),
    ("cs", "Čeština", "Czech"),
    ("da", "Dansk", "Danish"),
    ("de", "Deutsch", "German"),
    ("el", "Ελληνικά", "Greek"),
    ("es", "Español", "Spanish"),
    ("fa", "فارسی", "Persian"),
    ("fi", "Suomi", "Finnish"),
    ("fr", "Français", "French"),
    ("gu", "ગુજરાતી", "Gujarati"),
    ("ha", "Hausa", "Hausa"),
    ("he", "עברית", "Hebrew"),
    ("hi", "हिन्दी", "Hindi"),
    ("hu", "Magyar", "Hungarian"),
    ("id", "Bahasa Indonesia", "Indonesian"),
    ("it", "Italiano", "Italian"),
    ("ja", "日本語", "Japanese"),
    ("ko", "한국어", "Korean"),
    ("mr", "मराठी", "Marathi"),
    ("ms", "Bahasa Melayu", "Malay"),
    ("nl", "Nederlands", "Dutch"),
    ("pa", "ਪੰਜਾਬੀ", "Punjabi"),
    ("pl", "Polski", "Polish"),
    ("pt", "Português", "Portuguese"),
    ("ro", "Română", "Romanian"),
    ("ru", "Русский", "Russian"),
    ("sv", "Svenska", "Swedish"),
    ("sw", "Kiswahili", "Swahili"),
    ("ta", "தமிழ்", "Tamil"),
    ("te", "తెలుగు", "Telugu"),
    ("th", "ไทย", "Thai"),
    ("tl", "Filipino", "Filipino"),
    ("tr", "Türkçe", "Turkish"),
    ("uk", "Українська", "Ukrainian"),
    ("ur", "اردو", "Urdu"),
    ("vi", "Tiếng Việt", "Vietnamese"),
    ("yo", "Yorùbá", "Yoruba"),
    ("zh", "中文", "Chinese"),
)

UNKNOWN_LANGUAGE = ("und", "Not sure", "I don't know the language code")

ENGLISH_PROMPTS = (
    {"id": "en-01", "text": "The quick brown fox jumps over the lazy dog."},
    {"id": "en-02", "text": "Please leave the package by the front door."},
    {"id": "en-03", "text": "Tomorrow morning the train leaves at seven."},
    {"id": "en-04", "text": "I brought apples, bread, and cold water."},
    {"id": "en-05", "text": "Can you hear the rain against the window?"},
    {"id": "en-06", "text": "Red birds gathered along the stone wall."},
    {"id": "en-07", "text": "We finished the work before sunset."},
    {"id": "en-08", "text": "The kitchen clock ticks a little fast."},
)

OTHER_PROMPTS = {
    "ar": {"id": "ar-01", "text": "الجو جميل اليوم."},
    "de": {"id": "de-01", "text": "Das Fenster steht einen Spalt offen."},
    "es": {"id": "es-01", "text": "Hoy el cielo está despejado."},
    "fr": {"id": "fr-01", "text": "Le train part demain matin."},
    "hi": {"id": "hi-01", "text": "आज मौसम अच्छा है।"},
    "id": {"id": "id-01", "text": "Hari ini cuacanya cerah."},
    "it": {"id": "it-01", "text": "Il caffè è ancora caldo."},
    "ja": {"id": "ja-01", "text": "窓の外で雨が降っています。"},
    "ko": {"id": "ko-01", "text": "창밖으로 비가 내리고 있어요."},
    "nl": {"id": "nl-01", "text": "De trein vertrekt morgenochtend."},
    "pl": {"id": "pl-01", "text": "Deszcz pada od rana."},
    "pt": {"id": "pt-01", "text": "A chuva começou de manhã."},
    "ru": {"id": "ru-01", "text": "Утром поезд уходит в семь."},
    "sw": {"id": "sw-01", "text": "Leo kuna jua kali."},
    "tr": {"id": "tr-01", "text": "Yarın sabah tren erken kalkıyor."},
    "uk": {"id": "uk-01", "text": "Завтра вранці йде дощ."},
    "vi": {"id": "vi-01", "text": "Hôm nay trời đẹp."},
    "zh": {"id": "zh-01", "text": "今天天气很好。"},
}

LABEL_COPY = (
    {
        "id": "silence",
        "index": "01",
        "title": "Silence",
        "summary": "No sound.",
        "instructions": "Set it down. Do not speak.",
        "steps": [
            "Set the device down.",
            "Do not speak, type, or tap.",
        ],
    },
    {
        "id": "noise",
        "index": "02",
        "title": "Noise",
        "summary": "Sound. No speech.",
        "instructions": "Point at the sound. Do not talk.",
        "steps": [
            "Pick the noise. No voices.",
            "If you hear words, discard.",
        ],
    },
    {
        "id": "english",
        "index": "03",
        "title": "English",
        "summary": "English speech.",
        "instructions": "Read the line. No name.",
        "steps": [
            "Use only English.",
            "Read the line. No name.",
        ],
    },
    {
        "id": "non_english",
        "index": "04",
        "title": "Other language",
        "summary": "Not English.",
        "instructions": "Pick the language. Read the line.",
        "steps": [
            "Not English. Read the line.",
            "No code? Choose Not sure.",
        ],
    },
)

RECORD_FIELDS = (
    "schema_version",
    "submission_id",
    "label",
    "language",
    "noise_kind",
    "prompt_id",
    "duration_sec",
    "sample_rate",
    "channels",
    "format",
    "consent",
    "created_at",
    "path",
)

_PROMPT_ID = re.compile(r"^[a-z0-9-]{1,32}$")
_LANGUAGE_CODES = {code for code, _endonym, _english in LANGUAGES} | {UNKNOWN_LANGUAGE[0]}
_NOISE_IDS = {kind for kind, _title, _summary in NOISE_KINDS}


class LabelError(ValueError):
    """The submission breaks the label schema."""


@dataclass(frozen=True)
class NormalizedSubmission:
    label: str
    language: str | None
    noise_kind: str | None
    prompt_id: str | None


def class_index(label: str) -> int:
    return CLASS_ORDER.index(label)


def public_schema() -> dict:
    """JSON the website and the training import both treat as the label contract."""
    prompts = {"en": list(ENGLISH_PROMPTS)}
    prompts.update({code: [prompt] for code, prompt in OTHER_PROMPTS.items()})
    return {
        "schema_version": SCHEMA_VERSION,
        "record_seconds": RECORD_SECONDS,
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "labels": list(LABEL_COPY),
        "noise_kinds": [
            {"id": kind, "title": title, "summary": summary} for kind, title, summary in NOISE_KINDS
        ],
        "languages": [
            {"code": code, "endonym": endonym, "english": english}
            for code, endonym, english in sorted(LANGUAGES, key=lambda item: item[2])
        ],
        "unknown_language": {
            "code": UNKNOWN_LANGUAGE[0],
            "endonym": UNKNOWN_LANGUAGE[1],
            "english": UNKNOWN_LANGUAGE[2],
        },
        "prompts": prompts,
        "fields": list(RECORD_FIELDS),
        "rules": [
            "silence and noise do not have a language.",
            "english is stored as language en and is not a non_english clip.",
            "non_english requires an ISO 639-1 code other than en, or und when the code is unknown.",
            "noise requires a noise kind.",
            "Original filenames, names, and free-text comments are not stored.",
        ],
    }


def validate_submission(
    label: str,
    language: str | None,
    noise_kind: str | None,
    prompt_id: str | None,
    consent: bool,
) -> NormalizedSubmission:
    """Return a clean label record or raise LabelError. Empty strings count as missing."""
    if consent is not True:
        raise LabelError("Consent is required before a clip can be stored.")

    label_name = (label or "").strip().lower()
    if label_name not in CLASS_ORDER:
        allowed = ", ".join(CLASS_ORDER)
        raise LabelError(f"Unknown label {label!r}. Use one of: {allowed}.")

    language_code = _blank(language)
    kind = _blank(noise_kind)
    prompt = _blank(prompt_id)

    if label_name == "silence":
        if language_code or kind or prompt:
            raise LabelError("Silence is stored with no language, noise kind, or prompt.")
        return NormalizedSubmission("silence", None, None, None)

    if label_name == "noise":
        if language_code or prompt:
            raise LabelError("Noise does not take a language or a spoken prompt.")
        if kind not in _NOISE_IDS:
            allowed = ", ".join(sorted(_NOISE_IDS))
            raise LabelError(f"Noise needs a noise kind. Use one of: {allowed}.")
        return NormalizedSubmission("noise", None, kind, None)

    if label_name == "english":
        if kind:
            raise LabelError("English speech does not take a noise kind.")
        if language_code not in {None, "en"}:
            raise LabelError("English speech is stored as language en, not as another language.")
        _check_prompt("english", "en", prompt)
        return NormalizedSubmission("english", "en", None, prompt)

    if kind:
        raise LabelError("A non-English clip does not take a noise kind.")
    if not language_code or language_code not in _LANGUAGE_CODES:
        raise LabelError(
            "Choose the language for non-English speech. English belongs in the English task. Use und if you do not know the code."
        )
    _check_prompt("non_english", language_code, prompt)
    return NormalizedSubmission("non_english", language_code, None, prompt)


def _check_prompt(label: str, language: str, prompt_id: str | None) -> None:
    if prompt_id is None:
        return
    if _PROMPT_ID.fullmatch(prompt_id) is None:
        raise LabelError("Prompt ids are short codes such as en-01, not free text.")
    allowed = _prompt_ids(label, language)
    if prompt_id not in allowed:
        raise LabelError("That prompt does not belong to this label.")


def _prompt_ids(label: str, language: str) -> set[str]:
    if label == "english":
        return {item["id"] for item in ENGLISH_PROMPTS}
    prompt = OTHER_PROMPTS.get(language)
    if prompt is None:
        return set()
    return {prompt["id"]}


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None
