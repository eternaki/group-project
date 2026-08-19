"""
Konfiguracja kolektora wideo z TikToka.

Wartości progowe i listy dobrane pod kątem zbierania klatek z wyraźnie
widoczną mordą psa i odsiewania treści oznaczonej jako wygenerowana przez AI.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

# Hashtagi do przeszukania (kolejność = priorytet).
# Ogólne (duża objętość, głównie neutral/relaxed/happy) + hashtagi ukierunkowane
# na konkretne emocje (żeby rzadkie klasy typu angry/fearful/surprise nie musiały
# czekać, aż trafią się przypadkiem wśród ogólnych wideo).
# Usunięto: "dog"/"dogs" (zbyt szerokie, dużo treści bez wyraźnej mordy/memów),
# długie rosyjskie frazy złożone (собачкинаулице/наприроде/напрогулке - prawdopodobnie
# znikoma realna objętość na TikToku, tylko marnowały czas scrollowania do stagnacji).
DEFAULT_HASHTAGS: list[str] = [
    # Ogólne
    "dogsoftiktok",
    "dogfacecloseup",
    "dogexpressions",
    "doglife",
    "dogmood",
    "dogemotions",
    "dogfaces",
    "собака",
    "собаки",
    "собачка",
    "собачки",
    # Happy
    "happydog",
    "dogsmile",
    "dogtailwag",
    # Sad
    "saddog",
    "dogsadeyes",
    # Angry
    "angrydog",
    "growlingdog",
    # Fearful
    "scareddog",
    "anxiousdog",
    "nervousdog",
    # Surprise
    "surpriseddog",
    "shockeddog",
    # Submission
    "submissivedog",
    "shydog",
]

# Frazy/hashtagi wskazujące na treść wygenerowaną przez AI (filtr heurystyczny)
AI_CONTENT_MARKERS: list[str] = [
    "ai generated",
    "aigenerated",
    "#ai",
    "#aiart",
    "#aivideo",
    "#aidog",
    "#capcutai",
    "#veo",
    "#veo3",
    "#sora",
    "#midjourney",
    "#stablediffusion",
    "#runwayml",
    "#pika",
    "made with ai",
    "created with ai",
]

# Docelowa liczba wideo NA KAŻDĄ z 9 klas emocji (patrz packages/data/schemas.EMOTION_CLASSES)
TARGET_PER_EMOTION = 500

# Klasyfikacja emocji (pełny pipeline: keypoints -> neutral frame -> delta AU -> reguły)
# 14 zamiast 24 klatek - pomiar pokazał 73s/wideo na CPU dla two-pass keypoints,
# to główny koszt czasowy całego kolektora (patrz README: benchmark).
EMOTION_FRAME_SAMPLE_COUNT = 14
EMOTION_NUM_PEAKS = 5  # ile "peak frames" wybrać do głosowania nad finalną emocją
# Progi peak selectora POLUZOWANE względem domyślnych w peak_selector.py - materiał
# z TikToka jest bardziej zaszumiony/skompresowany niż źródło, pod które dopasowano
# oryginalne progi (30°/60 sharpness), przez co prawie każde wideo dawało 0 peaków.
EMOTION_MAX_HEAD_ANGLE = 60.0
EMOTION_MIN_KEYPOINT_CONF = 0.35
EMOTION_MIN_SHARPNESS = 15.0
# Próg jakości finalnej etykiety - odrzuca niepewne/niezgodne klasyfikacje zamiast
# wrzucać je do byle jakiej emocji (obserwacja: dużo fałszywych "neutral" przy
# niskim confidence pojedynczych peaków).
EMOTION_MIN_VOTE_RATIO = 0.5  # zwycięska emocja musi mieć >=50% głosów peaków
EMOTION_MIN_CONFIDENCE = 0.6  # średnie confidence zwycięskich peaków

BREED_WEIGHTS = PROJECT_ROOT / "models" / "breed.pt"
# Celowo nieistniejąca ścieżka - wyłącza klasyfikację rasy w kolektorze (niepotrzebna
# tu, a kosztuje 1 dodatkowe inference EfficientNet-B4 na każdą z 5 peak frames).
BREED_WEIGHTS_DISABLED = PROJECT_ROOT / "models" / "__breed_disabled__.pt"
FACE_DETECTOR_WEIGHTS = PROJECT_ROOT / "models" / "dogface_yolo.pt"
BREEDS_JSON = PROJECT_ROOT / "packages" / "models" / "breeds.json"

# Limity tempa zapytań do TikToka (sekundy) - bez proxy, jeden IP
MIN_REQUEST_DELAY_SECONDS = 4.0
MAX_REQUEST_DELAY_SECONDS = 9.0
VIDEOS_PER_HASHTAG_PER_ROUND = 30

# Przeglądarka (Playwright) - profil trwały, żeby sesja wyglądała jak
# powracający użytkownik, a nie nowe "czyste" urządzenie za każdym razem
CHROME_PROFILE_DIR = PROJECT_ROOT / "secrets" / "tiktok_browser_profile"
SCROLL_PAUSE_MIN_SECONDS = 1.5
SCROLL_PAUSE_MAX_SECONDS = 3.0
SCROLL_STAGNANT_ROUNDS_LIMIT = 6
CAPTCHA_WAIT_TIMEOUT_SECONDS = 900  # czas na ręczne rozwiązanie captchy
# Powiadomienie push (ntfy.sh, bez rejestracji) wysyłane, gdy pojawi się captcha -
# pozwala nie siedzieć przy komputerze "na wszelki wypadek". Temat losowy/tajny,
# bo tematy na ntfy.sh są publiczne (każdy znający nazwę może odczytać wiadomości).
NTFY_TOPIC = "dog-facs-captcha-b300b0ba0d75d3e7760e42159379f9b5"

# Po tylu błędach/rate-limit z rzędu skrypt robi długą przerwę
SOFTBAN_COOLDOWN_SECONDS = 20 * 60
MAX_CONSECUTIVE_ERRORS = 5

# Pobieranie
MAX_VIDEO_DURATION_SECONDS = 60
DOWNLOAD_TMP_DIR = PROJECT_ROOT / "data" / "tiktok_tmp"
STATE_FILE = PROJECT_ROOT / "data" / "tiktok_state.json"
CAPTCHA_FLAG_FILE = PROJECT_ROOT / "data" / "tiktok_captcha_pending.flag"

# Modele (te same wagi co reszta pipeline'u)
BBOX_WEIGHTS = PROJECT_ROOT / "models" / "yolov8m.pt"
KEYPOINTS_WEIGHTS = PROJECT_ROOT / "models" / "keypoints_dogflw.pt"

# Google Drive
GDRIVE_CREDENTIALS_PATH = PROJECT_ROOT / "secrets" / "gdrive_credentials.json"
GDRIVE_TOKEN_PATH = PROJECT_ROOT / "secrets" / "token.json"
GDRIVE_FOLDER_ID = "1jxUaN3Mq1ge8lFcPzwnN2ISl0E9k9mfQ"


@dataclass
class CollectorConfig:
    """
    Pełna konfiguracja przebiegu kolektora.

    Attributes:
        hashtags: Lista hashtagów do przeszukania
        target_per_emotion: Docelowa liczba zaakceptowanych wideo NA KAŻDĄ emocję
        ms_token: Token sesji TikTok (cookie ms_token zalogowanego konta)
        device: Urządzenie inference dla modeli ('cuda' lub 'cpu')
    """

    hashtags: list[str] = field(default_factory=lambda: list(DEFAULT_HASHTAGS))
    target_per_emotion: int = TARGET_PER_EMOTION
    ms_token: str = field(default_factory=lambda: os.environ.get("TIKTOK_MS_TOKEN", ""))
    device: str = field(default_factory=lambda: os.environ.get("DOG_FACS_DEVICE", "cpu"))

    def __post_init__(self) -> None:
        if not self.ms_token:
            raise ValueError(
                "Brak zmiennej środowiskowej TIKTOK_MS_TOKEN "
                "(wartość cookie ms_token z zalogowanej sesji TikTok w przeglądarce)."
            )
