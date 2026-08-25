"""
Wyszukiwanie wideo TikTok po hashtagach - bezpośrednio przez Playwright.

Otwiera realną stronę hashtagu w widocznej przeglądarce (z trwałym profilem,
żeby sesja wyglądała jak powracający użytkownik) i czyta linki do wideo z DOM,
zamiast podpisywać żądania do prywatnego API (co robił TikTokApi i co TikTok
konsekwentnie blokuje captchą).

Jeśli TikTok wyświetli captchę (slider "ułóż puzzle"), skrypt PAUZUJE i czeka,
aż użytkownik rozwiąże ją ręcznie w tym samym oknie - żadnego automatycznego
obchodzenia captchy.
"""

import random

from scripts.download.tiktok.common import VideoInfo
from scripts.download.tiktok.config import (
    CAPTCHA_FLAG_FILE,
    CAPTCHA_WAIT_TIMEOUT_SECONDS,
    CHROME_PROFILE_DIR,
    NTFY_TOPIC,
    SCROLL_PAUSE_MAX_SECONDS,
    SCROLL_PAUSE_MIN_SECONDS,
    SCROLL_STAGNANT_ROUNDS_LIMIT,
)

CAPTCHA_SELECTOR = "#captcha-verify-container, #captcha_container, [class*='captcha' i]"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class TikTokHashtagSearcher:
    """
    Przegląda stronę hashtagu w prawdziwej przeglądarce i zbiera linki wideo.

    Użycie:
        async with TikTokHashtagSearcher(ms_token) as searcher:
            async for video in searcher.search("dogsoftiktok", count=30):
                print(video.url)
    """

    def __init__(self, ms_token: str) -> None:
        self.ms_token = ms_token
        self._playwright = None
        self._context = None

    async def __aenter__(self) -> "TikTokHashtagSearcher":
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        CHROME_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self._context = await self._playwright.chromium.launch_persistent_context(
            str(CHROME_PROFILE_DIR),
            headless=False,
            viewport={"width": 1280, "height": 900},
            user_agent=_USER_AGENT,
        )
        await self._context.add_cookies(
            [{"name": "msToken", "value": self.ms_token, "domain": ".tiktok.com", "path": "/"}]
        )

        # Zachowanie bardziej podobne do człowieka: najpierw strona główna
        page = await self._context.new_page()
        await page.goto("https://www.tiktok.com/", wait_until="domcontentloaded", timeout=30000)
        await self._wait_out_captcha(page)
        await page.wait_for_timeout(random.randint(2000, 4000))
        await page.close()

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._context is not None:
            await self._context.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def search(self, hashtag: str, count: int):
        """
        Przewija stronę hashtagu i zwraca znalezione wideo.

        Args:
            hashtag: Nazwa hashtagu bez '#'
            count: Docelowa liczba unikalnych wideo do zebrania

        Yields:
            VideoInfo dla każdego znalezionego wideo
        """
        if self._context is None:
            raise RuntimeError("Użyj 'async with TikTokHashtagSearcher(...)'.")

        page = await self._context.new_page()
        try:
            await page.goto(
                f"https://www.tiktok.com/tag/{hashtag}",
                wait_until="domcontentloaded",
                timeout=30000,
            )
            await self._wait_out_captcha(page)

            found_urls: set[str] = set()
            stagnant_rounds = 0

            while len(found_urls) < count and stagnant_rounds < SCROLL_STAGNANT_ROUNDS_LIMIT:
                await self._wait_out_captcha(page)

                links = await page.eval_on_selector_all(
                    "a[href*='/video/']", "els => els.map(e => e.href)"
                )
                before = len(found_urls)
                found_urls.update(links)

                if len(found_urls) == before:
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0

                await page.mouse.wheel(0, random.randint(800, 1400))
                await page.wait_for_timeout(
                    random.randint(
                        int(SCROLL_PAUSE_MIN_SECONDS * 1000), int(SCROLL_PAUSE_MAX_SECONDS * 1000)
                    )
                )

            for url in list(found_urls)[:count]:
                video_id = url.rstrip("/").rsplit("/", 1)[-1]
                yield VideoInfo(video_id=video_id, url=url, source_label=hashtag, platform="tiktok")

        finally:
            await page.close()

    async def _wait_out_captcha(self, page) -> None:
        """Jeśli TikTok pokazuje captchę, czeka aż użytkownik rozwiąże ją ręcznie."""
        captcha = await page.query_selector(CAPTCHA_SELECTOR)
        if captcha is None:
            return

        print(
            "Wykryto captchę TikTok - rozwiąż ją ręcznie w otwartym oknie przeglądarki. "
            f"Czekam maks. {CAPTCHA_WAIT_TIMEOUT_SECONDS}s..."
        )
        CAPTCHA_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CAPTCHA_FLAG_FILE.write_text("pending", encoding="utf-8")
        self._notify_captcha()
        try:
            await page.wait_for_selector(
                CAPTCHA_SELECTOR,
                state="detached",
                timeout=CAPTCHA_WAIT_TIMEOUT_SECONDS * 1000,
            )
            print("Captcha rozwiązana, kontynuuję.")
        except Exception as e:
            raise RuntimeError("Captcha nie została rozwiązana w wyznaczonym czasie.") from e
        finally:
            CAPTCHA_FLAG_FILE.unlink(missing_ok=True)

    @staticmethod
    def _notify_captcha() -> None:
        """Wysyła push (ntfy.sh) o pojawieniu się captchy - nie blokuje przy błędzie sieci."""
        import requests

        try:
            requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data="TikTok pokazał captchę - otwórz okno przeglądarki i rozwiąż ją ręcznie.".encode("utf-8"),
                headers={"Title": "Dog FACS collector - captcha", "Priority": "urgent"},
                timeout=10,
            )
        except Exception:
            pass
