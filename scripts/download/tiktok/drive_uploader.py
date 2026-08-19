"""
Upload plików wideo na Google Drive przez OAuth (konto osobiste).

Przy pierwszym uruchomieniu otwiera przeglądarkę do zalogowania i zapisuje
token odświeżania lokalnie (secrets/token.json), żeby kolejne uruchomienia
nie wymagały ponownego logowania.
"""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class GoogleDriveUploader:
    """
    Klienta do uwierzytelniania i wysyłania plików na Google Drive.

    Użycie:
        uploader = GoogleDriveUploader(credentials_path, token_path, folder_id)
        uploader.authenticate()
        file_id = uploader.upload_file(Path("video.mp4"))
    """

    def __init__(
        self,
        credentials_path: Path,
        token_path: Path,
        folder_id: str,
    ) -> None:
        """
        Inicjalizuje uploader.

        Args:
            credentials_path: Ścieżka do pliku OAuth client (JSON z Google Cloud)
            token_path: Ścieżka, gdzie zapisany/wczytany będzie token sesji
            folder_id: ID docelowego folderu na Google Drive
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.folder_id = folder_id
        self._service = None

    def authenticate(self) -> None:
        """
        Uwierzytelnia użytkownika (interaktywnie przy pierwszym uruchomieniu).

        Raises:
            FileNotFoundError: Gdy brak pliku credentials_path
        """
        creds: Credentials | None = None

        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"Brak pliku OAuth credentials: {self.credentials_path}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")

        self._service = build("drive", "v3", credentials=creds)

    def ensure_folder(self, name: str, parent_id: str) -> str:
        """
        Zwraca ID podfolderu o danej nazwie w parent_id, tworząc go jeśli brak.

        Args:
            name: Nazwa folderu (np. nazwa klasy emocji)
            parent_id: ID folderu nadrzędnego na Drive

        Returns:
            ID istniejącego lub nowo utworzonego folderu

        Raises:
            RuntimeError: Gdy authenticate() nie zostało wcześniej wywołane
        """
        if self._service is None:
            raise RuntimeError("Wywołaj authenticate() przed ensure_folder().")

        query = (
            f"name = '{name}' and '{parent_id}' in parents "
            "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        existing = self._service.files().list(q=query, fields="files(id)").execute()
        files = existing.get("files", [])
        if files:
            return files[0]["id"]

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        created = self._service.files().create(body=metadata, fields="id").execute()
        return created["id"]

    def upload_file(
        self, file_path: Path, remote_name: str | None = None, folder_id: str | None = None
    ) -> str:
        """
        Wysyła pojedynczy plik do folderu na Drive.

        Args:
            file_path: Ścieżka do lokalnego pliku
            remote_name: Nazwa pliku na Drive (domyślnie nazwa lokalna)
            folder_id: Docelowy folder (domyślnie self.folder_id)

        Returns:
            ID utworzonego pliku na Google Drive

        Raises:
            RuntimeError: Gdy authenticate() nie zostało wcześniej wywołane
        """
        if self._service is None:
            raise RuntimeError("Wywołaj authenticate() przed upload_file().")

        metadata = {
            "name": remote_name or file_path.name,
            "parents": [folder_id or self.folder_id],
        }
        media = MediaFileUpload(str(file_path), resumable=True)

        uploaded = (
            self._service.files()
            .create(body=metadata, media_body=media, fields="id")
            .execute()
        )
        return uploaded["id"]

    def list_files(self, folder_id: str) -> list[dict]:
        """
        Zwraca listę WSZYSTKICH plików (id, name) bezpośrednio w danym folderze.

        Obsługuje paginację - pojedyncze zapytanie do Drive API zwraca maks.
        ~100 wyników, więc bez tego duże foldery (100+ plików) byłyby ucinane.

        Args:
            folder_id: ID folderu na Drive

        Returns:
            Lista słowników {"id": ..., "name": ...}

        Raises:
            RuntimeError: Gdy authenticate() nie zostało wcześniej wywołane
        """
        if self._service is None:
            raise RuntimeError("Wywołaj authenticate() przed list_files().")

        files: list[dict] = []
        page_token: str | None = None
        while True:
            result = (
                self._service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id,name)",
                    pageSize=1000,
                    pageToken=page_token,
                )
                .execute()
            )
            files.extend(result.get("files", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        return files

    def download_file(self, file_id: str, destination: Path) -> None:
        """
        Pobiera plik z Drive do lokalnej ścieżki.

        Args:
            file_id: ID pliku na Drive
            destination: Lokalna ścieżka docelowa

        Raises:
            RuntimeError: Gdy authenticate() nie zostało wcześniej wywołane
        """
        if self._service is None:
            raise RuntimeError("Wywołaj authenticate() przed download_file().")

        destination.parent.mkdir(parents=True, exist_ok=True)
        request = self._service.files().get_media(fileId=file_id)
        with open(destination, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
