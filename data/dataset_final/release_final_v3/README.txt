Dog FACS Dataset - finalny release (Politechnika Gdanska, WETI)

STRUKTURA
  high_resolution/<emocja>/<emocja>_<NNN>_1.jpg   klatka POCZATKU emocji
  high_resolution/<emocja>/<emocja>_<NNN>_2.jpg   klatka KONCA emocji
  low_resolution/<emocja>/...                     j.w. dla nizszej rozdzielczosci
Emocje: neutral, sad, happy, surprise, angry, fearful (surprise i fearful = dwie
osobne klasy). Nic nie zostalo usuniete - podzial na dwa poziomy jakosci.

PODZIAL NA POZIOMY ROZDZIELCZOSCI
  high_resolution: krotszy bok REALNEJ TRESCI >= 720 px
  low_resolution : krotszy bok REALNEJ TRESCI <  720 px
Realna tresc = kadr bez rozmytych/czarnych pasow po bokach (lub u gory/dolu).
Wideo pionowe wklejone w kadr 16:9 z rozmytym tlem po bokach (np. 408x720
w ramce 1280x720) trafia wiec do low_resolution, choc sama ramka ma 720+ px.
Kolumny content_width/content_height/blurred_bars w video_manifest.csv i w COCO.

SPOJNE NAZEWNICTWO
  Wideo i obrazy maja te sama baze nazwy: <emocja>_<NNN>.
  Wideo: <emocja>_<NNN>.mp4 (lub .mov), klatki: <emocja>_<NNN>_1.jpg / _2.jpg.
  Numeracja ciagla w obrebie emocji (obejmuje oba poziomy).
  Wideo na Google Drive: DOGS/DATASET_final_videos/<poziom>/<emocja>/<emocja>_<NNN>.<ext>
  (kopie; oryginaly w DOGS/ bez zmian). Mapa: video_manifest.csv
  (video_name, source_video = oryginalna nazwa, drive_path, drive_file_id).

PLIKI
  annotations.json                 wszystko (COCO: bbox, 46 keypoints, rasa, 21 AU)
  annotations_<emocja>.json        per emocja
  annotations_high_resolution.json / annotations_low_resolution.json
  frames.csv, licenses.csv, video_manifest.csv
  double_evaluation.csv            PODWOJNA OCENA emocji przez dwie rozne osoby
