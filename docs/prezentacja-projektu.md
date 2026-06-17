---
marp: true
title: DogFACS Dataset Generator
author: Zespół Dogs-ai (WETI PG)
paginate: true
theme: default
---

<!--
Prezentacja projektu "DogFACS Dataset Generator" — styl zgodny z wersją zespołową.
Render:  npx @marp-team/marp-cli docs/prezentacja-projektu.md -o out.pdf --allow-local-files --html
         (PPTX: zmień rozszerzenie na .pptx)
-->

<style>
:root {
  --green:      #2e5d46;
  --green-dark: #234a37;
  --green-soft: #6f9e88;
  --orange:     #df9460;
  --orange-soft:#f0c9a8;
  --blue:       #7fb1da;
  --blue-soft:  #aacfe9;
  --paper:      #f3f1ea;
  --paper-2:    #eeae0;
  --ink:        #2c3330;
  --muted:      #6c7770;
}
section {
  font-family: 'Nunito Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  font-size: 23px; color: var(--ink); background: #ffffff;
  padding: 48px 60px; line-height: 1.4;
}
strong { color: var(--green); }
.eyebrow {
  color: var(--orange); font-weight: 800; text-transform: uppercase;
  letter-spacing: .18em; font-size: .62em; margin-bottom: .15em;
}
h1 { color: var(--green); font-size: 1.85em; margin: 0 0 .5em; font-weight: 800; }
h2 { color: var(--green); font-size: 1.25em; margin: 0 0 .4em; font-weight: 800; }
section::after { color: var(--green-soft); font-weight: 700; }

/* karty */
.grid { display: flex; gap: 20px; }
.grid.col2 > * { flex: 1; }
.grid.col3 > * { flex: 1; }
.card {
  background: var(--paper); border-radius: 16px; padding: 22px 24px;
  box-shadow: 0 2px 10px rgba(0,0,0,.05);
}
.card h3 { margin: 0 0 .35em; color: var(--green); font-size: .95em; display: flex; align-items: center; gap: .4em; }
.card p  { margin: .2em 0; font-size: .82em; color: var(--ink); }
.card .sub { color: var(--muted); font-size: .8em; }
.card.green  { background: var(--green); }
.card.green *  { color: #eef4f0 !important; }
.card.blue   { background: var(--blue-soft); }
.card.blue h3, .card.blue p { color: #234a52 !important; }
.card.orange { background: var(--orange); }
.card.orange * { color: #fff !important; }
.ico { display:inline-grid; place-items:center; width:1.7em; height:1.7em; border-radius:50%;
       background: var(--green); color:#fff; font-size:.9em; }
.ico.b{ background: var(--blue); } .ico.o{ background: var(--orange); }

/* listy z kropką */
ul { margin:.2em 0; } li { margin:.25em 0; font-size:.86em; }
ul > li::marker { color: var(--orange); }

/* pasek boczny akcentu */
.acc { border-left: 6px solid var(--green); padding-left: 14px; border-radius: 6px; }
.acc.b{ border-color: var(--blue); } .acc.o{ border-color: var(--orange); }

/* numerowane elementy (TOC / lista) */
.row { display:flex; align-items:center; gap:14px; background:var(--paper); border-radius:14px;
       padding:14px 18px; margin:8px 0; border-left:6px solid var(--green); }
.row.b{ border-color:var(--blue);} .row.o{ border-color:var(--orange);}
.num { display:inline-grid; place-items:center; min-width:38px; height:38px; border-radius:50%;
       background:var(--green); color:#fff; font-weight:800; font-size:.85em; }
.num.b{ background:var(--blue);} .num.o{ background:var(--orange);}
.row .t { font-weight:800; color:var(--green); font-size:.92em; }
.row .s { color:var(--muted); font-size:.78em; }

code { background:#eef1ee; color:var(--green); padding:.06em .35em; border-radius:5px; font-size:.9em; }
pre { background: var(--green-dark); border-radius:12px; padding:.8em 1em; }
pre code { background:transparent; color:#dfe9e2; font-size:.78em; }
.note { background:var(--paper-2); border-radius:12px; padding:12px 18px; font-size:.8em; color:var(--muted); }
.pill { display:inline-block; background:var(--orange); color:#fff; font-weight:800;
        padding:.35em 1.1em; border-radius:999px; font-size:.7em; letter-spacing:.04em; }

/* okładka */
section.cover { background: linear-gradient(135deg,#2e5d46 0%,#21412f 100%); color:#fff; padding:64px; }
section.cover h1 { color:#fff; font-size:3.1em; line-height:1.05; margin:.25em 0 .35em; }
section.cover .sub { color:#cfe0d6; font-size:1.05em; }
section.cover .pill { background: var(--orange-soft); color:#5a3a22; }
section.cover .meta { display:flex; gap:60px; margin-top:42px; border-top:1px solid rgba(255,255,255,.25); padding-top:22px; }
section.cover .meta .lbl { color:#9fbdac; font-size:.72em; } section.cover .meta b{ color:#fff; }
section.cover::after { display:none; }

/* przekładka rozdziału */
section.chapter { background: linear-gradient(120deg,#2e5d46 0%,#1c3a2b 100%); color:#fff; justify-content:center; }
section.chapter h1 { color:#fff; font-size:3em; margin:.2em 0 .25em; }
section.chapter .sub { color:#cfe0d6; font-size:1em; }
section.chapter::after { color: rgba(255,255,255,.4); }
</style>

<!-- _class: cover -->

<span class="pill">PROJEKT BADAWCZY</span>

# DogFACS<br>Dataset Generator

<span class="sub">System automatycznej anotacji emocji psów<br>z wykorzystaniem sztucznej inteligencji</span>

<div class="meta">
<div><div class="lbl">Zespół</div><b>Danylo Lohachov</b><br><b>Danylo Zherzdiev</b><br><b>Anton Shkrebela</b><br><b>Mariia Volkova</b></div>
<div><div class="lbl">Opiekun</div><b>dr hab. inż. Michał Czubenko</b></div>
<div><div class="lbl">Uczelnia</div><b>Politechnika Gdańska</b><br>Wydział ETI</div>
</div>

---

<div class="eyebrow">Informacje o projekcie</div>

# Skład grupy projektowej

<div class="grid col2">
<div>
<div class="card"><h3>🏷️ „Dog FACS"</h3>
<p><strong>Aplikacja AI do automatycznej anotacji zbioru danych</strong> z wykorzystaniem DogFACS.</p>
<p class="sub">Rozpoznawanie emocji psów oparte o analizę mikroekspresji pyska i sztuczną inteligencję.</p></div>
<div class="card"><h3>🎓 Opiekun projektu</h3>
<p><strong>dr hab. inż. Michał Czubenko</strong></p></div>
</div>
<div>
<div class="row"><span class="num">1</span><div><div class="t">Danylo Lohachov</div><div class="s">Koordynator / Dokumentacja / QA / Frontend</div></div></div>
<div class="row b"><span class="num b">2</span><div><div class="t">Anton Shkrebela</div><div class="s">AI/ML — modele Keypoints i DogFACS</div></div></div>
<div class="row o"><span class="num o">3</span><div><div class="t">Danylo Zherzdiev</div><div class="s">Backend — modele BBox i Ras, pipeline, COCO</div></div></div>
<div class="row"><span class="num">4</span><div><div class="t">Mariia Volkova</div><div class="s">Data Engineer — zbieranie danych i weryfikacja</div></div></div>
</div>
</div>

---

<div class="eyebrow">Nawigacja</div>

# Spis treści

<div class="grid col2">
<div>
<div class="row"><span class="num">01</span><div><div class="t">Wprowadzenie do DogFACS</div><div class="s">Czym jest DogFACS i dlaczego jest ważny?</div></div></div>
<div class="row b"><span class="num b">03</span><div><div class="t">Architektura systemu</div><div class="s">Jak działa nasz pipeline?</div></div></div>
<div class="row o"><span class="num o">05</span><div><div class="t">Action Units</div><div class="s">Jednostki ruchu w pysku psa</div></div></div>
<div class="row"><span class="num">07</span><div><div class="t">Aplikacja webowa</div><div class="s">Interfejs anotacji i weryfikacji</div></div></div>
</div>
<div>
<div class="row"><span class="num">02</span><div><div class="t">Cel projektu</div><div class="s">Główne cele i zakres prac</div></div></div>
<div class="row b"><span class="num b">04</span><div><div class="t">Wykorzystane technologie</div><div class="s">Stack technologiczny</div></div></div>
<div class="row o"><span class="num o">06</span><div><div class="t">Klasyfikacja emocji</div><div class="s">9 klas emocji psów</div></div></div>
<div class="row"><span class="num">08</span><div><div class="t">Wyniki i osiągnięcia</div><div class="s">Co udało się osiągnąć?</div></div></div>
</div>
</div>

---

<!-- _class: chapter -->

<span class="pill">Rozdział 01</span>

# Wprowadzenie

<span class="sub">Czym jest DogFACS i dlaczego jest ważny<br>w badaniach nad emocjami zwierząt?</span>

---

<div class="eyebrow">Wprowadzenie</div>

# Czym jest DogFACS?

<div class="grid col2">
<div>
<div class="card"><h3>📖 Geneza systemu</h3>
<p><strong>DogFACS</strong> (Dog Facial Action Coding System) pozwala na <strong>obiektywne</strong> kodowanie wyrazów pyska psa poprzez identyfikację mikro-poruszeń mięśni — <strong>Action Units (AU)</strong>.</p></div>
<div class="card"><h3>🎯 Problem</h3>
<p>Nie istnieje duży, otwarty zbiór danych z emocjami psów opisanymi <strong>obiektywnie</strong> — bez subiektywnej interpretacji człowieka.</p></div>
</div>
<div>
<div class="card blue"><h3>🧩 Nasz wkład</h3>
<p>Pipeline AI, który <strong>automatycznie</strong> tworzy zbiór danych w formacie COCO z ramką, rasą, 46 punktami pyska, jednostkami AU i etykietą emocji.</p>
<p>Człowiek tylko <strong>weryfikuje</strong> gotowy szkic w aplikacji.</p></div>
</div>
</div>

---

<div class="eyebrow">Cel</div>

# Główne cele projektu

<div class="grid col3">
<div class="card green"><h3>🗄️ Zbiór danych</h3><p>Publicznie dostępny zbiór <strong>~25 000</strong> zaadnotowanych klatek wideo psów wysokiej jakości.</p></div>
<div class="card blue"><h3>🤖 Aplikacja AI</h3><p>System AI do <strong>automatycznej anotacji</strong> emocji z metodologią DogFACS.</p></div>
<div class="card orange"><h3>📦 Format COCO</h3><p>Pełna zgodność ze standardem <strong>COCO</strong> — łatwa integracja z narzędziami ML.</p></div>
</div>

<div class="grid col2" style="margin-top:18px;">
<div class="card"><h3>🏷️ Typy anotacji</h3>
<ul><li><strong>Bounding box</strong> — lokalizacja psa</li><li><strong>Rasa</strong> — 120 klas (Stanford Dogs)</li><li><strong>Punkty kluczowe</strong> — 46 punktów DogFLW</li><li><strong>Emocje + AU</strong> — 9 emocji, 21 AU</li></ul></div>
<div class="card"><h3>📈 Kryteria sukcesu</h3>
<ul><li>Detekcja psów — mAP &gt; 85%</li><li>Klasyfikacja ras — Top-5 &gt; 80%</li><li>Lokalizacja punktów — PCK@0.1 &gt; 75%</li><li>Klasyfikacja emocji &gt; 70%</li></ul></div>
</div>

---

<!-- _class: chapter -->

<span class="pill">Rozdział 02</span>

# Architektura i technologie

<span class="sub">Jak działa nasz system?<br>Od wideo do rozpoznanych emocji</span>

---

<div class="eyebrow">Architektura</div>

# Pipeline przetwarzania

<div class="grid col2">
<div>
<div class="row"><span class="num">1</span><div><div class="t">Wideo psa</div><div class="s">Wejście: plik MP4 (zalecane ~20 s)</div></div></div>
<div class="row b"><span class="num b">2</span><div><div class="t">Detekcja — YOLOv8m</div><div class="s">Wykrycie psa + bounding box</div></div></div>
<div class="row o"><span class="num o">3</span><div><div class="t">Punkty kluczowe — DogFLW</div><div class="s">46 punktów pyska (detekcja dwuprzebiegowa)</div></div></div>
<div class="row b"><span class="num b">4</span><div><div class="t">Pozycja głowy</div><div class="s">Yaw / Pitch / Roll + filtrowanie</div></div></div>
<div class="row o"><span class="num o">5</span><div><div class="t">Action Units (delta)</div><div class="s">21 AU względem klatki neutralnej</div></div></div>
<div class="row"><span class="num">6</span><div><div class="t">Klasyfikacja emocji</div><div class="s">Reguły DogFACS → 9 emocji</div></div></div>
</div>
<div>
<div class="card"><h3>✨ Kluczowe cechy</h3>
<ul>
<li>Automatyczna detekcja klatki <strong>neutralnej</strong></li>
<li>Wybór <strong>klatek szczytowych</strong> (najwyższe TFM)</li>
<li><strong>Delta AU</strong> — różnica względem neutralnej</li>
<li>Adaptacyjny dobór liczby klatek</li>
</ul></div>
<div class="card green"><h3>🎯 Wynik</h3><p>Gotowe anotacje <strong>per pies, per klatka</strong> — eksportowalne do COCO JSON.</p></div>
</div>
</div>

---

<div class="eyebrow">Technologie</div>

# Stack technologiczny

<div class="grid col2">
<div>
<div class="card"><h3>🖥️ Backend</h3>
<ul><li><strong>Python</strong> — język główny</li><li><strong>FastAPI</strong> — framework webowy</li><li><strong>OpenCV</strong> — przetwarzanie obrazu</li><li><strong>PyTorch</strong> — framework ML</li></ul></div>
<div class="card blue"><h3>🌐 Frontend</h3>
<ul><li><strong>React + TypeScript</strong></li><li><strong>Vite</strong> — build tool</li><li><strong>Tailwind CSS</strong> — stylowanie</li><li><strong>Zustand</strong> — stan aplikacji</li></ul></div>
</div>
<div>
<div class="card"><h3>🧠 Modele AI</h3>
<div class="acc" style="margin:8px 0;"><strong>YOLOv8m</strong> — detekcja psów + bbox</div>
<div class="acc b" style="margin:8px 0;"><strong>SimpleBaseline / ResNet34</strong> — punkty kluczowe DogFLW</div>
<div class="acc o" style="margin:8px 0;"><strong>EfficientNet-B4</strong> — klasyfikacja ras</div>
<div class="acc" style="margin:8px 0;"><strong>DogFACS Rules</strong> — klasyfikacja emocji</div></div>
<div class="card green"><h3>📦 Format danych</h3><p><strong>COCO</strong> z rozszerzeniami DogFACS dla Action Units i emocji.</p></div>
</div>
</div>

---

<!-- _class: chapter -->

<span class="pill">Rozdział 03</span>

# Metodologia

<span class="sub">Jak rozpoznajemy emocje psów?<br>Punkty kluczowe, Action Units i klasyfikacja</span>

---

<div class="eyebrow">Metodologia</div>

# Punkty kluczowe — DogFLW (46 pkt)

<div class="grid col2">
<div>
<div class="card"><h3>🐶 Schemat punktów</h3>
<p>Model wykrywa <strong>46 punktów</strong> pyska wg oficjalnego schematu <strong>DogFLW</strong> (arXiv:2405.11501): uszy, brwi, oczy, nos, wargi, podbródek, język.</p></div>
<div class="card blue"><h3>🔍 Detekcja dwuprzebiegowa</h3>
<p>Pas 1 — zgrubnie na całym psie → region pyska. Pas 2 — dokładnie na <strong>przyciętym pysku</strong>. Punkty trafiają na realne cechy.</p></div>
</div>
<div>
<div class="card orange"><h3>🛠️ Naprawa kolejności punktów</h3>
<p>Pierwotna kolejność punktów nie zgadzała się z modelem → punkty „oczu" lądowały na policzku, a AU liczono na złych punktach.</p>
<p><strong>Rozwiązanie:</strong> przywrócenie kanonicznej kolejności DogFLW (potwierdzone empirycznie na 66 mordach).</p></div>
<div class="note">Efekt: oczy = na oczach, nos = na nosie; AU liczone na właściwych punktach.</div>
</div>
</div>

---

<div class="eyebrow">Metodologia</div>

# Action Units — jednostki akcji

<p style="font-size:.85em;margin:.2em 0 .5em;">AU to <strong>mikro-poruszenia mięśni</strong> pyska. <strong>21 AU wyliczamy za pomocą naszego modelu</strong> — każdy AU = znormalizowana proporcja geometryczna między wykrytymi punktami kluczowymi, liczona jako <strong>delta względem klatki neutralnej</strong>.</p>

<div class="grid col2">
<div>
<div class="acc" style="margin:7px 0;"><strong>AU101</strong> — Inner Brow Raiser <span class="s" style="color:var(--muted);font-size:.8em;">· zainteresowanie</span></div>
<div class="acc" style="margin:7px 0;"><strong>AU145</strong> — Blink <span style="color:var(--muted);font-size:.8em;">· stres / zmęczenie</span></div>
<div class="acc b" style="margin:7px 0;"><strong>AU25</strong> — Lips Part <span style="color:var(--muted);font-size:.8em;">· część ekspresji radości</span></div>
<div class="acc b" style="margin:7px 0;"><strong>AU26</strong> — Jaw Drop <span style="color:var(--muted);font-size:.8em;">· otwarty pysk</span></div>
<div class="acc o" style="margin:7px 0;"><strong>AU109/110</strong> — Nose Wrinkler <span style="color:var(--muted);font-size:.8em;">· emocje negatywne</span></div>
</div>
<div>
<div class="acc o" style="margin:7px 0;"><strong>AU12</strong> — Lip Corner Puller <span style="color:var(--muted);font-size:.8em;">· „uśmiech"</span></div>
<div class="acc" style="margin:7px 0;"><strong>EAD102</strong> — Ears Forward <span style="color:var(--muted);font-size:.8em;">· skupienie</span></div>
<div class="acc" style="margin:7px 0;"><strong>EAD103</strong> — Ears Flattener <span style="color:var(--muted);font-size:.8em;">· strach / uległość</span></div>
<div class="acc b" style="margin:7px 0;"><strong>AD19</strong> — Tongue Show <span style="color:var(--muted);font-size:.8em;">· pokaz języka</span></div>
<div class="acc o" style="margin:7px 0;"><strong>AD137</strong> — Nose Lick <span style="color:var(--muted);font-size:.8em;">· wskaźnik stresu</span></div>
</div>
</div>

<div class="note" style="margin-top:12px;">ℹ️ Podstawa naukowa: Action Units wg badań Mota-Rojas et al. (2021).</div>

---

<div class="eyebrow">Metodologia</div>

# Klasyfikacja emocji (DogFACS)

<div class="grid col2">
<div>
<div class="card"><h3>⚙️ Reguły DogFACS</h3>
<p>Zestaw aktywnych AU mapowany jest regułami na <strong>9 emocji</strong>:</p>
<p class="sub">happy · sad · angry · fearful · relaxed · neutral · surprise · pain · submission</p></div>
<div class="card green"><h3>🛡️ Stabilność (nasze poprawki)</h3>
<p>Przycięcie „wybuchu" wartości przy małym mianowniku + <strong>bramkowanie pewnością</strong> — AU nie aktywuje się, gdy punkty są niepewne (mniej fałszywych AU).</p></div>
</div>
<div>
<div class="card blue"><h3>🎞️ Klatka neutralna i szczytowe</h3>
<p><strong>Neutralna</strong> — automatycznie wykrywana (najspokojniejsza, frontalna).</p>
<p><strong>Szczytowe (peak)</strong> — wybierane po TFM (sumaryczny ruch twarzy) z separacją czasową.</p></div>
<div class="note">Uczciwie: klasyfikacja jest <strong>regułowa</strong> (bez uczenia maszynowego); na spokojnym psie zwykle „neutral".</div>
</div>
</div>

---

<!-- _class: chapter -->

<span class="pill">Rozdział 04</span>

# Aplikacja i wyniki

<span class="sub">Interfejs anotacji oraz osiągnięte rezultaty</span>

---

<div class="eyebrow">Produkt</div>

# Aplikacja webowa

<div class="grid col2">
<div>
<div class="card"><h3>🧩 Funkcje</h3>
<ul>
<li>Wgranie wideo i przetworzenie pipeline'em</li>
<li>Przegląd klatek szczytowych z emocją i panelem <strong>AU</strong></li>
<li><strong>Edytor punktów kluczowych</strong> (przeciąganie, widoczność)</li>
<li>Ręczna korekta rasy i emocji</li>
<li><strong>Eksport do COCO JSON</strong></li>
</ul></div>
</div>
<div>
<div class="card blue"><h3>🛠️ Architektura</h3>
<p><strong>Backend:</strong> FastAPI + SessionStore (REST API, sesje).</p>
<p><strong>Frontend:</strong> React + TypeScript (Vite, Zustand, Tailwind).</p></div>
<div class="card green"><h3>👤 Rola człowieka</h3><p>Pipeline tworzy szkic — człowiek <strong>weryfikuje i poprawia</strong>. Interfejs w całości po polsku.</p></div>
</div>
</div>

---

<div class="eyebrow">Wyniki</div>

# Wyniki i osiągnięcia

<div class="grid col3">
<div class="card green"><h3>✅ Działający system</h3><p>Kompletny pipeline <strong>end-to-end</strong> + aplikacja webowa do anotacji i weryfikacji.</p></div>
<div class="card blue"><h3>🐕 Rasa: 6/6</h3><p>Po naprawie mapowania klas modelu — <strong>100%</strong> poprawnych ras na próbie testowej.</p></div>
<div class="card orange"><h3>🎯 Punkty + AU</h3><p>Punkty trafiają na pysk; AU i emocje liczone na <strong>poprawnych</strong> punktach.</p></div>
</div>

<div class="grid col2" style="margin-top:18px;">
<div class="card"><h3>🔧 Rozwiązane problemy</h3>
<ul>
<li>Punkty kluczowe — detekcja dwuprzebiegowa + kolejność DogFLW</li>
<li>Rasa — odtworzenie mapowania klas (0/6 → 6/6)</li>
<li>AU — odporność na szum (clamp + bramkowanie pewnością)</li>
</ul></div>
<div class="card"><h3>🎬 Demonstracja</h3>
<p>Wideo psa → klatka neutralna + klatki szczytowe z AU i emocją → eksport COCO.</p>
<p class="sub">Wskazówki: jeden pies, ujęcie frontalne, ~8–10 FPS.</p></div>
</div>

---

<!-- _class: chapter -->

# Dziękujemy za uwagę

<span class="sub">Zespół Dogs-ai · DogFACS Dataset Generator · Politechnika Gdańska WETI</span>
