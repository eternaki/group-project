/**
 * FastReview — экран разметки пары кадров (нейтральный + пиковый).
 *
 * Четыре решения, на которых он держится:
 *
 * 1. **Ничего не отмечено заранее.** Автоматические метки видны серой
 *    подписью, но не заполняют ответ. Иначе разметка выродилась бы в
 *    подтверждение ошибки одним нажатием: аудит показал, что правила
 *    бывают прямо противоположны тому, что видно на кадре.
 * 2. **Отмечается только то, что видно.** Enter означает «я просмотрел все
 *    AU, активны отмеченные», и лишь тогда остальные пишутся как `inactive`.
 * 3. **Пара рядом.** AU по определению есть разница относительно нейтрального
 *    кадра, поэтому оба кадра должны быть перед глазами одновременно.
 * 4. **Все 21 AU, а не подмножество.** Видимость решается на каждом кадре
 *    отдельно состоянием «не видно» — это точнее, чем исключать AU глобально.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AU_NAMES_RU,
  DOG_BREEDS,
  EMOTION_CLASSES,
  EMOTION_EMOJI,
  EMOTION_NAMES_RU,
  VERIFIABLE_AU,
  getKeypointColor,
  type AUVerdict,
  type FrameAnnotation,
  type SessionData,
} from '../types';
import {
  exportSessionCOCO,
  getSession,
  importCoco,
  listDatasets,
  patchReview,
  type AvailableDataset,
} from '../utils/api';
import useStore from '../store/useStore';
import FullEditorModal from './FullEditorModal';

/** Клавиши 1-8 у первых AU из VERIFIABLE_AU; остальные размечаются кликом */
const AU_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8'];

/** Порядок анатомических групп на экране */
const AU_GROUP_ORDER = ['Пасть', 'Глаза', 'Нос', 'Уши'];

interface ReviewPair {
  trackId: number;
  neutral: FrameAnnotation;
  peak: FrameAnnotation;
}

/** Grupuje klatki sesji w pary (neutralna, szczytowa) po track_id. */
function buildPairs(session: SessionData): ReviewPair[] {
  const byTrack = new Map<number, { neutral?: FrameAnnotation; peak?: FrameAnnotation }>();
  for (const frame of session.frames) {
    const trackId = frame.track_id ?? -1;
    const entry = byTrack.get(trackId) ?? {};
    if (frame.frame_role === 'neutral') entry.neutral = frame;
    else entry.peak = frame;
    byTrack.set(trackId, entry);
  }
  return [...byTrack.entries()]
    .filter(([, entry]) => entry.neutral && entry.peak)
    .map(([trackId, entry]) => ({
      trackId,
      neutral: entry.neutral as FrameAnnotation,
      peak: entry.peak as FrameAnnotation,
    }))
    .sort((a, b) => (a.peak.review_order ?? 0) - (b.peak.review_order ?? 0));
}

/** Czy reguły uznały to AU za aktywne (tylko podpowiedź, nie odpowiedź). */
function ruleSaysActive(peak: FrameAnnotation, code: string): boolean {
  return Boolean(peak.aus?.[code]?.is_active);
}

/** Zapas wokół boksu psa przy kadrowaniu — ułamek jego szerokości i wysokości */
const CROP_PADDING = 0.18;

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Wylicza wycinek kadru pokazywany anotatorowi.
 *
 * Pełna klatka jest bezużyteczna do oceny AU: bramka dopuszcza mordy od 40 px
 * szerokości, a klatka 1920 px pokazana w kolumnie szerokiej na 380 px kurczy
 * je do ośmiu. Do tego na nagraniu bywa kilka psów i bez wycinka nie wiadomo,
 * którego dotyczy anotacja.
 */
function cropRect(frame: FrameAnnotation, natural: Rect | null): Rect | null {
  if (!natural || !frame.bbox) return null;
  const [x, y, width, height] = frame.bbox;
  const padX = width * CROP_PADDING;
  const padY = height * CROP_PADDING;
  const left = Math.max(0, x - padX);
  const top = Math.max(0, y - padY);
  return {
    x: left,
    y: top,
    width: Math.min(natural.width - left, width + 2 * padX),
    height: Math.min(natural.height - top, height + 2 * padY),
  };
}

/** Promień punktu keypoint jako ułamek szerokości wycinka — skaluje się z kadrem */
const KEYPOINT_RADIUS_RATIO = 0.006;

/** Poniżej tej pewności punkt rysujemy jako niepewny */
const WEAK_KEYPOINT_CONF = 0.5;

interface KeypointOverlayProps {
  keypoints: number[];
  region: Rect;
}

/**
 * Rysuje 46 punktów na kadrze.
 *
 * viewBox ustawiony na wycinek sprawia, że współrzędne z COCO (w pikselach
 * oryginału) trafiają na swoje miejsce bez żadnego przeliczania — a przy
 * zmianie rozmiaru okna skalują się razem z obrazem.
 */
function KeypointOverlay({ keypoints, region }: KeypointOverlayProps) {
  const radius = region.width * KEYPOINT_RADIUS_RATIO;
  const points = [];
  for (let i = 0; i < keypoints.length; i += 3) {
    const [x, y, confidence] = [keypoints[i], keypoints[i + 1], keypoints[i + 2]];
    points.push(
      <circle
        key={i}
        cx={x}
        cy={y}
        r={radius}
        fill={confidence < WEAK_KEYPOINT_CONF ? '#ef4444' : getKeypointColor(i / 3)}
        stroke="rgba(0,0,0,0.6)"
        strokeWidth={radius * 0.3}
      />
    );
  }
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox={`${region.x} ${region.y} ${region.width} ${region.height}`}
      preserveAspectRatio="none"
    >
      {points}
    </svg>
  );
}

interface FrameViewProps {
  frame: FrameAnnotation;
  label: string;
  accent: string;
  showFullFrame: boolean;
  showKeypoints: boolean;
}

function FrameView({ frame, label, accent, showFullFrame, showKeypoints }: FrameViewProps) {
  const [natural, setNatural] = useState<Rect | null>(null);
  const quality = frame.quality ?? {};

  // Pełna klatka to po prostu wycinek obejmujący cały obraz. Dzięki temu punkty
  // kluczowe rysują się tym samym mechanizmem w obu trybach — przy `object-contain`
  // obraz bywa opasany pustym tłem i nakładka rozjeżdżałaby się z nim.
  const region = showFullFrame ? natural : cropRect(frame, natural);

  // Klatka i pies się zmieniły — poprzedni rozmiar naturalny już nie obowiązuje
  useEffect(() => setNatural(null), [frame.image_url]);

  const onLoad = (event: React.SyntheticEvent<HTMLImageElement>) => {
    const image = event.currentTarget;
    setNatural({ x: 0, y: 0, width: image.naturalWidth, height: image.naturalHeight });
  };

  return (
    <div className="flex-1 min-w-0">
      <div className="flex items-baseline justify-between mb-1">
        <span className={`text-xs font-semibold uppercase tracking-wide ${accent}`}>{label}</span>
        {quality.asymmetry !== undefined && (
          <span className="text-[11px] text-gray-400 font-mono">
            asym {quality.asymmetry.toFixed(2)} · morda {Math.round(quality.face_width_px ?? 0)}px
          </span>
        )}
      </div>
      <div
        className="relative w-full overflow-hidden rounded-lg bg-gray-900"
        style={region ? { aspectRatio: `${region.width} / ${region.height}` } : undefined}
      >
        <img
          src={frame.image_url}
          alt={label}
          onLoad={onLoad}
          className={region && natural ? 'absolute' : 'w-full'}
          style={
            region && natural
              ? {
                  width: `${(100 * natural.width) / region.width}%`,
                  left: `${(-100 * region.x) / region.width}%`,
                  top: `${(-100 * region.y) / region.height}%`,
                  maxWidth: 'none',
                }
              : undefined
          }
        />
        {showKeypoints && frame.keypoints && region && (
          <KeypointOverlay keypoints={frame.keypoints} region={region} />
        )}
      </div>
    </div>
  );
}

interface AUButtonProps {
  index: number;
  code: string;
  hint: string;
  verdict: AUVerdict | undefined;
  ruleActive: boolean;
  onToggle: () => void;
  onNotObservable: () => void;
}

function AUButton({
  index,
  code,
  hint,
  verdict,
  ruleActive,
  onToggle,
  onNotObservable,
}: AUButtonProps) {
  const style =
    verdict === 'active'
      ? 'bg-amber-500 border-amber-600 text-white'
      : verdict === 'not_observable'
        ? 'bg-gray-700 border-gray-800 text-gray-300'
        : 'bg-white border-gray-200 text-gray-700 hover:border-amber-300';

  return (
    <button
      onClick={onToggle}
      onContextMenu={(event) => {
        event.preventDefault();
        onNotObservable();
      }}
      className={`text-left p-2.5 rounded-lg border-2 transition-colors ${style}`}
      title="Клик = активен · правый клик = не видно"
    >
      <div className="flex items-center gap-2">
        {AU_KEYS[index] && (
          <kbd className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/10 shrink-0">
            {AU_KEYS[index]}
          </kbd>
        )}
        <span className="font-mono text-xs font-bold">{code}</span>
        {ruleActive && verdict === undefined && (
          <span
            className="text-[10px] text-gray-400 ml-auto shrink-0"
            title="Автомат счёл этот AU активным — это только подсказка"
          >
            правило: да
          </span>
        )}
      </div>
      <div className="text-[11px] mt-1 opacity-80 leading-tight">{AU_NAMES_RU[code] ?? code}</div>
      <div className="text-[10px] mt-0.5 opacity-60 leading-tight">{hint}</div>
      {verdict === 'not_observable' && (
        <div className="text-[10px] mt-1 font-semibold">не видно</div>
      )}
    </button>
  );
}

/** Ile propozycji rasy pokazujemy obok pola tekstowego */
const BREED_SUGGESTIONS = 6;

// Rasy klasyfikator myli notorycznie (mediana pewności 0.33 na tym materiale),
// więc anotator musi mieć jak powiedzieć "nie wiem" zamiast zgadywać.
const UNKNOWN_BREED = 'unknown';

interface ChoiceRowProps {
  label: string;
  hint?: string;
  children: React.ReactNode;
}

function ChoiceRow({ label, hint, children }: ChoiceRowProps) {
  return (
    <div className="flex items-start gap-3 py-2 border-t border-gray-100">
      <div className="w-24 shrink-0 pt-1">
        <div className="text-xs font-semibold text-gray-600">{label}</div>
        {hint && <div className="text-[10px] text-gray-400 leading-tight">{hint}</div>}
      </div>
      <div className="flex-1 min-w-0 flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

interface PillProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
  tone?: 'amber' | 'green' | 'gray';
}

function Pill({ active, onClick, children, tone = 'amber' }: PillProps) {
  const activeStyle =
    tone === 'green'
      ? 'bg-green-600 border-green-700 text-white'
      : tone === 'gray'
        ? 'bg-gray-700 border-gray-800 text-white'
        : 'bg-amber-500 border-amber-600 text-white';
  return (
    <button
      onClick={onClick}
      className={`px-2.5 py-1 rounded-md border text-xs font-medium transition-colors ${
        active ? activeStyle : 'bg-white border-gray-200 text-gray-700 hover:border-amber-300'
      }`}
    >
      {children}
    </button>
  );
}

export default function FastReview() {
  const [datasets, setDatasets] = useState<AvailableDataset[] | null>(null);
  const [session, setSession] = useState<SessionData | null>(null);
  const [pairs, setPairs] = useState<ReviewPair[]>([]);
  const [position, setPosition] = useState(0);
  const [verdicts, setVerdicts] = useState<Record<string, AUVerdict>>({});
  const [verifiedCount, setVerifiedCount] = useState(0);
  const [rejectedCount, setRejectedCount] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Anotator czasem potrzebuje kontekstu sceny — np. gdy trzeba rozstrzygnąć,
  // czy pies reaguje na człowieka poza wycinkiem.
  const [showFullFrame, setShowFullFrame] = useState(false);
  // Punkty domyślnie WIDOCZNE: ich ocena jest częścią tego samego przejścia,
  // a niewidoczne trzeba by włączać przy każdej parze osobno.
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [keypointsOk, setKeypointsOk] = useState<boolean | null>(null);
  const [breed, setBreed] = useState<string | null>(null);
  const [emotion, setEmotion] = useState<string | null>(null);
  const [breedQuery, setBreedQuery] = useState('');
  // Pełny edytor (przeciąganie 46 punktów, zakładki AU/emocja/rasa) działa na
  // sesji trzymanej w store — dlatego ładujemy ją tam obok szybkiego trybu.
  const [editing, setEditing] = useState(false);
  const loadStoreSession = useStore((state) => state.loadSession);
  const storeSession = useStore((state) => state.sessionData);

  const current = pairs[position];

  const startSession = useCallback(async (path: string) => {
    setBusy(true);
    setError(null);
    try {
      const imported = await importCoco(path);
      const loaded = await getSession(imported.session_id);
      await loadStoreSession(imported.session_id);
      const built = buildPairs(loaded);
      setSession(loaded);
      setPairs(built);
      setVerdicts({});
      setVerifiedCount(built.filter((p) => p.peak.annotation_status === 'verified').length);
      setRejectedCount(built.filter((p) => p.peak.usable === false).length);
      // Wracamy tam, gdzie praca się urwała — a nie na początek kolejki.
      const firstOpen = built.findIndex((p) => p.peak.annotation_status !== 'verified');
      setPosition(firstOpen === -1 ? built.length : firstOpen);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Не удалось загрузить набор');
    } finally {
      setBusy(false);
    }
  }, [loadStoreSession]);

  // Zbiory znajdują się same. Gdy jest dokładnie jeden, od razu go otwieramy —
  // anotator ma zobaczyć pierwszą parę, a nie formularz.
  useEffect(() => {
    let cancelled = false;
    listDatasets()
      .then(({ datasets: found }) => {
        if (cancelled) return;
        setDatasets(found);
        if (found.length === 1) void startSession(found[0].path);
      })
      .catch((caught) =>
        setError(caught instanceof Error ? caught.message : 'Не удалось прочитать каталог данных')
      );
    return () => {
      cancelled = true;
    };
  }, [startSession]);

  const goTo = useCallback(
    (index: number) => {
      if (index < 0 || index >= pairs.length) return;
      const peak = pairs[index]?.peak;
      setPosition(index);
      setVerdicts(peak?.au_verdicts ?? {});
      setKeypointsOk(peak?.keypoints_ok ?? null);
      // Rasa i emocja startują od tego, co policzył automat — tu poprawiamy
      // gotową wartość, a nie etykietujemy od zera jak przy AU.
      setBreed(peak?.breed ?? null);
      setEmotion(peak?.emotion ?? null);
      setBreedQuery('');
    },
    [pairs]
  );

  const toggle = useCallback((code: string, target: AUVerdict) => {
    setVerdicts((previous) => {
      const next = { ...previous };
      if (next[code] === target) delete next[code];
      else next[code] = target;
      return next;
    });
  }, []);

  /**
   * Zatwierdza parę: wszystko niezaznaczone staje się `inactive`.
   *
   * To jest moment, w którym człowiek oświadcza „obejrzałem wszystkie osiem".
   * Dlatego dopiero tutaj powstają werdykty negatywne — wcześniej brak
   * zaznaczenia znaczy „jeszcze nie patrzyłem", a nie „nieaktywne".
   */
  const commit = useCallback(
    async (usable = true) => {
      if (!session || !current || busy) return;
      setBusy(true);
      setError(null);
      try {
        const complete: Record<string, AUVerdict> = {};
        if (usable) {
          for (const { code } of VERIFIABLE_AU) complete[code] = verdicts[code] ?? 'inactive';
        } else {
          // Kadr odrzucony: żadne AU nie jest „nieaktywne", wszystkie są
          // NIEOBSERWOWALNE. Zapisanie zer nauczyłoby sieć, że pies z głową
          // w dół ma mimikę spoczynkową.
          for (const { code } of VERIFIABLE_AU) complete[code] = 'not_observable';
        }
        await patchReview(session.session_id, current.peak.frame_idx, current.trackId, {
          verdicts: complete,
          usable,
          keypoints_ok: usable ? keypointsOk : false,
          breed,
          emotion,
        });
        current.peak.au_verdicts = complete;
        current.peak.keypoints_ok = usable ? keypointsOk : false;
        current.peak.breed = breed;
        current.peak.emotion = emotion;
        current.peak.usable = usable;
        if (current.peak.annotation_status !== 'verified') {
          current.peak.annotation_status = 'verified';
          setVerifiedCount((count) => count + 1);
        }
        if (!usable) setRejectedCount((count) => count + 1);
        goTo(position + 1);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Не удалось сохранить');
      } finally {
        setBusy(false);
      }
    },
    [session, current, busy, verdicts, position, goTo, keypointsOk, breed, emotion]
  );

  useEffect(() => {
    if (!current) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;
      const digit = AU_KEYS.indexOf(event.key);
      if (digit >= 0) {
        event.preventDefault();
        toggle(VERIFIABLE_AU[digit].code, event.shiftKey ? 'not_observable' : 'active');
        return;
      }
      if (event.key === 'e' || event.key === 'E') {
        event.preventDefault();
        setEditing(true);
      } else if (event.key === 'k' || event.key === 'K') {
        event.preventDefault();
        setShowKeypoints((shown) => !shown);
      } else if (event.key === 'f' || event.key === 'F') {
        event.preventDefault();
        setShowFullFrame((shown) => !shown);
      } else if (event.key === 'x' || event.key === 'X') {
        event.preventDefault();
        void commit(false);
      } else if (event.key === 'Enter') {
        event.preventDefault();
        void commit();
      } else if (event.key === 'ArrowLeft') {
        goTo(position - 1);
      } else if (event.key === 'ArrowRight' || event.key === ' ') {
        event.preventDefault();
        goTo(position + 1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [current, toggle, commit, goTo, position]);

  const activeCount = useMemo(
    () => Object.values(verdicts).filter((value) => value === 'active').length,
    [verdicts]
  );

  // Pełny edytor pracuje na klatce ze store'u — tej samej, którą sam zapisuje.
  // Podanie mu kopii z szybkiego trybu znaczyłoby, że zapis idzie w próżnię.
  const editorFrame = useMemo(() => {
    if (!current || !storeSession) return null;
    return (
      storeSession.frames.find(
        (frame) =>
          frame.frame_idx === current.peak.frame_idx && frame.track_id === current.trackId
      ) ?? null
    );
  }, [current, storeSession]);

  if (!session) {
    return (
      <div className="max-w-xl mx-auto mt-16 p-6 bg-white rounded-xl border border-gray-200">
        <h2 className="text-lg font-bold text-gray-800">Разметка AU</h2>
        {datasets === null && !error && (
          <p className="text-sm text-gray-500 mt-2">Ищу наборы для разметки…</p>
        )}
        {busy && <p className="text-sm text-gray-500 mt-2">Загружаю набор…</p>}
        {datasets !== null && datasets.length === 0 && (
          <div className="mt-3 text-sm text-gray-600 space-y-2">
            <p>Не нашёл ни одного готового набора.</p>
            <p className="text-gray-500">
              Набор готовится командой:
              <code className="block mt-1 p-2 bg-gray-50 rounded font-mono text-xs">
                python -m scripts.annotation.curate_for_review
              </code>
            </p>
          </div>
        )}
        {datasets !== null && datasets.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-sm text-gray-500">Выбери набор — работа сохраняется и продолжается сама.</p>
            {datasets.map((dataset) => {
              const done = dataset.pairs > 0 ? (100 * dataset.verified) / dataset.pairs : 0;
              return (
                <button
                  key={dataset.path}
                  onClick={() => void startSession(dataset.path)}
                  disabled={busy}
                  className="w-full text-left p-3 rounded-lg border-2 border-gray-200 hover:border-amber-400 transition-colors disabled:opacity-50"
                >
                  <div className="flex items-baseline justify-between">
                    <span className="font-semibold text-gray-800">{dataset.name}</span>
                    <span className="text-xs text-gray-500 font-mono">
                      {dataset.verified} / {dataset.pairs} пар
                    </span>
                  </div>
                  <div className="h-1.5 bg-gray-200 rounded-full mt-2 overflow-hidden">
                    <div className="h-full bg-green-500" style={{ width: `${done}%` }} />
                  </div>
                </button>
              );
            })}
          </div>
        )}
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      </div>
    );
  }

  if (!current) {
    return (
      <div className="max-w-xl mx-auto mt-16 p-6 bg-white rounded-xl border border-gray-200 text-center">
        <h2 className="text-lg font-bold text-gray-800">Очередь закончилась</h2>
        <p className="text-sm text-gray-500 mt-1">
          Размечено {verifiedCount} из {pairs.length} пар.
        </p>
        <button
          onClick={() => exportSessionCOCO(session.session_id, session.video_filename)}
          className="mt-4 px-4 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700"
        >
          Экспорт COCO
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm text-gray-600">
          Пара <strong>{position + 1}</strong> / {pairs.length} · размечено{' '}
          <strong className="text-green-700">{verifiedCount}</strong> · отброшено{' '}
          <strong className="text-gray-600">{rejectedCount}</strong> · активных AU{' '}
          <strong className="text-amber-700">{activeCount}</strong>
        </div>
        <button
          onClick={() => exportSessionCOCO(session.session_id, session.video_filename)}
          className="text-xs px-3 py-1.5 bg-green-600 text-white rounded hover:bg-green-700"
        >
          Экспорт COCO
        </button>
      </div>

      <div className="h-1.5 bg-gray-200 rounded-full mb-4 overflow-hidden">
        <div
          className="h-full bg-green-500 transition-all"
          style={{ width: `${(verifiedCount / Math.max(pairs.length, 1)) * 100}%` }}
        />
      </div>

      <div className="flex gap-4 mb-4">
        <FrameView
          frame={current.neutral}
          label="Нейтральный (база AU)"
          accent="text-blue-600"
          showFullFrame={showFullFrame}
          showKeypoints={showKeypoints}
        />
        <FrameView
          frame={current.peak}
          label="Пиковый (оцениваем)"
          accent="text-amber-600"
          showFullFrame={showFullFrame}
          showKeypoints={showKeypoints}
        />
      </div>

      {/* Все 21 AU, сгруппированные по анатомии. Клавиши 1-8 у самых частых,
          остальные кликом — цифр всего девять, а AU двадцать одна. */}
      <div className="space-y-2 mb-3">
        {AU_GROUP_ORDER.map((group) => {
          const items = VERIFIABLE_AU.map((au, index) => ({ ...au, index })).filter(
            (au) => au.group === group
          );
          if (!items.length) return null;
          return (
            <div key={group}>
              <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">
                {group}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {items.map(({ code, hint, index }) => (
                  <AUButton
                    key={code}
                    index={index}
                    code={code}
                    hint={hint}
                    verdict={verdicts[code]}
                    ruleActive={ruleSaysActive(current.peak, code)}
                    onToggle={() => toggle(code, 'active')}
                    onNotObservable={() => toggle(code, 'not_observable')}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div className="bg-white rounded-lg border border-gray-200 px-3 mb-3">
        <ChoiceRow label="Точки" hint="лежат ли на морде">
          <Pill active={keypointsOk === true} onClick={() => setKeypointsOk(true)} tone="green">
            dobre
          </Pill>
          <Pill active={keypointsOk === false} onClick={() => setKeypointsOk(false)} tone="gray">
            złe
          </Pill>
          <button
            onClick={() => setShowKeypoints((shown) => !shown)}
            className="px-2.5 py-1 rounded-md border border-dashed border-gray-300 text-xs text-gray-500 hover:border-gray-400"
          >
            {showKeypoints ? 'скрыть точки (K)' : 'показать точки (K)'}
          </button>
        </ChoiceRow>

        <ChoiceRow label="Эмоция" hint="автомат, поправь если не так">
          {EMOTION_CLASSES.map((value) => (
            <Pill key={value} active={emotion === value} onClick={() => setEmotion(value)}>
              {EMOTION_EMOJI[value]} {EMOTION_NAMES_RU[value] ?? value}
            </Pill>
          ))}
        </ChoiceRow>

        <ChoiceRow label="Rasa" hint={`автомат: ${(current.peak.breed_confidence * 100).toFixed(0)}% уверенности`}>
          <input
            value={breedQuery}
            onChange={(event) => setBreedQuery(event.target.value)}
            placeholder={breed ?? 'начни вводить для поиска'}
            className="px-2 py-1 border border-gray-200 rounded-md text-xs w-44"
          />
          {breedQuery
            ? DOG_BREEDS.filter((name) =>
                name.toLowerCase().includes(breedQuery.toLowerCase())
              )
                .slice(0, BREED_SUGGESTIONS)
                .map((name) => (
                  <Pill
                    key={name}
                    active={breed === name}
                    onClick={() => {
                      setBreed(name);
                      setBreedQuery('');
                    }}
                  >
                    {name}
                  </Pill>
                ))
            : (
                <>
                  <Pill active={breed === current.peak.breed} onClick={() => setBreed(current.peak.breed ?? null)}>
                    {current.peak.breed ?? 'нет'}
                  </Pill>
                  <Pill active={breed === 'Mixed Breed'} onClick={() => setBreed('Mixed Breed')}>
                    Mixed Breed
                  </Pill>
                  <Pill active={breed === UNKNOWN_BREED} onClick={() => setBreed(UNKNOWN_BREED)} tone="gray">
                    nie wiem
                  </Pill>
                </>
              )}
        </ChoiceRow>
      </div>

      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">1–8</kbd> активен ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">Shift+1–8</kbd> не видно ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">Enter</kbd> сохранить и дальше ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">X</kbd> кадр не годится ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">K</kbd> точки ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">E</kbd> полное редактирование ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">F</kbd>{' '}
          {showFullFrame ? 'вернуть кадр собаки' : 'весь кадр'} ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">←→</kbd> навигация
        </span>
        <div className="flex gap-2 shrink-0">
          <button
            onClick={() => void commit(false)}
            disabled={busy}
            className="px-3 py-2 bg-gray-200 text-gray-700 rounded-lg font-semibold hover:bg-gray-300 disabled:opacity-50"
            title="Голова опущена, собака отвернулась, морды не видно"
          >
            Не годится ✕
          </button>
          <button
            onClick={() => void commit()}
            disabled={busy}
            className="px-4 py-2 bg-amber-600 text-white rounded-lg font-semibold hover:bg-amber-700 disabled:opacity-50"
          >
            {busy ? 'Сохраняю…' : 'Сохранить и дальше ⏎'}
          </button>
        </div>
      </div>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

      {editing && editorFrame && (
        <FullEditorModal
          frame={editorFrame}
          onClose={async () => {
            setEditing(false);
            // Edytor zapisuje przez store, więc świeże keypoints i przeliczone
            // AU trzeba wciągnąć z powrotem do szybkiego trybu — inaczej
            // anotator widziałby wersję sprzed poprawki.
            if (session) {
              const refreshed = await getSession(session.session_id);
              setPairs(buildPairs(refreshed));
            }
          }}
        />
      )}
    </div>
  );
}
