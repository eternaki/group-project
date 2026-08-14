/**
 * FastReview — szybka weryfikacja AU na parach (klatka neutralna, szczytowa).
 *
 * Ekran zaprojektowany pod jedno: liczbę par zweryfikowanych na godzinę.
 * Trzy decyzje, które o tym przesądzają:
 *
 * 1. **Nic nie jest zaznaczone z góry.** Autometki reguł są pokazane jako
 *    szara podpowiedź, ale NIE wypełniają odpowiedzi. Gdyby wypełniały,
 *    weryfikacja zamieniłaby się w zatwierdzanie błędu jednym kliknięciem —
 *    audyt pokazał, że reguły bywają odwrotnością tego, co widać.
 * 2. **Zaznacza się tylko to, co widać.** Enter oznacza „obejrzałem wszystkie
 *    osiem, aktywne są te zaznaczone" i dopiero wtedy reszta zapisuje się jako
 *    `inactive`. Przy medianie 2-3 aktywnych AU to około czterech klawiszy
 *    na parę.
 * 3. **Para obok siebie.** AU jest różnicą względem klatki neutralnej, więc
 *    człowiek musi widzieć obie naraz, inaczej ocenia z pamięci.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AU_NAMES,
  VERIFIABLE_AU,
  type AUVerdict,
  type FrameAnnotation,
  type SessionData,
} from '../types';
import { exportSessionCOCO, getSession, importCoco, patchAUVerdicts } from '../utils/api';

/**
 * Domyślna ścieżka zbioru po kuracji — WZGLĘDEM katalogu danych backendu
 * (`DOGFACS_IMPORT_ROOT`, domyślnie `data/`), a nie względem korzenia repo.
 * Backend nie przyjmie ścieżki wychodzącej poza ten katalog.
 */
const DEFAULT_DATASET_PATH = 'dataset_v2/curated.json';

/** Klawisze przypisane do AU: pozycja w VERIFIABLE_AU → cyfra */
const AU_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8'];

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

interface FrameViewProps {
  frame: FrameAnnotation;
  label: string;
  accent: string;
  showFullFrame: boolean;
}

function FrameView({ frame, label, accent, showFullFrame }: FrameViewProps) {
  const [natural, setNatural] = useState<Rect | null>(null);
  const quality = frame.quality ?? {};
  const crop = showFullFrame ? null : cropRect(frame, natural);

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
        style={crop ? { aspectRatio: `${crop.width} / ${crop.height}` } : undefined}
      >
        <img
          src={frame.image_url}
          alt={label}
          onLoad={onLoad}
          className={crop ? 'absolute' : 'w-full max-h-[46vh] object-contain'}
          style={
            crop && natural
              ? {
                  width: `${(100 * natural.width) / crop.width}%`,
                  left: `${(-100 * crop.x) / crop.width}%`,
                  top: `${(-100 * crop.y) / crop.height}%`,
                  maxWidth: 'none',
                }
              : undefined
          }
        />
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
      title="Klik = aktywne · prawy klik = niewidoczne"
    >
      <div className="flex items-center gap-2">
        <kbd className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/10 shrink-0">
          {AU_KEYS[index]}
        </kbd>
        <span className="font-mono text-xs font-bold">{code}</span>
        {ruleActive && verdict === undefined && (
          <span
            className="text-[10px] text-gray-400 ml-auto shrink-0"
            title="Reguła uznała to AU za aktywne — to tylko podpowiedź"
          >
            reguła: tak
          </span>
        )}
      </div>
      <div className="text-[11px] mt-1 opacity-80 leading-tight">{AU_NAMES[code] ?? code}</div>
      <div className="text-[10px] mt-0.5 opacity-60 leading-tight">{hint}</div>
      {verdict === 'not_observable' && (
        <div className="text-[10px] mt-1 font-semibold">niewidoczne</div>
      )}
    </button>
  );
}

export default function FastReview() {
  const [datasetPath, setDatasetPath] = useState(DEFAULT_DATASET_PATH);
  const [session, setSession] = useState<SessionData | null>(null);
  const [pairs, setPairs] = useState<ReviewPair[]>([]);
  const [position, setPosition] = useState(0);
  const [verdicts, setVerdicts] = useState<Record<string, AUVerdict>>({});
  const [verifiedCount, setVerifiedCount] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Anotator czasem potrzebuje kontekstu sceny — np. gdy trzeba rozstrzygnąć,
  // czy pies reaguje na człowieka poza wycinkiem.
  const [showFullFrame, setShowFullFrame] = useState(false);

  const current = pairs[position];

  const startSession = async () => {
    setBusy(true);
    setError(null);
    try {
      const imported = await importCoco(datasetPath);
      const loaded = await getSession(imported.session_id);
      const built = buildPairs(loaded);
      setSession(loaded);
      setPairs(built);
      setPosition(0);
      setVerdicts({});
      setVerifiedCount(built.filter((p) => p.peak.annotation_status === 'verified').length);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Nie udało się wczytać zbioru');
    } finally {
      setBusy(false);
    }
  };

  const goTo = useCallback(
    (index: number) => {
      if (index < 0 || index >= pairs.length) return;
      setPosition(index);
      setVerdicts(pairs[index].peak.au_verdicts ?? {});
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
  const commit = useCallback(async () => {
    if (!session || !current || busy) return;
    setBusy(true);
    setError(null);
    try {
      const complete: Record<string, AUVerdict> = {};
      for (const { code } of VERIFIABLE_AU) complete[code] = verdicts[code] ?? 'inactive';
      await patchAUVerdicts(session.session_id, current.peak.frame_idx, current.trackId, complete, true);
      current.peak.au_verdicts = complete;
      if (current.peak.annotation_status !== 'verified') {
        current.peak.annotation_status = 'verified';
        setVerifiedCount((count) => count + 1);
      }
      goTo(position + 1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Nie udało się zapisać');
    } finally {
      setBusy(false);
    }
  }, [session, current, busy, verdicts, position, goTo]);

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
      if (event.key === 'f' || event.key === 'F') {
        event.preventDefault();
        setShowFullFrame((shown) => !shown);
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

  if (!session) {
    return (
      <div className="max-w-xl mx-auto mt-16 p-6 bg-white rounded-xl border border-gray-200">
        <h2 className="text-lg font-bold text-gray-800">Weryfikacja AU — start</h2>
        <p className="text-sm text-gray-500 mt-1">
          Ścieżka do zbioru po kuracji (<code>curate_for_review.py</code>), względem katalogu
          danych backendu.
        </p>
        <input
          value={datasetPath}
          onChange={(event) => setDatasetPath(event.target.value)}
          className="w-full mt-3 px-3 py-2 border border-gray-300 rounded-lg font-mono text-sm"
        />
        <button
          onClick={startSession}
          disabled={busy}
          className="mt-3 w-full py-2.5 bg-amber-600 text-white rounded-lg font-semibold hover:bg-amber-700 disabled:opacity-50"
        >
          {busy ? 'Wczytywanie…' : 'Rozpocznij weryfikację'}
        </button>
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      </div>
    );
  }

  if (!current) {
    return (
      <div className="max-w-xl mx-auto mt-16 p-6 bg-white rounded-xl border border-gray-200 text-center">
        <h2 className="text-lg font-bold text-gray-800">Koniec kolejki</h2>
        <p className="text-sm text-gray-500 mt-1">
          Zweryfikowano {verifiedCount} z {pairs.length} par.
        </p>
        <button
          onClick={() => exportSessionCOCO(session.session_id, session.video_filename)}
          className="mt-4 px-4 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700"
        >
          Eksportuj COCO
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm text-gray-600">
          Para <strong>{position + 1}</strong> / {pairs.length} · zweryfikowane{' '}
          <strong className="text-green-700">{verifiedCount}</strong> · aktywne AU{' '}
          <strong className="text-amber-700">{activeCount}</strong>
        </div>
        <button
          onClick={() => exportSessionCOCO(session.session_id, session.video_filename)}
          className="text-xs px-3 py-1.5 bg-green-600 text-white rounded hover:bg-green-700"
        >
          Eksportuj COCO
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
          label="Neutralna (baza AU)"
          accent="text-blue-600"
          showFullFrame={showFullFrame}
        />
        <FrameView
          frame={current.peak}
          label="Szczytowa (oceniana)"
          accent="text-amber-600"
          showFullFrame={showFullFrame}
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
        {VERIFIABLE_AU.map(({ code, hint }, index) => (
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

      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">1–8</kbd> aktywne ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">Shift+1–8</kbd> niewidoczne ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">Enter</kbd> zapisz i dalej ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">F</kbd>{' '}
          {showFullFrame ? 'wróć do kadru psa' : 'cała klatka'} ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">←→</kbd> nawigacja
        </span>
        <button
          onClick={() => void commit()}
          disabled={busy}
          className="px-4 py-2 bg-amber-600 text-white rounded-lg font-semibold hover:bg-amber-700 disabled:opacity-50"
        >
          {busy ? 'Zapisywanie…' : 'Zapisz i dalej ⏎'}
        </button>
      </div>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}
