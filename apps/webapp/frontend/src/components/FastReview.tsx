/**
 * FastReview — экран разметки пары кадров (нейтральный + пиковый).
 *
 * Четыре решения, на которых он держится:
 *
 * 1. **Ничего не отмечено заранее.** Сводка сверху показывает, что насчитал
 *    автомат, но ответы не заполняет. Измерено, почему: правила ставят по 12
 *    активных AU из 21 на спокойной собаке, включая EAD101 («уши вперёд») и
 *    EAD103 («уши прижаты») ОДНОВРЕМЕННО — физически несовместимые. С
 *    предзаполнением разметка выродилась бы в подтверждение этих ошибок одним
 *    нажатием, и датасет унаследовал бы ровно то, от чего проверка защищает.
 * 2. **Отмечаются только активные.** Что разметчик не отметил — уходит как
 *    `inactive` в момент сохранения. Требовать ответа на каждую из 21 AU
 *    значило бы менять одно нажатие на двадцать одно без единого нового факта.
 *    Правый клик ставит «не видно» там, где действительно не видно.
 * 3. **Пара рядом.** AU по определению есть разница относительно нейтрального
 *    кадра, поэтому оба кадра должны быть перед глазами одновременно.
 * 4. **Все 21 AU, а не подмножество.** Видимость решается на каждом кадре
 *    отдельно состоянием «не видно» — это точнее, чем исключать AU глобально.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AU_NAMES_RU,
  DOG_BREEDS,
  EMOTION_CLASSES,
  EMOTION_EMOJI,
  EMOTION_NAMES_RU,
  KEYPOINT_NAMES_RU,
  KEYPOINT_VISIBLE_THRESHOLD,
  VERIFIABLE_AU,
  getKeypointColor,
  type AUVerdict,
  type FrameAnnotation,
  type SessionData,
} from '../types';
import {
  getSession,
  importCoco,
  listDatasets,
  listTeam,
  patchKeypoints,
  patchReview,
  recomputeAUs,
  recomputeEmotion,
  type AvailableDataset,
  type TeamMember,
} from '../utils/api';
import useStore from '../store/useStore';

/** Клавиши 1-8 у первых AU из VERIFIABLE_AU; остальные размечаются кликом */
const AU_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8'];

/** Порядок анатомических групп на экране */
const AU_GROUP_ORDER = ['Пасть', 'Глаза', 'Нос', 'Уши'];

/**
 * Где помним выбранного разметчика.
 *
 * Без запоминания каждый заход начинался бы с выбора имени, а промах по кнопке
 * означал бы работу над чужой очередью и записи в чужой файл этикеток.
 */
const ANNOTATOR_STORAGE_KEY = 'dogfacs.annotator';

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

/** Kody AU, które reguły uznały za aktywne — w kolejności ekranu. */
function activeByRules(peak: FrameAnnotation): string[] {
  return VERIFIABLE_AU.map(({ code }) => code).filter((code) => ruleSaysActive(peak, code));
}

/**
 * Wylicza, czego brakuje, żeby parę dało się zatwierdzić.
 *
 * NIE ma tu AU i to jest decyzja, nie przeoczenie. Anotator klika tylko te
 * aktywne, a niezaznaczone stają się `inactive` w chwili zapisu — wymaganie
 * odpowiedzi na każdą z 21 sztuk zamieniłoby jedno kliknięcie w dwadzieścia
 * jeden bez żadnej nowej informacji. `not_observable` zostaje pod prawym
 * klikiem dla tego, czego naprawdę nie widać.
 *
 * Zostają pytania, na które milczenie NIE jest odpowiedzią: położenie punktów
 * i emocja. Tam brak wyboru znaczy „nie patrzyłem", a nie wartość domyślną.
 *
 * Zwraca listę braków, a nie samo `true/false`: „nie da się zapisać" bez powodu
 * zmusza do zgadywania, czego szukać na ekranie.
 */
function missingAnswers(keypointsOk: boolean | null, emotion: string | null): string[] {
  const gaps: string[] = [];
  if (keypointsOk === null) gaps.push('не отмечено, верно ли лежат точки');
  if (!emotion) gaps.push('не выбрана эмоция');
  return gaps;
}

/** Zapas wokół boksu psa przy kadrowaniu — ułamek jego szerokości i wysokości */
const CROP_PADDING = 0.18;

/**
 * Zapas wokół rozpiętości punktów przy kadrowaniu mordy.
 *
 * Ćwierć szerokości z każdej strony zostawia ucho i podgardle — sam obrys punktów
 * ucina je i anotator traci odniesienie, na którą część psa patrzy.
 */
const FACE_CROP_PADDING = 0.25;

/** Punkt poniżej tej pewności nie wyznacza kadru mordy */
const FACE_CROP_MIN_VISIBILITY = 0.3;

/** Poniżej tylu pewnych punktów kadr mordy jest zgadywaniem — wracamy do boksu psa */
const FACE_CROP_MIN_POINTS = 6;

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Wylicza wycinek wokół boksu CAŁEGO psa.
 *
 * Zapasowy sposób kadrowania: używany, gdy punktów jest za mało, żeby wyznaczyć
 * z nich mordę.
 */
function bodyRect(frame: FrameAnnotation, natural: Rect | null): Rect | null {
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

/**
 * Wylicza wycinek MORDY z rozpiętości pewnych punktów.
 *
 * Kadr po boksie psa jest za szeroki do pracy: na leżącym psie ma około 1000 px,
 * a morda w nim jakieś 50 — czyli 46 punktów zlewa się w plamę i trafienie
 * w konkretny jest loterią. Kadr liczony z samych punktów daje mordę na całą
 * kolumnę niezależnie od tego, ile psa jest w kadrze.
 *
 * Pełna klatka zostaje pod klawiszem `F` — potrzebna, żeby zobaczyć kontekst
 * (który to pies, co robi), ale nie do oceny ruchu mięśni.
 */
function faceRect(frame: FrameAnnotation, natural: Rect | null): Rect | null {
  const keypoints = frame.keypoints;
  if (!natural || !keypoints) return null;

  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < keypoints.length / 3; i++) {
    if (keypoints[i * 3 + 2] <= FACE_CROP_MIN_VISIBILITY) continue;
    xs.push(keypoints[i * 3]);
    ys.push(keypoints[i * 3 + 1]);
  }
  if (xs.length < FACE_CROP_MIN_POINTS) return null;

  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const padX = Math.max((x1 - x0) * FACE_CROP_PADDING, 1);
  const padY = Math.max((y1 - y0) * FACE_CROP_PADDING, 1);
  const left = Math.max(0, x0 - padX);
  const top = Math.max(0, y0 - padY);
  return {
    x: left,
    y: top,
    width: Math.min(natural.width - left, x1 - x0 + 2 * padX),
    height: Math.min(natural.height - top, y1 - y0 + 2 * padY),
  };
}

/**
 * Wybiera kadr roboczy: morda, a gdy jej nie da się wyznaczyć — cały pies.
 */
function cropRect(frame: FrameAnnotation, natural: Rect | null): Rect | null {
  return faceRect(frame, natural) ?? bodyRect(frame, natural);
}

/** Promien punktu keypoint jako ułamek szerokości wycinka — skaluje się z kadrem */
const KEYPOINT_RADIUS_RATIO = 0.006;

interface KeypointOverlayProps {
  keypoints: number[];
  region: Rect;
  /** Можно ли двигать точки прямо здесь */
  editable?: boolean;
  /** Отдаёт новые точки; `commit` = пора сохранять на сервер */
  onChange?: (keypoints: number[], commit: boolean) => void;
  /** Показывать ли скрытые точки (пустыми кольцами), чтобы их можно было вернуть */
  showHidden?: boolean;
}

/**
 * Рисует точки, которые ПОЙДУТ В ДАТАСЕТ, — и только их.
 *
 * Скрытые точки (видимость ниже порога) не рисуются вовсе. Раньше они
 * рисовались красным как «ненадёжные», и разметчик, спрятав точку в редакторе,
 * видел её здесь ярко-красной — то есть два окна показывали разные данные.
 *
 * viewBox по вырезу кадра означает, что координаты из COCO (в пикселях
 * оригинала) попадают на своё место без пересчёта и масштабируются вместе
 * с картинкой при изменении окна.
 */
function KeypointOverlay({
  keypoints,
  region,
  editable,
  onChange,
  showHidden,
}: KeypointOverlayProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const radius = region.width * KEYPOINT_RADIUS_RATIO * (editable ? 1.6 : 1);

  /** Переводит событие мыши в координаты исходного кадра. */
  const toImage = (event: React.PointerEvent): [number, number] => {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box) return [0, 0];
    return [
      region.x + ((event.clientX - box.left) / box.width) * region.width,
      region.y + ((event.clientY - box.top) / box.height) * region.height,
    ];
  };

  const movePoint = (index: number, event: React.PointerEvent) => {
    if (!onChange) return;
    const [x, y] = toImage(event);
    const next = [...keypoints];
    next[index * 3] = x;
    next[index * 3 + 1] = y;
    onChange(next, false);
  };

  const toggleVisibility = (index: number) => {
    if (!onChange) return;
    const next = [...keypoints];
    const visible = next[index * 3 + 2] > KEYPOINT_VISIBLE_THRESHOLD;
    next[index * 3 + 2] = visible ? 0 : 1;
    onChange(next, true);
  };

  const points = [];
  for (let i = 0; i < keypoints.length / 3; i++) {
    const [x, y, visibility] = [keypoints[i * 3], keypoints[i * 3 + 1], keypoints[i * 3 + 2]];
    const hidden = visibility <= KEYPOINT_VISIBLE_THRESHOLD;
    if (hidden && !showHidden) continue;
    points.push(
      <circle
        key={i}
        cx={x}
        cy={y}
        r={i === hovered ? radius * 1.5 : radius}
        fill={hidden ? 'none' : getKeypointColor(i)}
        stroke={i === hovered ? '#fff' : 'rgba(0,0,0,0.6)'}
        strokeWidth={radius * 0.35}
        style={editable ? { cursor: 'grab', pointerEvents: 'all' } : undefined}
        onPointerDown={
          editable
            ? (event) => {
                event.preventDefault();
                (event.target as Element).setPointerCapture(event.pointerId);
                setDragging(i);
              }
            : undefined
        }
        onPointerMove={editable && dragging === i ? (event) => movePoint(i, event) : undefined}
        onPointerUp={
          editable
            ? () => {
                if (dragging === i && onChange) onChange(keypoints, true);
                setDragging(null);
              }
            : undefined
        }
        onPointerEnter={editable ? () => setHovered(i) : undefined}
        onPointerLeave={editable ? () => setHovered(null) : undefined}
        onContextMenu={
          editable
            ? (event) => {
                event.preventDefault();
                toggleVisibility(i);
              }
            : undefined
        }
      />
    );
  }

  return (
    <svg
      ref={svgRef}
      className={`absolute inset-0 w-full h-full ${editable ? '' : 'pointer-events-none'}`}
      viewBox={`${region.x} ${region.y} ${region.width} ${region.height}`}
      preserveAspectRatio="none"
      style={editable ? { pointerEvents: 'none' } : undefined}
    >
      {points}
      {hovered !== null && (
        <text
          x={keypoints[hovered * 3] + radius * 2}
          y={keypoints[hovered * 3 + 1] - radius}
          fill="#fff"
          stroke="rgba(0,0,0,0.85)"
          strokeWidth={radius * 0.5}
          paintOrder="stroke"
          fontSize={radius * 3.5}
          style={{ pointerEvents: 'none' }}
        >
          {KEYPOINT_NAMES_RU[hovered] ?? `kp${hovered}`}
        </text>
      )}
    </svg>
  );
}

interface FrameViewProps {
  frame: FrameAnnotation;
  label: string;
  accent: string;
  showFullFrame: boolean;
  showKeypoints: boolean;
  keypoints: number[] | null;
  editable?: boolean;
  showHidden?: boolean;
  onChange?: (keypoints: number[], commit: boolean) => void;
}

function FrameView({
  frame,
  label,
  accent,
  showFullFrame,
  showKeypoints,
  keypoints,
  editable,
  showHidden,
  onChange,
}: FrameViewProps) {
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
        {showKeypoints && keypoints && region && (
          <KeypointOverlay
            keypoints={keypoints}
            region={region}
            editable={editable}
            showHidden={showHidden}
            onChange={onChange}
          />
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

interface RulesSummaryProps {
  peak: FrameAnnotation;
  recomputing: boolean;
  onRecompute: () => void;
}

/**
 * Сводка того, что насчитал автомат, — в одном месте и без раскопок по карточкам.
 *
 * Раньше замер правил был размазан подписью «правило: да» по 21 карточке: чтобы
 * понять, что вообще увидела модель, приходилось глазами собирать список.
 * Здесь он собран, рядом стоит пересчёт (после правки точек замер устаревает)
 * и кнопка принять всё разом.
 */
function RulesSummary({ peak, recomputing, onRecompute }: RulesSummaryProps) {
  const active = activeByRules(peak);

  return (
    <div className="mb-3 p-3 rounded-lg border border-gray-200 bg-gray-50">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-gray-500 shrink-0">
          Автомат насчитал
        </span>

        {active.length > 0 ? (
          <span className="flex flex-wrap gap-1">
            {active.map((code) => (
              <span
                key={code}
                className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200"
                title={AU_NAMES_RU[code] ?? code}
              >
                {code}
              </span>
            ))}
          </span>
        ) : (
          <span className="text-xs text-gray-500">активных AU нет</span>
        )}

        <span className="text-xs text-gray-600">
          эмоция{' '}
          <b className="text-gray-800">
            {peak.emotion ? (EMOTION_NAMES_RU[peak.emotion] ?? peak.emotion) : '—'}
          </b>
        </span>

        <div className="flex gap-2 ml-auto shrink-0">
          <button
            onClick={onRecompute}
            disabled={recomputing}
            className="px-3 py-1.5 rounded-md border border-amber-300 bg-amber-50 text-xs text-amber-800 hover:border-amber-500 disabled:opacity-50"
            title="Пересчитает замер по текущим точкам. Твои ответы не трогает."
          >
            {recomputing ? 'Считаю…' : '↻ Пересчитать'}
          </button>
        </div>
      </div>

      {/* Мера, а не мнение: шум правил измерен и он выше порога активации на
          большинстве пар трек–AU. Подпись держит это перед глазами — список
          сверху подсказка, а не готовый ответ. */}
      <p className="mt-2 text-[10px] leading-tight text-gray-400">
        Это замер правил по геометрии точек, а не ответ: измеренный шум правил выше порога
        активации у 68.9% пар трек–AU. Отмечай активные сам — что не отметил, уйдёт как
        неактивное.
      </p>
    </div>
  );
}

/** Ile propozycji rasy pokazujemy obok pola tekstowego */
const BREED_SUGGESTIONS = 6;

// Rasy klasyfikator myli notorycznie (mediana pewności 0.33 na tym materiale),
// więc anotator musi mieć jak powiedzieć "nie wiem" zamiast zgadywać.
const UNKNOWN_BREED = 'unknown';

interface AnnotatorPickerProps {
  team: TeamMember[];
  current: string | null;
  onPick: (key: string) => void;
}

/**
 * Кто размечает. От выбора зависит, какие видео попадут в очередь и в чей файл
 * этикеток уйдут вердикты, поэтому имя видно всегда, а не прячется в настройках.
 */
function AnnotatorPicker({ team, current, onPick }: AnnotatorPickerProps) {
  if (!team.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-3">
      {team.map((member) => (
        <button
          key={member.key}
          onClick={() => onPick(member.key)}
          className={`px-3 py-1.5 rounded-md border text-sm font-medium transition-colors ${
            current === member.key
              ? 'bg-primary-700 border-primary-800 text-white'
              : 'bg-white border-gray-200 text-gray-700 hover:border-primary-400'
          }`}
        >
          {member.display}
        </button>
      ))}
    </div>
  );
}

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
  const [team, setTeam] = useState<TeamMember[]>([]);
  const [annotator, setAnnotator] = useState<string | null>(
    () => localStorage.getItem(ANNOTATOR_STORAGE_KEY)
  );
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
  const [rolesSwapped, setRolesSwapped] = useState(false);
  // Правка точек идёт ЗДЕСЬ, а не в отдельном окне: переход в него стоил
  // времени на каждой паре, а показывал те же данные в другом масштабе.
  const [editPoints, setEditPoints] = useState(false);
  const [showHiddenPoints, setShowHiddenPoints] = useState(false);
  const [editedNeutral, setEditedNeutral] = useState<number[] | null>(null);
  const [editedPeak, setEditedPeak] = useState<number[] | null>(null);
  const [recomputing, setRecomputing] = useState(false);
  // Pełny edytor (przeciąganie 46 punktów, zakładki AU/emocja/rasa) działa na
  // sesji trzymanej w store — dlatego ładujemy ją tam obok szybkiego trybu.
  const loadStoreSession = useStore((state) => state.loadSession);

  const current = pairs[position];

  const pickAnnotator = useCallback((key: string) => {
    localStorage.setItem(ANNOTATOR_STORAGE_KEY, key);
    setAnnotator(key);
    // Смена человека меняет очередь целиком — сессию сбрасываем, иначе
    // на экране осталась бы чужая пара.
    setSession(null);
    setPairs([]);
    setDatasets(null);
  }, []);

  const startSession = useCallback(async (path: string) => {
    setBusy(true);
    setError(null);
    try {
      const imported = await importCoco(path, annotator ?? undefined);
      const loaded = await getSession(imported.session_id);
      await loadStoreSession(imported.session_id);
      const built = buildPairs(loaded);
      setSession(loaded);
      setPairs(built);
      setVerifiedCount(built.filter((p) => p.peak.annotation_status === 'verified').length);
      setRejectedCount(built.filter((p) => p.peak.usable === false).length);
      // Wracamy tam, gdzie praca się urwała — a nie na początek kolejki.
      const firstOpen = built.findIndex((p) => p.peak.annotation_status !== 'verified');
      const start = firstOpen === -1 ? built.length : firstOpen;
      setPosition(start);
      // Pierwsza para po otwarciu musi dostać to samo co każda następna. `goTo`
      // tu nie zadziała — czyta `pairs` ze stanu, którego React jeszcze nie
      // zaktualizował — więc wypełniamy z listy policzonej przed chwilą.
      const first = built[start]?.peak;
      setVerdicts(first?.au_verdicts ?? {});
      setKeypointsOk(first?.keypoints_ok ?? null);
      setBreed(first?.breed ?? null);
      setEmotion(first?.emotion ?? null);
      setRolesSwapped(first?.roles_swapped ?? false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Не удалось загрузить набор');
    } finally {
      setBusy(false);
    }
  }, [loadStoreSession, annotator]);

  // Zbiory znajdują się same. Gdy jest dokładnie jeden, od razu go otwieramy —
  // anotator ma zobaczyć pierwszą parę, a nie formularz.
  useEffect(() => {
    listTeam().then(setTeam).catch(() => setTeam([]));
  }, []);

  useEffect(() => {
    if (!annotator) return;
    let cancelled = false;
    listDatasets(annotator)
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
  }, [startSession, annotator]);

  const goTo = useCallback(
    (index: number) => {
      if (index < 0 || index >= pairs.length) return;
      const peak = pairs[index]?.peak;
      setPosition(index);
      // Pusto, a nie „jak policzyły reguły". Zaznaczenie z góry zamieniało
      // weryfikację w zatwierdzanie: reguły potrafią wskazać 12 z 21 AU naraz na
      // spokojnym psie, w tym EAD101 („uszy do przodu") i EAD103 („uszy
      // położone") JEDNOCZEŚNIE. Zbiór odziedziczyłby dokładnie te usterki,
      // przed którymi weryfikacja ma chronić.
      setVerdicts(peak?.au_verdicts ?? {});
      setKeypointsOk(peak?.keypoints_ok ?? null);
      // Rasa i emocja startują od tego, co policzył automat — tu poprawiamy
      // gotową wartość, a nie etykietujemy od zera jak przy AU.
      setBreed(peak?.breed ?? null);
      setEmotion(peak?.emotion ?? null);
      setBreedQuery('');
      setRolesSwapped(peak?.roles_swapped ?? false);
      setEditedNeutral(null);
      setEditedPeak(null);
      setShowHiddenPoints(false);
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
      const blocking = missingAnswers(keypointsOk, emotion);
      if (usable && blocking.length > 0) {
        setError(`Не хватает: ${blocking.join(' · ')}`);
        return;
      }
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
          roles_swapped: rolesSwapped,
          annotator,
        });
        current.peak.au_verdicts = complete;
        current.peak.keypoints_ok = usable ? keypointsOk : false;
        current.peak.breed = breed;
        current.peak.emotion = emotion;
        current.peak.usable = usable;
        current.peak.roles_swapped = rolesSwapped;
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
    [session, current, busy, verdicts, position, goTo, keypointsOk, breed, emotion, rolesSwapped]
  );

  useEffect(() => {
    if (!current) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;

      // Читаем ФИЗИЧЕСКУЮ клавишу (`code`), а не символ (`key`): на русской
      // раскладке «S» даёт «ы», «K» даёт «л», и разметка переставала работать
      // ровно у тех, кто ей пользуется.
      const digit = AU_KEYS.indexOf(event.code.replace(/^(Digit|Numpad)/, ''));
      if (digit >= 0) {
        event.preventDefault();
        toggle(VERIFIABLE_AU[digit].code, event.shiftKey ? 'not_observable' : 'active');
        return;
      }

      switch (event.code) {
        case 'KeyS':
          event.preventDefault();
          setRolesSwapped((swapped) => !swapped);
          break;
        case 'KeyP':
          event.preventDefault();
          setEditPoints((on) => !on);
          break;
        case 'KeyK':
          event.preventDefault();
          setShowKeypoints((shown) => !shown);
          break;
        case 'KeyF':
          event.preventDefault();
          setShowFullFrame((shown) => !shown);
          break;
        case 'KeyX':
          event.preventDefault();
          void commit(false);
          break;
        case 'Enter':
        case 'NumpadEnter':
          event.preventDefault();
          void commit();
          break;
        case 'ArrowLeft':
          goTo(position - 1);
          break;
        case 'ArrowRight':
        case 'Space':
          event.preventDefault();
          goTo(position + 1);
          break;
        default:
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [current, toggle, commit, goTo, position]);

  /**
   * Сохраняет поправленные точки и, для нейтрального кадра, обновляет базу AU.
   *
   * `commit=false` приходит на каждом движении мыши — тогда только обновляем
   * картинку, чтобы не слать на сервер сотню запросов за одно перетаскивание.
   */
  const saveKeypoints = useCallback(
    async (role: 'neutral' | 'peak', points: number[], commit: boolean) => {
      if (role === 'neutral') setEditedNeutral(points);
      else setEditedPeak(points);
      if (!commit || !session || !current) return;
      const frame = role === 'neutral' ? current.neutral : current.peak;
      try {
        await patchKeypoints(session.session_id, frame.frame_idx, points, annotator ?? undefined);
        frame.keypoints = points;
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Не удалось сохранить точки');
      }
    },
    [session, current, annotator]
  );

  /**
   * Пересчитывает AU и эмоцию по исправленным точкам.
   *
   * Это подсказка, а не ответ: пересчёт обновляет ЗАМЕР правил, который виден
   * серой подписью «правило: да». Метку по-прежнему ставит человек.
   */
  const recompute = useCallback(async () => {
    if (!session || !current || recomputing) return;
    setRecomputing(true);
    setError(null);
    try {
      await recomputeAUs(session.session_id, current.peak.frame_idx, current.trackId);
      const result = await recomputeEmotion(
        session.session_id,
        current.peak.frame_idx,
        current.trackId
      );
      const refreshed = await getSession(session.session_id);
      setPairs(buildPairs(refreshed));
      setEmotion(result.emotion);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Не удалось пересчитать');
    } finally {
      setRecomputing(false);
    }
  }, [session, current, recomputing]);

  const activeCount = useMemo(
    () => Object.values(verdicts).filter((value) => value === 'active').length,
    [verdicts]
  );

  /**
   * Czego brakuje, żeby zapisać parę.
   *
   * Dotyczy tylko zapisu „nadaje się". Odrzucenie kadru (`X`) jest kompletną
   * odpowiedzią samo w sobie: skoro kadr się nie nadaje, nie ma czego na nim
   * oceniać i wymaganie wypełnienia 21 AU byłoby żądaniem zmyślania.
   */
  const gaps = useMemo(() => missingAnswers(keypointsOk, emotion), [keypointsOk, emotion]);

  if (!session) {
    return (
      <div className="max-w-xl mx-auto mt-16 p-6 bg-white rounded-xl border border-gray-200">
        <h2 className="text-lg font-bold text-gray-800">Разметка AU</h2>

        <AnnotatorPicker team={team} current={annotator} onPick={pickAnnotator} />

        {!annotator && (
          <p className="text-sm text-gray-500 mt-3">
            Выбери, кто размечает — у каждого своя часть видео, чтобы не делать одно и то же.
          </p>
        )}
        {annotator && datasets === null && !error && (
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
                    {/* Один набор виден дважды только у того, кто его сгенерировал:
                        сырой материал лежит локально, паковка приходит из git.
                        Имя у них общее — по нему сходится разметка команды. */}
                    {dataset.variant === 'work' && (
                      <span
                        className="ml-2 text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200"
                        title="Из репозитория — это же видят коллеги"
                      >
                        из git
                      </span>
                    )}
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

        {annotator && (
          <p className="mt-4 text-xs text-gray-400">
            Новые видео добавляются на вкладке «Загрузка видео».
          </p>
        )}
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
        {/* Раньше здесь была кнопка «Экспорт COCO». Она отдавала только текущую
            сессию — без width/height и лицензий, то есть COCO хуже пакетного, —
            и работала лишь пока набор открыт в браузере. Готовый датасет
            собирает `python -m scripts.annotation.build_final_dataset`, и он
            же кладёт его в `data/dataset_final/release/`. */}
        <p className="text-xs text-gray-400 mt-3">
          Разметка сохраняется сама. Готовый датасет собирается командой{' '}
          <code className="px-1 py-0.5 bg-gray-100 rounded">build_final_dataset</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm text-gray-600">
          <span className="font-semibold text-primary-700">
            {team.find((m) => m.key === annotator)?.display ?? annotator}
          </span>{' '}
          · Пара <strong>{position + 1}</strong> / {pairs.length} · размечено{' '}
          <strong className="text-green-700">{verifiedCount}</strong> · отброшено{' '}
          <strong className="text-gray-600">{rejectedCount}</strong> · активных AU{' '}
          <strong className="text-amber-700">{activeCount}</strong>
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            onClick={() => {
              localStorage.removeItem(ANNOTATOR_STORAGE_KEY);
              setAnnotator(null);
              setSession(null);
              setPairs([]);
              setDatasets(null);
            }}
            className="text-xs px-3 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
          >
            Сменить пользователя
          </button>
        </div>
      </div>

      <div className="h-1.5 bg-gray-200 rounded-full mb-4 overflow-hidden">
        <div
          className="h-full bg-green-500 transition-all"
          style={{ width: `${(verifiedCount / Math.max(pairs.length, 1)) * 100}%` }}
        />
      </div>

      {/* Порядок кадров меняется вместе с ролями: если разметчик сказал, что
          пайплайн перепутал, база отсчёта встаёт слева, как и должна. */}
      <div className="flex gap-4 mb-2">
        {(rolesSwapped
          ? ([['peak', current.peak], ['neutral', current.neutral]] as const)
          : ([['neutral', current.neutral], ['peak', current.peak]] as const)
        ).map(([role, frame], index) => (
          <FrameView
            key={frame.frame_idx}
            frame={frame}
            label={index === 0 ? 'Нейтральный (база AU)' : 'Пиковый (оцениваем)'}
            accent={index === 0 ? 'text-blue-600' : 'text-amber-600'}
            showFullFrame={showFullFrame}
            showKeypoints={showKeypoints}
            keypoints={
              (role === 'neutral' ? editedNeutral : editedPeak) ?? frame.keypoints ?? null
            }
            editable={editPoints}
            showHidden={editPoints && showHiddenPoints}
            onChange={(points, commit) => void saveKeypoints(role, points, commit)}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2 mb-3">
        <button
          onClick={() => setEditPoints((on) => !on)}
          className={`px-3 py-1.5 rounded-md border text-xs font-medium transition-colors ${
            editPoints
              ? 'bg-green-600 border-green-700 text-white'
              : 'bg-white border-gray-200 text-gray-600 hover:border-green-300'
          }`}
          title="Точки можно перетаскивать; правый клик скрывает или возвращает точку"
        >
          ✎ Править точки (P)
        </button>
        {editPoints && (
          <button
            onClick={() => setShowHiddenPoints((on) => !on)}
            className="px-3 py-1.5 rounded-md border border-gray-200 text-xs text-gray-600 hover:border-gray-400"
          >
            {showHiddenPoints ? 'скрыть спрятанные' : 'показать спрятанные'}
          </button>
        )}
        <button
          onClick={() => setRolesSwapped((swapped) => !swapped)}
          className={`px-3 py-1.5 rounded-md border text-xs font-medium transition-colors ${
            rolesSwapped
              ? 'bg-blue-600 border-blue-700 text-white'
              : 'bg-white border-gray-200 text-gray-600 hover:border-blue-300'
          }`}
          title="Если спокойным выглядит правый кадр — роли перепутаны"
        >
          ⇄ Поменять роли (S)
        </button>
        {rolesSwapped && (
          <span className="text-xs text-blue-700">
            роли переставлены — база отсчёта слева
          </span>
        )}
      </div>

      <RulesSummary
        peak={current.peak}
        recomputing={recomputing}
        onRecompute={() => void recompute()}
      />

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
            верно
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

        <ChoiceRow label="Порода" hint={`автомат: ${(current.peak.breed_confidence * 100).toFixed(0)}% уверенности`}>
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
                    не знаю
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
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">S</kbd> поменять роли ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">K</kbd> точки ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">P</kbd> править точки ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">F</kbd>{' '}
          {showFullFrame ? 'вернуться к морде' : 'весь кадр'} ·{' '}
          <kbd className="px-1.5 py-0.5 bg-gray-100 rounded font-mono">←→</kbd> навигация
          {gaps.length > 0 && (
            <span className="block mt-1 text-amber-700 font-medium">
              Не хватает: {gaps.join(' · ')}
            </span>
          )}
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
            disabled={busy || gaps.length > 0}
            title={gaps.length > 0 ? `Не хватает: ${gaps.join(' · ')}` : undefined}
            className="px-4 py-2 bg-amber-600 text-white rounded-lg font-semibold hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {busy ? 'Сохраняю…' : 'Сохранить и дальше ⏎'}
          </button>
        </div>
      </div>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

    </div>
  );
}
