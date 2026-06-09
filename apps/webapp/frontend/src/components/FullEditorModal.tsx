/**
 * FullEditorModal — pełnoekranowy edytor anotacji klatki.
 *
 * Lewa część: canvas z obrazem psa, przycięty do bbox.
 * Prawa część: zakładki — Punkty kluczowe, AU, Emocja, Rasa.
 *
 * Przekształcenie współrzędnych:
 *   Punkty kluczowe przechowywane są we współrzędnych pełnej klatki.
 *   Na canvas wyświetlany jest tylko obszar bbox → offsetX = -bboxX * scale.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  KEYPOINT_NAMES,
  NUM_KEYPOINTS,
  SKELETON_CONNECTIONS,
  getKeypointColor,
  type FrameAnnotation,
} from '../types';
import useStore from '../store/useStore';
import AUPanel from './AUPanel';
import BreedPicker from './BreedPicker';
import EmotionSelector from './EmotionSelector';

interface FullEditorModalProps {
  frame: FrameAnnotation;
  onClose: () => void;
}

type TabId = 'keypoints' | 'au' | 'emotion' | 'breed';

const TABS: { id: TabId; label: string }[] = [
  { id: 'keypoints', label: 'Punkty kluczowe' },
  { id: 'au',        label: 'AU' },
  { id: 'emotion',   label: 'Emocja' },
  { id: 'breed',     label: 'Rasa' },
];

const POINT_RADIUS = 3;
const HIT_RADIUS = 10;

function parseKeypoints(flat: number[]): { x: number; y: number; v: number }[] {
  const result = [];
  for (let i = 0; i < NUM_KEYPOINTS; i++) {
    result.push({ x: flat[i * 3], y: flat[i * 3 + 1], v: flat[i * 3 + 2] });
  }
  return result;
}

function flattenKeypoints(kps: { x: number; y: number; v: number }[]): number[] {
  return kps.flatMap((kp) => [kp.x, kp.y, kp.v]);
}

export default function FullEditorModal({ frame, onClose }: FullEditorModalProps) {
  const { updateFrameKeypoints, recomputeFrameAUs, saving } = useStore();
  const [activeTab, setActiveTab] = useState<TabId>('keypoints');

  // Canvas state
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const scaleRef = useRef({ x: 1, y: 1, offsetX: 0, offsetY: 0 });
  const [localKPs, setLocalKPs] = useState(() =>
    frame.keypoints ? parseKeypoints(frame.keypoints) : []
  );
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  const bbox = frame.bbox;

  // Esc = zamknij
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  // Blokada przewijania
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; };
  }, []);

  // Rysowanie canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Rysujemy przyciety obszar bbox
    const bboxX = bbox ? bbox[0] : 0;
    const bboxY = bbox ? bbox[1] : 0;
    const bboxW = bbox ? bbox[2] : img.naturalWidth;
    const bboxH = bbox ? bbox[3] : img.naturalHeight;
    ctx.drawImage(img, bboxX, bboxY, bboxW, bboxH, 0, 0, canvas.width, canvas.height);

    if (localKPs.length === 0) return;

    const { x: sx, y: sy, offsetX, offsetY } = scaleRef.current;

    // Szkielet
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    for (const [a, b] of SKELETON_CONNECTIONS) {
      const kpA = localKPs[a];
      const kpB = localKPs[b];
      if (kpA.v > 0.3 && kpB.v > 0.3) {
        ctx.beginPath();
        ctx.moveTo(kpA.x * sx + offsetX, kpA.y * sy + offsetY);
        ctx.lineTo(kpB.x * sx + offsetX, kpB.y * sy + offsetY);
        ctx.stroke();
      }
    }

    // Punkty
    for (let i = 0; i < localKPs.length; i++) {
      const kp = localKPs[i];
      const cx = kp.x * sx + offsetX;
      const cy = kp.y * sy + offsetY;
      const color = getKeypointColor(i);
      const alpha = kp.v < 0.3 ? 0.25 : 1;
      const radius = i === hoveredIdx ? POINT_RADIUS + 2 : POINT_RADIUS;

      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = i === hoveredIdx ? '#fff' : '#000';
      ctx.lineWidth = i === hoveredIdx ? 1.5 : 0.8;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Etykieta przy najechaniu
    if (hoveredIdx !== null && localKPs[hoveredIdx]) {
      const kp = localKPs[hoveredIdx];
      const cx = kp.x * sx + offsetX;
      const cy = kp.y * sy + offsetY;
      const label = KEYPOINT_NAMES[hoveredIdx] ?? `kp${hoveredIdx}`;
      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(0,0,0,0.8)';
      const w = ctx.measureText(label).width + 8;
      ctx.fillRect(cx + 8, cy - 14, w, 17);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, cx + 12, cy - 1);
    }
  }, [localKPs, hoveredIdx, bbox]);

  // Ladowanie obrazu i konfiguracja canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = frame.image_url;
    img.onload = () => {
      imageRef.current = img;

      const bboxX = bbox ? bbox[0] : 0;
      const bboxY = bbox ? bbox[1] : 0;
      const bboxW = bbox ? bbox[2] : img.naturalWidth;
      const bboxH = bbox ? bbox[3] : img.naturalHeight;

      // Lewa czesc zajmuje ~65% szerokosci, wysokosc — prawie caly ekran
      const maxW = Math.floor((window.innerWidth - 320) * 0.97);
      const maxH = Math.floor(window.innerHeight * 0.84);
      const scale = Math.min(maxW / bboxW, maxH / bboxH, 4);

      canvas.width = Math.round(bboxW * scale);
      canvas.height = Math.round(bboxH * scale);

      // offsetX = -bboxX * scale → formuly pozostaja bez zmian
      scaleRef.current = {
        x: scale,
        y: scale,
        offsetX: -bboxX * scale,
        offsetY: -bboxY * scale,
      };
      draw();
    };
  }, [frame.image_url, bbox, draw]);

  useEffect(() => { draw(); }, [draw]);

  const findHitPoint = useCallback((cx: number, cy: number): number => {
    const { x: sx, y: sy, offsetX, offsetY } = scaleRef.current;
    for (let i = 0; i < localKPs.length; i++) {
      const kp = localKPs[i];
      const kx = kp.x * sx + offsetX;
      const ky = kp.y * sy + offsetY;
      if (Math.hypot(cx - kx, cy - ky) <= HIT_RADIUS) return i;
    }
    return -1;
  }, [localKPs]);

  const canvasToImage = useCallback((cx: number, cy: number) => {
    const { x: sx, y: sy, offsetX, offsetY } = scaleRef.current;
    return { x: (cx - offsetX) / sx, y: (cy - offsetY) / sy };
  }, []);

  const getCanvasPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { cx: e.clientX - rect.left, cy: e.clientY - rect.top };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const { cx, cy } = getCanvasPos(e);
    const idx = findHitPoint(cx, cy);
    if (idx !== -1) setDraggingIdx(idx);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { cx, cy } = getCanvasPos(e);
    const hit = findHitPoint(cx, cy);
    setHoveredIdx(draggingIdx !== null ? draggingIdx : hit === -1 ? null : hit);
    if (draggingIdx === null) return;
    const { x, y } = canvasToImage(cx, cy);
    setLocalKPs((prev) => {
      const next = [...prev];
      next[draggingIdx] = { ...next[draggingIdx], x, y, v: 1.0 };
      return next;
    });
    setIsDirty(true);
  };

  const handleMouseUp = () => setDraggingIdx(null);

  const handleContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const { cx, cy } = getCanvasPos(e);
    const idx = findHitPoint(cx, cy);
    if (idx === -1) return;
    setLocalKPs((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], v: next[idx].v > 0.3 ? 0 : 1.0 };
      return next;
    });
    setIsDirty(true);
  };

  const handleSave = async () => {
    if (localKPs.length === 0) return;
    await updateFrameKeypoints(frame.frame_idx, flattenKeypoints(localKPs));
    setIsDirty(false);
  };

  const handleRecompute = async () => {
    if (isDirty) await handleSave();
    await recomputeFrameAUs(frame.frame_idx);
  };

  const handleClose = async () => {
    if (isDirty) await handleSave();
    onClose();
  };

  const visibleCount = localKPs.filter((k) => k.v > 0.3).length;

  return createPortal(
    <div className="fixed inset-0 z-50 flex bg-black/90">
      {/* Lewa czesc: canvas */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Naglowek */}
        <div className="flex items-center justify-between px-4 py-2 bg-gray-800 flex-shrink-0">
          <span className="text-white text-sm font-medium">Klatka #{frame.frame_idx}</span>
          <span className="text-gray-400 text-xs">PPM = przełącz widoczność · Esc = zamknij</span>
        </div>

        {/* Canvas */}
        <div className="flex-1 bg-gray-900 overflow-auto flex items-center justify-center p-3">
          {localKPs.length === 0 ? (
            <p className="text-gray-500">Brak punktów kluczowych dla tej klatki</p>
          ) : (
            <canvas
              ref={canvasRef}
              className="block cursor-crosshair"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={() => { setDraggingIdx(null); setHoveredIdx(null); }}
              onContextMenu={handleContextMenu}
            />
          )}
        </div>

        {/* Stopka */}
        <div className="flex items-center gap-3 px-4 py-2.5 bg-gray-800 flex-shrink-0">
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-gray-400 flex-1">
            {[
              { label: 'Oczy', color: '#00ff00' },
              { label: 'Brwi', color: '#ff0080' },
              { label: 'Uszy', color: '#00a5ff' },
              { label: 'Nos', color: '#ff3030' },
              { label: 'Usta', color: '#00ffff' },
              { label: 'Pysk', color: '#ff00ff' },
              { label: 'Kontur', color: '#4488ff' },
            ].map(({ label, color }) => (
              <span key={label} className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                {label}
              </span>
            ))}
            <span className="text-gray-500">· Widocznych: {visibleCount}/{NUM_KEYPOINTS}</span>
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <button
              onClick={handleRecompute}
              disabled={saving || localKPs.length === 0}
              className="px-3 py-1.5 text-sm bg-amber-500 text-white rounded hover:bg-amber-600 disabled:opacity-40 transition-colors"
            >
              {saving ? '…' : 'Przelicz AU'}
            </button>
            <button
              onClick={handleClose}
              disabled={saving}
              className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-40 transition-colors"
            >
              {saving ? 'Zapisuję…' : isDirty ? 'Zapisz i zamknij *' : 'Zamknij'}
            </button>
          </div>
        </div>
      </div>

      {/* Prawy panel: zakladki */}
      <div className="w-80 bg-white flex flex-col flex-shrink-0 border-l border-gray-200">
        {/* Zakladki + przycisk zamkniecia */}
        <div className="flex items-stretch border-b border-gray-200 bg-gray-50 flex-shrink-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 py-2 text-xs font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-700 border-b-2 border-blue-500 bg-white'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.label}
            </button>
          ))}
          <button
            onClick={handleClose}
            className="px-3 text-gray-400 hover:text-gray-700 text-lg leading-none border-l border-gray-200"
            title="Zamknij"
          >
            ✕
          </button>
        </div>

        {/* Zawartosc zakladki */}
        <div className="flex-1 overflow-y-auto p-3">
          {activeTab === 'keypoints' && (
            <div className="space-y-3 text-sm">
              <p className="text-gray-500 text-xs">
                Przeciągaj punkty na obrazie po lewej stronie.<br />
                PPM = przełącz widoczność punktu.
              </p>
              {isDirty && (
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="w-full py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-40"
                >
                  {saving ? 'Zapisuję…' : 'Zapisz punkty kluczowe *'}
                </button>
              )}
            </div>
          )}

          {activeTab === 'au' && (
            <AUPanel frameIdx={frame.frame_idx} aus={frame.aus} />
          )}

          {activeTab === 'emotion' && (
            <EmotionSelector
              frameIdx={frame.frame_idx}
              currentEmotion={frame.emotion}
              confidence={frame.emotion_confidence}
              ruleApplied={frame.emotion_rule_applied}
              aus={frame.aus}
            />
          )}

          {activeTab === 'breed' && (
            <BreedPicker
              frameIdx={frame.frame_idx}
              currentBreed={frame.breed}
              confidence={frame.breed_confidence}
            />
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
