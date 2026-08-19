/**
 * KeypointEditorModal — fullscreen overlay do precyzyjnego edytowania keypoints.
 *
 * Otwierany przyciskiem "↗" w KeypointEditor.
 * Canvas zajmuje ~90vw × ~80vh ekranu, co ułatwia klikanie małych punktów.
 * Logika drag&drop identyczna jak w KeypointEditor.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  KEYPOINT_VISIBLE_THRESHOLD,
  KEYPOINT_NAMES_RU,
  NUM_KEYPOINTS,
  SKELETON_CONNECTIONS,
  getKeypointColor,
} from '../types';
import useStore from '../store/useStore';

interface KeypointEditorModalProps {
  frameIdx: number;
  imageUrl: string;
  keypoints: number[];
  onClose: () => void;
}

/** Parsuje flat array [x0,y0,v0,...] do tablicy {x,y,v}[46] */
function parseKeypoints(flat: number[]): { x: number; y: number; v: number }[] {
  const result = [];
  for (let i = 0; i < NUM_KEYPOINTS; i++) {
    result.push({ x: flat[i * 3], y: flat[i * 3 + 1], v: flat[i * 3 + 2] });
  }
  return result;
}

/** Konwertuje {x,y,v}[46] z powrotem na flat array */
function flattenKeypoints(kps: { x: number; y: number; v: number }[]): number[] {
  return kps.flatMap((kp) => [kp.x, kp.y, kp.v]);
}

const POINT_RADIUS = 6;
const HIT_RADIUS = 14;

export default function KeypointEditorModal({
  frameIdx,
  imageUrl,
  keypoints,
  onClose,
}: KeypointEditorModalProps) {
  const { updateFrameKeypoints, recomputeFrameAUs, saving } = useStore();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const scaleRef = useRef({ x: 1, y: 1, offsetX: 0, offsetY: 0 });
  const [localKPs, setLocalKPs] = useState(() => parseKeypoints(keypoints));
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  // Zamknij po naciśnięciu Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  // Zablokuj przewijanie strony gdy modal otwarty
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
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    if (localKPs.length === 0) return;

    const { x: sx, y: sy, offsetX, offsetY } = scaleRef.current;

    // Szkielet
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    for (const [a, b] of SKELETON_CONNECTIONS) {
      const kpA = localKPs[a];
      const kpB = localKPs[b];
      if (kpA.v > KEYPOINT_VISIBLE_THRESHOLD && kpB.v > KEYPOINT_VISIBLE_THRESHOLD) {
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
      const alpha = kp.v < KEYPOINT_VISIBLE_THRESHOLD ? 0.25 : 1;
      const radius = i === hoveredIdx ? POINT_RADIUS + 3 : POINT_RADIUS;

      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = i === hoveredIdx ? '#fff' : '#000';
      ctx.lineWidth = i === hoveredIdx ? 2 : 1;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Etykieta hovered punktu
    if (hoveredIdx !== null && localKPs[hoveredIdx]) {
      const kp = localKPs[hoveredIdx];
      const cx = kp.x * sx + offsetX;
      const cy = kp.y * sy + offsetY;
      const label = KEYPOINT_NAMES_RU[hoveredIdx] ?? `kp${hoveredIdx}`;
      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(0,0,0,0.75)';
      const w = ctx.measureText(label).width + 8;
      ctx.fillRect(cx + 10, cy - 14, w, 18);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, cx + 14, cy);
    }
  }, [localKPs, hoveredIdx]);

  // Załaduj obraz i dopasuj canvas do okna
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageUrl;
    img.onload = () => {
      imageRef.current = img;

      // Dopasuj canvas do dostępnego miejsca w modalu (90vw × 80vh)
      const maxW = Math.floor(window.innerWidth * 0.88);
      const maxH = Math.floor(window.innerHeight * 0.78);
      const scale = Math.min(maxW / img.width, maxH / img.height, 1);

      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      scaleRef.current = { x: scale, y: scale, offsetX: 0, offsetY: 0 };
      draw();
    };
  }, [imageUrl, draw]);

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
      next[idx] = { ...next[idx], v: next[idx].v > KEYPOINT_VISIBLE_THRESHOLD ? 0 : 1.0 };
      return next;
    });
    setIsDirty(true);
  };

  const handleSave = async () => {
    if (localKPs.length === 0) return;
    await updateFrameKeypoints(frameIdx, flattenKeypoints(localKPs));
    setIsDirty(false);
  };

  const handleRecompute = async () => {
    if (isDirty) await handleSave();
    await recomputeFrameAUs(frameIdx);
  };

  const handleSaveAndClose = async () => {
    if (isDirty) await handleSave();
    onClose();
  };

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/80"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="flex flex-col bg-gray-900 rounded-xl shadow-2xl overflow-hidden max-w-[95vw]">
        {/* Nagłówek */}
        <div className="flex items-center justify-between px-4 py-2 bg-gray-800">
          <span className="text-white text-sm font-medium">
            Edycja keypoints — Frame #{frameIdx}
          </span>
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-xs">PPM = toggle widoczność · Esc = zamknij</span>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white text-lg leading-none px-1"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Canvas */}
        <div className="relative bg-gray-900">
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { setDraggingIdx(null); setHoveredIdx(null); }}
            onContextMenu={handleContextMenu}
            className="block cursor-crosshair"
          />
        </div>

        {/* Stopka z przyciskami */}
        <div className="flex items-center gap-3 px-4 py-3 bg-gray-800">
          {/* Legenda */}
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
                <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                {label}
              </span>
            ))}
          </div>
          {/* Przyciski akcji */}
          <div className="flex gap-2">
            <button
              onClick={handleRecompute}
              disabled={saving}
              className="px-3 py-1.5 text-sm bg-amber-500 text-white rounded hover:bg-amber-600 disabled:opacity-40 transition-colors"
            >
              {saving ? '…' : 'Przelicz AU'}
            </button>
            <button
              onClick={handleSaveAndClose}
              disabled={saving}
              className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-40 transition-colors"
            >
              {saving ? 'Zapisuję…' : isDirty ? 'Zapisz i zamknij *' : 'Zamknij'}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
