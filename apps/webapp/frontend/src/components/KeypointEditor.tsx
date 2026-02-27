/**
 * KeypointEditor — canvas editor 46 punktów kluczowych DogFLW.
 *
 * Funkcje:
 * - Rysuje obraz psa z nałożonymi punktami i szkieletem
 * - Drag&drop dla przesuwania punktów (lewy przycisk myszy)
 * - Prawy przycisk myszy — toggle widoczności punktu
 * - Przyciski: "Zapisz keypoints" → PATCH /keypoints
 *              "Przelicz AU" → POST /recompute_aus
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  KEYPOINT_NAMES,
  NUM_KEYPOINTS,
  SKELETON_CONNECTIONS,
  getKeypointColor,
} from '../types';
import useStore from '../store/useStore';

interface KeypointEditorProps {
  frameIdx: number;
  imageUrl: string;
  keypoints: number[] | null;
}

const CANVAS_MAX_WIDTH = 520;
const POINT_RADIUS = 5;
const HIT_RADIUS = 10;

/** Парсит flat array [x0,y0,v0,...] в массив {x,y,v}[46] */
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

export default function KeypointEditor({ frameIdx, imageUrl, keypoints }: KeypointEditorProps) {
  const { updateFrameKeypoints, recomputeFrameAUs, saving } = useStore();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const scaleRef = useRef({ x: 1, y: 1, offsetX: 0, offsetY: 0 });

  // Lokalna kopia punktów do edycji
  const [localKPs, setLocalKPs] = useState<{ x: number; y: number; v: number }[]>(() =>
    keypoints ? parseKeypoints(keypoints) : []
  );
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  // Synchronizuj props → local state gdy zmienia się frame
  useEffect(() => {
    setLocalKPs(keypoints ? parseKeypoints(keypoints) : []);
    setIsDirty(false);
  }, [keypoints, frameIdx]);

  // Rysowanie canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Obraz
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    if (localKPs.length === 0) return;

    const { x: sx, y: sy, offsetX, offsetY } = scaleRef.current;

    // Szkielet
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
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

      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }, [localKPs]);

  // Załaduj obraz i ustaw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageUrl;
    img.onload = () => {
      imageRef.current = img;

      // Skaluj canvas do max szerokości zachowując proporcje
      const scale = Math.min(CANVAS_MAX_WIDTH / img.width, 1);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);

      scaleRef.current = { x: scale, y: scale, offsetX: 0, offsetY: 0 };
      draw();
    };
  }, [imageUrl, draw]);

  // Przerysuj przy zmianie punktów
  useEffect(() => { draw(); }, [draw]);

  // Pomocnicza: znajdź indeks punktu w pobliżu (x,y) canvas
  const findHitPoint = useCallback(
    (cx: number, cy: number): number => {
      const { x: sx, y: sy, offsetX, offsetY } = scaleRef.current;
      for (let i = 0; i < localKPs.length; i++) {
        const kp = localKPs[i];
        const kx = kp.x * sx + offsetX;
        const ky = kp.y * sy + offsetY;
        const dist = Math.hypot(cx - kx, cy - ky);
        if (dist <= HIT_RADIUS) return i;
      }
      return -1;
    },
    [localKPs]
  );

  // Konwersja pozycji canvas → współrzędne obrazu
  const canvasToImage = useCallback((cx: number, cy: number) => {
    const { x: sx, y: sy, offsetX, offsetY } = scaleRef.current;
    return { x: (cx - offsetX) / sx, y: (cy - offsetY) / sy };
  }, []);

  // Pobierz pozycję myszy względem canvas
  const getCanvasPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { cx: e.clientX - rect.left, cy: e.clientY - rect.top };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return; // tylko lewy przycisk
    const { cx, cy } = getCanvasPos(e);
    const idx = findHitPoint(cx, cy);
    if (idx !== -1) setDraggingIdx(idx);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingIdx === null) return;
    const { cx, cy } = getCanvasPos(e);
    const { x, y } = canvasToImage(cx, cy);
    setLocalKPs((prev) => {
      const next = [...prev];
      next[draggingIdx] = { ...next[draggingIdx], x, y, v: 1.0 };
      return next;
    });
    setIsDirty(true);
  };

  const handleMouseUp = () => setDraggingIdx(null);

  // Prawy przycisk — toggle widoczności
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
    await updateFrameKeypoints(frameIdx, flattenKeypoints(localKPs));
    setIsDirty(false);
  };

  const handleRecompute = async () => {
    if (isDirty) await handleSave();
    await recomputeFrameAUs(frameIdx);
  };

  return (
    <div className="space-y-3">
      {/* Canvas */}
      <div className="relative border border-gray-200 rounded-lg overflow-hidden bg-gray-900">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onContextMenu={handleContextMenu}
          className="block w-full cursor-crosshair"
          style={{ imageRendering: 'pixelated' }}
        />
        {localKPs.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">
            Brak keypoints dla tej klatki
          </div>
        )}
      </div>

      {/* Legenda */}
      <div className="flex flex-wrap gap-2 text-[10px] text-gray-500">
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
        <span className="text-gray-400">· PPM = toggle widoczność</span>
      </div>

      {/* Przyciski */}
      <div className="flex gap-2">
        <button
          onClick={handleSave}
          disabled={!isDirty || saving || localKPs.length === 0}
          className="flex-1 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-40 transition-colors"
        >
          {saving ? 'Zapisuję…' : isDirty ? 'Zapisz keypoints *' : 'Zapisano'}
        </button>
        <button
          onClick={handleRecompute}
          disabled={saving || localKPs.length === 0}
          className="flex-1 py-1.5 text-sm bg-amber-500 text-white rounded hover:bg-amber-600 disabled:opacity-40 transition-colors"
        >
          {saving ? '…' : 'Przelicz AU'}
        </button>
      </div>

      {/* Info o liczbie punktów */}
      {localKPs.length > 0 && (
        <p className="text-[10px] text-gray-400 text-center">
          Widocznych: {localKPs.filter((k) => k.v > 0.3).length} / {NUM_KEYPOINTS} · {KEYPOINT_NAMES.length} nazw
        </p>
      )}
    </div>
  );
}
