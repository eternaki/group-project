/**
 * PeakFrameCard — карточка аннотации одного кадра.
 *
 * Превью: полный кадр с наложенными кейпоинтами и скелетом (canvas).
 * Вкладки: Кейпоинты (инфо), AU (слайдеры), Эмоция, Порода.
 * Полноэкранный редактор — FullEditorModal.
 */

import { useEffect, useRef, useState } from 'react';
import type { FrameAnnotation } from '../types';
import {
  EMOTION_EMOJI,
  EMOTION_NAMES,
  NUM_KEYPOINTS,
  SKELETON_CONNECTIONS,
  getKeypointColor,
} from '../types';
import AnnotationStatusBadge from './AnnotationStatusBadge';
import AUPanel from './AUPanel';
import BreedPicker from './BreedPicker';
import DogSelector from './DogSelector';
import EmotionSelector from './EmotionSelector';
import FullEditorModal from './FullEditorModal';

interface PeakFrameCardProps {
  frame: FrameAnnotation;
}

type TabId = 'keypoints' | 'au' | 'emotion' | 'breed';

const TABS: { id: TabId; label: string }[] = [
  { id: 'keypoints', label: 'Кейпоинты' },
  { id: 'au',        label: 'AU' },
  { id: 'emotion',   label: 'Эмоция' },
  { id: 'breed',     label: 'Порода' },
];

/** Превью кадра: полный кадр с кейпоинтами и скелетом */
function KeypointPreview({ imageUrl, keypoints }: { imageUrl: string; keypoints: number[] | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageUrl;
    img.onload = () => {
      const PREVIEW_W = 480;
      const scale = PREVIEW_W / img.naturalWidth;
      canvas.width = PREVIEW_W;
      canvas.height = Math.round(img.naturalHeight * scale);

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      if (!keypoints) return;

      // Скелет — тонкие белые линии
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
      for (const [a, b] of SKELETON_CONNECTIONS) {
        const ax = keypoints[a * 3] * scale;
        const ay = keypoints[a * 3 + 1] * scale;
        const av = keypoints[a * 3 + 2];
        const bx = keypoints[b * 3] * scale;
        const by = keypoints[b * 3 + 1] * scale;
        const bv = keypoints[b * 3 + 2];
        if (av < 0.3 || bv < 0.3) continue;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }

      // Точки
      for (let i = 0; i < NUM_KEYPOINTS; i++) {
        const x = keypoints[i * 3] * scale;
        const y = keypoints[i * 3 + 1] * scale;
        const v = keypoints[i * 3 + 2];
        if (v < 0.3) continue;

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = getKeypointColor(i);
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    };
  }, [imageUrl, keypoints]);

  return <canvas ref={canvasRef} className="w-full block" />;
}

export default function PeakFrameCard({ frame }: PeakFrameCardProps) {
  const [activeTab, setActiveTab] = useState<TabId>('keypoints');
  const [selectedDogIdx, setSelectedDogIdx] = useState(0);
  const [editorOpen, setEditorOpen] = useState(false);

  const activeAUCount = Object.values(frame.aus).filter((au) => au.is_active).length;
  const visibleKPs = frame.keypoints
    ? frame.keypoints.reduce((acc, v, i) => (i % 3 === 2 && v > 0.3 ? acc + 1 : acc), 0)
    : 0;

  return (
    <>
      <div className="bg-white rounded-xl shadow-md overflow-hidden flex flex-col">
        {/* Превью кадра с кейпоинтами и кнопкой раскрытия */}
        <div
          className="relative bg-gray-900 cursor-pointer group"
          onClick={() => setEditorOpen(true)}
        >
          <KeypointPreview imageUrl={frame.image_url} keypoints={frame.keypoints ?? null} />

          {/* Бейджи */}
          <div className="absolute top-2 left-2">
            <AnnotationStatusBadge status={frame.annotation_status} />
          </div>
          <div className="absolute top-2 right-2 bg-black bg-opacity-60 text-white px-2 py-0.5 rounded text-xs">
            #{frame.frame_idx}
          </div>

          {/* Кнопка раскрытия (видна при наведении) */}
          <div className="absolute top-2 right-11 opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="bg-blue-600/80 text-white rounded px-1.5 py-0.5 text-xs">↗</span>
          </div>

          {/* Эмоция + TFM */}
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent px-3 py-2 flex items-end justify-between">
            <div className="flex items-center gap-1.5">
              {frame.emotion && (
                <>
                  <span className="text-lg">{EMOTION_EMOJI[frame.emotion] ?? '❓'}</span>
                  <span className="text-white text-sm font-semibold">
                    {EMOTION_NAMES[frame.emotion] ?? frame.emotion}
                  </span>
                  {frame.emotion_confidence > 0 && (
                    <span className="text-gray-300 text-xs">
                      {(frame.emotion_confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </>
              )}
            </div>
            <div className="text-gray-300 text-xs">TFM {frame.tfm_score.toFixed(2)}</div>
          </div>
        </div>

        {/* Селектор собаки + счётчик AU */}
        <div className="px-3 py-2 border-b border-gray-100 flex items-center justify-between gap-2 flex-wrap">
          <DogSelector dogCount={1} selectedDogIdx={selectedDogIdx} onSelect={setSelectedDogIdx} />
          <span className="text-xs text-gray-400">
            AU: <strong className="text-amber-600">{activeAUCount}</strong>/21
            {frame.breed && <> · <span className="text-indigo-600">{frame.breed}</span></>}
          </span>
        </div>

        {/* Вкладки */}
        <div className="flex border-b border-gray-200 bg-gray-50">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 py-2 text-xs font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-700 border-b-2 border-blue-500 bg-white'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Содержимое вкладки */}
        <div className="p-3 flex-1 overflow-y-auto max-h-80">
          {activeTab === 'keypoints' && (
            <div className="space-y-2">
              <div className="text-xs text-gray-500">
                Видимых точек: <strong className="text-gray-800">{visibleKPs}</strong> / 46
              </div>
              {frame.keypoints ? (
                <button
                  onClick={() => setEditorOpen(true)}
                  className="w-full py-2 text-sm border border-dashed border-blue-300 text-blue-600 rounded-lg hover:bg-blue-50 transition-colors"
                >
                  ↗ Открыть редактор кейпоинтов
                </button>
              ) : (
                <p className="text-xs text-gray-400">Кейпоинты отсутствуют</p>
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

      {/* Полноэкранный редактор */}
      {editorOpen && (
        <FullEditorModal frame={frame} onClose={() => setEditorOpen(false)} />
      )}
    </>
  );
}
