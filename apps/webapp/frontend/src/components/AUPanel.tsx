/**
 * AUPanel — panel edycji 21 Action Units (DogFACS).
 *
 * Wyświetla AU w 4 grupach anatomicznych z suwakami intensywności 0-5.
 * Suwak mapuje na wartość delta: intensity/5 → delta (0.0–1.0).
 */

import { useState } from 'react';
import { AU_GROUPS, AU_NAMES, type DeltaActionUnit } from '../types';
import useStore from '../store/useStore';

interface AUPanelProps {
  frameIdx: number;
  aus: Record<string, DeltaActionUnit>;
}

/**
 * Konwertuje wartość delta AU na intensywność FACS (0-5).
 * 0 = brak aktywności, 5 = maksymalna intensywność.
 */
function deltaToIntensity(delta: number): number {
  return Math.min(5, Math.round(Math.abs(delta) * 5));
}

/** Konwertuje intensywność FACS (0-5) na wartość delta. */
function intensityToDelta(intensity: number): number {
  return intensity / 5;
}

interface AURowProps {
  code: string;
  au: DeltaActionUnit | undefined;
  onIntensityChange: (intensity: number) => void;
}

function AURow({ code, au, onIntensityChange }: AURowProps) {
  const intensity = au ? deltaToIntensity(au.delta) : 0;
  const isActive = intensity > 0;

  return (
    <div className={`flex items-center gap-3 p-2 rounded-lg transition-colors ${
      isActive ? 'bg-amber-50 border border-amber-200' : 'bg-gray-50 border border-gray-100'
    }`}>
      {/* Kod AU */}
      <div className="w-16 shrink-0">
        <span className={`text-xs font-mono font-bold ${isActive ? 'text-amber-700' : 'text-gray-500'}`}>
          {code}
        </span>
      </div>

      {/* Nazwa AU */}
      <div className="flex-1 min-w-0">
        <span className="text-xs text-gray-600 truncate block">
          {AU_NAMES[code] ?? code}
        </span>
      </div>

      {/* Suwak intensywności 0-5 */}
      <div className="flex items-center gap-2 shrink-0">
        <input
          type="range"
          min={0}
          max={5}
          step={1}
          value={intensity}
          onChange={(e) => onIntensityChange(Number(e.target.value))}
          className="w-20 h-1.5 accent-amber-500 cursor-pointer"
        />
        <span className={`text-xs font-bold w-4 text-center ${isActive ? 'text-amber-700' : 'text-gray-400'}`}>
          {intensity}
        </span>
      </div>
    </div>
  );
}

export default function AUPanel({ frameIdx, aus }: AUPanelProps) {
  const { updateFrameAUs, recomputeFrameEmotion, saving } = useStore();
  const [localAUs, setLocalAUs] = useState<Record<string, DeltaActionUnit>>(aus);

  const handleIntensityChange = async (code: string, intensity: number) => {
    const delta = intensityToDelta(intensity);
    const current = localAUs[code] ?? { ratio: 1.0, delta: 0, is_active: false, confidence: 0 };
    const updated: DeltaActionUnit = {
      ...current,
      delta,
      ratio: 1.0 + delta,
      is_active: intensity > 0,
    };

    const newAUs = { ...localAUs, [code]: updated };
    setLocalAUs(newAUs);
    await updateFrameAUs(frameIdx, newAUs);
  };

  const handleRecomputeEmotion = () => recomputeFrameEmotion(frameIdx);

  const activeCount = Object.values(localAUs).filter((au) => au.is_active).length;

  return (
    <div className="space-y-4">
      {/* Nagłówek z licznikiem aktywnych AU */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">
          Активных AU: <strong className="text-amber-700">{activeCount}</strong> / 21
        </span>
        <button
          onClick={handleRecomputeEmotion}
          disabled={saving}
          className="text-xs px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 transition-colors"
        >
          {saving ? 'Сохранение…' : 'Пересчитать эмоцию →'}
        </button>
      </div>

      {/* Grupy AU */}
      {AU_GROUPS.map((group) => (
        <div key={group.label}>
          <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            {group.label}
          </h5>
          <div className="space-y-1">
            {group.codes.map((code) => (
              <AURow
                key={code}
                code={code}
                au={localAUs[code]}
                onIntensityChange={(intensity) => handleIntensityChange(code, intensity)}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
