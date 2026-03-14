/**
 * EmotionSelector — wybór emocji z 9 klas DogFACS.
 *
 * Wyświetla siatkę emocji z emoji i podpowiedziami AU.
 * Kliknięcie zapisuje emocję przez PATCH /emotion.
 * Sekcja diagnostyczna pokazuje aktywne AU i dlaczego dana emocja została wybrana.
 */

import { useState } from 'react';
import { EMOTION_AU_HINTS, EMOTION_CLASSES, EMOTION_EMOJI, EMOTION_NAMES, type DeltaActionUnit, type EmotionClass } from '../types';
import useStore from '../store/useStore';

/** Kluczowe AU wymagane dla każdej emocji (do diagnostyki) */
const EMOTION_REQUIRED_AUS: Record<string, string[]> = {
  happy:      ['AU12', 'EAD101'],
  sad:        ['EAD103'],
  angry:      ['AU26', 'AU109'],
  fearful:    ['EAD103', 'AU101'],
  relaxed:    [],
  neutral:    [],
  surprise:   ['AU25', 'EAD101', 'AU101'],
  pain:       ['AU143', 'AU101', 'AU116'],
  submission: ['EAD103', 'EAD102'],
};

interface EmotionSelectorProps {
  frameIdx: number;
  currentEmotion: string | null;
  confidence: number;
  ruleApplied: string | null;
  aus: Record<string, DeltaActionUnit>;
}

export default function EmotionSelector({
  frameIdx,
  currentEmotion,
  confidence,
  ruleApplied,
  aus,
}: EmotionSelectorProps) {
  const { updateFrameEmotion, saving } = useStore();
  const [showDiag, setShowDiag] = useState(false);

  const handleSelect = async (emotion: EmotionClass) => {
    if (emotion === currentEmotion) return;
    await updateFrameEmotion(frameIdx, emotion);
  };

  // Lista aktywnych AU
  const activeAUs = Object.entries(aus)
    .filter(([, au]) => au.is_active)
    .map(([name]) => name)
    .sort();

  // Sprawdź czy wymagane AU są aktywne dla danej emocji
  const checkRequiredAUs = (emotion: string): { au: string; active: boolean }[] =>
    (EMOTION_REQUIRED_AUS[emotion] ?? []).map((au) => ({
      au,
      active: Boolean(aus[au]?.is_active),
    }));

  return (
    <div className="space-y-3">
      {/* Aktualnie ustawiona emocja */}
      {currentEmotion && (
        <div className="flex items-center gap-2 p-2 bg-gray-50 rounded-lg border border-gray-200">
          <span className="text-2xl">{EMOTION_EMOJI[currentEmotion]}</span>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-bold text-gray-800">{EMOTION_NAMES[currentEmotion] ?? currentEmotion}</div>
            <div className="text-xs text-gray-500">
              {confidence > 0 ? `Уверенность: ${(confidence * 100).toFixed(0)}%` : ''}
              {ruleApplied && ruleApplied !== 'manual' ? ` · ${ruleApplied}` : ''}
            </div>
          </div>
          <button
            onClick={() => setShowDiag((v) => !v)}
            className="text-[10px] text-blue-500 hover:text-blue-700 whitespace-nowrap"
            title="Показать диагностику AU"
          >
            {showDiag ? 'скрыть' : 'почему?'}
          </button>
        </div>
      )}

      {/* Diagnostyka AU */}
      {showDiag && (
        <div className="p-2 bg-amber-50 border border-amber-200 rounded-lg space-y-2 text-[11px]">
          {/* Aktywne AU */}
          <div>
            <span className="font-semibold text-gray-700">Активные AU ({activeAUs.length}/21): </span>
            {activeAUs.length > 0
              ? activeAUs.map((name) => (
                  <span key={name} className="inline-block bg-green-100 text-green-800 rounded px-1 mr-1 mb-0.5">
                    {name}
                  </span>
                ))
              : <span className="text-gray-400">нет</span>
            }
          </div>

          {/* Требования к текущей эмоции */}
          {currentEmotion && currentEmotion !== 'neutral' && (
            <div>
              <span className="font-semibold text-gray-700">
                Нужно для «{EMOTION_NAMES[currentEmotion] ?? currentEmotion}»:{' '}
              </span>
              {checkRequiredAUs(currentEmotion).length === 0
                ? <span className="text-gray-400">нет требований (всегда активна)</span>
                : checkRequiredAUs(currentEmotion).map(({ au, active }) => (
                    <span
                      key={au}
                      className={`inline-block rounded px-1 mr-1 mb-0.5 ${
                        active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {active ? '✓' : '✗'} {au}
                    </span>
                  ))
              }
            </div>
          )}

          {/* Если neutral — объясняем */}
          {currentEmotion === 'neutral' && (
            <div className="text-gray-600">
              Ни одно правило не совпало — «neutral» это fallback.
              <br />Проверьте: точность кейпоинтов и корректность нейтрального кадра.
            </div>
          )}
        </div>
      )}

      {/* Siatka emocji */}
      <div className="grid grid-cols-3 gap-2">
        {EMOTION_CLASSES.map((emotion) => {
          const isSelected = emotion === currentEmotion;
          return (
            <button
              key={emotion}
              onClick={() => handleSelect(emotion)}
              disabled={saving}
              title={EMOTION_AU_HINTS[emotion]}
              className={`flex flex-col items-center p-2 rounded-lg border transition-all text-center ${
                isSelected
                  ? 'bg-purple-100 border-purple-400 ring-2 ring-purple-300'
                  : 'bg-white border-gray-200 hover:bg-purple-50 hover:border-purple-300'
              } disabled:opacity-50`}
            >
              <span className="text-xl mb-1">{EMOTION_EMOJI[emotion]}</span>
              <span className="text-xs font-medium leading-tight">{EMOTION_NAMES[emotion] ?? emotion}</span>
              <span className="text-[10px] text-gray-400 leading-tight mt-0.5">
                {EMOTION_AU_HINTS[emotion]}
              </span>
            </button>
          );
        })}
      </div>

      {saving && (
        <p className="text-xs text-center text-gray-400">Сохранение…</p>
      )}
    </div>
  );
}
