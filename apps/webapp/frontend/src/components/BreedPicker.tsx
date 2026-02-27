/**
 * BreedPicker — wyszukiwanie i wybór rasy psa.
 *
 * Filtruje listę 120+ ras na podstawie wpisanego tekstu.
 * Zapisuje wybraną rasę przez PATCH /breed.
 */

import { useState } from 'react';
import { DOG_BREEDS } from '../types';
import useStore from '../store/useStore';

interface BreedPickerProps {
  frameIdx: number;
  currentBreed: string | null;
  confidence: number;
}

export default function BreedPicker({ frameIdx, currentBreed, confidence }: BreedPickerProps) {
  const { updateFrameBreed, saving } = useStore();
  const [query, setQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);

  const filtered = query.length > 0
    ? DOG_BREEDS.filter((b) => b.toLowerCase().includes(query.toLowerCase())).slice(0, 10)
    : [];

  const handleSelect = async (breed: string) => {
    setQuery('');
    setShowDropdown(false);
    if (breed !== currentBreed) {
      await updateFrameBreed(frameIdx, breed);
    }
  };

  return (
    <div className="space-y-3">
      {/* Aktualna rasa */}
      {currentBreed && (
        <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-indigo-800">{currentBreed}</p>
              {confidence > 0 && (
                <p className="text-xs text-indigo-500">
                  Pewność: {(confidence * 100).toFixed(0)}%
                </p>
              )}
            </div>
            <button
              onClick={() => updateFrameBreed(frameIdx, '')}
              className="text-xs text-indigo-400 hover:text-indigo-600"
            >
              ✕ wyczyść
            </button>
          </div>
        </div>
      )}

      {/* Wyszukiwarka */}
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setShowDropdown(true); }}
          onFocus={() => setShowDropdown(true)}
          onBlur={() => setTimeout(() => setShowDropdown(false), 150)}
          placeholder="Wyszukaj rasę psa…"
          disabled={saving}
          className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg
                     focus:outline-none focus:ring-2 focus:ring-indigo-300 disabled:opacity-50"
        />

        {/* Dropdown z wynikami */}
        {showDropdown && filtered.length > 0 && (
          <ul className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-48 overflow-y-auto">
            {filtered.map((breed) => (
              <li key={breed}>
                <button
                  onClick={() => handleSelect(breed)}
                  className={`w-full text-left px-3 py-2 text-sm hover:bg-indigo-50 transition-colors ${
                    breed === currentBreed ? 'font-semibold text-indigo-700' : 'text-gray-700'
                  }`}
                >
                  {breed}
                </button>
              </li>
            ))}
          </ul>
        )}

        {showDropdown && query.length > 0 && filtered.length === 0 && (
          <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow p-3 text-sm text-gray-400">
            Nie znaleziono rasy „{query}"
          </div>
        )}
      </div>

      <p className="text-xs text-gray-400 text-center">
        {DOG_BREEDS.length} ras w bazie
      </p>
    </div>
  );
}
