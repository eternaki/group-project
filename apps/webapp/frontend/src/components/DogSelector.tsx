/**
 * DogSelector — przełączanie między psami w klatce.
 *
 * Aktualnie obsługuje jeden wybrany pies (multi-dog planowany w Sprint 11).
 * Komponent przygotowany do rozszerzenia.
 */

interface DogSelectorProps {
  dogCount: number;
  selectedDogIdx: number;
  onSelect: (idx: number) => void;
}

export default function DogSelector({ dogCount, selectedDogIdx, onSelect }: DogSelectorProps) {
  if (dogCount <= 1) {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-500 bg-gray-50 border border-gray-200 rounded px-3 py-1.5">
        <span>🐕</span>
        <span>Собака 1 из {dogCount}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <span className="text-xs text-gray-500 mr-1">Собака:</span>
      {Array.from({ length: dogCount }, (_, i) => (
        <button
          key={i}
          onClick={() => onSelect(i)}
          className={`px-2.5 py-1 text-xs rounded-full font-medium transition-colors ${
            i === selectedDogIdx
              ? 'bg-orange-500 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-orange-100'
          }`}
        >
          🐕 #{i + 1}
        </button>
      ))}
    </div>
  );
}
