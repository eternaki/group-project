/**
 * Main App component dla DogFACS Dataset Generator.
 */

import VideoUpload from './components/VideoUpload';
import Timeline from './components/Timeline';
import PeakFramesGrid from './components/PeakFramesGrid';
import ExportPanel from './components/ExportPanel';
import useStore from './store/useStore';

export default function App() {
  const { error, videoData } = useStore();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-primary-700 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">
            🐕 DogFACS Dataset Generator
          </h1>
          <p className="text-primary-100 mt-1">
Narzędzie do anotacji emocji psów oparte na regułach (rule-based)
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-300 text-red-800 px-4 py-3 rounded">
            <strong className="font-bold">Błąd:</strong>
            <span className="ml-2">{error}</span>
          </div>
        )}

        {/* Video Upload */}
        <VideoUpload />

        {/* Results */}
        {videoData && (
          <>
            <Timeline />
            <PeakFramesGrid />
            <ExportPanel />
          </>
        )}

        {/* Instructions */}
        {!videoData && (
          <div className="mt-8 bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold mb-3">Jak to działa</h2>
            <ol className="list-decimal list-inside space-y-2 text-gray-700">
              <li>Wgraj wideo psa (zalecane 20 sekund)</li>
              <li>
                Pipeline automatycznie wykryje klatkę neutralną i wybierze 10 klatek szczytowych ekspresji
              </li>
              <li>Przejrzyj delta Action Units (AU) dla każdej klatki szczytowej</li>
              <li>W razie potrzeby ręcznie przełącz aktywacje AU</li>
              <li>Emocje są klasyfikowane przy użyciu dopasowania poselet DogFACS opartego na regułach</li>
              <li>Wyeksportuj zanotowany dataset do formatu COCO JSON</li>
            </ol>

            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
              <p className="text-sm text-blue-800">
                <strong>Uwaga:</strong> To narzędzie używa wyłącznie klasyfikacji emocji
                opartej na regułach (BEZ uczenia maszynowego). Emocje są ustalane na
                podstawie oficjalnych kombinacji Action Units DogFACS.
              </p>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 border-t mt-12">
        <div className="container mx-auto px-4 py-6 text-center text-gray-600 text-sm">
          <p>DogFACS Dataset Generator v1.0 | Politechnika Gdańska WETI</p>
          <p className="mt-1">
            Podstawa naukowa:{' '}
            <a
              href="https://www.animalfacs.com/dogfacs"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-600 hover:underline"
            >
              DogFACS Manual
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}
