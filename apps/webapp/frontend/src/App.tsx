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
            Rule-based emotion annotation tool for dog videos
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-300 text-red-800 px-4 py-3 rounded">
            <strong className="font-bold">Error:</strong>
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
            <h2 className="text-xl font-bold mb-3">How it works</h2>
            <ol className="list-decimal list-inside space-y-2 text-gray-700">
              <li>Upload a dog video (20 seconds recommended)</li>
              <li>
                Pipeline will auto-detect neutral frame and select 10 peak expression frames
              </li>
              <li>Review delta Action Units (AU) for each peak frame</li>
              <li>Toggle AU activations manually if needed</li>
              <li>Emotions are classified using rule-based DogFACS poselet matching</li>
              <li>Export annotated dataset to COCO JSON format</li>
            </ol>

            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
              <p className="text-sm text-blue-800">
                <strong>Note:</strong> This tool uses strictly rule-based emotion
                classification (NO machine learning). Emotions are determined by
                official DogFACS Action Unit combinations.
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
            Scientific basis:{' '}
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
