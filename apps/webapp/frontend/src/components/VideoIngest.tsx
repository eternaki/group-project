/**
 * VideoIngest — добавление новых видео в общую папку и фоновая обработка.
 *
 * Обработка идёт ОТДЕЛЬНЫМ процессом, а не в бэкенде: один ролик это около
 * семидесяти секунд, и запуск внутри сервера остановил бы разметку на весь
 * прогон — ровно то, чему эта функция должна мешать.
 *
 * Делить материал между людьми не нужно: папка одна, а владелец ролика
 * считается из хэша его имени. Поэтому досыпка не трогает то, что уже в работе.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { getIngestStatus, startIngest, uploadVideos, type IngestStatus } from '../utils/api';

/** Как часто опрашиваем состояние обработки, мс */
const POLL_INTERVAL_MS = 5000;

export default function VideoIngest() {
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    getIngestStatus().then(setStatus).catch(() => undefined);
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const send = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      if (!list.length) return;
      setBusy(true);
      setError(null);
      setMessage(null);
      try {
        const result = await uploadVideos(list);
        const skipped = result.skipped.length
          ? `, пропущено ${result.skipped.length} (${result.skipped[0].reason})`
          : '';
        setMessage(`Добавлено ${result.saved.length} видео${skipped}`);
        refresh();
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Не удалось загрузить видео');
      } finally {
        setBusy(false);
      }
    },
    [refresh]
  );

  const launch = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      await startIngest();
      setMessage('Обработка запущена — можно продолжать разметку');
      refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Не удалось запустить обработку');
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  const pending = status ? status.videos_total - status.videos_processed : 0;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-sm font-bold text-gray-800">Добавить видео</h3>
      <p className="text-xs text-gray-500 mt-0.5">
        Папка общая. Кто какой ролик размечает, определяется автоматически, поэтому
        досыпка не сбивает уже начатую работу.
      </p>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void send(event.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`mt-3 p-6 rounded-lg border-2 border-dashed text-center cursor-pointer transition-colors ${
          dragging ? 'border-amber-500 bg-amber-50' : 'border-gray-300 hover:border-amber-400'
        }`}
      >
        <p className="text-sm text-gray-600">
          Перетащи сюда видео или нажми, чтобы выбрать
        </p>
        <p className="text-[11px] text-gray-400 mt-1">mp4 · webm · mkv · avi · mov</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="video/*"
          className="hidden"
          onChange={(event) => {
            if (event.target.files) void send(event.target.files);
            event.target.value = '';
          }}
        />
      </div>

      {status && (
        <div className="mt-3 text-xs text-gray-600 space-y-1">
          <div>
            В папке <strong>{status.videos_total}</strong> · обработано{' '}
            <strong>{status.videos_processed}</strong> · ждёт{' '}
            <strong className="text-amber-700">{pending}</strong>
          </div>
          {status.pairs_ready > 0 && (
            <div className="text-green-700">
              Готово к разметке: {status.pairs_ready} пар
            </div>
          )}
          {status.running && (
            <div className="text-blue-700">Обработка идёт в фоне — разметке не мешает</div>
          )}
        </div>
      )}

      <button
        onClick={() => void launch()}
        disabled={busy || !status || status.running || pending === 0}
        className="mt-3 w-full py-2 bg-amber-600 text-white rounded-lg text-sm font-semibold hover:bg-amber-700 disabled:opacity-40"
      >
        {status?.running ? 'Обработка идёт…' : `Добавить в обработку (${pending})`}
      </button>

      {message && <p className="mt-2 text-xs text-green-700">{message}</p>}
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
    </div>
  );
}
