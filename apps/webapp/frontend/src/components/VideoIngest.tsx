/**
 * VideoIngest — добавление видео в общую папку и фоновая обработка.
 *
 * Обработка идёт ОТДЕЛЬНЫМ процессом, а не в бэкенде: один ролик это около
 * семидесяти секунд, и запуск внутри сервера остановил бы разметку на весь
 * прогон — ровно то, чему эта функция должна мешать. Поэтому страницу можно
 * закрыть, обработка продолжится.
 *
 * Делить материал между людьми не нужно: папка одна, а владелец ролика
 * считается из хэша его имени. Досыпка не трогает то, что уже в работе.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { getIngestStatus, startIngest, uploadVideos, type IngestStatus } from '../utils/api';

/** Как часто опрашиваем состояние обработки, мс */
const POLL_INTERVAL_MS = 4000;

/** Этапы, на которых работа реально идёт */
const ACTIVE_STAGES = new Set(['starting', 'processing', 'curating']);

export default function VideoIngest() {
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploadedBytes, setUploadedBytes] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const filesRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);

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
      // Из папки приходит всё подряд, включая обложки и субтитры — фильтруем
      // на месте, чтобы не гонять по сети то, что бэкенд и так отвергнет.
      const list = Array.from(files).filter((file) =>
        /\.(mp4|webm|mkv|avi|mov)$/i.test(file.name)
      );
      if (!list.length) {
        setError('В выбранном нет ни одного видео поддерживаемого формата');
        return;
      }
      setBusy(true);
      setError(null);
      setMessage(null);
      setUploadedBytes(0);
      try {
        const result = await uploadVideos(list, setUploadedBytes);
        const skipped = result.skipped.length ? `, пропущено ${result.skipped.length}` : '';
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
      setMessage('Обработка запущена — страницу можно закрыть');
      refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Не удалось запустить обработку');
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  const pending = status ? status.videos_total - status.videos_processed : 0;
  const done = status && status.videos_total > 0
    ? (100 * status.videos_processed) / status.videos_total
    : 0;
  const active = status ? ACTIVE_STAGES.has(status.stage) : false;

  return (
    <div className="max-w-2xl mx-auto mt-8 space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <h2 className="text-lg font-bold text-gray-800">Загрузка видео</h2>
        <p className="text-sm text-gray-500 mt-1">
          Папка общая для всех. Кто какой ролик размечает, определяется автоматически
          по имени файла, поэтому досыпка не сбивает уже начатую работу.
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
          className={`mt-4 p-8 rounded-lg border-2 border-dashed text-center transition-colors ${
            dragging ? 'border-amber-500 bg-amber-50' : 'border-gray-300'
          }`}
        >
          <p className="text-sm text-gray-600">Перетащи сюда видео или целую папку</p>
          <p className="text-[11px] text-gray-400 mt-1">mp4 · webm · mkv · avi · mov</p>
          <div className="flex gap-2 justify-center mt-3">
            <button
              onClick={() => filesRef.current?.click()}
              disabled={busy}
              className="px-3 py-1.5 text-xs rounded-md border border-gray-300 hover:border-amber-400 disabled:opacity-50"
            >
              Выбрать файлы
            </button>
            <button
              onClick={() => folderRef.current?.click()}
              disabled={busy}
              className="px-3 py-1.5 text-xs rounded-md border border-gray-300 hover:border-amber-400 disabled:opacity-50"
            >
              Выбрать папку
            </button>
          </div>
          <input
            ref={filesRef}
            type="file"
            multiple
            accept="video/*"
            className="hidden"
            onChange={(event) => {
              if (event.target.files) void send(event.target.files);
              event.target.value = '';
            }}
          />
          {/* webkitdirectory отдаёт всю папку рекурсивно — так можно закинуть
              скачанный архив с собаками одним действием */}
          <input
            ref={folderRef}
            type="file"
            multiple
            className="hidden"
            // @ts-expect-error нестандартные, но поддерживаемые всеми браузерами атрибуты
            webkitdirectory=""
            directory=""
            onChange={(event) => {
              if (event.target.files) void send(event.target.files);
              event.target.value = '';
            }}
          />
        </div>

        {busy && uploadedBytes > 0 && (
          <p className="mt-3 text-xs text-gray-500">
            Загружено {(uploadedBytes / 1024 / 1024).toFixed(0)} МБ…
          </p>
        )}
        {message && <p className="mt-3 text-sm text-green-700">{message}</p>}
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <h3 className="text-sm font-bold text-gray-800">Обработка</h3>

        {status && (
          <>
            <div className="flex items-baseline justify-between mt-3 text-sm text-gray-600">
              <span>
                В папке <strong>{status.videos_total}</strong> · обработано{' '}
                <strong>{status.videos_processed}</strong> · ждёт{' '}
                <strong className="text-amber-700">{pending}</strong>
              </span>
              <span className={active ? 'text-blue-700 text-xs' : 'text-gray-400 text-xs'}>
                {status.stage_label}
              </span>
            </div>

            <div className="h-2 bg-gray-200 rounded-full mt-2 overflow-hidden">
              <div
                className={`h-full transition-all ${active ? 'bg-blue-500' : 'bg-green-500'}`}
                style={{ width: `${done}%` }}
              />
            </div>

            {status.pairs_ready > 0 && (
              <p className="mt-2 text-sm text-green-700">
                Готово к разметке: {status.pairs_ready} пар — появятся в списке наборов
              </p>
            )}
            {status.stage === 'failed' && (
              <p className="mt-2 text-sm text-red-600">
                Обработка прервалась. Нажми «Продолжить обработку» — она начнёт
                с того ролика, на котором остановилась.
              </p>
            )}
          </>
        )}

        <button
          onClick={() => void launch()}
          disabled={busy || active || pending === 0}
          className="mt-4 w-full py-2.5 bg-amber-600 text-white rounded-lg text-sm font-semibold hover:bg-amber-700 disabled:opacity-40"
        >
          {active
            ? 'Обработка идёт…'
            : status?.videos_processed
              ? `Продолжить обработку (${pending})`
              : `Добавить в обработку (${pending})`}
        </button>

        <div className="mt-4 text-xs text-gray-500 space-y-1 border-t border-gray-100 pt-3">
          <p>
            <strong>Страницу можно закрыть.</strong> Обработка идёт отдельным процессом
            и не зависит ни от браузера, ни от вкладки.
          </p>
          <p>
            <strong>Выключение компьютера</strong> обработку прервёт, но не обнулит:
            уже обработанные ролики запомнены, кнопка продолжит с места остановки.
          </p>
          <p>
            Можно и просто скопировать файлы в папку <code>data/videos/DOGS</code> —
            кнопка увидит их так же, как загруженные через браузер.
          </p>
        </div>
      </div>
    </div>
  );
}
