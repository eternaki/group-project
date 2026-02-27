/**
 * AnnotationStatusBadge — wyświetla status anotacji klatki.
 *
 * Stany:
 *   auto     — wygenerowane automatycznie przez AI
 *   reviewed — ręcznie zmienione przez użytkownika
 *   verified — zweryfikowane i zatwierdzone
 */

import type { AnnotationStatus } from '../types';

interface AnnotationStatusBadgeProps {
  status: AnnotationStatus;
}

const STATUS_CONFIG: Record<AnnotationStatus, { label: string; className: string }> = {
  auto:     { label: 'AI auto',     className: 'bg-blue-100 text-blue-700 border border-blue-300' },
  reviewed: { label: 'Reviewed',    className: 'bg-yellow-100 text-yellow-700 border border-yellow-300' },
  verified: { label: '✓ Verified',  className: 'bg-green-100 text-green-700 border border-green-300' },
};

export default function AnnotationStatusBadge({ status }: AnnotationStatusBadgeProps) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.auto;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${config.className}`}>
      {config.label}
    </span>
  );
}
