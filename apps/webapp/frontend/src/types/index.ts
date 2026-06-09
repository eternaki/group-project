/**
 * TypeScript types dla DogFACS Dataset Generator.
 *
 * Oparte na:
 * - DogFLW: 46 facial keypoints
 * - DogFACS: 21 Action Units (Waller et al. 2013)
 * - Mota-Rojas et al. 2021: 9 kategorii emocji
 */

// =============================================================================
// Keypoints (DogFLW — 46 punktów)
// =============================================================================

/** Jeden punkt kluczowy: x, y w pikselach obrazu, visibility 0..1 */
export interface Keypoint {
  x: number;
  y: number;
  visibility: number;
}

/** Nazwy 46 keypoints według schematu DogFLW */
// Kolejność ZGODNA z oficjalnym schematem DogFLW (martvelge/DogFLW, arXiv:2405.11501) —
// to kolejność kanałów wyjściowych modelu keypoints_dogflw.pt.
export const KEYPOINT_NAMES: string[] = [
  // Uszy (0-13)
  'lewe ucho nasada góra', 'prawe ucho nasada góra', 'lewe ucho zgięcie', 'prawe ucho zgięcie',
  'lewe ucho środek góra', 'prawe ucho środek góra', 'lewe ucho czubek', 'prawe ucho czubek',
  'lewe ucho środek dół', 'prawe ucho środek dół', 'lewe ucho dół 2/3', 'prawe ucho dół 2/3',
  'lewe ucho nasada dół', 'prawe ucho nasada dół',
  // Brwi (14-15)
  'lewa brew', 'prawa brew',
  // Oczy (16-23)
  'lewe oko wewn.', 'prawe oko wewn.', 'lewe oko zewn.', 'prawe oko zewn.',
  'lewe oko góra', 'prawe oko góra', 'lewe oko dół', 'prawe oko dół',
  // Nos / pysk (24-37)
  'nos środek (stop)', 'nos góra', 'nos lewa krawędź', 'nos prawa krawędź',
  'pysk (lewa)', 'pysk (prawa)', 'kość policzkowa (lewa)', 'kość policzkowa (prawa)',
  'między nozdrzami', 'nozdrze (lewe)', 'nozdrze (prawe)', 'czubek nosa',
  'poduszka wibrysów (lewa)', 'poduszka wibrysów (prawa)',
  // Pysk / wargi / podbródek / język (38-45)
  'środek górnej wargi', 'kącik ust (lewy)', 'kącik ust (prawy)', 'środek dolnej wargi',
  'podbródek', 'warga lewa (do podbr.)', 'warga prawa (do podbr.)', 'czubek języka',
];

export const NUM_KEYPOINTS = 46;

/**
 * Zwraca kolor CSS dla keypoint według grupy anatomicznej.
 * Odpowiednik get_keypoint_color() z packages/data/schemas.py (konwersja BGR→RGB).
 */
export function getKeypointColor(index: number): string {
  if (index <= 13) return '#ffa500';  // Uszy — pomarańczowy
  if (index <= 15) return '#ff0080';  // Brwi — różowy
  if (index <= 23) return '#00ff00';  // Oczy — zielony
  if (index <= 37) return '#ff3030';  // Nos / pysk — czerwony
  return '#00ffff';                    // Pysk / wargi — cyjan
}

/** Połączenia szkieletu (indeksy keypoints w kolejności DogFLW) */
export const SKELETON_CONNECTIONS: [number, number][] = [
  // Uszy (lewe / prawe — łańcuch wzdłuż konturu)
  [0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12],
  [1, 3], [3, 5], [5, 7], [7, 9], [9, 11], [11, 13],
  // Oczy (pętla: wewn-góra-zewn-dół)
  [16, 20], [20, 18], [18, 22], [22, 16],
  [17, 21], [21, 19], [19, 23], [23, 17],
  // Brwi do oczu
  [14, 16], [15, 17],
  // Nos
  [25, 24], [24, 32], [26, 25], [27, 25], [33, 32], [34, 32], [32, 35],
  // Pysk / wibrysy
  [28, 36], [29, 37], [28, 39], [29, 40],
  // Pysk
  [39, 38], [38, 40], [39, 41], [41, 40], [41, 42],
  [43, 42], [44, 42], [39, 43], [40, 44], [45, 41],
];

// =============================================================================
// Action Units (DogFACS — 21 AU)
// =============================================================================

/** Dane jednego Action Unit (obliczone przez DeltaActionUnitsExtractor) */
export interface DeltaActionUnit {
  ratio: number;       // wartość absolutna względem klatki neutralnej
  delta: number;       // zmiana: (ratio - 1.0), wartość ujemna = mniejszy, dodatnia = większy
  is_active: boolean;  // czy AU jest aktywny (|delta| > threshold)
  confidence: number;  // pewność obliczenia (0-1)
}

/** Oficjalne nazwy 21 AU (DogFACS — Waller et al. 2013) */
export const AU_NAMES: Record<string, string> = {
  // Górna część
  AU101:  'Unoszenie wewnętrznej brwi',
  AU143:  'Napinanie powieki',
  AU145:  'Mruganie / Zamknięcie oczu',
  // Dolna część
  AU109:  'Marszczenie nosa (lewa)',
  AU110:  'Marszczenie nosa (prawa)',
  AU12:   'Unoszenie kącików ust',
  AU116:  'Opuszczanie dolnej wargi',
  AU118:  'Rozciąganie ust',
  AU25:   'Rozchylenie ust',
  AU26:   'Opuszczanie żuchwy',
  AU27:   'Rozciąganie pyska',
  // Deskryptory
  AD19:   'Pokazywanie języka',
  AD33:   'Nadymanie',
  AD35:   'Przygryzanie wargi',
  AD37:   'Oblizywanie wargi',
  AD137:  'Oblizywanie nosa',
  // Uszy
  EAD101: 'Uszy do przodu',
  EAD102: 'Uszy zsunięte',
  EAD103: 'Uszy przyciśnięte',
  EAD104: 'Obrót uszu',
  EAD105: 'Uszy na boki',
};

/** Grupy AU do wyświetlania w panelu */
export const AU_GROUPS: { label: string; codes: string[] }[] = [
  {
    label: 'Górna część',
    codes: ['AU101', 'AU143', 'AU145'],
  },
  {
    label: 'Dolna część',
    codes: ['AU109', 'AU110', 'AU12', 'AU116', 'AU118', 'AU25', 'AU26', 'AU27'],
  },
  {
    label: 'Deskryptory',
    codes: ['AD19', 'AD33', 'AD35', 'AD37', 'AD137'],
  },
  {
    label: 'Uszy',
    codes: ['EAD101', 'EAD102', 'EAD103', 'EAD104', 'EAD105'],
  },
];

// =============================================================================
// Emocje (9 klas — Mota-Rojas et al. 2021)
// =============================================================================

export const EMOTION_CLASSES = [
  'happy', 'sad', 'angry', 'fearful', 'relaxed',
  'neutral', 'surprise', 'pain', 'submission',
] as const;

export type EmotionClass = typeof EMOTION_CLASSES[number];

export const EMOTION_EMOJI: Record<string, string> = {
  happy:      '😊',
  sad:        '😢',
  angry:      '😠',
  fearful:    '😨',
  relaxed:    '😌',
  neutral:    '😐',
  surprise:   '😲',
  pain:       '😣',
  submission: '🐾',
};

/** Polskie nazwy 9 emocji */
export const EMOTION_NAMES: Record<string, string> = {
  happy:      'Szczęśliwy',
  sad:        'Smutny',
  angry:      'Zły',
  fearful:    'Przestraszony',
  relaxed:    'Zrelaksowany',
  neutral:    'Neutralny',
  surprise:   'Zaskoczenie',
  pain:       'Ból',
  submission: 'Uległość',
};

/** Kluczowe AU charakterystyczne dla każdej emocji (do podpowiedzi w UI) */
export const EMOTION_AU_HINTS: Record<string, string> = {
  happy:      'AU12 · AD19 · EAD101',
  sad:        'AU101 · AU116 · EAD103',
  angry:      'AU101 · AU109 · AU110',
  fearful:    'AU101 · AU143 · EAD103',
  relaxed:    'AU25 · AU26 · EAD101',
  neutral:    'brak aktywnych AU',
  surprise:   'AU101 · AU145 · AU27',
  pain:       'AU101 · AU143 · AU116',
  submission: 'AU101 · EAD103 · AD37',
};

// =============================================================================
// Status anotacji
// =============================================================================

export type AnnotationStatus = 'auto' | 'reviewed' | 'verified';

// =============================================================================
// Anotacja klatki (odpowiednik FrameAnnotation z backend session_store.py)
// =============================================================================

/** Anotacja jednej klatki wideo — pełny format z SessionStore */
export interface FrameAnnotation {
  frame_idx: number;
  image_url: string;
  annotation_status: AnnotationStatus;
  source: 'ai' | 'manual';
  bbox: [number, number, number, number] | null;   // [x, y, w, h]
  keypoints: number[] | null;                       // flat 138 = 46×[x, y, visibility]
  aus: Record<string, DeltaActionUnit>;
  emotion: string | null;
  emotion_confidence: number;
  emotion_rule_applied: string | null;
  breed: string | null;
  breed_confidence: number;
  tfm_score: number;
}

/** Dane sesji (odpowiednik SessionData z backend session_store.py) */
export interface SessionData {
  session_id: string;
  video_filename: string;
  created_at: string;
  total_frames: number;
  neutral_frame_idx: number;
  neutral_keypoints: number[] | null;
  frames: FrameAnnotation[];
}

// =============================================================================
// Stare typy (kompatybilność z /api/process_video)
// =============================================================================

/** Peak frame z odpowiedzi process_video (uproszczony format) */
export interface PeakFrame {
  frame_idx: number;
  image_url: string;
  aus: Record<string, DeltaActionUnit>;
  emotion: string;
  emotion_confidence: number;
  emotion_rule_applied: string;
  tfm_score: number;
}

export interface ProcessVideoResponse {
  session_id: string;
  video_filename: string;
  neutral_frame_idx: number;
  neutral_frame_url: string;
  peak_frames: PeakFrame[];
  total_frames: number;
}

export interface ExportCOCORequest {
  peak_frames: PeakFrame[];
  neutral_frame_idx: number;
  video_filename: string;
}

export interface ProcessingSettings {
  fps_sample: number;
  num_peaks: number;
  min_separation_frames: number;
  neutral_mode: 'auto' | 'manual';
  manual_neutral_idx: number | null;
}

export interface ProcessVideoOptions {
  fps_sample?: number;
  num_peaks?: number;
  neutral_idx?: number | null;
  min_separation_frames?: number;
}

// =============================================================================
// Lista ras (120+ popularnych ras psów)
// =============================================================================

export const DOG_BREEDS: string[] = [
  'Affenpinscher', 'Afghan Hound', 'Airedale Terrier', 'Akita', 'Alaskan Malamute',
  'American Bulldog', 'American Eskimo Dog', 'American Foxhound', 'American Pit Bull Terrier',
  'American Staffordshire Terrier', 'Australian Cattle Dog', 'Australian Shepherd',
  'Australian Terrier', 'Basenji', 'Basset Hound', 'Beagle', 'Bearded Collie',
  'Belgian Malinois', 'Belgian Sheepdog', 'Bernese Mountain Dog', 'Bichon Frise',
  'Bloodhound', 'Border Collie', 'Border Terrier', 'Boston Terrier', 'Boxer',
  'Boykin Spaniel', 'Brittany', 'Brussels Griffon', 'Bull Terrier', 'Bullmastiff',
  'Cairn Terrier', 'Cane Corso', 'Cavalier King Charles Spaniel', 'Chesapeake Bay Retriever',
  'Chihuahua', 'Chinese Crested', 'Chinese Shar-Pei', 'Chow Chow', 'Clumber Spaniel',
  'Cocker Spaniel', 'Collie', 'Corgi (Pembroke Welsh)', 'Corgi (Cardigan Welsh)',
  'Dachshund', 'Dalmatian', 'Doberman Pinscher', 'Dogo Argentino', 'English Bulldog',
  'English Setter', 'English Springer Spaniel', 'Field Spaniel', 'Finnish Spitz',
  'Flat-Coated Retriever', 'French Bulldog', 'German Pinscher', 'German Shepherd',
  'German Shorthaired Pointer', 'Giant Schnauzer', 'Golden Retriever', 'Gordon Setter',
  'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain Dog', 'Greyhound',
  'Havanese', 'Irish Setter', 'Irish Terrier', 'Irish Water Spaniel', 'Irish Wolfhound',
  'Italian Greyhound', 'Jack Russell Terrier', 'Japanese Chin', 'Keeshond',
  'Kerry Blue Terrier', 'Komondor', 'Kuvasz', 'Labrador Retriever', 'Leonberger',
  'Lhasa Apso', 'Maltese', 'Manchester Terrier', 'Mastiff', 'Miniature Pinscher',
  'Miniature Schnauzer', 'Mixed Breed', 'Neapolitan Mastiff', 'Newfoundland',
  'Norfolk Terrier', 'Norwegian Elkhound', 'Norwich Terrier', 'Nova Scotia Duck Tolling Retriever',
  'Old English Sheepdog', 'Papillon', 'Pekingese', 'Plott Hound', 'Pointer',
  'Pomeranian', 'Poodle (Miniature)', 'Poodle (Standard)', 'Poodle (Toy)',
  'Portuguese Water Dog', 'Pug', 'Rat Terrier', 'Rhodesian Ridgeback', 'Rottweiler',
  'Saint Bernard', 'Saluki', 'Samoyed', 'Schipperke', 'Scottish Deerhound',
  'Scottish Terrier', 'Shetland Sheepdog', 'Shiba Inu', 'Shih Tzu', 'Siberian Husky',
  'Silky Terrier', 'Skye Terrier', 'Soft Coated Wheaten Terrier', 'Staffordshire Bull Terrier',
  'Standard Schnauzer', 'Sussex Spaniel', 'Tibetan Mastiff', 'Tibetan Spaniel',
  'Tibetan Terrier', 'Toy Fox Terrier', 'Vizsla', 'Weimaraner', 'Welsh Springer Spaniel',
  'Welsh Terrier', 'West Highland White Terrier', 'Whippet', 'Wire Fox Terrier',
  'Wirehaired Pointing Griffon', 'Xoloitzcuintli', 'Yorkshire Terrier',
];
