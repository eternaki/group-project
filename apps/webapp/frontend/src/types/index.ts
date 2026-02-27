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
export const KEYPOINT_NAMES: string[] = [
  // Oczy (0-7)
  'left_eye_inner', 'left_eye_top', 'left_eye_outer', 'left_eye_bottom',
  'right_eye_inner', 'right_eye_top', 'right_eye_outer', 'right_eye_bottom',
  // Brwi (8-13)
  'left_brow_inner', 'left_brow_center', 'left_brow_outer',
  'right_brow_inner', 'right_brow_center', 'right_brow_outer',
  // Uszy (14-21)
  'left_ear_base_front', 'left_ear_base_back', 'left_ear_mid', 'left_ear_tip',
  'right_ear_base_front', 'right_ear_base_back', 'right_ear_mid', 'right_ear_tip',
  // Nos (22-25)
  'nose_tip', 'nose_left_wing', 'nose_right_wing', 'nose_bridge',
  // Usta (26-33)
  'mouth_left_corner', 'upper_lip_left', 'upper_lip_center', 'upper_lip_right',
  'mouth_right_corner', 'lower_lip_right', 'lower_lip_center', 'lower_lip_left',
  // Pysk (34-37)
  'muzzle_top', 'muzzle_left', 'muzzle_right', 'chin',
  // Kontur (38-45)
  'forehead_center', 'forehead_left', 'forehead_right',
  'left_cheek_upper', 'left_cheek_lower',
  'right_cheek_upper', 'right_cheek_lower', 'jaw_center',
];

export const NUM_KEYPOINTS = 46;

/**
 * Zwraca kolor CSS dla keypoint według grupy anatomicznej.
 * Odpowiednik get_keypoint_color() z packages/data/schemas.py (konwersja BGR→RGB).
 */
export function getKeypointColor(index: number): string {
  if (index <= 7)  return '#00ff00';  // Oczy — zielony
  if (index <= 13) return '#ff0080';  // Brwi — różowo-fioletowy
  if (index <= 21) return '#00a5ff';  // Uszy — niebieskawym
  if (index <= 25) return '#ff3030';  // Nos — czerwony
  if (index <= 33) return '#00ffff';  // Usta — cyjan
  if (index <= 37) return '#ff00ff';  // Pysk — magenta
  return '#4488ff';                    // Kontur — niebieski
}

/** Połączenia szkieletu do wizualizacji (indeksy keypoints) */
export const SKELETON_CONNECTIONS: [number, number][] = [
  // Oczy
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  // Brwi
  [8, 9], [9, 10], [11, 12], [12, 13],
  // Uszy
  [14, 16], [16, 17], [18, 20], [20, 21],
  // Nos
  [25, 22], [23, 22], [24, 22],
  // Usta — górna warga
  [26, 27], [27, 28], [28, 29], [29, 30],
  // Usta — dolna warga
  [26, 33], [33, 32], [32, 31], [31, 30],
  // Pysk
  [22, 34], [34, 28], [35, 26], [36, 30], [37, 32],
  // Kontur twarzy
  [38, 39], [38, 40], [39, 10], [40, 13],
  [41, 42], [43, 44], [42, 45], [44, 45], [45, 37],
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
  // Górna twarz
  AU101:  'Inner Brow Raiser',
  AU143:  'Lid Tightener',
  AU145:  'Blink / Eye Closure',
  // Dolna twarz
  AU109:  'Nose Wrinkler Left',
  AU110:  'Nose Wrinkler Right',
  AU12:   'Lip Corner Puller',
  AU116:  'Lower Lip Depressor',
  AU118:  'Lip Stretcher',
  AU25:   'Lips Part',
  AU26:   'Jaw Drop',
  AU27:   'Mouth Stretch',
  // Action Descriptors
  AD19:   'Tongue Show',
  AD33:   'Blow',
  AD35:   'Lip Bite',
  AD37:   'Lip Wipe',
  AD137:  'Nose Lick',
  // Ear Action Descriptors
  EAD101: 'Ears Forward',
  EAD102: 'Ears Adductor',
  EAD103: 'Ears Flattener',
  EAD104: 'Ears Rotator',
  EAD105: 'Ears Apart',
};

/** Grupy AU do wyświetlania w panelu */
export const AU_GROUPS: { label: string; codes: string[] }[] = [
  {
    label: 'Górna twarz',
    codes: ['AU101', 'AU143', 'AU145'],
  },
  {
    label: 'Dolna twarz',
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
