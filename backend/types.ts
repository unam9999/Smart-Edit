export interface ImageFile {
  id: string;
  file: File;
  name: string;
  size: number;
  preview: string;
  status: 'pending' | 'processing' | 'classified' | 'enhanced' | 'error';
  category?: string;
  confidence?: number;
  editedPreview?: string;
  enhancements?: ImageEnhancements;
  faceDetected?: boolean;
  faceRegions?: FaceRegion[];
}

export interface FaceRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ImageEnhancements {
  brightness: number;
  contrast: number;
  saturation: number;
  sharpness: number;
  warmth: number;
  skinSmoothing: number;
  autoEnhance: boolean;
}

export interface Category {
  id: string;
  name: string;
  icon: string;
  color: string;
  enabled: boolean;
  count: number;
}

export type PageView = 'upload' | 'categories' | 'processing' | 'results' | 'editor';

export const DEFAULT_CATEGORIES: Category[] = [
  { id: 'portrait', name: 'Portraits', icon: '👤', color: '#8B5CF6', enabled: true, count: 0 },
  { id: 'group', name: 'Group Photos', icon: '👥', color: '#EC4899', enabled: true, count: 0 },
  { id: 'landscape', name: 'Landscapes', icon: '🏔️', color: '#10B981', enabled: true, count: 0 },
  { id: 'architecture', name: 'Architecture', icon: '🏛️', color: '#F59E0B', enabled: true, count: 0 },
  { id: 'food', name: 'Food', icon: '🍽️', color: '#EF4444', enabled: true, count: 0 },
  { id: 'animal', name: 'Animals', icon: '🐾', color: '#6366F1', enabled: true, count: 0 },
  { id: 'nature', name: 'Nature', icon: '🌿', color: '#22C55E', enabled: true, count: 0 },
  { id: 'vehicle', name: 'Vehicles', icon: '🚗', color: '#3B82F6', enabled: true, count: 0 },
  { id: 'product', name: 'Products', icon: '📦', color: '#A855F7', enabled: true, count: 0 },
  { id: 'event', name: 'Events', icon: '🎉', color: '#F97316', enabled: true, count: 0 },
  { id: 'selfie', name: 'Selfies', icon: '🤳', color: '#14B8A6', enabled: true, count: 0 },
  { id: 'document', name: 'Documents', icon: '📄', color: '#64748B', enabled: true, count: 0 },
  { id: 'night', name: 'Night Shots', icon: '🌙', color: '#1E293B', enabled: true, count: 0 },
  { id: 'uncertain', name: 'Uncertain', icon: '❓', color: '#94A3B8', enabled: true, count: 0 },
];

export const DEFAULT_ENHANCEMENTS: ImageEnhancements = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  sharpness: 0,
  warmth: 0,
  skinSmoothing: 0,
  autoEnhance: false,
};
