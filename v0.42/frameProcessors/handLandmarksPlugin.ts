// frameProcessors/handLandmarksPlugin.ts
import { VisionCameraProxy, Frame } from 'react-native-vision-camera';

export interface Landmark {
  x: number; // normalisé [0,1]
  y: number; // normalisé [0,1]
}

export type Hand = Landmark[];

// Initialisation du plugin natif pour la détection des mains
// Ce plugin utilise les capacités de vision par ordinateur de l'appareil
let plugin: any = null;

try {
  plugin = VisionCameraProxy.initFrameProcessorPlugin('handLandmarks', {});
  console.log(
    '[HandLandmarksPlugin] plugin initialisé ?',
    plugin != null ? '✅ OK' : '❌ null',
  );
} catch (error) {
  console.warn(
    '[HandLandmarksPlugin] Impossible d’initialiser le plugin natif handLandmarks :',
    error,
  );
}

// Fonction principale de détection des mains utilisée dans le worklet React Native Reanimated
// Prend un frame de la caméra et retourne un tableau de mains détectées
export function detectHands(_frame: Frame): Hand[] {
  'worklet';

  // MODE MOCK: Si le plugin natif n'est pas disponible (sur simulateur ou erreur d'initialisation)
  // On utilise des données fictives pour tester l'interface et le rendu graphique
  if (plugin == null) {
    // Création d'une main stylisée positionnée au centre de l'écran
    // Les coordonnées sont normalisées [0,1] par rapport à la taille du frame
    // Structure des 21 points de repère de la main (format MediaPipe Hand Landmarks)
    // Chaque doigt a 4 points (extrémité, milieu, base, articulation)
    const mockHand: Hand = [
      // 0: poignet (point central de référence)
      { x: 0.5, y: 0.8 },
      // pouce (points 1-4): de la base au bout du pouce
      { x: 0.45, y: 0.75 },
      { x: 0.40, y: 0.7 },
      { x: 0.37, y: 0.65 },
      { x: 0.35, y: 0.62 },
      // index (points 5-8): de la base au bout de l'index
      { x: 0.52, y: 0.75 },
      { x: 0.53, y: 0.65 },
      { x: 0.54, y: 0.55 },
      { x: 0.55, y: 0.5 },
      // majeur (points 9-12): de la base au bout du majeur
      { x: 0.56, y: 0.76 },
      { x: 0.57, y: 0.64 },
      { x: 0.585, y: 0.52 },
      { x: 0.595, y: 0.47 },
      // annulaire (points 13-16): de la base au bout de l'annulaire
      { x: 0.60, y: 0.78 },
      { x: 0.615, y: 0.67 },
      { x: 0.63, y: 0.57 },
      { x: 0.64, y: 0.52 },
      // auriculaire (points 17-20): de la base au bout de l'auriculaire
      { x: 0.63, y: 0.80 },
      { x: 0.645, y: 0.71 },
      { x: 0.66, y: 0.62 },
      { x: 0.67, y: 0.57 },
    ];

    // Retourne un tableau contenant une seule main mock pour les tests
    return [mockHand];
  }

  // MODE RÉEL: Utilisation du plugin natif pour la détection réelle des mains
  // Le plugin analyse le frame vidéo avec des algorithmes de vision par ordinateur
  // et détecte automatiquement les positions des mains présentes dans l'image
  // @ts-ignore – le type du plugin n'est pas connu de TypeScript
  const result = plugin(_frame) as unknown as Hand[];

  // Retourne le tableau des mains détectées, ou un tableau vide si aucune main trouvée
  return result ?? [];
}