import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  useWindowDimensions,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  runAtTargetFps,
} from 'react-native-vision-camera';
import type { FrameAnalysisResult } from './frameProcessors/useEntropyFrameProcessor';
import { useEntropyFrameProcessor } from './frameProcessors/useEntropyFrameProcessor';
import { loadHandDetectorModel, loadHandLandmarksModel } from './ml/handModels';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRunOnJS } from 'react-native-worklets-core';

type HandPoint = { x: number; y: number };
type HandLandmarksPayload = {
  landmarks: Array<Array<HandPoint>>;
  frameWidth: number;
  frameHeight: number;
};

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4], // pouce
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8], // index
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12], // majeur
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16], // annulaire
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20], // auriculaire
  [5, 9],
  [9, 13],
  [13, 17], // paume
];

const LANDMARK_LINE_THICKNESS = 2;

export default function App() {
  const ultraWideDevice = useCameraDevice('back', {
    physicalDevices: ['ultra-wide-angle-camera'],
  });
  const wideDevice = useCameraDevice('back', {
    physicalDevices: ['wide-angle-camera'],
  });
  const fallbackDevice = useCameraDevice('back');
  const windowDimensions = useWindowDimensions();
  const previewSize = React.useMemo(
    () => Math.min(windowDimensions.width, windowDimensions.height),
    [windowDimensions.width, windowDimensions.height],
  );
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);

  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  const [detectionCount, setDetectionCount] = useState(0);
  const [lastFrameAnalysis, setLastFrameAnalysis] =
    useState<FrameAnalysisResult | null>(null);
  const [detectedHandsCount, setDetectedHandsCount] = useState(0);
  const [detectedLandmarks, setDetectedLandmarks] = useState<Array<Array<HandPoint>>>([]);
  const [smoothedLandmarks, setSmoothedLandmarks] = useState<Array<Array<HandPoint>>>([]);
  const [useSmoothedLandmarks, setUseSmoothedLandmarks] = useState(true); // Par défaut activé pour stabilité
  const [currentLens, setCurrentLens] = useState<'ultra' | 'wide'>('ultra');
  const frameDimensionsRef = useRef({ width: 0, height: 0 });
  const landmarkHistoryRef = useRef<Array<Array<Array<HandPoint>>>>([]);
  const maxHistorySize = 3;

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    if (currentLens === 'ultra' && !ultraWideDevice && wideDevice) {
      setCurrentLens('wide');
    } else if (currentLens === 'wide' && !wideDevice && ultraWideDevice) {
      setCurrentLens('ultra');
    }
  }, [currentLens, ultraWideDevice, wideDevice]);

  // Debug : chargement “manuel” des modèles pour inspecter les shapes
  useEffect(() => {
    (async () => {
      try {
        const detector = await loadHandDetectorModel();
        const landmarks = await loadHandLandmarksModel();

        console.log('Hand detector model loaded:');
        console.log('  inputs:', detector.inputs);
        console.log('  input[0].shape:', detector.inputs[0].shape);
        console.log('  outputs:', detector.outputs);

        console.log('Hand landmarks model loaded:');
        console.log('  inputs:', landmarks.inputs);
        console.log('  input[0].shape:', landmarks.inputs[0].shape);
        console.log('  outputs:', landmarks.outputs);
      } catch (e) {
        console.error('Error loading TFLite models', e);
        Alert.alert('ML error', 'Impossible de charger les modèles TFLite');
      }
    })();
  }, []);

  const activeDevice = React.useMemo(() => {
    if (currentLens === 'ultra') {
      return ultraWideDevice ?? wideDevice ?? fallbackDevice;
    }
    return wideDevice ?? ultraWideDevice ?? fallbackDevice;
  }, [currentLens, ultraWideDevice, wideDevice, fallbackDevice]);

  const isUltraAvailable = ultraWideDevice != null;
  const isWideAvailable = wideDevice != null;

  const handleFrameAnalyzed = useCallback((result: FrameAnalysisResult) => {
    setDetectionCount((c) => c + 1);
    setLastFrameAnalysis(result);
  }, []);

  // Charger le modèle de DETECTION (main entière) via fast-tflite
  const handDetector = useTensorflowModel(
    require('./assets/models/hand_detector.tflite'),
  );
  const detectorModel =
    handDetector.state === 'loaded' ? handDetector.model : undefined;

  // Charger le modèle de LANDMARKS (points de repère de la main) via fast-tflite
  const handLandmarks = useTensorflowModel(
    require('./assets/models/hand_landmarks_detector.tflite'),
  );
  const landmarkModel =
    handLandmarks.state === 'loaded' ? handLandmarks.model : undefined;

  // On extrait la taille d'entrée du modèle de détection (1, H, W, 3)
  const detectorInputSize = React.useMemo(() => {
    if (handDetector.state !== 'loaded') return null;
    const shape = handDetector.model.inputs[0].shape; // [1, 192, 192, 3]
    const [, height, width] = shape;
    return {
      width,
      height,
      dataType: handDetector.model.inputs[0].dataType,
    };
  }, [handDetector]);

  // On extrait la taille d'entrée du modèle de landmarks (1, H, W, 3)
  const landmarksInputSize = React.useMemo(() => {
    if (handLandmarks.state !== 'loaded') return null;
    const shape = handLandmarks.model.inputs[0].shape; // [1, 224, 224, 3]
    const [, height, width] = shape;
    return {
      width,
      height,
      dataType: handLandmarks.model.inputs[0].dataType,
    };
  }, [handLandmarks]);

  // Plugin de resize (utilisé côté worklet)
  const { resize } = useResizePlugin();

  // Callback JS quand on a compté les mains
  const onHandsDetected = useRunOnJS((numHands: number) => {
    console.log('Hands détectées (heuristique):', numHands);
    setDetectedHandsCount(numHands);
  },
    []
  );

  // Callback JS pour compter les frames traitées (juste pour l’affichage)
  const onFrameProcessed = useRunOnJS(() => {
    setDetectionCount((c) => c + 1);
  }, 
    []
  );

  const handleToggleLensMode = useCallback(() => {
    if (currentLens === 'ultra') {
      if (isWideAvailable) {
        setCurrentLens('wide');
      }
    } else if (isUltraAvailable) {
      setCurrentLens('ultra');
    }
  }, [currentLens, isUltraAvailable, isWideAvailable]);

  const cloneLandmarks = (hands: Array<Array<HandPoint>>): Array<Array<HandPoint>> =>
    hands.map((hand) => hand.map((point) => ({ ...point })));

const hasSignificantMovement = (
  current: Array<Array<HandPoint>>,
  previous: Array<Array<HandPoint>>,
  threshold: number,
): boolean => {
  for (let handIndex = 0; handIndex < current.length; handIndex++) {
    const currentHand = current[handIndex];
    const previousHand = previous[handIndex];

    if (!previousHand || currentHand.length !== previousHand.length) {
      continue;
    }

    for (let i = 0; i < currentHand.length; i++) {
      const dx = currentHand[i].x - previousHand[i].x;
      const dy = currentHand[i].y - previousHand[i].y;
      if (Math.hypot(dx, dy) > threshold) {
        return true;
      }
    }
  }
  return false;
};

  // Fonction pour lisser les landmarks (réduire le bruit avec un buffer circulaire)
  const smoothLandmarks = (current: Array<Array<HandPoint>>): Array<Array<HandPoint>> => {
    if (current.length === 0) {
      landmarkHistoryRef.current = [];
      return current;
    }

    const previousFrame =
      landmarkHistoryRef.current[landmarkHistoryRef.current.length - 1];

    const frameMinSize =
      Math.min(
        frameDimensionsRef.current.width || previewSize || 0,
        frameDimensionsRef.current.height || previewSize || 0,
      ) || 1;
    const movementThreshold = frameMinSize * 0.08; // 8% du plus petit côté

    if (
      previousFrame &&
      hasSignificantMovement(current, previousFrame, movementThreshold)
    ) {
      landmarkHistoryRef.current = [cloneLandmarks(current)];
      return current;
    }

    // Ajouter le frame actuel à l'historique
    landmarkHistoryRef.current.push(cloneLandmarks(current));
    if (landmarkHistoryRef.current.length > maxHistorySize) {
      landmarkHistoryRef.current.shift(); // Garder seulement les N derniers
    }

    if (landmarkHistoryRef.current.length < 2) return current;

    const smoothed: Array<Array<HandPoint>> = current.map((hand, handIndex) =>
      hand.map((point, pointIndex) => {
        let sumX = 0;
        let sumY = 0;
        let count = 0;

        for (const historyFrame of landmarkHistoryRef.current) {
          const historyHand = historyFrame[handIndex];
          if (historyHand && historyHand[pointIndex]) {
            sumX += historyHand[pointIndex].x;
            sumY += historyHand[pointIndex].y;
            count++;
          }
        }

        if (count === 0) {
          return point;
        }

        return {
          x: sumX / count,
          y: sumY / count,
        };
      }),
    );

    return smoothed;
  };

  // Supprimé : la correction d'orientation forcée
  // Le modèle MediaPipe landmarks gère naturellement les différentes orientations

  // Callback JS pour mettre à jour les landmarks détectés
  const onLandmarksDetected = useRunOnJS((payload: HandLandmarksPayload) => {
    frameDimensionsRef.current = {
      width: payload.frameWidth,
      height: payload.frameHeight,
    };

    const smoothed = smoothLandmarks(payload.landmarks);

    setDetectedLandmarks(payload.landmarks);
    setSmoothedLandmarks(smoothed);
  },
  []
  );

  const mapPointToScreen = useCallback(
    (point: HandPoint) => {
      const frameWidth = frameDimensionsRef.current.width;
      const frameHeight = frameDimensionsRef.current.height;
      const displayWidth = previewSize || 1;
      const displayHeight = previewSize || 1;

      if (frameWidth === 0 || frameHeight === 0) {
        return { left: 0, top: 0 };
      }

      const sensorIsLandscape = frameWidth >= frameHeight;
      const displayIsPortrait = displayHeight >= displayWidth;

      let x = point.x;
      let y = point.y;
      let imageWidth = frameWidth;
      let imageHeight = frameHeight;

      if (sensorIsLandscape && displayIsPortrait) {
        const rotatedX = frameHeight - y;
        const rotatedY = x;
        x = rotatedX;
        y = rotatedY;
        imageWidth = frameHeight;
        imageHeight = frameWidth;
      }

      const scale = Math.max(displayWidth / imageWidth, displayHeight / imageHeight);
      const scaledWidth = imageWidth * scale;
      const scaledHeight = imageHeight * scale;
      const offsetX = (scaledWidth - displayWidth) / 2;
      const offsetY = (scaledHeight - displayHeight) / 2;

      let pixelX = x * scale - offsetX;
      let pixelY = y * scale - offsetY;

      if (activeDevice?.position === 'front') {
        pixelX = displayWidth - pixelX;
      }

      return {
        left: Math.max(0, Math.min(displayWidth, pixelX)),
        top: Math.max(0, Math.min(displayHeight, pixelY)),
      };
    },
    [activeDevice?.position, previewSize],
  );

  // Nouveau frame processor pour la détection de mains et landmarks
  const handDetectionFrameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';

      if (detectorModel == null || detectorInputSize == null || landmarkModel == null || landmarksInputSize == null) {
        return;
      }

      const { width: detectorWidth, height: detectorHeight, dataType: detectorDataType } = detectorInputSize;
      const { width: landmarksWidth, height: landmarksHeight, dataType: landmarksDataType } = landmarksInputSize;

      runAtTargetFps(20, () => {

        // 1) Resize la frame au format attendu par le modèle de détection (192x192)
        const resized = resize(frame, {
          scale: {
            width: detectorWidth,
            height: detectorHeight,
          },
          pixelFormat: 'rgb',
          dataType: detectorDataType === 'float32' ? 'float32' : 'uint8',
        });

        // 2) Inference TFLite synchrone du modèle de détection
        const detectorOutputs = detectorModel.runSync([resized]);

        // outputs[0] : [1, 2016, 18] → boîtes englobantes et points de repère de paume
        // outputs[1] : [1, 2016, 1] → scores (logits)
        const rawBoxes = detectorOutputs[0] as Float32Array; // [1, 2016, 18]
        const rawScores = detectorOutputs[1] as Float32Array; // [1, 2016, 1]

        // 3) Appliquer sigmoid aux scores
        const scores = new Float32Array(rawScores.length);
        for (let i = 0; i < rawScores.length; i++) {
          scores[i] = 1 / (1 + Math.exp(-rawScores[i]));
        }

        // 4) Approche simplifiée : utiliser les positions approximatives des détections
        // Pour MediaPipe, on peut estimer la position de la main à partir des scores élevés
        // et faire un crop conservateur autour du centre de la frame
        const highScoreIndices: number[] = [];
        for (let i = 0; i < scores.length; i++) {
          if (scores[i] >= 0.7) {
            highScoreIndices.push(i);
          }
        }

        // Prendre max 1 main avec un score très élevé pour plus de stabilité
        highScoreIndices.sort((a, b) => scores[b] - scores[a]);
        const topIndices = highScoreIndices.slice(0, 1).filter(index => scores[index] >= 0.8); // Seuil plus strict

        if (topIndices.length === 0) {
          onHandsDetected(0);
          onLandmarksDetected({
            landmarks: [],
            frameWidth: frame.width,
            frameHeight: frame.height,
          });
          onFrameProcessed();
          return;
        }

        const allLandmarks: Array<Array<{x: number, y: number}>> = [];

        // 5) Approche simplifiée : traiter seulement la première main détectée
        // Pour l'instant, on fait un crop centré qui devrait contenir la main principale
        if (topIndices.length > 0) {
          const frameWidth = frame.width;
          const frameHeight = frame.height;

          // Crop d'environ 70% de la frame centrée (devrait contenir une main normale)
          const cropSize = Math.min(frameWidth, frameHeight) * 0.7;
          const cropX = Math.max(0, (frameWidth - cropSize) / 2);
          const cropY = Math.max(0, (frameHeight - cropSize) / 2);
          const cropWidth = Math.min(cropSize, frameWidth - cropX);
          const cropHeight = Math.min(cropSize, frameHeight - cropY);

          // 6) Recadrer l'image originale sur la région de la main
          const handPatch = resize(frame, {
            crop: {
              x: cropX,
              y: cropY,
              width: cropWidth,
              height: cropHeight,
            },
            scale: {
              width: landmarksWidth, // 224x224 pour le modèle de landmarks
              height: landmarksHeight,
            },
            pixelFormat: 'rgb',
            dataType: landmarksDataType === 'float32' ? 'float32' : 'uint8',
          });

          // 7) Exécuter le modèle de landmarks sur le patch
          const landmarksOutputs = landmarkModel.runSync([handPatch]);

          // Le modèle retourne généralement [1, 63] pour 21 points (x,y,z)
          const landmarksData = landmarksOutputs[0] as Float32Array;
          const handLandmarks: Array<{x: number, y: number}> = [];

          // 7) Extraire les 21 points de repère (x,y seulement pour l'affichage)
          for (let i = 0; i < 21; i++) {
            const baseIdx = i * 3; // chaque point a x,y,z
            const x = landmarksData[baseIdx] / landmarksWidth; // normalisé 0-1 dans le patch
            const y = landmarksData[baseIdx + 1] / landmarksHeight;

            // Convertir les coordonnées du patch vers l'image caméra originale
            // Le patch est centré dans la frame, donc on translate simplement
            const screenX = cropX + (x * cropWidth);
            const screenY = cropY + (y * cropHeight);

            handLandmarks.push({x: screenX, y: screenY});
          }

          allLandmarks.push(handLandmarks);
        }

        // 8) Mettre à jour les callbacks
        onHandsDetected(Math.min(topIndices.length, 1)); // Pour l'instant limité à 1 main
        onLandmarksDetected({
          landmarks: allLandmarks,
          frameWidth: frame.width,
          frameHeight: frame.height,
        });
        onFrameProcessed();

        // Debug supprimé pour réduire la surcharge
      });
    },
    [detectorModel, detectorInputSize, landmarkModel, landmarksInputSize, resize, onHandsDetected, onLandmarksDetected, onFrameProcessed],
  );

  // Utiliser le nouveau frame processor pour la détection de mains
  const frameProcessor = handDetectionFrameProcessor;

  const isUltraActive = activeDevice === ultraWideDevice && ultraWideDevice != null;
  const zoomLabel = isUltraActive ? 'x0.5' : 'x1';
  const canToggleLens = isUltraAvailable && isWideAvailable;

  const handleTakePhoto = useCallback(async () => {
    if (!cameraRef.current || isTakingPhoto) return;

    try {
      setIsTakingPhoto(true);
      const photo = await cameraRef.current.takePhoto();
      console.log('Photo prise (manual) :', photo);
    } catch (err) {
      console.error('Erreur lors de la prise de photo :', err);
    } finally {
      setIsTakingPhoto(false);
    }
  }, [isTakingPhoto]);

  if (!activeDevice) {
    return (
      <View style={styles.center}>
        <Text>Aucun device caméra trouvé</Text>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text>Permission caméra en attente...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View
        style={[
          styles.previewContainer,
          { width: previewSize, height: previewSize },
        ]}
      >
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
          device={activeDevice}
        isActive={!isTakingPhoto}
        photo
        frameProcessor={frameProcessor}
      />

        {/* Overlay pour afficher les landmarks des mains */}
        <View pointerEvents="none" style={StyleSheet.absoluteFill}>
          {(useSmoothedLandmarks ? smoothedLandmarks : detectedLandmarks).map(
            (handLandmarks, handIndex) => {
              const mappedPoints = handLandmarks.map(mapPointToScreen);
              return (
                <View key={handIndex} style={StyleSheet.absoluteFill}>
                  {HAND_CONNECTIONS.map(([startIdx, endIdx], connectionIndex) => {
                    const start = mappedPoints[startIdx];
                    const end = mappedPoints[endIdx];
                    if (!start || !end) return null;

                    const dx = end.left - start.left;
                    const dy = end.top - start.top;
                    const length = Math.hypot(dx, dy);
                    if (length === 0) return null;

                    const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
                    const centerX = (start.left + end.left) / 2;
                    const centerY = (start.top + end.top) / 2;

                    return (
                      <View
                        key={`line-${handIndex}-${connectionIndex}`}
                        style={[
                          styles.landmarkLine,
                          {
                            width: length,
                            left: centerX - length / 2,
                            top: centerY - LANDMARK_LINE_THICKNESS / 2,
                            transform: [{ rotate: `${angle}deg` }],
                          },
                        ]}
                      />
                    );
                  })}

                  {mappedPoints.map((point, pointIndex) => (
                    <View
                      key={`point-${handIndex}-${pointIndex}`}
                      style={[
                        styles.landmarkPoint,
                        {
                          left: point.left,
                          top: point.top,
                        },
                      ]}
                    />
                  ))}
                </View>
              );
            },
          )}
        </View>
      </View>

      <View style={styles.topOverlay}>
        <Text style={styles.infoText}>Frames traitées: {detectionCount}</Text>
        <Text style={styles.infoText}>Mains détectées: {detectedHandsCount}</Text>
        <TouchableOpacity
          style={styles.toggleButton}
          onPress={() => setUseSmoothedLandmarks(!useSmoothedLandmarks)}
        >
          <Text style={styles.toggleText}>
            {useSmoothedLandmarks ? 'LISSÉ' : 'BRUT'}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.bottomOverlay}>
        <TouchableOpacity
          style={[
            styles.zoomButton,
            !canToggleLens && styles.zoomButtonDisabled,
          ]}
          onPress={handleToggleLensMode}
          disabled={!canToggleLens}
        >
          <Text style={styles.zoomButtonText}>{zoomLabel}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.shutterButton,
            isTakingPhoto && styles.shutterButtonDisabled,
          ]}
          onPress={handleTakePhoto}
          disabled={isTakingPhoto}
        >
          <Text style={styles.shutterText}>
            {isTakingPhoto ? '...' : 'SHOT'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
    alignItems: 'center',
    justifyContent: 'center',
  },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  previewContainer: {
    position: 'relative',
    overflow: 'hidden',
    borderRadius: 16,
    backgroundColor: 'black',
  },
  topOverlay: {
    position: 'absolute',
    top: 60,
    width: '100%',
    alignItems: 'center',
  },
  bottomOverlay: {
    position: 'absolute',
    bottom: 60,
    width: '100%',
    alignItems: 'center',
  },
  infoText: { color: 'white', fontSize: 14, marginBottom: 4 },
  zoomButton: {
    marginBottom: 16,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#ffffff80',
    backgroundColor: '#00000080',
  },
  zoomButtonDisabled: {
    opacity: 0.5,
  },
  zoomButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  shutterButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 4,
    borderColor: 'white',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff20',
  },
  shutterButtonDisabled: { opacity: 0.5 },
  shutterText: { color: 'white', fontWeight: 'bold' },
  landmarkPoint: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'red',
    borderWidth: 2,
    borderColor: 'white',
  },
  landmarkLine: {
    position: 'absolute',
    height: LANDMARK_LINE_THICKNESS,
    backgroundColor: 'white',
    borderRadius: LANDMARK_LINE_THICKNESS / 2,
  },
  toggleButton: {
    backgroundColor: '#ffffff40',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    marginTop: 4,
  },
  toggleText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
});