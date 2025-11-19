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
import {
  interpretFrame,
  palmDetectionAnchors,
  prepareDetections,
  type FrameMeta,
  type HandState,
  type Landmark3D,
  type MLRawDetectorOutput,
  type MLRawLandmarksOutput,
  type SceneInterpretation,
} from './vision/interpretation';

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
const MAX_HANDS = 2;
const DETECTION_SCORE_THRESHOLD = 0.55;

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
  const [sceneInterpretation, setSceneInterpretation] =
    useState<SceneInterpretation | null>(null);
  const [useSmoothedLandmarks, setUseSmoothedLandmarks] = useState(true); // Par défaut activé pour stabilité
  const [currentLens, setCurrentLens] = useState<'ultra' | 'wide'>('ultra');
  const frameDimensionsRef = useRef({ width: 0, height: 0 });
  const handHistoryRef = useRef<Record<string, Landmark3D[][]>>({});
  const maxHistorySize = 3;
  const anchors = React.useMemo(() => palmDetectionAnchors, []);

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

  // Callback JS pour compter les frames traitées (juste pour l’affichage)
  const onFrameProcessed = useRunOnJS(() => {
    setDetectionCount((c) => c + 1);
  }, 
    []
  );

  const onSceneUpdated = useRunOnJS((interpretation: SceneInterpretation) => {
    frameDimensionsRef.current = {
      width: interpretation.rawMeta.frameWidth,
      height: interpretation.rawMeta.frameHeight,
    };

    if (interpretation.hands.length === 0) {
      handHistoryRef.current = {};
    }

    setDetectedHandsCount(interpretation.hands.length);
    setSceneInterpretation(interpretation);
  }, []);

  const handleToggleLensMode = useCallback(() => {
    if (currentLens === 'ultra') {
      if (isWideAvailable) {
        setCurrentLens('wide');
      }
    } else if (isUltraAvailable) {
      setCurrentLens('ultra');
    }
  }, [currentLens, isUltraAvailable, isWideAvailable]);

  const hasSignificantMovement3D = React.useCallback(
    (current: Landmark3D[], previous: Landmark3D[], threshold: number) => {
      for (let i = 0; i < current.length; i++) {
        const prev = previous[i];
        if (!prev) continue;
        const dx = current[i].x - prev.x;
        const dy = current[i].y - prev.y;
        if (Math.hypot(dx, dy) > threshold) {
          return true;
        }
      }
      return false;
    },
    [],
  );

  const handsForDisplay = React.useMemo<HandState[]>(() => {
    if (!sceneInterpretation) {
      handHistoryRef.current = {};
      return [];
    }

    const hands = sceneInterpretation.hands;
    if (hands.length === 0) {
      handHistoryRef.current = {};
      return [];
    }

    if (!useSmoothedLandmarks) {
      const nextHistory: Record<string, Landmark3D[][]> = {};
      hands.forEach((hand) => {
        nextHistory[hand.id] = [hand.landmarks.map((point) => ({ ...point }))];
      });
      handHistoryRef.current = nextHistory;
      return hands;
    }

    const frameMinSize =
      Math.min(
        frameDimensionsRef.current.width || previewSize || 0,
        frameDimensionsRef.current.height || previewSize || 0,
      ) || 1;
    const movementThreshold = frameMinSize * 0.08;
    const nextHistory: Record<string, Landmark3D[][]> = { ...handHistoryRef.current };

    const smoothed = hands.map((hand) => {
      const history = nextHistory[hand.id] ?? [];
      if (
        history.length > 0 &&
        hasSignificantMovement3D(hand.landmarks, history[history.length - 1], movementThreshold)
      ) {
        nextHistory[hand.id] = [];
      }
      const updatedHistory = [...(nextHistory[hand.id] ?? [])];
      updatedHistory.push(hand.landmarks.map((point) => ({ ...point })));
      if (updatedHistory.length > maxHistorySize) {
        updatedHistory.shift();
      }
      nextHistory[hand.id] = updatedHistory;

      if (updatedHistory.length < 2) {
        return hand;
      }

      const averaged = hand.landmarks.map((_, index) => {
        let sumX = 0;
        let sumY = 0;
        let sumZ = 0;
        let count = 0;
        for (const framePoints of updatedHistory) {
          const point = framePoints[index];
          if (point) {
            sumX += point.x;
            sumY += point.y;
            sumZ += point.z;
            count++;
          }
        }
        if (count === 0) {
          return hand.landmarks[index];
        }
        return {
          x: sumX / count,
          y: sumY / count,
          z: sumZ / count,
        };
      });

      return {
        ...hand,
        landmarks: averaged,
      };
    });

    const activeIds = new Set(hands.map((hand) => hand.id));
    Object.keys(nextHistory).forEach((id) => {
      if (!activeIds.has(id)) {
        delete nextHistory[id];
      }
    });
    handHistoryRef.current = nextHistory;

    return smoothed;
  }, [sceneInterpretation, useSmoothedLandmarks, previewSize, hasSignificantMovement3D, maxHistorySize]);

  // Supprimé : la correction d'orientation forcée
  // Le modèle MediaPipe landmarks gère naturellement les différentes orientations

  const mapPointToScreen = useCallback(
    (point: { x: number; y: number }) => {
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

      if (
        detectorModel == null ||
        detectorInputSize == null ||
        landmarkModel == null ||
        landmarksInputSize == null
      ) {
        return;
      }

      const { width: detectorWidth, height: detectorHeight, dataType: detectorDataType } =
        detectorInputSize;
      const {
        width: landmarksWidth,
        height: landmarksHeight,
        dataType: landmarksDataType,
      } = landmarksInputSize;

      const frameId =
        typeof frame.timestamp === 'number'
          ? frame.timestamp
          : Date.now();

      const meta: FrameMeta = {
        frameWidth: frame.width,
        frameHeight: frame.height,
        modelInputWidth: detectorWidth,
        modelInputHeight: detectorHeight,
        frameId,
      };

      runAtTargetFps(20, () => {
        const resized = resize(frame, {
          scale: {
            width: detectorWidth,
            height: detectorHeight,
          },
          pixelFormat: 'rgb',
          dataType: detectorDataType === 'float32' ? 'float32' : 'uint8',
        });

        const detectorOutputs = detectorModel.runSync([resized]) as Array<Float32Array>;
        const rawBoxes = detectorOutputs[0] as Float32Array;
        const rawScores = detectorOutputs[1] as Float32Array;

        const detectorOutput: MLRawDetectorOutput = {
          rawBoxes,
          rawScores,
        };

        const prepared = prepareDetections(detectorOutput, anchors, meta, {
          maxHands: MAX_HANDS,
          scoreThreshold: DETECTION_SCORE_THRESHOLD,
        });

        const landmarkResults: MLRawLandmarksOutput[] = [];
        for (let i = 0; i < prepared.roiBoxesOnFrame.length; i++) {
          if (i >= MAX_HANDS) break;
          const roi = prepared.roiBoxesOnFrame[i];

          const cropX = Math.max(0, Math.min(roi.xMin, frame.width - 1));
          const cropY = Math.max(0, Math.min(roi.yMin, frame.height - 1));
          const cropWidth = Math.max(1, Math.min(roi.wPx, frame.width - cropX));
          const cropHeight = Math.max(1, Math.min(roi.hPx, frame.height - cropY));

          const handPatch = resize(frame, {
            crop: {
              x: cropX,
              y: cropY,
              width: cropWidth,
              height: cropHeight,
            },
            scale: {
              width: landmarksWidth,
              height: landmarksHeight,
            },
            pixelFormat: 'rgb',
            dataType: landmarksDataType === 'float32' ? 'float32' : 'uint8',
          });

          const landmarkOutputs = landmarkModel.runSync([handPatch]) as Array<Float32Array>;
          landmarkResults.push(parseLandmarkResult(landmarkOutputs));
        }

        const interpretation = interpretFrame(detectorOutput, landmarkResults, anchors, meta, {
          precomputed: prepared,
          includePerformanceMetrics: true,
        });

        onSceneUpdated(interpretation);
        onFrameProcessed();
      });
    },
    [
      detectorModel,
      detectorInputSize,
      landmarkModel,
      landmarksInputSize,
      resize,
      anchors,
      onSceneUpdated,
      onFrameProcessed,
    ],
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
          {handsForDisplay.map((hand) => {
            const mappedPoints = hand.landmarks.map(mapPointToScreen);
            const accentColor = hand.handedness === 'Left' ? '#5AD1FF' : '#FF7A5C';
            const topLeft = mapPointToScreen({ x: hand.detectionBox.xMin, y: hand.detectionBox.yMin });
            const bottomRight = mapPointToScreen({
              x: hand.detectionBox.xMax,
              y: hand.detectionBox.yMax,
            });
            const boxWidth = Math.max(0, bottomRight.left - topLeft.left);
            const boxHeight = Math.max(0, bottomRight.top - topLeft.top);

            return (
              <View key={hand.id} style={StyleSheet.absoluteFill}>
                <View
                  style={[
                    styles.handBox,
                    {
                      left: topLeft.left,
                      top: topLeft.top,
                      width: boxWidth,
                      height: boxHeight,
                      borderColor: accentColor,
                    },
                  ]}
                />
                <View
                  style={[
                    styles.handLabel,
                    {
                      left: topLeft.left,
                      top: Math.max(0, topLeft.top - 28),
                      borderColor: accentColor,
                      backgroundColor: `${accentColor}33`,
                    },
                  ]}
                >
                  <Text style={styles.handLabelText}>
                    {hand.handedness} · {(hand.qualityScore * 100).toFixed(0)}%
                  </Text>
                  <Text style={styles.handLabelSubText}>
                    {hand.pose.isPinching
                      ? `Pinch ${(hand.pose.pinchStrength * 100).toFixed(0)}%`
                      : `Écart ${(hand.pose.spread * 100).toFixed(0)}%`}
                  </Text>
                </View>

                {HAND_CONNECTIONS.map(([startIdx, endIdx]) => {
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
                      key={`line-${hand.id}-${startIdx}-${endIdx}`}
                      style={[
                        styles.landmarkLine,
                        {
                          width: length,
                          left: centerX - length / 2,
                          top: centerY - LANDMARK_LINE_THICKNESS / 2,
                          transform: [{ rotate: `${angle}deg` }],
                          backgroundColor: accentColor,
                        },
                      ]}
                    />
                  );
                })}

                {mappedPoints.map((point, pointIndex) => (
                  <View
                    key={`point-${hand.id}-${pointIndex}`}
                    style={[
                      styles.landmarkPoint,
                      {
                        left: point.left,
                        top: point.top,
                        backgroundColor: accentColor,
                      },
                    ]}
                  />
                ))}
              </View>
            );
          })}
          {sceneInterpretation?.interHandRelations
            ?.filter((relation) => relation.type !== 'Unknown' && relation.confidence > 0.5)
            .map((relation) => (
              <View
                key={`${relation.handIdA}-${relation.handIdB}-${relation.type}`}
                style={styles.relationBadge}
              >
                <Text style={styles.relationBadgeText}>
                  {relation.type === 'HandshakeCandidate' ? '🤝' : '✋'} {relation.type}
                </Text>
              </View>
            ))}
        </View>
      </View>

      <View style={styles.topOverlay}>
        <Text style={styles.infoText}>Frames traitées: {detectionCount}</Text>
        <Text style={styles.infoText}>Mains détectées: {detectedHandsCount}</Text>
        {sceneInterpretation?.rawMeta.processingTimeMs != null && (
          <Text style={styles.infoText}>
            Interp: {sceneInterpretation.rawMeta.processingTimeMs.toFixed(1)} ms
          </Text>
        )}
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

const clampProbability = (value: number): number => {
  'worklet';
  if (Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
};

const parseLandmarkResult = (outputs: Float32Array[]): MLRawLandmarksOutput => {
  'worklet';
  const landmarks = outputs[0] ?? new Float32Array(63);
  let handednessScore = 0.5;
  let handedness: 'Left' | 'Right' = 'Right';
  let presenceScore: number | undefined;
  let visibility: Float32Array | undefined;
  let scalarConsumed = false;

  for (let i = 1; i < outputs.length; i++) {
    const tensor = outputs[i];
    if (!tensor) continue;
    if (tensor.length === 1) {
      if (!scalarConsumed) {
        presenceScore = clampProbability(tensor[0]);
        scalarConsumed = true;
      } else {
        handednessScore = clampProbability(tensor[0]);
      }
    } else if (tensor.length === 2) {
      handednessScore = clampProbability(tensor[1]);
    } else if (!visibility && (tensor.length === 21 || tensor.length === 63)) {
      visibility = tensor;
    }
  }

  handedness = handednessScore >= 0.5 ? 'Right' : 'Left';

  return {
    landmarks,
    handedness,
    handednessScore,
    presenceScore,
    visibility,
    landmarkScore: presenceScore ?? handednessScore,
  };
};

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
  handBox: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 10,
  },
  handLabel: {
    position: 'absolute',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    borderWidth: 1,
    backgroundColor: '#ffffff40',
  },
  handLabelText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  handLabelSubText: {
    color: 'white',
    fontSize: 10,
  },
  relationBadge: {
    alignSelf: 'center',
    marginTop: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: '#ffffff30',
  },
  relationBadgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
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