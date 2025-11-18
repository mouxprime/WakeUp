import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, StyleSheet, Text, View, TouchableOpacity } from 'react-native';
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

export default function App() {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);

  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  const [detectionCount, setDetectionCount] = useState(0);
  const [lastFrameAnalysis, setLastFrameAnalysis] =
    useState<FrameAnalysisResult | null>(null);
  const [detectedHandsCount, setDetectedHandsCount] = useState(0);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

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

  // On extrait la taille d'entrée du modèle (1, H, W, 3)
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

  // Nouveau frame processor pour la détection de mains
  const handDetectionFrameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';

      if (detectorModel == null || detectorInputSize == null) {
        return;
      }

      const { width, height, dataType } = detectorInputSize;

      runAtTargetFps(10, () => {
        // 1) Resize la frame au format attendu par le modèle
        const resized = resize(frame, {
          scale: {
            width,
            height,
          },
          pixelFormat: 'rgb',
          dataType: dataType === 'float32' ? 'float32' : 'uint8',
        });

        // 2) Inference TFLite synchrone
        const outputs = detectorModel.runSync([resized]);

        // outputs[1] : [1, 2016, 1] → scores (logits)
        const rawScores = outputs[1] as Float32Array;

        // 3) Analyse détaillée des scores pour déboguer
        const sigmoidScores = new Float32Array(rawScores.length);
        for (let i = 0; i < rawScores.length; i++) {
          sigmoidScores[i] = 1 / (1 + Math.exp(-rawScores[i]));
        }

        // Compter combien dépassent différents seuils
        let count_05 = 0, count_07 = 0, count_08 = 0, count_09 = 0;
        let maxScore = 0, secondMaxScore = 0;

        for (let i = 0; i < sigmoidScores.length; i++) {
          const s = sigmoidScores[i];
          maxScore = Math.max(maxScore, s);
          if (s > secondMaxScore && s < maxScore) secondMaxScore = s;

          if (s >= 0.5) count_05++;
          if (s >= 0.7) count_07++;
          if (s >= 0.8) count_08++;
          if (s >= 0.9) count_09++;
        }

        console.log('Score analysis:', {
          maxScore: maxScore.toFixed(3),
          secondMaxScore: secondMaxScore.toFixed(3),
          count_05, count_07, count_08, count_09,
          top5: Array.from(sigmoidScores.slice().sort((a,b)=>b-a).slice(0,5)).map(x=>x.toFixed(3))
        });

        // 4) Heuristique ajustée : utiliser les top scores avec seuils plus stricts
        const sortedScores = Array.from(sigmoidScores).sort((a,b)=>b-a);
        let hands = 0;

        // Si le meilleur score est > 0.8, compter comme 1 main
        if (sortedScores[0] >= 0.8) {
          hands = 1;
          // Si le deuxième meilleur est aussi > 0.9 (beaucoup plus strict), compter comme 2 mains
          if (sortedScores[1] >= 0.9) {
            hands = 2;
          }
        }

        onHandsDetected(hands);
        onFrameProcessed();
      });
    },
    [detectorModel, detectorInputSize, resize, onHandsDetected, onFrameProcessed],
  );

  // Utiliser le nouveau frame processor pour la détection de mains
  const frameProcessor = handDetectionFrameProcessor;

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

  if (!device) {
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
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={!isTakingPhoto}
        photo
        frameProcessor={frameProcessor}
      />

      <View style={styles.topOverlay}>
        <Text style={styles.infoText}>Frames traitées: {detectionCount}</Text>
        <Text style={styles.infoText}>Mains détectées: {detectedHandsCount}</Text>
      </View>

      <View style={styles.bottomOverlay}>
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
  container: { flex: 1, backgroundColor: 'black' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
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
});