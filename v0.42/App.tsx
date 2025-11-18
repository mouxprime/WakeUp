import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import type { FrameAnalysisResult } from './frameProcessors/useEntropyFrameProcessor';
import { useEntropyFrameProcessor } from './frameProcessors/useEntropyFrameProcessor';
import { loadHandDetectorModel, loadHandLandmarksModel } from './ml/handModels';

export default function App() {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);

  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  const [detectionCount, setDetectionCount] = useState(0);
  const [lastFrameAnalysis, setLastFrameAnalysis] = useState<FrameAnalysisResult | null>(null);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    (async () => {
      try {
        const detector = await loadHandDetectorModel();
        const landmarks = await loadHandLandmarksModel();

        console.log('Hand detector model loaded:', {
          inputs: detector.inputs,
          outputs: detector.outputs,
        });

        console.log('Hand landmarks model loaded:', {
          inputs: landmarks.inputs,
          outputs: landmarks.outputs,
        });
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

  const frameProcessor = useEntropyFrameProcessor(handleFrameAnalyzed);

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
        <Text style={styles.infoText}>
          Luminosité moyenne:{' '}
          {lastFrameAnalysis?.averageLuma != null
            ? lastFrameAnalysis.averageLuma.toFixed(0)
            : '—'}
        </Text>
      </View>

      <View style={styles.bottomOverlay}>
        <TouchableOpacity
          style={[styles.shutterButton, isTakingPhoto && styles.shutterButtonDisabled]}
          onPress={handleTakePhoto}
          disabled={isTakingPhoto}
        >
          <Text style={styles.shutterText}>{isTakingPhoto ? '...' : 'SHOT'}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  topOverlay: { position: 'absolute', top: 60, width: '100%', alignItems: 'center' },
  bottomOverlay: { position: 'absolute', bottom: 60, width: '100%', alignItems: 'center' },
  infoText: { color: 'white', fontSize: 14, marginBottom: 4 },
  shutterButton: {
    width: 80, height: 80, borderRadius: 40, borderWidth: 4, borderColor: 'white',
    alignItems: 'center', justifyContent: 'center', backgroundColor: '#ffffff20',
  },
  shutterButtonDisabled: { opacity: 0.5 },
  shutterText: { color: 'white', fontWeight: 'bold' },
});