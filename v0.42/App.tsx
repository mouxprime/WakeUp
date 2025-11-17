import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';

export default function App() {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);

  const [isTakingPhoto, setIsTakingPhoto] = useState(false);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

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

  const handleTakePhoto = async () => {
    if (!cameraRef.current || isTakingPhoto) return;

    try {
      setIsTakingPhoto(true);
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'balanced',
      });
      console.log('Photo prise :', photo);
      // plus tard : sauvegarde / envoi / affichage
    } catch (err) {
      console.error('Erreur lors de la prise de photo :', err);
    } finally {
      setIsTakingPhoto(false);
    }
  };

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo
      />

      {/* Overlay UI */}
      <View style={styles.overlay}>
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
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  overlay: {
    position: 'absolute',
    bottom: 60,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
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
  shutterButtonDisabled: {
    opacity: 0.5,
  },
  shutterText: {
    color: 'white',
    fontWeight: 'bold',
  },
});
