import React, { useRef, useEffect } from 'react';
import { WebView } from 'react-native-webview';
import { StyleSheet } from 'react-native';

interface MediaPipeWebViewProps {
  onDetectionData: (data: any) => void;
  onTakePhoto: () => void;
  isActive: boolean;
}

export const MediaPipeWebView: React.FC<MediaPipeWebViewProps> = ({
  onDetectionData,
  onTakePhoto,
  isActive,
}) => {
  const webViewRef = useRef<WebView>(null);

  const htmlContent = `
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <style>
    body {
      margin: 0;
      padding: 0;
      background: transparent;
      overflow: hidden;
    }
    #video {
      position: absolute;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      object-fit: cover;
      transform: scaleX(-1);
    }
    #canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>

  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm/vision_bundle.js"></script>

  <script>
    let handLandmarker = null;
    let poseLandmarker = null;
    let video = null;
    let canvas = null;
    let ctx = null;
    let animationFrame = null;

    const initMediaPipe = async () => {
      try {
        // Initialiser les détecteurs MediaPipe
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        // Détecteur de mains
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        // Détecteur de pose
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        console.log('MediaPipe initialisé');
      } catch (error) {
        console.error('Erreur lors de l\\'initialisation de MediaPipe:', error);
      }
    };

    const startCamera = async () => {
      try {
        video = document.getElementById('video');
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          console.log('Caméra démarrée:', video.videoWidth, 'x', video.videoHeight);
          detectFrame();
        };
      } catch (error) {
        console.error('Erreur caméra:', error);
      }
    };

    const detectFrame = async () => {
      if (!handLandmarker || !poseLandmarker || !video || video.readyState !== 4) {
        animationFrame = requestAnimationFrame(detectFrame);
        return;
      }

      try {
        const startTime = Date.now();

        // Détection des mains
        const handResults = await handLandmarker.detectForVideo(video, startTime);

        // Détection de la pose
        const poseResults = await poseLandmarker.detectForVideo(video, startTime);

        // Analyser les contacts mains-pose
        const contacts = analyzeContacts(handResults, poseResults);

        // Préparer les données à envoyer
        const detectionData = {
          timestamp: Date.now(),
          hands: handResults.landmarks.map((hand, index) => ({
            index,
            landmarks: hand.map(point => ({
              x: point.x,
              y: point.y,
              z: point.z
            })),
            handedness: handResults.handednesses[index][0].categoryName
          })),
          pose: poseResults.landmarks.length > 0 ? {
            landmarks: poseResults.landmarks[0].map(point => ({
              x: point.x,
              y: point.y,
              z: point.z,
              visibility: point.visibility
            }))
          } : null,
          contacts
        };

        // Envoyer les données à React Native
        window.ReactNativeWebView.postMessage(JSON.stringify({
          type: 'detection',
          data: detectionData
        }));

        // Déclencher photo si nécessaire (ex: geste spécifique détecté)
        if (shouldTakePhoto(detectionData)) {
          window.ReactNativeWebView.postMessage(JSON.stringify({
            type: 'takePhoto'
          }));
        }

        // Dessiner les résultats pour debug
        drawResults(handResults, poseResults);

      } catch (error) {
        console.error('Erreur détection:', error);
      }

      animationFrame = requestAnimationFrame(detectFrame);
    };

    const analyzeContacts = (handResults, poseResults) => {
      const contacts = [];

      if (handResults.landmarks.length > 0 && poseResults.landmarks.length > 0) {
        const pose = poseResults.landmarks[0];

        handResults.landmarks.forEach((hand, handIndex) => {
          // Vérifier contact avec le visage (points 0-10 du pose)
          const facePoints = pose.slice(0, 11);
          const handPoints = hand.slice(4, 9); // doigts

          handPoints.forEach(handPoint => {
            facePoints.forEach(facePoint => {
              const distance = Math.sqrt(
                Math.pow(handPoint.x - facePoint.x, 2) +
                Math.pow(handPoint.y - facePoint.y, 2)
              );

              if (distance < 0.1) { // seuil de contact
                contacts.push({
                  type: 'face_touch',
                  handIndex,
                  handPoint,
                  facePoint,
                  distance
                });
              }
            });
          });
        });
      }

      return contacts;
    };

    const shouldTakePhoto = (detectionData) => {
      // Logique simple : prendre une photo si les deux mains sont détectées
      return detectionData.hands.length >= 2;
    };

    const drawResults = (handResults, poseResults) => {
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Dessiner les mains
      handResults.landmarks.forEach((hand) => {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;

        // Dessiner les connexions des mains
        const connections = [
          [0,1],[1,2],[2,3],[3,4], // pouce
          [0,5],[5,6],[6,7],[7,8], // index
          [0,9],[9,10],[10,11],[11,12], // majeur
          [0,13],[13,14],[14,15],[15,16], // annulaire
          [0,17],[17,18],[18,19],[19,20], // auriculaire
          [5,9],[9,13],[13,17] // paume
        ];

        connections.forEach(([start, end]) => {
          const startPoint = hand[start];
          const endPoint = hand[end];
          ctx.beginPath();
          ctx.moveTo(startPoint.x * canvas.width, startPoint.y * canvas.height);
          ctx.lineTo(endPoint.x * canvas.width, endPoint.y * canvas.height);
          ctx.stroke();
        });

        // Dessiner les points
        hand.forEach(point => {
          ctx.fillStyle = '#ff0000';
          ctx.beginPath();
          ctx.arc(point.x * canvas.width, point.y * canvas.height, 3, 0, 2 * Math.PI);
          ctx.fill();
        });
      });

      // Dessiner la pose
      if (poseResults.landmarks.length > 0) {
        const pose = poseResults.landmarks[0];
        ctx.strokeStyle = '#0000ff';
        ctx.lineWidth = 2;

        // Connexions de pose simplifiées
        const poseConnections = [
          [11,12],[11,13],[13,15],[15,17],[15,19],[15,21], // bras gauche
          [12,14],[14,16],[16,18],[16,20],[16,22], // bras droit
          [11,23],[12,24],[23,24], // torse
          [23,25],[25,27],[27,29],[27,31], // jambe gauche
          [24,26],[26,28],[28,30],[28,32] // jambe droite
        ];

        poseConnections.forEach(([start, end]) => {
          if (pose[start] && pose[end]) {
            const startPoint = pose[start];
            const endPoint = pose[end];
            ctx.beginPath();
            ctx.moveTo(startPoint.x * canvas.width, startPoint.y * canvas.height);
            ctx.lineTo(endPoint.x * canvas.width, endPoint.y * canvas.height);
            ctx.stroke();
          }
        });
      }
    };

    const stopCamera = () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
      if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
      }
    };

    // Écouter les messages de React Native
    window.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'start') {
        startCamera();
      } else if (message.type === 'stop') {
        stopCamera();
      }
    });

    // Initialisation
    initMediaPipe();
  </script>
</body>
</html>`;

  useEffect(() => {
    if (isActive) {
      // Démarrer la détection
      webViewRef.current?.injectJavaScript(`
        window.postMessage(JSON.stringify({ type: 'start' }));
        true;
      `);
    } else {
      // Arrêter la détection
      webViewRef.current?.injectJavaScript(`
        window.postMessage(JSON.stringify({ type: 'stop' }));
        true;
      `);
    }
  }, [isActive]);

  const handleMessage = (event: any) => {
    try {
      const message = JSON.parse(event.nativeEvent.data);

      if (message.type === 'detection') {
        onDetectionData(message.data);
      } else if (message.type === 'takePhoto') {
        onTakePhoto();
      }
    } catch (error) {
      console.error('Erreur parsing message WebView:', error);
    }
  };

  return (
    <WebView
      ref={webViewRef}
      source={{ html: htmlContent }}
      style={styles.webview}
      onMessage={handleMessage}
      javaScriptEnabled={true}
      domStorageEnabled={true}
      allowsInlineMediaPlayback={true}
      mediaPlaybackRequiresUserAction={false}
      mixedContentMode="always"
      originWhitelist={['*']}
      onError={(syntheticEvent) => {
        const { nativeEvent } = syntheticEvent;
        console.warn('WebView error: ', nativeEvent);
      }}
      onHttpError={(syntheticEvent) => {
        const { nativeEvent } = syntheticEvent;
        console.warn('WebView HTTP error: ', nativeEvent);
      }}
    />
  );
};

const styles = StyleSheet.create({
  webview: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'transparent',
  },
});
