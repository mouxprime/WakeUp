import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';

const HAND_DETECTOR_SRC = require('../assets/models/hand_detector.tflite');
const HAND_LANDMARKS_SRC = require('../assets/models/hand_landmarks_detector.tflite');

export function loadHandDetectorModel(): Promise<TensorflowModel> {
  return loadTensorflowModel(HAND_DETECTOR_SRC);
}

export function loadHandLandmarksModel(): Promise<TensorflowModel> {
  return loadTensorflowModel(HAND_LANDMARKS_SRC);
}

