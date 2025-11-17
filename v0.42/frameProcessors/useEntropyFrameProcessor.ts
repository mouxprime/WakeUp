import { useRunOnJS } from 'react-native-worklets-core';
import { useFrameProcessor } from 'react-native-vision-camera';
import type { Frame } from 'react-native-vision-camera';

type FrameAnalysisResult = {
  timestamp: number;
  averageLuma: number | null;
};

const SAMPLE_TARGET = 1500;

const computeAverageLuma = (frame: Frame): number | null => {
  'worklet';

  if (frame.pixelFormat !== 'rgb' || typeof frame.toArrayBuffer !== 'function') {
    return null;
  }

  const buffer = frame.toArrayBuffer();
  const data = new Uint8Array(buffer);
  const stride = Math.max(1, Math.floor(data.length / SAMPLE_TARGET));

  let sum = 0;
  let sampleCount = 0;

  for (let index = 0; index < data.length; index += stride) {
    sum += data[index];
    sampleCount += 1;
  }

  if (sampleCount === 0) {
    return null;
  }

  return sum / sampleCount;
};

export const useEntropyFrameProcessor = (
  onFrameAnalyzed: (result: FrameAnalysisResult) => void,
) => {
  const runOnJSHandler = useRunOnJS(onFrameAnalyzed, [onFrameAnalyzed]);

  return useFrameProcessor(
    (frame) => {
      'worklet';
      const averageLuma = computeAverageLuma(frame);
      runOnJSHandler({
        averageLuma,
        timestamp: Number(frame.timestamp ?? Date.now()),
      });
    },
    [runOnJSHandler],
  );
};

export type { FrameAnalysisResult };

